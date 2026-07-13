// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Sticky session routing with pluggable affinity storage.
//!
//! Provides router-side session affinity so that all requests within
//! a multi-turn session are routed to the same `(worker, dp_rank)`. The
//! affinity store is trait-based: the default [`InMemoryAffinityStore`] uses a
//! `DashMap` with a background reaper, but implementations backed by
//! Redis, etcd, or NATS KV can be swapped in for multi-router deployments.
//!
//! Affinity is bound at `(worker, dp_rank)` granularity so that multi-DP-rank
//! engines (e.g. SGLang DEP) keep a conversation pinned to a single DP rank,
//! which is where its prefix stays warm in the rank-local radix cache. This is
//! purely a routing-layer decision -- no RPC is sent to the worker.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use dashmap::DashMap;
use dynamo_kv_router::protocols::WorkerWithDpRank;

/// Interval between sweeps of the background reaper that removes expired entries.
const REAPER_INTERVAL: Duration = Duration::from_secs(30);
static NEXT_AFFINITY_REVISION: AtomicU64 = AtomicU64::new(1);

type ExpiryHandler = Arc<dyn Fn(String, u64) + Send + Sync>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AffinityKind {
    RouterOnly,
    EngineBacked,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AffinityBinding {
    pub worker: WorkerWithDpRank,
    pub kind: AffinityKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AffinityBindingToken {
    pub binding: AffinityBinding,
    pub revision: u64,
}

/// Token for a pending compare-and-rebind transition.
///
/// The old binding remains visible until this token is committed. Both commit
/// and rollback compare the old and pending revisions, so a concurrent
/// bind/remove always wins without being overwritten by the transition.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AffinityRebindToken {
    pub previous: AffinityBindingToken,
    pub pending: AffinityBindingToken,
}

/// Trait for session affinity storage backends.
///
/// Stores `(worker, dp_rank)` as an atomic routing target. `get` is the
/// action-less-turn path and refreshes sliding TTL; `peek` is for lifecycle
/// actions that must not extend affinity until the worker confirms success.
pub trait AffinityStore: Send + Sync {
    /// Look up the `(worker, dp_rank)` for a session. Returns `None` if unknown
    /// or expired. Implementations should refresh the TTL on hit.
    fn get(&self, session_id: &str) -> Option<WorkerWithDpRank>;

    /// Look up the `(worker, dp_rank)` for a session without refreshing TTL.
    fn peek(&self, session_id: &str) -> Option<WorkerWithDpRank>;

    /// Return the current binding revision without refreshing TTL. Unlike a
    /// rebind candidate lookup, this remains available while a shadow exists.
    /// This is required because conditional affinity cleanup is part of core
    /// sticky routing; an implementation must not silently degrade to a miss.
    fn peek_binding_token(&self, session_id: &str) -> Option<AffinityBindingToken>;

    /// Return the current binding token only when no rebind is already pending.
    /// The default disables rebind for stores without atomic transition support.
    fn peek_rebind_token(&self, _session_id: &str) -> Option<AffinityBindingToken> {
        None
    }

    /// Bind a session to a `(worker, dp_rank)` with the given TTL and kind.
    fn put(
        &self,
        session_id: &str,
        worker: WorkerWithDpRank,
        ttl: Duration,
        kind: AffinityKind,
    ) -> AffinityBindingToken;

    /// Remove a session binding and return its metadata.
    fn remove(&self, session_id: &str) -> Option<AffinityBinding>;

    /// Remove a binding only if it is still the binding created by this attempt.
    fn remove_if_token(&self, session_id: &str, token: AffinityBindingToken) -> bool;

    /// Atomically start a shadow rebind if `expected` is still current.
    fn begin_rebind(
        &self,
        _session_id: &str,
        _expected: AffinityBindingToken,
        _cold: WorkerWithDpRank,
        _ttl: Duration,
    ) -> Option<AffinityRebindToken> {
        None
    }

    /// Publish the pending cold binding if the transition is still current.
    fn commit_rebind(&self, _session_id: &str, _token: AffinityRebindToken) -> bool {
        false
    }

    /// Remove the pending cold binding if the transition is still current.
    fn rollback_rebind(&self, _session_id: &str, _token: AffinityRebindToken) -> bool {
        false
    }
}

struct PendingAffinityBinding {
    token: AffinityBindingToken,
    ttl: Duration,
}

/// In-memory affinity entry with sliding-window TTL.
struct AffinityEntry {
    worker: WorkerWithDpRank,
    ttl: Duration,
    expires_at: Instant,
    kind: AffinityKind,
    revision: u64,
    pending_rebind: Option<PendingAffinityBinding>,
}

impl AffinityEntry {
    fn binding(&self) -> AffinityBinding {
        AffinityBinding {
            worker: self.worker,
            kind: self.kind,
        }
    }

    fn token(&self) -> AffinityBindingToken {
        AffinityBindingToken {
            binding: self.binding(),
            revision: self.revision,
        }
    }
}

/// Default in-memory affinity store backed by `DashMap`.
///
/// A background tokio task sweeps expired entries every [`REAPER_INTERVAL`].
#[derive(Clone)]
pub struct InMemoryAffinityStore {
    map: Arc<DashMap<String, AffinityEntry>>,
    on_expire: Option<ExpiryHandler>,
}

impl Default for InMemoryAffinityStore {
    fn default() -> Self {
        Self::new()
    }
}

impl InMemoryAffinityStore {
    pub fn new() -> Self {
        Self::new_with_on_expire(None)
    }

    pub fn new_with_on_expire(on_expire: Option<ExpiryHandler>) -> Self {
        let map = Arc::new(DashMap::new());

        let store = InMemoryAffinityStore { map, on_expire };

        let reaper_store = store.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(REAPER_INTERVAL);
            loop {
                interval.tick().await;
                reaper_store.reap_expired(Instant::now());
            }
        });

        store
    }

    fn reap_expired(&self, now: Instant) {
        let on_expire = self.on_expire.clone();
        self.map.retain(|session_id, entry: &mut AffinityEntry| {
            let alive = entry.expires_at > now;
            if !alive {
                tracing::debug!(%session_id, "Session affinity expired, removing");
                if entry.kind == AffinityKind::EngineBacked
                    && let Some(handler) = &on_expire
                {
                    handler(session_id.clone(), entry.worker.worker_id);
                }
            }
            alive
        });
    }

    fn lookup_token(
        &self,
        session_id: &str,
        refresh: bool,
        require_no_pending_rebind: bool,
    ) -> Option<AffinityBindingToken> {
        let now = Instant::now();
        let mut entry = self.map.get_mut(session_id)?;
        if entry.expires_at <= now {
            let binding = entry.binding();
            let expires_at = entry.expires_at;
            drop(entry);
            self.remove_expired_if_current(session_id, binding, expires_at);
            return None;
        }

        if require_no_pending_rebind && entry.pending_rebind.is_some() {
            return None;
        }

        let token = entry.token();
        if refresh {
            entry.expires_at = now + entry.ttl;
        }
        tracing::info!(
            %session_id,
            worker_id = token.binding.worker.worker_id,
            dp_rank = token.binding.worker.dp_rank,
            refreshed = refresh,
            "Sticky session hit"
        );
        Some(token)
    }

    fn remove_expired_if_current(
        &self,
        session_id: &str,
        binding: AffinityBinding,
        expires_at: Instant,
    ) {
        let removed = self.map.remove_if(session_id, |_, entry| {
            entry.worker == binding.worker
                && entry.expires_at == expires_at
                && entry.expires_at <= Instant::now()
        });
        if removed.is_none() {
            return;
        }

        tracing::debug!(%session_id, "Session affinity expired during resolve");
        if binding.kind == AffinityKind::EngineBacked
            && let Some(handler) = &self.on_expire
        {
            handler(session_id.to_owned(), binding.worker.worker_id);
        }
    }
}

impl AffinityStore for InMemoryAffinityStore {
    fn get(&self, session_id: &str) -> Option<WorkerWithDpRank> {
        self.lookup_token(session_id, true, false)
            .map(|token| token.binding.worker)
    }

    fn peek(&self, session_id: &str) -> Option<WorkerWithDpRank> {
        self.lookup_token(session_id, false, false)
            .map(|token| token.binding.worker)
    }

    fn peek_binding_token(&self, session_id: &str) -> Option<AffinityBindingToken> {
        self.lookup_token(session_id, false, false)
    }

    fn peek_rebind_token(&self, session_id: &str) -> Option<AffinityBindingToken> {
        self.lookup_token(session_id, false, true)
    }

    fn put(
        &self,
        session_id: &str,
        worker: WorkerWithDpRank,
        ttl: Duration,
        kind: AffinityKind,
    ) -> AffinityBindingToken {
        let revision = NEXT_AFFINITY_REVISION.fetch_add(1, Ordering::Relaxed);
        let binding = AffinityBinding { worker, kind };
        self.map.insert(
            session_id.to_owned(),
            AffinityEntry {
                worker,
                ttl,
                expires_at: Instant::now() + ttl,
                kind,
                revision,
                pending_rebind: None,
            },
        );
        AffinityBindingToken { binding, revision }
    }

    fn remove(&self, session_id: &str) -> Option<AffinityBinding> {
        self.map
            .remove(session_id)
            .map(|(_, entry)| entry.binding())
    }

    fn remove_if_token(&self, session_id: &str, token: AffinityBindingToken) -> bool {
        self.map
            .remove_if(session_id, |_, entry| {
                entry.revision == token.revision && entry.binding() == token.binding
            })
            .is_some()
    }

    fn begin_rebind(
        &self,
        session_id: &str,
        expected: AffinityBindingToken,
        cold: WorkerWithDpRank,
        ttl: Duration,
    ) -> Option<AffinityRebindToken> {
        let now = Instant::now();
        let mut entry = self.map.get_mut(session_id)?;
        if entry.expires_at <= now || entry.token() != expected || entry.pending_rebind.is_some() {
            return None;
        }

        let pending = AffinityBindingToken {
            binding: AffinityBinding {
                worker: cold,
                kind: expected.binding.kind,
            },
            revision: NEXT_AFFINITY_REVISION.fetch_add(1, Ordering::Relaxed),
        };
        entry.pending_rebind = Some(PendingAffinityBinding {
            token: pending,
            ttl,
        });
        Some(AffinityRebindToken {
            previous: expected,
            pending,
        })
    }

    fn commit_rebind(&self, session_id: &str, token: AffinityRebindToken) -> bool {
        let now = Instant::now();
        let Some(mut entry) = self.map.get_mut(session_id) else {
            return false;
        };
        if entry.expires_at <= now || entry.token() != token.previous {
            return false;
        }
        let Some(pending) = entry.pending_rebind.as_ref() else {
            return false;
        };
        if pending.token != token.pending {
            return false;
        }

        let pending = entry
            .pending_rebind
            .take()
            .expect("pending rebind checked above");
        entry.worker = pending.token.binding.worker;
        entry.kind = pending.token.binding.kind;
        entry.revision = pending.token.revision;
        entry.ttl = pending.ttl;
        entry.expires_at = now + pending.ttl;
        true
    }

    fn rollback_rebind(&self, session_id: &str, token: AffinityRebindToken) -> bool {
        let Some(mut entry) = self.map.get_mut(session_id) else {
            return false;
        };
        if entry.token() != token.previous
            || entry.pending_rebind.as_ref().map(|pending| pending.token) != Some(token.pending)
        {
            return false;
        }

        entry.pending_rebind = None;
        true
    }
}

/// Routes requests to workers based on session affinity.
///
/// Wraps an [`AffinityStore`] and provides session-id-level affinity helpers.
pub struct StickySessionRouter {
    store: Box<dyn AffinityStore>,
    /// Last Layer-2 rebind time per session, used by [`Self::in_rebind_cooldown`]
    /// to throttle thrash. Kept as a side-map (rather than a field on
    /// `AffinityEntry`) so it does not touch the entry's `Eq`/serde contract.
    /// Entries are best-effort: a session that expires from `store` leaves a
    /// stale timestamp here that is overwritten on the next rebind; this never
    /// resurrects affinity, only gates a future rebind, so no separate reaper
    /// is required.
    last_rebind: DashMap<String, Instant>,
    rebind_counts: DashMap<String, u32>,
}

impl StickySessionRouter {
    pub fn new(store: impl AffinityStore + 'static) -> Self {
        tracing::debug!("StickySessionRouter initialized");
        StickySessionRouter {
            store: Box::new(store),
            last_rebind: DashMap::new(),
            rebind_counts: DashMap::new(),
        }
    }

    /// Resolve a session id directly and refresh its sticky TTL.
    pub fn resolve_session(&self, session_id: &str) -> Option<WorkerWithDpRank> {
        self.store.get(session_id)
    }

    /// Resolve a session id directly without refreshing its sticky TTL.
    pub fn peek_session(&self, session_id: &str) -> Option<WorkerWithDpRank> {
        self.store.peek(session_id)
    }

    /// Return the visible binding and its revision, including while a shadow
    /// transition is pending.
    pub(crate) fn peek_binding_token(&self, session_id: &str) -> Option<AffinityBindingToken> {
        self.store.peek_binding_token(session_id)
    }

    /// Return a revision token suitable for an atomic rebind decision.
    /// A session with another rebind already in flight is not a candidate.
    pub(crate) fn peek_rebind_token(&self, session_id: &str) -> Option<AffinityBindingToken> {
        self.store.peek_rebind_token(session_id)
    }

    /// Bind a session to a `(worker, dp_rank)` with the given TTL and kind.
    pub fn bind(
        &self,
        session_id: &str,
        worker: WorkerWithDpRank,
        ttl: Duration,
        kind: AffinityKind,
    ) -> AffinityBindingToken {
        tracing::info!(
            %session_id,
            worker_id = worker.worker_id,
            dp_rank = worker.dp_rank,
            ttl_secs = ttl.as_secs(),
            kind = ?kind,
            "Binding session affinity"
        );
        self.store.put(session_id, worker, ttl, kind)
    }

    /// Start a shadow rebind without changing the target returned by resolve.
    /// The expected revision makes this a compare-and-set operation.
    pub(crate) fn begin_rebind(
        &self,
        session_id: &str,
        expected: AffinityBindingToken,
        cold: WorkerWithDpRank,
        ttl: Duration,
    ) -> Option<AffinityRebindToken> {
        if expected.binding.kind == AffinityKind::EngineBacked
            && expected.binding.worker.worker_id != cold.worker_id
        {
            tracing::warn!(
                %session_id,
                hot_worker_id = expected.binding.worker.worker_id,
                hot_dp_rank = expected.binding.worker.dp_rank,
                cold_worker_id = cold.worker_id,
                cold_dp_rank = cold.dp_rank,
                "Refusing cross-worker rebind for engine-backed session"
            );
            return None;
        }

        let token = self.store.begin_rebind(session_id, expected, cold, ttl)?;
        tracing::info!(
            %session_id,
            old_worker_id = expected.binding.worker.worker_id,
            old_dp_rank = expected.binding.worker.dp_rank,
            new_worker_id = cold.worker_id,
            new_dp_rank = cold.dp_rank,
            kind = ?expected.binding.kind,
            "Sticky rebind prepared as shadow transition"
        );
        Some(token)
    }

    /// Atomically publish a previously prepared shadow rebind.
    pub(crate) fn commit_rebind(&self, session_id: &str, token: AffinityRebindToken) -> bool {
        if !self.store.commit_rebind(session_id, token) {
            return false;
        }
        self.last_rebind
            .insert(session_id.to_owned(), Instant::now());
        *self.rebind_counts.entry(session_id.to_owned()).or_insert(0) += 1;
        tracing::info!(
            %session_id,
            old_worker_id = token.previous.binding.worker.worker_id,
            old_dp_rank = token.previous.binding.worker.dp_rank,
            new_worker_id = token.pending.binding.worker.worker_id,
            new_dp_rank = token.pending.binding.worker.dp_rank,
            kind = ?token.pending.binding.kind,
            "Sticky rebind committed"
        );
        true
    }

    /// Discard a shadow rebind without disturbing a concurrent newer binding.
    ///
    /// Resets `last_rebind` to now so the cooldown window starts from the
    /// rollback time, not from the original initiation.  Without this reset,
    /// a session whose rebind always rolls back accumulates zero cooldown time
    /// and can be rebinded repeatedly, creating a cold-prefill cascade.
    pub(crate) fn rollback_rebind(&self, session_id: &str, token: AffinityRebindToken) -> bool {
        let rolled = self.store.rollback_rebind(session_id, token);
        // Reset the cooldown clock from the rollback instant so the trigger
        // cannot fire again until the full cooldown has elapsed from *here*.
        self.last_rebind
            .insert(session_id.to_owned(), Instant::now());
        tracing::info!(
            %session_id,
            "Layer-2 rebind cooldown reset on rollback"
        );
        rolled
    }

    /// Number of committed rebinds for this session (bounce cap input).
    pub fn rebind_count(&self, session_id: &str) -> u32 {
        self.rebind_counts.get(session_id).map(|c| *c).unwrap_or(0)
    }

    /// Returns true if this session was rebound less than `cooldown` ago.
    /// Used by the Layer-2 trigger to avoid rebind thrash.
    pub fn in_rebind_cooldown(&self, session_id: &str, cooldown: Duration) -> bool {
        self.last_rebind
            .get(session_id)
            .map(|last| last.elapsed() < cooldown)
            .unwrap_or(false)
    }

    /// Remove a session binding.
    pub fn unbind(&self, session_id: &str) -> Option<AffinityBinding> {
        tracing::info!(%session_id, "Removing session affinity");
        self.store.remove(session_id)
    }

    pub(super) fn unbind_if_token(&self, session_id: &str, token: AffinityBindingToken) -> bool {
        self.store.remove_if_token(session_id, token)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;

    fn worker(worker_id: u64, dp_rank: u32) -> WorkerWithDpRank {
        WorkerWithDpRank::new(worker_id, dp_rank)
    }

    #[test]
    fn resolve_returns_none_for_unknown_session() {
        let store = InMemoryAffinityStore {
            map: Arc::new(DashMap::new()),
            on_expire: None,
        };
        let router = StickySessionRouter::new(store);
        assert!(router.resolve_session("unknown-session").is_none());
    }

    #[test]
    fn bind_then_resolve_returns_worker_and_dp_rank() {
        let store = InMemoryAffinityStore {
            map: Arc::new(DashMap::new()),
            on_expire: None,
        };
        let router = StickySessionRouter::new(store);
        router.bind(
            "sess-1",
            worker(42, 3),
            Duration::from_secs(300),
            AffinityKind::EngineBacked,
        );

        assert_eq!(router.resolve_session("sess-1"), Some(worker(42, 3)));
    }

    #[test]
    fn peek_returns_worker_without_refreshing_ttl() {
        let map = Arc::new(DashMap::new());
        let ttl = Duration::from_secs(60);
        let expires_at = Instant::now() + Duration::from_secs(5);
        map.insert(
            "sess-peek".to_owned(),
            AffinityEntry {
                worker: worker(7, 2),
                ttl,
                expires_at,
                kind: AffinityKind::EngineBacked,
                revision: 1,
                pending_rebind: None,
            },
        );
        let store = InMemoryAffinityStore {
            map: map.clone(),
            on_expire: None,
        };
        let router = StickySessionRouter::new(store);

        assert_eq!(router.peek_session("sess-peek"), Some(worker(7, 2)));

        let entry = map.get("sess-peek").unwrap();
        assert_eq!(entry.expires_at, expires_at);
    }

    #[test]
    fn bind_overwrites_worker_rank_and_ttl() {
        let map = Arc::new(DashMap::new());
        let store = InMemoryAffinityStore {
            map: map.clone(),
            on_expire: None,
        };
        let router = StickySessionRouter::new(store);
        router.bind(
            "sess-1",
            worker(1, 0),
            Duration::from_secs(10),
            AffinityKind::EngineBacked,
        );
        router.bind(
            "sess-1",
            worker(2, 3),
            Duration::from_secs(90),
            AffinityKind::RouterOnly,
        );

        assert_eq!(router.peek_session("sess-1"), Some(worker(2, 3)));

        let entry = map.get("sess-1").unwrap();
        assert_eq!(entry.worker, worker(2, 3));
        assert_eq!(entry.ttl, Duration::from_secs(90));
        assert_eq!(entry.kind, AffinityKind::RouterOnly);
        assert!(entry.expires_at > Instant::now() + Duration::from_secs(80));
    }

    #[test]
    fn shadow_rebind_stays_hot_until_atomic_commit() {
        let map = Arc::new(DashMap::new());
        let store = InMemoryAffinityStore {
            map: map.clone(),
            on_expire: None,
        };
        let router = StickySessionRouter::new(store);
        let expected = router.bind(
            "sess-1",
            worker(42, 3),
            Duration::from_secs(300),
            AffinityKind::EngineBacked,
        );

        let transition = router
            .begin_rebind("sess-1", expected, worker(42, 7), Duration::from_secs(300))
            .unwrap();

        assert_eq!(router.peek_session("sess-1"), Some(worker(42, 3)));
        assert!(router.peek_rebind_token("sess-1").is_none());
        assert!(router.commit_rebind("sess-1", transition));
        assert_eq!(router.peek_session("sess-1"), Some(worker(42, 7)));
        let entry = map.get("sess-1").unwrap();
        assert_eq!(entry.worker, worker(42, 7));
        assert_eq!(entry.kind, AffinityKind::EngineBacked);
        assert!(entry.pending_rebind.is_none());
        assert!(router.in_rebind_cooldown("sess-1", Duration::from_secs(5)));
    }

    /// The Layer-2 rebind trigger gates on a stable existing binding token.
    /// A fresh session is a MISS and must not trigger a rebind; an already-bound
    /// session is a HIT and is eligible when no transition is pending.
    #[test]
    fn rebind_gate_fires_on_existing_binding_not_on_fresh_session() {
        let store = InMemoryAffinityStore {
            map: Arc::new(DashMap::new()),
            on_expire: None,
        };
        let router = StickySessionRouter::new(store);

        // Fresh session (a brand-new conversation's first turn): no prior home
        // rank -> peek is a MISS -> the gate short-circuits, no rebind.
        assert!(
            router.peek_session("fresh-session").is_none(),
            "a never-bound session must be a MISS so the rebind gate skips it"
        );

        // Bind it (simulating a prior turn's on_routed), then the same session
        // is a HIT and the rebind gate would proceed.
        router.bind(
            "bound-session",
            worker(11, 2),
            Duration::from_secs(300),
            AffinityKind::RouterOnly,
        );
        assert_eq!(
            router.peek_session("bound-session"),
            Some(worker(11, 2)),
            "an already-bound session must be a HIT so the rebind gate proceeds"
        );

        // And the peek used by the gate must NOT have created a binding for the
        // fresh session as a side effect.
        assert!(
            router.peek_session("fresh-session").is_none(),
            "gate peek must be side-effect-free (no implicit bind on MISS)"
        );
    }

    #[test]
    fn stale_rebind_revision_does_not_overwrite_newer_binding() {
        let store = InMemoryAffinityStore {
            map: Arc::new(DashMap::new()),
            on_expire: None,
        };
        let router = StickySessionRouter::new(store);
        let stale = router.bind(
            "sess-1",
            worker(1, 0),
            Duration::from_secs(300),
            AffinityKind::RouterOnly,
        );
        router.bind(
            "sess-1",
            worker(2, 0),
            Duration::from_secs(300),
            AffinityKind::RouterOnly,
        );

        assert!(
            router
                .begin_rebind("sess-1", stale, worker(3, 0), Duration::from_secs(300),)
                .is_none()
        );
        assert_eq!(router.peek_session("sess-1"), Some(worker(2, 0)));
    }

    #[test]
    fn shadow_rebind_rollback_preserves_hot_and_not_newer_binding() {
        let store = InMemoryAffinityStore {
            map: Arc::new(DashMap::new()),
            on_expire: None,
        };
        let router = StickySessionRouter::new(store);
        let expected = router.bind(
            "sess-1",
            worker(1, 0),
            Duration::from_secs(300),
            AffinityKind::RouterOnly,
        );
        let transition = router
            .begin_rebind("sess-1", expected, worker(1, 1), Duration::from_secs(300))
            .unwrap();

        assert!(router.rollback_rebind("sess-1", transition));
        assert_eq!(router.peek_session("sess-1"), Some(worker(1, 0)));
        assert!(router.peek_rebind_token("sess-1").is_some());
        assert!(!router.in_rebind_cooldown("sess-1", Duration::from_secs(5)));

        let transition = router
            .begin_rebind("sess-1", expected, worker(1, 1), Duration::from_secs(300))
            .unwrap();
        router.bind(
            "sess-1",
            worker(9, 0),
            Duration::from_secs(300),
            AffinityKind::RouterOnly,
        );
        assert!(!router.rollback_rebind("sess-1", transition));
        assert_eq!(router.peek_session("sess-1"), Some(worker(9, 0)));
    }

    #[test]
    fn shadow_commit_does_not_overwrite_concurrent_put_or_remove() {
        let store = InMemoryAffinityStore {
            map: Arc::new(DashMap::new()),
            on_expire: None,
        };
        let router = StickySessionRouter::new(store);
        let expected = router.bind(
            "sess-put",
            worker(1, 0),
            Duration::from_secs(300),
            AffinityKind::RouterOnly,
        );
        let transition = router
            .begin_rebind("sess-put", expected, worker(1, 1), Duration::from_secs(300))
            .unwrap();
        router.bind(
            "sess-put",
            worker(9, 0),
            Duration::from_secs(300),
            AffinityKind::RouterOnly,
        );
        assert!(!router.commit_rebind("sess-put", transition));
        assert_eq!(router.peek_session("sess-put"), Some(worker(9, 0)));

        let expected = router.bind(
            "sess-remove",
            worker(2, 0),
            Duration::from_secs(300),
            AffinityKind::RouterOnly,
        );
        let transition = router
            .begin_rebind(
                "sess-remove",
                expected,
                worker(2, 1),
                Duration::from_secs(300),
            )
            .unwrap();
        assert!(router.unbind("sess-remove").is_some());
        assert!(!router.commit_rebind("sess-remove", transition));
        assert_eq!(router.peek_session("sess-remove"), None);
    }

    #[test]
    fn engine_backed_cross_worker_rebind_fails_closed() {
        let store = InMemoryAffinityStore {
            map: Arc::new(DashMap::new()),
            on_expire: None,
        };
        let router = StickySessionRouter::new(store);
        let expected = router.bind(
            "sess-1",
            worker(1, 0),
            Duration::from_secs(300),
            AffinityKind::EngineBacked,
        );

        assert!(
            router
                .begin_rebind("sess-1", expected, worker(2, 0), Duration::from_secs(300),)
                .is_none()
        );
        assert_eq!(router.peek_session("sess-1"), Some(worker(1, 0)));
    }

    #[test]
    fn rollback_token_does_not_remove_newer_binding() {
        let store = InMemoryAffinityStore {
            map: Arc::new(DashMap::new()),
            on_expire: None,
        };
        let router = StickySessionRouter::new(store);
        let stale = router.bind(
            "sess-1",
            worker(1, 0),
            Duration::from_secs(30),
            AffinityKind::RouterOnly,
        );
        router.bind(
            "sess-1",
            worker(2, 0),
            Duration::from_secs(30),
            AffinityKind::RouterOnly,
        );

        assert!(!router.unbind_if_token("sess-1", stale));
        assert_eq!(router.peek_session("sess-1"), Some(worker(2, 0)));
    }

    #[test]
    fn rollback_token_removes_its_own_binding() {
        let store = InMemoryAffinityStore {
            map: Arc::new(DashMap::new()),
            on_expire: None,
        };
        let router = StickySessionRouter::new(store);
        let token = router.bind(
            "sess-1",
            worker(1, 0),
            Duration::from_secs(30),
            AffinityKind::RouterOnly,
        );

        assert!(router.unbind_if_token("sess-1", token));
        assert_eq!(router.peek_session("sess-1"), None);
    }

    #[test]
    fn unbind_removes_affinity() {
        let store = InMemoryAffinityStore {
            map: Arc::new(DashMap::new()),
            on_expire: None,
        };
        let router = StickySessionRouter::new(store);
        router.bind(
            "sess-1",
            worker(42, 1),
            Duration::from_secs(300),
            AffinityKind::EngineBacked,
        );
        assert_eq!(
            router.unbind("sess-1"),
            Some(AffinityBinding {
                worker: worker(42, 1),
                kind: AffinityKind::EngineBacked,
            })
        );

        assert!(router.resolve_session("sess-1").is_none());
    }

    #[test]
    fn expired_entry_returns_none() {
        let store = InMemoryAffinityStore {
            map: Arc::new(DashMap::new()),
            on_expire: None,
        };
        // Insert with zero TTL so it's already expired
        store.map.insert(
            "sess-expired".to_owned(),
            AffinityEntry {
                worker: worker(99, 0),
                ttl: Duration::from_secs(0),
                expires_at: Instant::now() - Duration::from_secs(1),
                kind: AffinityKind::EngineBacked,
                revision: 1,
                pending_rebind: None,
            },
        );
        let router = StickySessionRouter::new(store);

        assert!(router.resolve_session("sess-expired").is_none());
        // Entry should be cleaned up
        assert!(router.store.peek("sess-expired").is_none());
    }

    #[test]
    fn resolve_refreshes_ttl() {
        let map = Arc::new(DashMap::new());
        let ttl = Duration::from_secs(60);
        map.insert(
            "sess-refresh".to_owned(),
            AffinityEntry {
                worker: worker(7, 2),
                ttl,
                // Expires in 5 seconds (simulating time passing since bind)
                expires_at: Instant::now() + Duration::from_secs(5),
                kind: AffinityKind::EngineBacked,
                revision: 1,
                pending_rebind: None,
            },
        );
        let store = InMemoryAffinityStore {
            map: map.clone(),
            on_expire: None,
        };
        let router = StickySessionRouter::new(store);

        assert_eq!(router.resolve_session("sess-refresh"), Some(worker(7, 2)));

        // After resolve, expires_at should be refreshed to now + ttl (60s),
        // so it should be at least 50s from now (not the original 5s).
        let entry = map.get("sess-refresh").unwrap();
        let remaining = entry.expires_at.duration_since(Instant::now());
        assert!(
            remaining > Duration::from_secs(50),
            "TTL should have been refreshed, but remaining={remaining:?}"
        );
    }

    #[test]
    fn expired_entry_triggers_close_callback_on_resolve() {
        let expired_sessions = Arc::new(Mutex::new(Vec::new()));
        let on_expire = {
            let expired_sessions = expired_sessions.clone();
            Arc::new(move |session_id: String, worker_id: u64| {
                expired_sessions
                    .lock()
                    .unwrap()
                    .push((session_id, worker_id));
            })
        };
        let store = InMemoryAffinityStore {
            map: Arc::new(DashMap::new()),
            on_expire: Some(on_expire),
        };
        store.map.insert(
            "sess-expired".to_owned(),
            AffinityEntry {
                worker: worker(99, 0),
                ttl: Duration::from_secs(0),
                expires_at: Instant::now() - Duration::from_secs(1),
                kind: AffinityKind::EngineBacked,
                revision: 1,
                pending_rebind: None,
            },
        );
        let router = StickySessionRouter::new(store);

        assert!(router.resolve_session("sess-expired").is_none());
        assert_eq!(
            expired_sessions.lock().unwrap().as_slice(),
            &[("sess-expired".to_string(), 99)]
        );
    }

    #[test]
    fn expired_router_only_entry_drops_without_close_callback_on_resolve() {
        let expired_sessions = Arc::new(Mutex::new(Vec::new()));
        let on_expire = {
            let expired_sessions = expired_sessions.clone();
            Arc::new(move |session_id: String, worker_id: u64| {
                expired_sessions
                    .lock()
                    .unwrap()
                    .push((session_id, worker_id));
            })
        };
        let store = InMemoryAffinityStore {
            map: Arc::new(DashMap::new()),
            on_expire: Some(on_expire),
        };
        store.map.insert(
            "sess-router-only".to_owned(),
            AffinityEntry {
                worker: worker(11, 0),
                ttl: Duration::from_secs(0),
                expires_at: Instant::now() - Duration::from_secs(1),
                kind: AffinityKind::RouterOnly,
                revision: 1,
                pending_rebind: None,
            },
        );
        let router = StickySessionRouter::new(store);

        assert!(router.resolve_session("sess-router-only").is_none());
        assert!(expired_sessions.lock().unwrap().is_empty());
    }

    #[test]
    fn expired_lookup_does_not_remove_newer_binding() {
        let expired_sessions = Arc::new(Mutex::new(Vec::new()));
        let on_expire = {
            let expired_sessions = expired_sessions.clone();
            Arc::new(move |session_id: String, worker_id: u64| {
                expired_sessions
                    .lock()
                    .unwrap()
                    .push((session_id, worker_id));
            })
        };
        let store = InMemoryAffinityStore {
            map: Arc::new(DashMap::new()),
            on_expire: Some(on_expire),
        };
        store.map.insert(
            "sess-race".to_owned(),
            AffinityEntry {
                worker: worker(1, 0),
                ttl: Duration::from_secs(1),
                expires_at: Instant::now() - Duration::from_secs(1),
                kind: AffinityKind::EngineBacked,
                revision: 1,
                pending_rebind: None,
            },
        );

        let stale = store.map.get("sess-race").unwrap();
        let stale_binding = stale.binding();
        let stale_expires_at = stale.expires_at;
        drop(stale);

        store.put(
            "sess-race",
            worker(2, 1),
            Duration::from_secs(300),
            AffinityKind::EngineBacked,
        );
        store.remove_expired_if_current("sess-race", stale_binding, stale_expires_at);

        assert_eq!(store.peek("sess-race"), Some(worker(2, 1)));
        assert!(expired_sessions.lock().unwrap().is_empty());
    }

    #[test]
    fn reaper_triggers_close_callback_for_expired_entry() {
        let expired_sessions = Arc::new(Mutex::new(Vec::new()));
        let on_expire = {
            let expired_sessions = expired_sessions.clone();
            Arc::new(move |session_id: String, worker_id: u64| {
                expired_sessions
                    .lock()
                    .unwrap()
                    .push((session_id, worker_id));
            })
        };
        let store = InMemoryAffinityStore {
            map: Arc::new(DashMap::new()),
            on_expire: Some(on_expire),
        };
        store.map.insert(
            "sess-reaped".to_owned(),
            AffinityEntry {
                worker: worker(17, 0),
                ttl: Duration::from_secs(30),
                expires_at: Instant::now() - Duration::from_secs(1),
                kind: AffinityKind::EngineBacked,
                revision: 1,
                pending_rebind: None,
            },
        );

        store.reap_expired(Instant::now());

        assert!(store.map.get("sess-reaped").is_none());
        assert_eq!(
            expired_sessions.lock().unwrap().as_slice(),
            &[("sess-reaped".to_string(), 17)]
        );
    }

    #[test]
    fn reaper_drops_router_only_entry_without_close_callback() {
        let expired_sessions = Arc::new(Mutex::new(Vec::new()));
        let on_expire = {
            let expired_sessions = expired_sessions.clone();
            Arc::new(move |session_id: String, worker_id: u64| {
                expired_sessions
                    .lock()
                    .unwrap()
                    .push((session_id, worker_id));
            })
        };
        let store = InMemoryAffinityStore {
            map: Arc::new(DashMap::new()),
            on_expire: Some(on_expire),
        };
        store.map.insert(
            "sess-router-only-reaped".to_owned(),
            AffinityEntry {
                worker: worker(18, 0),
                ttl: Duration::from_secs(30),
                expires_at: Instant::now() - Duration::from_secs(1),
                kind: AffinityKind::RouterOnly,
                revision: 1,
                pending_rebind: None,
            },
        );

        store.reap_expired(Instant::now());

        assert!(store.map.get("sess-router-only-reaped").is_none());
        assert!(expired_sessions.lock().unwrap().is_empty());
    }
}
