// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Coordination layer for sticky routing and backend session lifecycle.

use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use dashmap::DashMap;
use dynamo_kv_router::protocols::WorkerWithDpRank;
use dynamo_runtime::component::Component;

use crate::{
    preprocessor::PreprocessedRequest,
    protocols::common::{
        extensions::SessionAction, preprocessor::RoutingHints, timing::RequestPhase,
    },
};

use super::{
    lifecycle::{SessionCloseAction, SessionLifecycleController},
    router::{
        AffinityBinding, AffinityBindingToken, AffinityKind, AffinityRebindToken,
        InMemoryAffinityStore, StickySessionRouter,
    },
};

pub struct SessionRoutingResult {
    pub deferred_close: Option<SessionCloseAction>,
    pub rollback: Option<SessionRoutingRollback>,
}

pub struct SessionRoutingRollback {
    session_id: String,
    binding_token: AffinityBindingToken,
    close_opened_session: Option<SessionCloseAction>,
}

/// Owns a shadow rebind until the backend confirms cold prefill completion.
/// Dropping it conditionally rolls the transition back.
pub(crate) struct SessionRebindGuard {
    router: Arc<StickySessionRouter>,
    session_id: String,
    token: Option<AffinityRebindToken>,
}

/// Serializes prefill turns for one sticky session on this frontend.
///
/// A rebind snapshots the hot prefix and may publish a cold owner when prefill
/// completes. Allowing another turn to mutate the hot owner in that window can
/// make the cold snapshot stale even when the affinity revision itself did not
/// change. Only the turn owner is rebind-eligible; an overlapping turn follows
/// the visible binding instead of being rejected (non-closed-loop clients
/// legitimately overlap turns of one session).
pub(crate) struct SessionTurnGuard {
    turns: Arc<DashMap<String, String>>,
    session_id: String,
    owner: String,
}

/// Outcome of a turn-ownership attempt for a sticky session.
pub(crate) enum TurnAcquisition {
    /// Request has no sticky turn semantics for this phase.
    NotApplicable,
    /// This request now owns the session's turn and is rebind-eligible.
    Owner(SessionTurnGuard),
    /// Another request owns the turn. The caller must follow the visible
    /// binding and skip rebind evaluation so the owner's snapshot window
    /// cannot be raced.
    Overlap,
}

impl Drop for SessionTurnGuard {
    fn drop(&mut self) {
        self.turns
            .remove_if(&self.session_id, |_, owner| owner == &self.owner);
    }
}

impl SessionRebindGuard {
    pub(crate) fn commit(mut self) -> bool {
        let token = self.token.expect("pending rebind token must be present");
        if !self.router.commit_rebind(&self.session_id, token) {
            return false;
        }
        self.token = None;
        true
    }
}

impl Drop for SessionRebindGuard {
    fn drop(&mut self) {
        if let Some(token) = self.token.take() {
            self.router.rollback_rebind(&self.session_id, token);
        }
    }
}

pub struct StickySessionCoordinator {
    router: Arc<StickySessionRouter>,
    lifecycle: Arc<SessionLifecycleController>,
    prefill_turns: Arc<DashMap<String, String>>,
}

impl StickySessionCoordinator {
    pub fn new(component: Component) -> Self {
        let lifecycle = Arc::new(SessionLifecycleController::new(component));
        let on_expire = {
            let lifecycle = lifecycle.clone();
            Arc::new(move |session_id: String, worker_id: u64| {
                lifecycle
                    .clone()
                    .close_expired_session(session_id, worker_id);
            }) as Arc<dyn Fn(String, u64) + Send + Sync>
        };
        let router = Arc::new(StickySessionRouter::new(
            InMemoryAffinityStore::new_with_on_expire(Some(on_expire)),
        ));

        StickySessionCoordinator {
            router,
            lifecycle,
            prefill_turns: Arc::new(DashMap::new()),
        }
    }

    /// Acquire ownership of a sticky session turn for `phase`.
    /// Requests without session control do not need a guard.
    pub(crate) fn acquire_turn_for_phase(
        &self,
        request: &PreprocessedRequest,
        phase: RequestPhase,
        owner: &str,
    ) -> TurnAcquisition {
        use dashmap::mapref::entry::Entry;

        let Some(session_id) = turn_session_id_for_phase(request, phase) else {
            return TurnAcquisition::NotApplicable;
        };
        let session_id = session_id.to_owned();
        match self.prefill_turns.entry(session_id.clone()) {
            Entry::Vacant(entry) => {
                entry.insert(owner.to_owned());
                TurnAcquisition::Owner(SessionTurnGuard {
                    turns: self.prefill_turns.clone(),
                    session_id,
                    owner: owner.to_owned(),
                })
            }
            Entry::Occupied(entry) => {
                let current_owner = entry.get().to_owned();
                tracing::info!(
                    %session_id,
                    phase = %phase,
                    request_id = %owner,
                    %current_owner,
                    "sticky session turn overlap: following existing binding, not rebind-eligible"
                );
                TurnAcquisition::Overlap
            }
        }
    }

    pub fn worker_for_phase(
        &self,
        request: &PreprocessedRequest,
        phase: RequestPhase,
    ) -> Option<WorkerWithDpRank> {
        let session_id = binding_session_id_for_phase(request, phase)?;
        self.router.peek_session(session_id)
    }

    /// Return the visible binding and revision in one store read. Lifecycle
    /// actions intentionally resolve an existing owner even when the request
    /// carries new constraints or explicit pins.
    pub(crate) fn binding_token_for_phase(
        &self,
        request: &PreprocessedRequest,
        phase: RequestPhase,
    ) -> Option<AffinityBindingToken> {
        let session_id = binding_session_id_for_phase(request, phase)?;
        self.router.peek_binding_token(session_id)
    }

    /// Phase-gated sticky session id for this request, or `None` when sticky
    /// routing is not applicable (no session_control, or explicit pins set).
    /// Reuses the same gating as the resolve/bind paths so the Layer-2 rebind
    /// trigger never fires on a request the rest of the sticky layer ignores.
    pub(crate) fn session_id_for_phase<'a>(
        &self,
        request: &'a PreprocessedRequest,
        phase: RequestPhase,
    ) -> Option<&'a str> {
        sticky_session_id_for_phase(request, phase)
    }

    /// Return a stable revision for a Layer-2 rebind decision. A pending
    /// transition is deliberately not eligible for another rebind.
    pub(crate) fn rebind_token_for_phase(
        &self,
        request: &PreprocessedRequest,
        phase: RequestPhase,
    ) -> Option<AffinityBindingToken> {
        let session_id = sticky_session_id_for_phase(request, phase)?;
        self.router.peek_rebind_token(session_id)
    }

    /// Atomically prepare a shadow rebind. The old binding remains visible
    /// until the returned guard is committed after cold prefill completion.
    pub(crate) fn begin_rebind(
        &self,
        session_id: &str,
        expected: AffinityBindingToken,
        cold: WorkerWithDpRank,
        ttl: Duration,
    ) -> Option<SessionRebindGuard> {
        let token = self.router.begin_rebind(session_id, expected, cold, ttl)?;
        Some(SessionRebindGuard {
            router: self.router.clone(),
            session_id: session_id.to_owned(),
            token: Some(token),
        })
    }

    /// True if `session_id` was rebound within `cooldown`; used to throttle
    /// Layer-2 rebinds.
    pub(crate) fn in_rebind_cooldown(&self, session_id: &str, cooldown: Duration) -> bool {
        self.router.in_rebind_cooldown(session_id, cooldown)
    }

    /// Committed rebind count for `session_id`; input to the Layer-2 bounce cap.
    pub(crate) fn rebind_count(&self, session_id: &str) -> u32 {
        self.router.rebind_count(session_id)
    }

    pub fn refresh_worker_for_phase(&self, request: &PreprocessedRequest, phase: RequestPhase) {
        let Some(session_id) = sticky_session_id_for_phase(request, phase) else {
            return;
        };
        let Some(sc) = request
            .routing
            .as_ref()
            .and_then(|routing| routing.session_control.as_ref())
        else {
            return;
        };
        if sc.action.is_some() {
            return;
        }

        self.router.resolve_session(session_id);
    }

    pub fn unbind_for_phase<'a>(
        &self,
        request: &'a PreprocessedRequest,
        phase: RequestPhase,
    ) -> Option<(&'a str, Option<AffinityBinding>)> {
        let session_id = sticky_session_id_for_phase(request, phase)?;
        Some((session_id, self.router.unbind(session_id)))
    }

    pub(crate) fn unbind_if_token_for_phase(
        &self,
        request: &PreprocessedRequest,
        phase: RequestPhase,
        token: AffinityBindingToken,
    ) -> bool {
        let Some(session_id) = binding_session_id_for_phase(request, phase) else {
            return false;
        };
        self.router.unbind_if_token(session_id, token)
    }

    pub async fn on_routed(
        &self,
        request: &PreprocessedRequest,
        worker: WorkerWithDpRank,
        context_id: &str,
    ) -> Result<SessionRoutingResult> {
        let sc = request
            .routing
            .as_ref()
            .and_then(|r| r.session_control.as_ref());

        let Some(sc) = sc else {
            return Ok(SessionRoutingResult {
                deferred_close: None,
                rollback: None,
            });
        };
        let Some(action) = sc.action.as_ref() else {
            return Ok(SessionRoutingResult {
                deferred_close: None,
                rollback: None,
            });
        };

        match action {
            SessionAction::Open => {
                let opened = self
                    .lifecycle
                    .open_session(&sc.session_id, sc.timeout, worker.worker_id, context_id)
                    .await?;
                let rollback = if opened {
                    let binding_token = self.router.bind(
                        &sc.session_id,
                        worker,
                        Duration::from_secs(sc.timeout),
                        AffinityKind::EngineBacked,
                    );
                    let close_opened_session = self
                        .lifecycle
                        .deferred_close(sc.session_id.clone(), worker.worker_id)
                        .await;
                    Some(SessionRoutingRollback {
                        session_id: sc.session_id.clone(),
                        binding_token,
                        close_opened_session,
                    })
                } else {
                    None
                };
                Ok(SessionRoutingResult {
                    deferred_close: None,
                    rollback,
                })
            }
            SessionAction::Bind => {
                let binding_token = self.router.bind(
                    &sc.session_id,
                    worker,
                    Duration::from_secs(sc.timeout),
                    AffinityKind::RouterOnly,
                );
                Ok(SessionRoutingResult {
                    deferred_close: None,
                    rollback: Some(SessionRoutingRollback {
                        session_id: sc.session_id.clone(),
                        binding_token,
                        close_opened_session: None,
                    }),
                })
            }
            SessionAction::Close => {
                let removed = self.router.unbind(&sc.session_id);
                let should_close_worker_session = removed
                    .map(|binding| binding.kind == AffinityKind::EngineBacked)
                    .unwrap_or(true);
                let close_worker_id = removed
                    .map(|binding| binding.worker.worker_id)
                    .unwrap_or(worker.worker_id);
                let deferred_close = if should_close_worker_session {
                    self.lifecycle
                        .deferred_close(sc.session_id.clone(), close_worker_id)
                        .await
                } else {
                    None
                };
                Ok(SessionRoutingResult {
                    deferred_close,
                    rollback: None,
                })
            }
        }
    }

    pub fn rollback_routed(&self, rollback: SessionRoutingRollback, context_id: &str) {
        self.router
            .unbind_if_token(&rollback.session_id, rollback.binding_token);
        if let Some(close) = rollback.close_opened_session {
            close.execute(context_id);
        }
    }
}

pub(crate) fn sticky_allowed_for_phase(
    phase: RequestPhase,
    routing: Option<&RoutingHints>,
) -> bool {
    let Some(routing) = routing else {
        return false;
    };
    if routing.session_control.is_none() {
        return false;
    }

    match phase {
        RequestPhase::Prefill => {
            routing.prefill_worker_id.is_none()
                && routing.prefill_dp_rank.is_none()
                && routing.backend_instance_id.is_none()
        }
        RequestPhase::Decode => {
            routing.decode_worker_id.is_none()
                && routing.dp_rank.is_none()
                && routing.backend_instance_id.is_none()
        }
        RequestPhase::Aggregated => {
            routing.backend_instance_id.is_none() && routing.dp_rank.is_none()
        }
    }
}

fn sticky_session_id_for_phase(request: &PreprocessedRequest, phase: RequestPhase) -> Option<&str> {
    let routing = request.routing.as_ref()?;
    if !sticky_allowed_for_phase(phase, Some(routing)) {
        return None;
    }

    routing
        .session_control
        .as_ref()
        .map(|sc| sc.session_id.as_str())
}

fn turn_session_id_for_phase(request: &PreprocessedRequest, _phase: RequestPhase) -> Option<&str> {
    // Explicit pins disable sticky target selection, but the request still
    // mutates the same engine session and must remain serialized with it.
    request
        .routing
        .as_ref()?
        .session_control
        .as_ref()
        .map(|sc| sc.session_id.as_str())
}

fn binding_session_id_for_phase(
    request: &PreprocessedRequest,
    phase: RequestPhase,
) -> Option<&str> {
    let routing = request.routing.as_ref()?;
    let session_control = routing.session_control.as_ref()?;
    // An existing lifecycle action belongs to the worker that owns the
    // session. Close must not be redirected before the old engine session is
    // freed, even if this request carries new routing constraints.
    if session_control.action.is_some() {
        Some(session_control.session_id.as_str())
    } else {
        sticky_session_id_for_phase(request, phase)
    }
}

#[cfg(test)]
mod tests {
    use super::{sticky_allowed_for_phase, turn_session_id_for_phase};
    use crate::protocols::common::extensions::SessionControl;
    use crate::protocols::common::{
        preprocessor::{PreprocessedRequest, RoutingHints},
        timing::RequestPhase,
    };

    fn session_control() -> SessionControl {
        SessionControl {
            session_id: "sess-1".to_string(),
            action: None,
            timeout: 300,
        }
    }

    #[test]
    fn sticky_is_noop_without_session_control() {
        let routing = RoutingHints::default();
        assert!(!sticky_allowed_for_phase(
            RequestPhase::Aggregated,
            Some(&routing)
        ));
    }

    #[test]
    fn sticky_allowed_when_only_session_control_is_present() {
        let routing = RoutingHints {
            session_control: Some(session_control()),
            ..Default::default()
        };
        assert!(sticky_allowed_for_phase(
            RequestPhase::Aggregated,
            Some(&routing)
        ));
        assert!(sticky_allowed_for_phase(
            RequestPhase::Prefill,
            Some(&routing)
        ));
        assert!(sticky_allowed_for_phase(
            RequestPhase::Decode,
            Some(&routing)
        ));
    }

    #[test]
    fn sticky_skips_phase_specific_explicit_pins() {
        let prefill = RoutingHints {
            session_control: Some(session_control()),
            prefill_worker_id: Some(1),
            ..Default::default()
        };
        assert!(!sticky_allowed_for_phase(
            RequestPhase::Prefill,
            Some(&prefill)
        ));

        let prefill_rank = RoutingHints {
            session_control: Some(session_control()),
            prefill_dp_rank: Some(2),
            ..Default::default()
        };
        assert!(!sticky_allowed_for_phase(
            RequestPhase::Prefill,
            Some(&prefill_rank)
        ));

        let decode = RoutingHints {
            session_control: Some(session_control()),
            decode_worker_id: Some(3),
            ..Default::default()
        };
        assert!(!sticky_allowed_for_phase(
            RequestPhase::Decode,
            Some(&decode)
        ));

        let decode_rank = RoutingHints {
            session_control: Some(session_control()),
            dp_rank: Some(4),
            ..Default::default()
        };
        assert!(!sticky_allowed_for_phase(
            RequestPhase::Decode,
            Some(&decode_rank)
        ));

        let aggregated = RoutingHints {
            session_control: Some(session_control()),
            backend_instance_id: Some(5),
            ..Default::default()
        };
        assert!(!sticky_allowed_for_phase(
            RequestPhase::Aggregated,
            Some(&aggregated)
        ));
    }

    #[test]
    fn turn_serialization_keeps_session_id_with_explicit_pins() {
        let prefill = RoutingHints {
            session_control: Some(session_control()),
            prefill_worker_id: Some(1),
            prefill_dp_rank: Some(2),
            ..Default::default()
        };
        let backend = RoutingHints {
            session_control: Some(session_control()),
            backend_instance_id: Some(5),
            dp_rank: Some(3),
            ..Default::default()
        };

        for (phase, routing) in [
            (RequestPhase::Prefill, prefill),
            (RequestPhase::Prefill, backend.clone()),
            (RequestPhase::Aggregated, backend),
        ] {
            let request = PreprocessedRequest::builder()
                .model("test-model".to_string())
                .token_ids(vec![1])
                .stop_conditions(Default::default())
                .sampling_options(Default::default())
                .output_options(Default::default())
                .routing(Some(routing))
                .build()
                .unwrap();
            assert_eq!(turn_session_id_for_phase(&request, phase), Some("sess-1"));
        }
    }
}
