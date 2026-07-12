// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use dynamo_kv_router::protocols::{TokensWithHashes, WorkerWithDpRank};
use dynamo_runtime::{
    component::TransportType,
    metrics::frontend_perf::{STAGE_ROUTE, StageGuard},
    pipeline::{
        AsyncEngine, AsyncEngineContextProvider, Error, ManyOut, PushRouter, ResponseStream,
        SingleIn, async_trait,
    },
    protocols::annotated::Annotated,
};
use futures::stream::{self, StreamExt};
use tracing::Instrument;

use crate::{
    kv_router::{
        KvRouter,
        metrics::{RebindTransition, RouterRequestMetrics, record_rebind_transition},
        sticky::{
            coordinator::{SessionRebindGuard, SessionTurnGuard, StickySessionCoordinator},
            router::{AffinityBindingToken, AffinityKind},
        },
    },
    preprocessor::PreprocessedRequest,
    protocols::common::{
        FinishReason,
        llm_backend::LLMEngineOutput,
        preprocessor::MigrateFrom,
        timing::{RequestPhase, RoutingData},
    },
};

mod cancellation;
mod request_guard;
mod selection;

use cancellation::{cancel_on_stop, cancelled_error};
use request_guard::RequestGuard;
use selection::{RoutingRequestParts, WorkerSelection};

/// Layer-2 rebind hysteresis: the hot rank's potential prefill tokens must
/// exceed the coldest rank's by more than this before a rebind fires.
/// Production default 4096; overridable via `DYN_REBIND_HYSTERESIS_TOKENS`
/// for controlled test/demo runs where a smaller imbalance must trigger.
// Raw-backlog gap threshold: the deferral a migration costs is roughly one
// prefill iteration, so one chunk of tokens is the natural unit.
const REBIND_HYSTERESIS_TOKENS_DEFAULT: usize = 32768;
// Greedy load-shedding: only turns that bring at least this much NEW
// (uncached-on-hot) prefill work are worth a migration's fixed overhead.
const REBIND_MIN_NEW_TOKENS_DEFAULT: usize = 4096;
// Bounce cap: each rebind costs a deferred turn + a transfer with
// diminishing returns; stop ping-ponging a session after this many.
const REBIND_MAX_PER_SESSION_DEFAULT: u32 = 4;

fn rebind_min_new_tokens() -> usize {
    use std::sync::OnceLock;
    static V: OnceLock<usize> = OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("DYN_REBIND_MIN_NEW_TOKENS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(REBIND_MIN_NEW_TOKENS_DEFAULT)
    })
}

fn rebind_max_per_session() -> u32 {
    use std::sync::OnceLock;
    static V: OnceLock<u32> = OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("DYN_REBIND_MAX_PER_SESSION")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(REBIND_MAX_PER_SESSION_DEFAULT)
    })
}

fn rebind_hysteresis_tokens() -> usize {
    use std::sync::OnceLock;
    static V: OnceLock<usize> = OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("DYN_REBIND_HYSTERESIS_TOKENS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(REBIND_HYSTERESIS_TOKENS_DEFAULT)
    })
}

/// Optional override for the source worker's migration peer host, used when the
/// Dynamo request plane is NATS (the worker's registry address is a subject, not
/// a routable IP, so the host is otherwise unknown). Valid when all prefill ranks
/// share one host — a single prefill node, the common case, and always true for
/// intra-worker cross-dp-rank migration. The production fix is for SGLang to
/// advertise its migration endpoint at registration; until then this env unblocks
/// migration under a NATS request plane.
fn migration_peer_host_override() -> Option<std::net::IpAddr> {
    use std::sync::OnceLock;
    static V: OnceLock<Option<std::net::IpAddr>> = OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("DYN_MIGRATION_PEER_HOST")
            .ok()
            .and_then(|s| s.trim().parse().ok())
    })
}

/// Warn (once) that a cross-worker rebind cannot carry a migrate_from directive.
/// Migration is scoped to intra-worker cross-dp-rank rebalancing; a cross-worker
/// source host is remote and unknowable under a NATS request plane, so we keep the
/// load-balancing rebind but drop the KV-reuse directive (unless an explicit
/// `DYN_MIGRATION_PEER_HOST` opts into a known topology).
fn warn_cross_worker_migration_skipped(src: WorkerWithDpRank, dst: WorkerWithDpRank) {
    use std::sync::atomic::{AtomicBool, Ordering};
    static WARNED: AtomicBool = AtomicBool::new(false);
    if !WARNED.swap(true, Ordering::Relaxed) {
        tracing::warn!(
            src_worker_id = src.worker_id,
            src_dp_rank = src.dp_rank,
            dst_worker_id = dst.worker_id,
            dst_dp_rank = dst.dp_rank,
            "Layer-2 rebind: cross-worker rebind — migrate_from skipped (migration is \
             intra-worker only unless DYN_MIGRATION_PEER_HOST is set); proceeding with the \
             cold-worker rebind for load balancing"
        );
    }
}
/// Minimum interval between rebinds of the same session, to avoid thrash.
const REBIND_COOLDOWN: Duration = Duration::from_secs(5);
/// Fallback TTL for a rebind when the request carries no session_control
/// timeout (in practice sticky gating guarantees one is present).
const REBIND_DEFAULT_TTL: Duration = Duration::from_secs(300);

#[derive(Debug, Clone)]
struct RebindDecision {
    session_id: String,
    expected: AffinityBindingToken,
    cold: WorkerWithDpRank,
    ttl: Duration,
}

struct PrefillSelection {
    selection: WorkerSelection,
    pending_rebind: Option<SessionRebindGuard>,
    session_turn: Option<SessionTurnGuard>,
}

/// Internal SGLang prefill completion ACK carried in the existing
/// `LLMEngineOutput.extra_args` field. The initial disaggregated-params chunk
/// only confirms bootstrap-room registration and is deliberately not an ACK.
const PREFILL_COMPLETE_MARKER_KEY: &str = "dynamo_prefill_complete";
/// Worker runtime-data capability required on both sides of a Layer-2 rebind.
/// The cold adapter must emit `PREFILL_COMPLETE_MARKER_KEY`, and the published
/// disaggregated endpoint guarantees PrefillRouter will use its bootstrap path.
const SESSION_REBIND_CAPABILITY_KEY: &str = "kv_session_rebind_v1";

#[derive(Debug, PartialEq, Eq)]
enum RebindStreamAction {
    Forward,
    SuppressControlFrame,
    Committed,
    RolledBack(&'static str),
    ReplaceWithError(String),
}

/// Holds a shadow affinity until the cold prefill worker explicitly confirms
/// that migration or its full-prefill fallback has completed.
struct RebindResponseGate {
    pending: Option<SessionRebindGuard>,
    saw_bootstrap_data: bool,
    transitions: prometheus::IntCounterVec,
}

impl RebindResponseGate {
    fn new(pending: Option<SessionRebindGuard>, transitions: prometheus::IntCounterVec) -> Self {
        Self {
            pending,
            saw_bootstrap_data: false,
            transitions,
        }
    }

    fn observe(&mut self, item: &Annotated<LLMEngineOutput>) -> RebindStreamAction {
        if self.pending.is_none() {
            return if item.event.is_none()
                && item.error.is_none()
                && item.data.as_ref().is_some_and(is_prefill_complete_marker)
            {
                RebindStreamAction::SuppressControlFrame
            } else {
                RebindStreamAction::Forward
            };
        }

        if item.is_error() || item.error.is_some() {
            drop(self.pending.take());
            self.record(RebindTransition::AbortAnnotatedError);
            return RebindStreamAction::RolledBack("annotated_error");
        }

        let Some(output) = item.data.as_ref() else {
            // Annotation-only frames are not proof that the cold prefill ran.
            return RebindStreamAction::Forward;
        };

        match output.finish_reason.as_ref() {
            Some(FinishReason::Error(message)) => {
                drop(self.pending.take());
                self.record(RebindTransition::AbortLlmError);
                return RebindStreamAction::ReplaceWithError(format!(
                    "cold prefill failed before completion ACK: {message}"
                ));
            }
            Some(FinishReason::Cancelled) => {
                drop(self.pending.take());
                self.record(RebindTransition::AbortCancelled);
                return RebindStreamAction::ReplaceWithError(
                    "cold prefill was cancelled before completion ACK".to_string(),
                );
            }
            _ => {}
        }

        if is_prefill_complete_marker(output) {
            if !self.saw_bootstrap_data {
                drop(self.pending.take());
                self.record(RebindTransition::AbortAckBeforeBootstrap);
                return RebindStreamAction::ReplaceWithError(
                    "cold prefill completion ACK arrived before bootstrap data".to_string(),
                );
            }

            let rebind = self
                .pending
                .take()
                .expect("pending rebind must exist while processing its ACK");
            return if rebind.commit() {
                self.record(RebindTransition::CommitPrefillCompleteAck);
                RebindStreamAction::Committed
            } else {
                self.record(RebindTransition::AbortCasConflict);
                RebindStreamAction::ReplaceWithError(
                    "session affinity changed before cold prefill completion; refusing to overwrite concurrent affinity"
                        .to_string(),
                )
            };
        }

        if output.disaggregated_params.is_some() {
            self.saw_bootstrap_data = true;
        }
        RebindStreamAction::Forward
    }

    fn finish(&mut self) -> Option<String> {
        self.pending.take()?;
        self.record(RebindTransition::AbortMissingAck);
        Some(if self.saw_bootstrap_data {
            "cold prefill stream ended without a completion ACK".to_string()
        } else {
            "cold prefill stream ended before bootstrap data or completion ACK".to_string()
        })
    }

    fn cancel(&mut self) -> bool {
        let cancelled = self.pending.take().is_some();
        if cancelled {
            self.record(RebindTransition::AbortContextStopped);
        }
        cancelled
    }

    fn record(&self, transition: RebindTransition) {
        record_rebind_transition(&self.transitions, transition);
    }
}

impl Drop for RebindResponseGate {
    fn drop(&mut self) {
        if self.pending.take().is_some() {
            self.record(RebindTransition::AbortStreamDropped);
        }
    }
}

fn is_prefill_complete_marker(output: &LLMEngineOutput) -> bool {
    output
        .extra_args
        .as_ref()
        .and_then(|extra_args| extra_args.get(PREFILL_COMPLETE_MARKER_KEY))
        .and_then(serde_json::Value::as_bool)
        == Some(true)
}

/// Base TCP port for SGLang per-rank migration peer channels: source attention
/// dp_rank `r` binds `MIGRATION_PEER_PORT_BASE + r`. This MUST equal the value
/// SGLang is launched with (`--prefill-migration-peer-port-base`); a silent
/// mismatch sends the migrate_from directive to a dead port.
///
/// TODO(layer2): plumb this from prefill-router config instead of hardcoding so
/// the router and SGLang values are configured from one source. Tracked as an
/// open item before merge.
const MIGRATION_PEER_PORT_BASE: u16 = 20000;

pub struct KvPushRouter {
    inner: PushRouter<PreprocessedRequest, Annotated<LLMEngineOutput>>,
    pub chooser: Arc<KvRouter>,
    /// Sticky session routing. Lazily activated when requests carry session_control.
    pub(super) sticky: Arc<StickySessionCoordinator>,
}

impl KvPushRouter {
    pub fn new(
        inner: PushRouter<PreprocessedRequest, Annotated<LLMEngineOutput>>,
        chooser: Arc<KvRouter>,
    ) -> Self {
        // Eagerly register router request metrics (as zeros) so they are
        // scrapeable before any requests arrive. Both the frontend pipeline
        // and the standalone router create KvPushRouter, so this covers both.
        RouterRequestMetrics::from_component(chooser.client().endpoint.component());

        let component = chooser.client().endpoint.component().clone();
        let sticky = Arc::new(StickySessionCoordinator::new(component));

        // F2: the Layer-2 rebind hysteresis compares potential_prefill_tokens,
        // which is only populated when router_track_prefill_tokens is enabled.
        // Warn once at startup so an inert rebind path isn't mistaken for "no
        // overload ever happened". This runs once per router construction, so
        // "one-time" is free (no Once/atomic needed).
        if !chooser.kv_router_config().router_track_prefill_tokens {
            tracing::warn!(
                "Layer-2 KV-migration rebind: router_track_prefill_tokens=false, so the \
                 rebind hysteresis (compares potential_prefill_tokens) will never trip and \
                 migration is inert. Set router_track_prefill_tokens=true to activate."
            );
        }

        KvPushRouter {
            inner,
            chooser,
            sticky,
        }
    }

    fn record_rebind_transition(&self, transition: RebindTransition) {
        RouterRequestMetrics::from_component(self.chooser.client().endpoint.component())
            .record_rebind_transition(transition);
    }

    fn worker_supports_session_rebind(&self, worker_id: u64) -> bool {
        let configs = self.chooser.workers_with_configs.borrow();
        let Some(config) = configs.get(&worker_id) else {
            return false;
        };
        let advertises_ack = config
            .runtime_data
            .get(SESSION_REBIND_CAPABILITY_KEY)
            .and_then(serde_json::Value::as_bool)
            == Some(true);
        let has_bootstrap_endpoint =
            config
                .disaggregated_endpoint
                .as_ref()
                .is_some_and(|endpoint| {
                    endpoint.bootstrap_host.is_some() && endpoint.bootstrap_port.is_some()
                });
        advertises_ack && has_bootstrap_endpoint
    }

    async fn select_request(
        &self,
        request: &SingleIn<PreprocessedRequest>,
        phase: RequestPhase,
        is_query_only: bool,
    ) -> Result<WorkerSelection, Error> {
        let context_id = request.context().id().to_string();
        let preserve_sticky_lifecycle = request
            .routing
            .as_ref()
            .and_then(|routing| routing.session_control.as_ref())
            .and_then(|session| session.action.as_ref())
            .is_some();
        let sticky_binding = self.resolve_sticky_binding_for_phase(
            &context_id,
            request,
            phase,
            preserve_sticky_lifecycle,
        )?;
        let sticky_worker = sticky_binding.map(|token| token.binding.worker);

        self.select_request_inner(
            request,
            phase,
            is_query_only,
            sticky_worker,
            true,
            !preserve_sticky_lifecycle,
        )
        .await
    }

    async fn select_request_for_final_target(
        &self,
        request: &SingleIn<PreprocessedRequest>,
        phase: RequestPhase,
        target: WorkerWithDpRank,
    ) -> Result<WorkerSelection, Error> {
        self.select_request_inner(request, phase, false, Some(target), false, false)
            .await
    }

    async fn select_request_inner(
        &self,
        request: &SingleIn<PreprocessedRequest>,
        phase: RequestPhase,
        is_query_only: bool,
        pinned_worker: Option<WorkerWithDpRank>,
        refresh_sticky: bool,
        allow_sticky_fallback: bool,
    ) -> Result<WorkerSelection, Error> {
        let context_id = request.context().id().to_string();
        let routing_parts = RoutingRequestParts::new(request);
        let request_context = request.context().clone();
        let mut selection_future = Box::pin(async {
            match self
                .select_worker(
                    &context_id,
                    request,
                    routing_parts,
                    phase,
                    is_query_only,
                    pinned_worker,
                )
                .instrument(tracing::info_span!("kv_router.select_worker"))
                .await
            {
                Ok(selection) => {
                    if refresh_sticky && pinned_worker.is_some() && !is_query_only {
                        self.sticky.refresh_worker_for_phase(request, phase);
                    }
                    Ok(selection)
                }
                Err(error) if allow_sticky_fallback && pinned_worker.is_some() => {
                    if let Some(worker) = pinned_worker {
                        let unbound = self
                            .sticky
                            .binding_token_for_phase(request, phase)
                            .filter(|token| token.binding.worker == worker)
                            .map(|token| {
                                self.remove_ineligible_sticky_binding_for_phase(
                                    &context_id,
                                    request,
                                    phase,
                                    token,
                                )
                            })
                            .unwrap_or(false);
                        if !unbound {
                            tracing::warn!(
                                request_id = %context_id,
                                worker_id = worker.worker_id,
                                dp_rank = worker.dp_rank,
                                error = %error,
                                "Sticky worker routing failed while affinity remains current; \
                                 refusing a mismatched fallback target"
                            );
                            return Err(error);
                        }
                        tracing::warn!(
                            request_id = %context_id,
                            worker_id = worker.worker_id,
                            dp_rank = worker.dp_rank,
                            error = %error,
                            "Ineligible sticky binding was removed; falling back to normal routing"
                        );
                    }
                    self.select_worker(
                        &context_id,
                        request,
                        routing_parts,
                        phase,
                        is_query_only,
                        None,
                    )
                    .instrument(tracing::info_span!("kv_router.select_worker_fallback"))
                    .await
                }
                Err(error) => Err(error),
            }
        });
        let selection_result = tokio::select! {
            biased;

            _ = request_context.stopped() => None,
            result = &mut selection_future => Some(result),
        };
        drop(selection_future);

        match selection_result {
            Some(result) => result,
            None => {
                if !is_query_only && let Err(error) = self.chooser.free(&context_id).await {
                    tracing::warn!(
                        request_id = %context_id,
                        %error,
                        "Failed to free scheduler state after cancellation during worker selection"
                    );
                }
                Err(cancelled_error(&context_id))
            }
        }
    }

    async fn reject_unexpected_final_target(
        &self,
        request: &SingleIn<PreprocessedRequest>,
        selection: &WorkerSelection,
        expected: WorkerWithDpRank,
    ) -> Result<(), Error> {
        if selection.instance_id == expected.worker_id && selection.dp_rank == expected.dp_rank {
            return Ok(());
        }

        let request_id = request.context().id().to_string();
        if selection.scheduler_tracked
            && let Err(cleanup_error) = self.chooser.free(&request_id).await
        {
            tracing::error!(
                %request_id,
                selected_worker_id = selection.instance_id,
                selected_dp_rank = selection.dp_rank,
                expected_worker_id = expected.worker_id,
                expected_dp_rank = expected.dp_rank,
                %cleanup_error,
                "Failed to clean up scheduler state after final-target mismatch"
            );
        }

        Err(anyhow::anyhow!(
            "stateful final-target selection mismatch for request {request_id}: \
             expected worker {} dp_rank {}, got worker {} dp_rank {}",
            expected.worker_id,
            expected.dp_rank,
            selection.instance_id,
            selection.dp_rank,
        )
        .into())
    }

    async fn track_selection(
        &self,
        request: &SingleIn<PreprocessedRequest>,
        selection: &mut WorkerSelection,
    ) -> Result<RequestGuard, Error> {
        let context_id = request.context().id().to_string();
        let request_context = request.context().clone();
        let routing_parts = RoutingRequestParts::new(request);
        let block_size = self.chooser.block_size() as usize;
        let mut guard = RequestGuard::new(
            self.chooser.clone(),
            context_id.clone(),
            request,
            selection.scheduler_tracked,
        );

        let record_result: Result<(), Error> = async {
            if self.chooser.indexer().records_routing_decisions() {
                let worker = WorkerWithDpRank::new(selection.instance_id, selection.dp_rank);
                let record_result = if let Some(hashes) = selection.routing_hashes.take() {
                    cancel_on_stop(
                        request_context.as_ref(),
                        &context_id,
                        self.chooser.record_routing_decision_hashes(hashes, worker),
                    )
                    .await?
                } else {
                    let lora_name = request.routing.as_ref().and_then(|r| r.lora_name.clone());
                    let mut tokens_with_hashes = TokensWithHashes::new(
                        routing_parts.token_ids.to_vec(),
                        self.chooser.block_size(),
                    )
                    .with_is_eagle(self.chooser.is_eagle());
                    if let Some(infos) = routing_parts.block_mm_infos {
                        tokens_with_hashes = tokens_with_hashes.with_mm_infos(infos.to_vec());
                    }
                    if let Some(lora_name) = lora_name {
                        tokens_with_hashes = tokens_with_hashes.with_lora_name(lora_name);
                    }
                    cancel_on_stop(
                        request_context.as_ref(),
                        &context_id,
                        self.chooser
                            .record_routing_decision(tokens_with_hashes, worker),
                    )
                    .await?
                };
                if let Err(error) = record_result {
                    tracing::warn!(
                        request_id = %context_id,
                        worker_id = selection.instance_id,
                        dp_rank = selection.dp_rank,
                        error = %error,
                        "Failed to record routing decision"
                    );
                }
            }

            if let Some(ref tracker) = request.tracker {
                let isl_blocks = routing_parts.token_ids.len().div_ceil(block_size);
                tracker.record_kv_hit(selection.effective_overlap_blocks, isl_blocks);
                tracker.record_isl(routing_parts.token_ids.len(), Some(selection.cached_tokens));
                tracker.record_worker(
                    selection.instance_id,
                    Some(selection.dp_rank),
                    self.chooser.worker_type(),
                );
                tracker.record_router_queue_depth(self.chooser.pending_count());
                if let Some(hit_rate) = tracker.kv_hit_rate() {
                    guard.request_metrics().kv_hit_rate.observe(hit_rate);
                }
            }
            guard
                .request_metrics()
                .input_sequence_tokens
                .observe(request.token_ids.len() as f64);
            Ok(())
        }
        .await;

        if let Err(error) = record_result {
            guard.abort().await;
            return Err(error);
        }
        Ok(guard)
    }

    async fn dispatch_selection(
        &self,
        request: SingleIn<PreprocessedRequest>,
        selection: WorkerSelection,
        mut guard: RequestGuard,
        exact: bool,
        mut pending_rebind: Option<SessionRebindGuard>,
        session_turn: Option<SessionTurnGuard>,
    ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
        let context_id = request.context().id().to_string();
        let request_context = request.context().clone();
        let phase = request
            .tracker
            .as_ref()
            .map(|tracker| tracker.phase())
            .unwrap_or(RequestPhase::Aggregated);
        let phase_label = phase.to_string();
        guard.start_dispatch(&phase_label);

        let worker = WorkerWithDpRank::new(selection.instance_id, selection.dp_rank);
        let route_outcome = cancel_on_stop(
            request_context.as_ref(),
            &context_id,
            self.sticky.on_routed(&request, worker, &context_id),
        )
        .await
        .and_then(|result| result);
        let route_outcome = match route_outcome {
            Ok(outcome) => outcome,
            Err(error) => {
                if pending_rebind.is_some() {
                    self.record_rebind_transition(RebindTransition::AbortRouteError);
                }
                drop(pending_rebind.take());
                guard.abort().await;
                return Err(error);
            }
        };
        guard.set_deferred_close(route_outcome.deferred_close);
        let mut rollback = route_outcome.rollback;

        let (mut backend_input, context) = request.into_parts();
        backend_input.routing_mut().dp_rank = Some(selection.dp_rank);
        let updated_request = context.map(|_| backend_input);
        guard.record_prefill_start();

        let dispatch = async {
            if exact {
                self.inner
                    .dispatch_exact(updated_request, selection.instance_id)
                    .await
            } else {
                self.inner
                    .direct(updated_request, selection.instance_id)
                    .await
            }
        };
        let dispatch_result = cancel_on_stop(
            request_context.as_ref(),
            &context_id,
            dispatch.instrument(tracing::info_span!(
                "kv_router.route_request",
                request_id = %context_id,
                worker_id = selection.instance_id,
                dp_rank = selection.dp_rank,
                overlap_blocks = selection.overlap_amount,
                phase = ?phase,
            )),
        )
        .await
        .and_then(|result| result);
        let mut response_stream = match dispatch_result {
            Ok(stream) => stream,
            Err(error) => {
                if let Some(rollback) = rollback.take() {
                    self.sticky.rollback_routed(rollback, &context_id);
                }
                if pending_rebind.is_some() {
                    self.record_rebind_transition(RebindTransition::AbortDispatchError);
                }
                drop(pending_rebind.take());
                guard.abort().await;
                return Err(error);
            }
        };

        guard.mark_dispatched();
        let stream_context = response_stream.context();
        let context_for_monitoring = stream_context.clone();
        let wrapped_stream = Box::pin(async_stream::stream! {
            let mut guard = guard;
            let session_turn = session_turn;
            let transitions = guard.request_metrics().rebind_transitions_total.clone();
            let mut rebind_gate = RebindResponseGate::new(pending_rebind.take(), transitions);

            loop {
                tokio::select! {
                    biased;

                    _ = context_for_monitoring.stopped() => {
                        if rebind_gate.cancel() {
                            tracing::info!(
                                request_id = %context_id,
                                reason = "context_stopped",
                                "Rolled back shadow session rebind"
                            );
                        }
                        tracing::debug!("Request {context_id} cancelled, ending stream");
                        break;
                    }

                    item = response_stream.next() => {
                        let Some(item) = item else {
                            break;
                        };
                        match rebind_gate.observe(&item) {
                            RebindStreamAction::Forward => {
                                guard.on_item(&item).await;
                                yield item;
                            }
                            RebindStreamAction::SuppressControlFrame => {}
                            RebindStreamAction::Committed => {
                                tracing::info!(
                                    request_id = %context_id,
                                    reason = "prefill_complete_ack",
                                    "Committed shadow session rebind"
                                );
                                // Internal control frame: do not expose it to
                                // PrefillRouter or client-facing processing.
                            }
                            RebindStreamAction::RolledBack(reason) => {
                                tracing::info!(
                                    request_id = %context_id,
                                    reason,
                                    "Rolled back shadow session rebind"
                                );
                                guard.on_item(&item).await;
                                yield item;
                            }
                            RebindStreamAction::ReplaceWithError(message) => {
                                tracing::warn!(
                                    request_id = %context_id,
                                    reason = "invalid_prefill_completion",
                                    error = %message,
                                    "Rolled back shadow session rebind"
                                );
                                let error = Annotated::from_error(message);
                                guard.on_item(&error).await;
                                yield error;
                                break;
                            }
                        }
                    }
                }
            }

            if let Some(message) = rebind_gate.finish() {
                tracing::warn!(
                    request_id = %context_id,
                    reason = "missing_prefill_complete_ack",
                    error = %message,
                    "Rolled back shadow session rebind"
                );
                let error = Annotated::from_error(message);
                guard.on_item(&error).await;
                yield error;
            }

            guard.finish().await;
            drop(session_turn);
        });
        Ok(ResponseStream::new(wrapped_stream, stream_context))
    }

    /// Layer-2 KV-migration read-only decision. If the sticky-pinned (hot) rank
    /// is overloaded relative to the coldest eligible rank, return the proposed
    /// final target without mutating sticky state or scheduler accounting.
    /// Returns `None` for a non-prefill phase, a first/unbound turn, a load gap
    /// within hysteresis, or an active cooldown.
    ///
    /// Runs on the LIVE prefill path (`select_and_dispatch_prefill`), the only
    /// place with both `self.sticky` and `self.chooser` load data in scope.
    async fn check_rebind_decision(
        &self,
        request: &SingleIn<PreprocessedRequest>,
        phase: RequestPhase,
        expected: Option<AffinityBindingToken>,
    ) -> Option<RebindDecision> {
        if phase != RequestPhase::Prefill {
            return None;
        }
        // Rebind is a data-plane optimization for ordinary session turns.
        // Lifecycle actions must reach the currently visible binding so Open,
        // Bind, and Close cannot race a shadow transition.
        let session_control = request.routing.as_ref()?.session_control.as_ref()?;
        // Lifecycle mutations (Open/Close) must reach the currently visible
        // binding untouched. A per-turn idempotent Bind — what conv-aware
        // clients send on every turn — is an ordinary data-plane turn and
        // stays rebind-eligible; the shadow CAS still serializes any race.
        if matches!(
            session_control.action,
            Some(crate::protocols::common::extensions::SessionAction::Open)
                | Some(crate::protocols::common::extensions::SessionAction::Close)
        ) {
            return None;
        }
        // Phase-gated sticky session id; also confirms session_control exists.
        let session_id = self
            .sticky
            .session_id_for_phase(request, phase)?
            .to_string();
        // DIAG (bounded to sticky requests only): confirms nvext.session_control
        // round-tripped into routing and reached the live rebind check.
        tracing::info!(
            %session_id,
            "Layer-2 diag: sticky prefill reached rebind check (session_control present)"
        );

        // Only a stable existing binding is eligible. The revision captured here
        // is the CAS condition used after the final-target scheduler booking.
        let Some(expected) = expected else {
            tracing::info!(
                %session_id,
                "Layer-2 diag: no stable prior binding (first turn / pending transition) — skip rebind"
            );
            return None;
        };
        let hot = expected.binding.worker;
        if !self.worker_supports_session_rebind(hot.worker_id) {
            tracing::debug!(
                %session_id,
                hot_worker_id = hot.worker_id,
                "Layer-2 rebind skipped: hot worker does not advertise the completion-ACK capability"
            );
            return None;
        }

        // Per-(worker, dp_rank) prefill load for THIS request's tokens, so the
        // hot/cold comparison reflects what adding this request would cost.
        let routing_parts = RoutingRequestParts::new(request);
        let lora_name = request.routing.as_ref().and_then(|r| r.lora_name.clone());
        let loads = self
            .chooser
            .get_potential_loads(
                routing_parts.token_ids,
                request.router_config_override.as_ref(),
                routing_parts.block_mm_infos,
                lora_name.as_deref(),
            )
            .await
            .ok()?;

        let hot_entry = loads
            .iter()
            .find(|l| l.worker_id == hot.worker_id && l.dp_rank == hot.dp_rank)?;
        // Compare RAW scheduled backlog (cache discount removed). The
        // discounted potential makes a cache-rich hot rank look cheapest for
        // its own sessions, which inverts the trigger: it fires on young
        // sessions with nothing to migrate and never on mature ones.
        // Placement keeps the discounted potential; the trigger must not.
        let hot_queue = hot_entry
            .potential_prefill_tokens
            .saturating_sub(hot_entry.request_prefill_delta);
        let request_new_tokens = hot_entry.request_prefill_delta;
        let cold = loads
            .iter()
            .filter(|l| !(l.worker_id == hot.worker_id && l.dp_rank == hot.dp_rank))
            .filter(|load| {
                let candidate = WorkerWithDpRank::new(load.worker_id, load.dp_rank);
                (expected.binding.kind != AffinityKind::EngineBacked
                    || candidate.worker_id == hot.worker_id)
                    && self.worker_supports_session_rebind(candidate.worker_id)
                    && self
                        .worker_ineligibility_for_phase(request, phase, candidate)
                        .is_none()
            })
            .min_by_key(|l| {
                l.potential_prefill_tokens
                    .saturating_sub(l.request_prefill_delta)
            })?;

        let cold_queue = cold
            .potential_prefill_tokens
            .saturating_sub(cold.request_prefill_delta);
        let gap = hot_queue.saturating_sub(cold_queue);
        let hysteresis = rebind_hysteresis_tokens();
        let min_new_tokens = rebind_min_new_tokens();
        // DIAG (bounded to sticky requests with an existing binding): shows
        // whether the raw-backlog imbalance reached the trigger gates.
        tracing::info!(
            %session_id,
            hot_queue,
            cold_queue,
            request_new_tokens,
            gap,
            hysteresis,
            min_new_tokens,
            "Layer-2 diag: raw backlog gap before trigger gates"
        );
        // Greedy load-shedding: small turns keep locality on the hot rank.
        if request_new_tokens < min_new_tokens {
            return None;
        }
        if gap <= hysteresis {
            return None;
        }
        // Bounce cap + cooldown: avoid thrash on a session we keep moving.
        let bounce = self.sticky.rebind_count(&session_id);
        if bounce >= rebind_max_per_session() {
            tracing::info!(
                %session_id,
                bounce,
                "Layer-2 rebind skipped: per-session bounce cap reached"
            );
            return None;
        }
        if self.sticky.in_rebind_cooldown(&session_id, REBIND_COOLDOWN) {
            return None;
        }

        // TTL preserved from the session_control timeout (sticky gating already
        // guaranteed session_control is present via session_id_for_phase).
        let ttl = request
            .routing
            .as_ref()
            .and_then(|r| r.session_control.as_ref())
            .map(|sc| Duration::from_secs(sc.timeout))
            .unwrap_or(REBIND_DEFAULT_TTL);

        let cold_target = WorkerWithDpRank::new(cold.worker_id, cold.dp_rank);
        if expected.binding.kind == AffinityKind::EngineBacked
            && hot.worker_id != cold_target.worker_id
        {
            tracing::warn!(
                %session_id,
                hot_worker_id = hot.worker_id,
                hot_dp_rank = hot.dp_rank,
                cold_worker_id = cold_target.worker_id,
                cold_dp_rank = cold_target.dp_rank,
                "Layer-2 rebind skipped: engine-backed cross-worker transition requires lifecycle RPC"
            );
            return None;
        }
        Some(RebindDecision {
            session_id,
            expected,
            cold: cold_target,
            ttl,
        })
    }

    /// Resolve the source (hot) worker's deterministic SGLang migration peer
    /// endpoint, `tcp://{host}:{MIGRATION_PEER_PORT_BASE + dp_rank}`, for the
    /// `migrate_from` directive.
    ///
    /// The host comes from the source worker's registry entry when the request
    /// plane is TCP (`host:port[/endpoint]`, optional `tcp://`, IPv6-safe, parsed
    /// as a `SocketAddr`). Under a NATS request plane `transport` is a subject,
    /// not an `ip:port`, so the host falls back to `DYN_MIGRATION_PEER_HOST`
    /// (see [`migration_peer_host_override`]); without it we return `None` and the
    /// caller skips the directive but keeps the cold-rank rebind.
    fn resolve_migration_endpoint(&self, src: WorkerWithDpRank) -> Option<String> {
        // Port = base + dp_rank, guarded against u16 overflow (dp_rank is small
        // in practice; this only trips on a misconfigured base or absurd rank).
        let port = u16::try_from(src.dp_rank)
            .ok()
            .and_then(|r| MIGRATION_PEER_PORT_BASE.checked_add(r))?;

        let inst = self
            .chooser
            .client()
            .instances()
            .into_iter()
            .find(|i| i.instance_id == src.worker_id)?;

        let host: std::net::IpAddr = match &inst.transport {
            // Strip an optional `tcp://` scheme and any `/endpoint` suffix, then
            // parse the remaining `host:port` as a SocketAddr (IPv6-safe).
            TransportType::Tcp(addr) => {
                let trimmed = addr.strip_prefix("tcp://").unwrap_or(addr);
                let socket_part = trimmed.split('/').next()?;
                let socket: std::net::SocketAddr = socket_part.parse().ok()?;
                socket.ip()
            }
            // NATS request plane: subject carries no routable host. Prefer an
            // explicitly configured host; otherwise emit the unspecified host
            // 0.0.0.0 as a sentinel that tells the SGLang target to substitute
            // its own local IP — valid for intra-worker cross-dp-rank migration,
            // where source and target share a node.
            TransportType::Nats(_) => migration_peer_host_override()
                .unwrap_or(std::net::IpAddr::V4(std::net::Ipv4Addr::UNSPECIFIED)),
        };

        // SocketAddr's Display brackets IPv6 hosts; reconstruct so the migration
        // port (not the original request port) is used while keeping IPv6 valid.
        let migration_socket = std::net::SocketAddr::new(host, port);
        Some(format!("tcp://{migration_socket}"))
    }

    fn attach_migrate_from(
        &self,
        request: &mut SingleIn<PreprocessedRequest>,
        session_id: &str,
        old_hot: WorkerWithDpRank,
        cold: WorkerWithDpRank,
    ) {
        // Migration is currently safe by default only for an intra-worker
        // cross-dp-rank rebind. Cross-worker routing still keeps the cold target,
        // but omits KV reuse unless a topology-specific host override is explicit.
        let intra_worker = old_hot.worker_id == cold.worker_id;
        if !intra_worker && migration_peer_host_override().is_none() {
            warn_cross_worker_migration_skipped(old_hot, cold);
            return;
        }

        match self.resolve_migration_endpoint(old_hot) {
            Some(source_endpoint) => {
                request.migrate_from = Some(MigrateFrom {
                    source_endpoint,
                    source_dp_rank: old_hot.dp_rank,
                    session_id: session_id.to_string(),
                });
            }
            None => {
                tracing::warn!(
                    hot_worker_id = old_hot.worker_id,
                    hot_dp_rank = old_hot.dp_rank,
                    "Layer-2 rebind: could not resolve source migration endpoint \
                     (non-TCP request plane, or source worker gone); proceeding with \
                     the cold-rank rebind but without a migrate_from directive"
                );
            }
        }
    }

    async fn select_prefill_request(
        &self,
        request: &mut SingleIn<PreprocessedRequest>,
    ) -> Result<PrefillSelection, Error> {
        let phase = RequestPhase::Prefill;
        let context_id = request.context().id().to_string();
        let session_turn = self
            .sticky
            .acquire_turn_for_phase(request, phase, &context_id)?;
        if request
            .routing
            .as_ref()
            .and_then(|routing| routing.session_control.as_ref())
            .and_then(|session| session.action.as_ref())
            .is_some()
        {
            return Ok(PrefillSelection {
                selection: self.select_request(request, phase, false).await?,
                pending_rebind: None,
                session_turn,
            });
        }
        let hot_binding =
            self.resolve_sticky_binding_for_phase(&context_id, request, phase, false)?;
        let expected = hot_binding.and_then(|binding| {
            self.sticky
                .rebind_token_for_phase(request, phase)
                .filter(|token| *token == binding)
        });
        let Some(decision) = self.check_rebind_decision(request, phase, expected).await else {
            return Ok(PrefillSelection {
                selection: self.select_request(request, phase, false).await?,
                pending_rebind: None,
                session_turn,
            });
        };

        // The decision is deliberately read-only. Book exactly once, explicitly
        // against the final cold target; only a successful, matching booking may
        // publish the sticky rebind and the migrate_from directive.
        let selection = self
            .select_request_for_final_target(request, phase, decision.cold)
            .await?;
        self.reject_unexpected_final_target(request, &selection, decision.cold)
            .await?;

        let Some(pending_rebind) = self.sticky.begin_rebind(
            &decision.session_id,
            decision.expected,
            decision.cold,
            decision.ttl,
        ) else {
            if selection.scheduler_tracked
                && let Err(cleanup_error) = self.chooser.free(&context_id).await
            {
                tracing::error!(
                    request_id = %context_id,
                    %cleanup_error,
                    "Failed to clean up scheduler state after rebind CAS rejection"
                );
            }
            return Err(anyhow::anyhow!(
                "session {} changed while preparing rebind; refusing stale cold dispatch",
                decision.session_id
            )
            .into());
        };

        let hot = decision.expected.binding.worker;
        self.record_rebind_transition(RebindTransition::DecisionLoadImbalance);
        self.attach_migrate_from(request, &decision.session_id, hot, decision.cold);
        Ok(PrefillSelection {
            selection,
            pending_rebind: Some(pending_rebind),
            session_turn,
        })
    }

    pub(crate) async fn select_and_dispatch_prefill<M, F>(
        &self,
        mut request: SingleIn<PreprocessedRequest>,
        prepare: F,
    ) -> Result<(M, ManyOut<Annotated<LLMEngineOutput>>), Error>
    where
        F: FnOnce(&mut PreprocessedRequest, u64, Option<u32>) -> Result<M, Error>,
    {
        let phase = RequestPhase::Prefill;
        let phase_label = phase.to_string();
        let route_guard = StageGuard::new(STAGE_ROUTE, &phase_label);
        let PrefillSelection {
            mut selection,
            pending_rebind,
            session_turn,
        } = self.select_prefill_request(&mut request).await?;

        let mut guard = match self.track_selection(&request, &mut selection).await {
            Ok(guard) => guard,
            Err(error) => {
                if pending_rebind.is_some() {
                    self.record_rebind_transition(RebindTransition::AbortTrackingError);
                }
                drop(pending_rebind);
                return Err(error);
            }
        };
        let metadata = match prepare(&mut request, selection.instance_id, Some(selection.dp_rank)) {
            Ok(metadata) => metadata,
            Err(error) => {
                if pending_rebind.is_some() {
                    self.record_rebind_transition(RebindTransition::AbortPrepareError);
                }
                drop(pending_rebind);
                guard.abort().await;
                return Err(error);
            }
        };
        drop(route_guard);
        let stream = self
            .dispatch_selection(
                request,
                selection,
                guard,
                true,
                pending_rebind,
                session_turn,
            )
            .await?;
        Ok((metadata, stream))
    }
}

#[async_trait]
impl AsyncEngine<SingleIn<PreprocessedRequest>, ManyOut<Annotated<LLMEngineOutput>>, Error>
    for KvPushRouter
{
    /// Generate method that handles KV-aware routing with three distinct behaviors:
    ///
    /// 1. **If `query_instance_id` annotation is set**:
    ///    - Returns the best matching worker ID without routing the request
    ///    - Does NOT update any router local states
    ///    - Response includes worker_instance_id and token_data annotations
    ///
    /// 2. **If a phase-specific worker or `backend_instance_id` is set in the request**:
    ///    - Query-only requests return that worker selection without state updates
    ///    - Requests route through the scheduler as an exact pin when dp_rank is resolved
    ///    - If dp_rank cannot be resolved, the request is rejected instead of treating rank 0 as a sentinel
    ///
    /// 3. **If neither are set (default behavior)**:
    ///    - Finds the best worker based on KV cache overlap
    ///    - Updates router states to track the request
    ///    - Routes to the selected worker
    ///
    /// The router state updates include tracking active sequences and managing
    /// prefill/completion lifecycle for proper KV cache management.
    async fn generate(
        &self,
        request: SingleIn<PreprocessedRequest>,
    ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
        let is_query_only = request.get_annotation_value("query_instance_id").is_some();
        let phase = request
            .tracker
            .as_ref()
            .map(|tracker| tracker.phase())
            .unwrap_or(RequestPhase::Aggregated);
        let phase_label = phase.to_string();
        let route_guard = StageGuard::new(STAGE_ROUTE, &phase_label);
        let mut selection = self.select_request(&request, phase, is_query_only).await?;
        if is_query_only {
            let routing_parts = RoutingRequestParts::new(&request);
            if let Some(ref tracker) = request.tracker {
                let isl_blocks = routing_parts
                    .token_ids
                    .len()
                    .div_ceil(self.chooser.block_size() as usize);
                tracker.record_kv_hit(selection.effective_overlap_blocks, isl_blocks);
                tracker.record_isl(routing_parts.token_ids.len(), Some(selection.cached_tokens));
                tracker.record_worker(
                    selection.instance_id,
                    Some(selection.dp_rank),
                    self.chooser.worker_type(),
                );
                tracker.record_router_queue_depth(self.chooser.pending_count());
            }
            RouterRequestMetrics::from_component(self.chooser.client().endpoint.component())
                .input_sequence_tokens
                .observe(request.token_ids.len() as f64);
            let stream_context = request.context().clone();
            let worker_id_info = request
                .tracker
                .as_ref()
                .and_then(|tracker| tracker.get_worker_info());

            tracing::trace!(
                ?phase,
                worker_id = selection.instance_id,
                ?worker_id_info,
                "Returning worker selection (query-only mode)"
            );

            let output = LLMEngineOutput {
                routing_data: Some(RoutingData {
                    worker_id: worker_id_info,
                    token_ids: Some(request.token_ids.clone()),
                    ..Default::default()
                }),
                ..Default::default()
            };
            let response = Annotated::from_data(output);
            let stream = stream::iter(vec![response]);
            return Ok(ResponseStream::new(Box::pin(stream), stream_context));
        }

        let guard = self.track_selection(&request, &mut selection).await?;
        drop(route_guard);
        self.dispatch_selection(request, selection, guard, false, None, None)
            .await
    }
}

/// A direct routing wrapper for `RouterMode::Direct`.
///
/// This wraps a `PushRouter` and reads worker IDs from each request's routing hints,
/// then routes directly to the specified worker. Used when an external router
/// (e.g., EPP) handles worker selection.
pub struct DirectRoutingRouter {
    inner: PushRouter<PreprocessedRequest, Annotated<LLMEngineOutput>>,
}

impl DirectRoutingRouter {
    pub fn new(inner: PushRouter<PreprocessedRequest, Annotated<LLMEngineOutput>>) -> Self {
        DirectRoutingRouter { inner }
    }

    /// Extract worker ID from request routing hints.
    /// Returns an error if no worker ID is found (required in direct routing mode).
    fn get_worker_id(request: &PreprocessedRequest) -> Result<u64, Error> {
        let routing = request.routing.as_ref();
        let worker_id = routing.and_then(|r| r.decode_worker_id.or(r.backend_instance_id));

        worker_id.ok_or_else(|| {
            anyhow::anyhow!(
                "Worker ID required (--direct-route) but none found in request. \
                 Expected decode_worker_id or backend_instance_id to be set by external router (e.g., EPP)."
            )
        })
    }
}

#[async_trait]
impl AsyncEngine<SingleIn<PreprocessedRequest>, ManyOut<Annotated<LLMEngineOutput>>, Error>
    for DirectRoutingRouter
{
    async fn generate(
        &self,
        request: SingleIn<PreprocessedRequest>,
    ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
        let worker_id = Self::get_worker_id(&request)?;

        tracing::debug!(worker_id = worker_id, "Direct routing to specified worker");

        self.inner.direct(request, worker_id).await
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};
    use std::sync::atomic::{AtomicU64, Ordering};

    use dynamo_kv_router::config::KvRouterConfig;
    use dynamo_runtime::{
        DistributedRuntime, Runtime,
        distributed::DistributedConfig,
        pipeline::{Context, RouterMode},
    };
    use tokio::sync::watch;

    use super::*;
    use crate::{
        discovery::WORKER_TYPE_PREFILL,
        kv_router::scheduler::DefaultWorkerSelector,
        local_model::runtime_config::{DisaggregatedEndpoint, ModelRuntimeConfig},
        protocols::common::{
            extensions::{SessionAction, SessionControl},
            preprocessor::RoutingHints,
            timing::RequestTracker,
        },
    };

    static NEXT_TEST_ID: AtomicU64 = AtomicU64::new(1);

    /// Rebind-trigger knobs are process-wide OnceLocks; pin them to
    /// test-friendly values before the first check_rebind_decision call.
    /// Every rebind test sets the SAME values, so parallel init is benign.
    fn pin_rebind_trigger_env_for_tests() {
        // SAFETY: test-only, every caller sets identical values, and the
        // consumers read them once through OnceLock.
        unsafe {
            std::env::set_var("DYN_REBIND_MIN_NEW_TOKENS", "0");
            std::env::set_var("DYN_REBIND_HYSTERESIS_TOKENS", "256");
            std::env::set_var("DYN_REBIND_MAX_PER_SESSION", "100");
        }
    }

    fn session_request(
        session_id: &str,
        action: Option<SessionAction>,
        tokens: usize,
    ) -> PreprocessedRequest {
        PreprocessedRequest::builder()
            .model("test-model".to_string())
            .token_ids(vec![7; tokens])
            .stop_conditions(Default::default())
            .sampling_options(Default::default())
            .output_options(Default::default())
            .routing(Some(RoutingHints {
                session_control: Some(SessionControl {
                    session_id: session_id.to_string(),
                    action,
                    timeout: 300,
                }),
                ..Default::default()
            }))
            .build()
            .unwrap()
    }

    async fn make_dp_test_router_with_capability(
        worker_ids: &[u64],
        supports_rebind: bool,
    ) -> KvPushRouter {
        let runtime = Runtime::from_current().unwrap();
        let drt = DistributedRuntime::new(runtime, DistributedConfig::process_local())
            .await
            .unwrap();
        let test_id = NEXT_TEST_ID.fetch_add(1, Ordering::Relaxed);
        let namespace = drt
            .namespace(format!("test-rebind-final-target-{test_id}"))
            .unwrap();
        let component = namespace.component("router").unwrap();
        let endpoint = component.endpoint("backend");
        let client = endpoint.client().await.unwrap();

        let worker_configs = worker_ids
            .iter()
            .map(|worker_id| {
                let mut worker_config = ModelRuntimeConfig::default();
                worker_config.data_parallel_size = 2;
                if supports_rebind {
                    worker_config.runtime_data.insert(
                        SESSION_REBIND_CAPABILITY_KEY.to_string(),
                        serde_json::Value::Bool(true),
                    );
                    worker_config.disaggregated_endpoint = Some(DisaggregatedEndpoint {
                        bootstrap_host: Some("127.0.0.1".to_string()),
                        bootstrap_port: Some(18000),
                    });
                }
                (*worker_id, worker_config)
            })
            .collect();
        let (_tx, workers) = watch::channel(worker_configs);

        let config = KvRouterConfig {
            overlap_score_credit: 0.0,
            router_temperature: 0.0,
            use_kv_events: false,
            router_track_active_blocks: false,
            router_track_prefill_tokens: true,
            skip_initial_worker_wait: true,
            ..Default::default()
        };
        let selector = DefaultWorkerSelector::new(Some(config.clone()), WORKER_TYPE_PREFILL);
        let chooser = Arc::new(
            KvRouter::new(
                endpoint,
                client.clone(),
                workers,
                16,
                selector,
                Some(config),
                None,
                WORKER_TYPE_PREFILL,
                None,
                false,
                None,
            )
            .await
            .unwrap(),
        );
        let inner = PushRouter::from_client(client, RouterMode::KV)
            .await
            .unwrap();
        KvPushRouter::new(inner, chooser)
    }

    async fn make_dp_test_router_with_workers(worker_ids: &[u64]) -> KvPushRouter {
        make_dp_test_router_with_capability(worker_ids, true).await
    }

    async fn make_dp_test_router() -> KvPushRouter {
        make_dp_test_router_with_workers(&[42]).await
    }

    async fn load_snapshot(router: &KvPushRouter) -> HashMap<WorkerWithDpRank, (usize, usize)> {
        router
            .chooser
            .get_potential_loads(&[], None, None, None)
            .await
            .unwrap()
            .into_iter()
            .map(|load| {
                (
                    WorkerWithDpRank::new(load.worker_id, load.dp_rank),
                    (load.potential_prefill_tokens, load.active_requests),
                )
            })
            .collect()
    }

    async fn make_rebind_response_gate(
        router: &KvPushRouter,
        session_id: &str,
        hot: WorkerWithDpRank,
        cold: WorkerWithDpRank,
    ) -> RebindResponseGate {
        let bind_request = session_request(session_id, Some(SessionAction::Bind), 1);
        router
            .sticky
            .on_routed(&bind_request, hot, &format!("bind-{session_id}"))
            .await
            .unwrap();

        let request = session_request(session_id, None, 1);
        let expected = router
            .sticky
            .rebind_token_for_phase(&request, RequestPhase::Prefill)
            .unwrap();
        let pending = router
            .sticky
            .begin_rebind(session_id, expected, cold, Duration::from_secs(300))
            .unwrap();
        let transitions =
            RouterRequestMetrics::from_component(router.chooser.client().endpoint.component())
                .rebind_transitions_total
                .clone();
        RebindResponseGate::new(Some(pending), transitions)
    }

    fn visible_affinity(router: &KvPushRouter, session_id: &str) -> Option<WorkerWithDpRank> {
        router
            .sticky
            .worker_for_phase(&session_request(session_id, None, 1), RequestPhase::Prefill)
    }

    fn prefill_bootstrap_output() -> Annotated<LLMEngineOutput> {
        Annotated::from_data(LLMEngineOutput {
            disaggregated_params: Some(serde_json::json!({
                "bootstrap_host": "127.0.0.1",
                "bootstrap_port": 1234,
                "bootstrap_room": 5678,
            })),
            ..Default::default()
        })
    }

    fn prefill_complete_marker() -> Annotated<LLMEngineOutput> {
        Annotated::from_data(LLMEngineOutput {
            extra_args: Some(serde_json::json!({
                "dynamo_prefill_complete": true,
            })),
            ..Default::default()
        })
    }

    #[tokio::test]
    async fn rebind_response_gate_commits_only_after_bootstrap_and_completion_marker() {
        let router = make_dp_test_router().await;
        let hot = WorkerWithDpRank::new(42, 0);
        let cold = WorkerWithDpRank::new(42, 1);
        let session_id = "response-gate-success";
        let mut gate = make_rebind_response_gate(&router, session_id, hot, cold).await;

        assert_eq!(
            gate.observe(&prefill_bootstrap_output()),
            RebindStreamAction::Forward
        );
        assert_eq!(visible_affinity(&router, session_id), Some(hot));

        assert_eq!(
            gate.observe(&prefill_complete_marker()),
            RebindStreamAction::Committed
        );
        assert_eq!(visible_affinity(&router, session_id), Some(cold));
        assert!(gate.finish().is_none());
    }

    #[tokio::test]
    async fn rebind_response_gate_rolls_back_on_first_annotated_error() {
        let router = make_dp_test_router().await;
        let hot = WorkerWithDpRank::new(42, 0);
        let cold = WorkerWithDpRank::new(42, 1);
        let session_id = "response-gate-first-error";
        let mut gate = make_rebind_response_gate(&router, session_id, hot, cold).await;

        assert_eq!(
            gate.observe(&Annotated::from_error("prefill failed")),
            RebindStreamAction::RolledBack("annotated_error")
        );
        assert_eq!(visible_affinity(&router, session_id), Some(hot));
        assert!(gate.finish().is_none());
    }

    #[tokio::test]
    async fn rebind_response_gate_rejects_empty_stream() {
        let router = make_dp_test_router().await;
        let hot = WorkerWithDpRank::new(42, 0);
        let cold = WorkerWithDpRank::new(42, 1);
        let session_id = "response-gate-empty";
        let mut gate = make_rebind_response_gate(&router, session_id, hot, cold).await;

        let error = gate.finish().expect("empty stream must fail closed");
        assert!(error.contains("before bootstrap data"));
        assert_eq!(visible_affinity(&router, session_id), Some(hot));
    }

    #[tokio::test]
    async fn rebind_response_gate_rolls_back_on_cancel_before_bootstrap() {
        let router = make_dp_test_router().await;
        let hot = WorkerWithDpRank::new(42, 0);
        let cold = WorkerWithDpRank::new(42, 1);
        let session_id = "response-gate-cancel";
        let mut gate = make_rebind_response_gate(&router, session_id, hot, cold).await;

        let action = gate.observe(&Annotated::from_data(LLMEngineOutput::cancelled()));
        assert!(matches!(action, RebindStreamAction::ReplaceWithError(_)));
        assert_eq!(visible_affinity(&router, session_id), Some(hot));
        assert!(gate.finish().is_none());
    }

    #[tokio::test]
    async fn rebind_response_gate_cas_failure_preserves_concurrent_affinity() {
        let router = make_dp_test_router_with_workers(&[42, 99]).await;
        let hot = WorkerWithDpRank::new(42, 0);
        let cold = WorkerWithDpRank::new(42, 1);
        let concurrent = WorkerWithDpRank::new(99, 0);
        let session_id = "response-gate-cas";
        let mut gate = make_rebind_response_gate(&router, session_id, hot, cold).await;
        assert_eq!(
            gate.observe(&prefill_bootstrap_output()),
            RebindStreamAction::Forward
        );

        let concurrent_bind = session_request(session_id, Some(SessionAction::Bind), 1);
        router
            .sticky
            .on_routed(&concurrent_bind, concurrent, "concurrent-bind")
            .await
            .unwrap();
        assert_eq!(visible_affinity(&router, session_id), Some(concurrent));

        let action = gate.observe(&prefill_complete_marker());
        assert!(matches!(action, RebindStreamAction::ReplaceWithError(_)));
        assert_eq!(visible_affinity(&router, session_id), Some(concurrent));
        assert!(gate.finish().is_none());
    }

    #[tokio::test]
    async fn rebind_response_gate_keeps_cold_after_later_error() {
        let router = make_dp_test_router().await;
        let hot = WorkerWithDpRank::new(42, 0);
        let cold = WorkerWithDpRank::new(42, 1);
        let session_id = "response-gate-later-error";
        let mut gate = make_rebind_response_gate(&router, session_id, hot, cold).await;

        assert_eq!(
            gate.observe(&prefill_bootstrap_output()),
            RebindStreamAction::Forward
        );
        assert_eq!(
            gate.observe(&prefill_complete_marker()),
            RebindStreamAction::Committed
        );
        assert_eq!(
            gate.observe(&Annotated::from_error("late failure")),
            RebindStreamAction::Forward
        );
        assert_eq!(visible_affinity(&router, session_id), Some(cold));
    }

    #[tokio::test]
    async fn rebind_response_gate_does_not_accept_marker_after_unrelated_data() {
        let router = make_dp_test_router().await;
        let hot = WorkerWithDpRank::new(42, 0);
        let cold = WorkerWithDpRank::new(42, 1);
        let session_id = "response-gate-no-bootstrap";
        let mut gate = make_rebind_response_gate(&router, session_id, hot, cold).await;

        assert_eq!(
            gate.observe(&Annotated::from_data(LLMEngineOutput::default())),
            RebindStreamAction::Forward
        );
        let action = gate.observe(&prefill_complete_marker());
        assert!(matches!(action, RebindStreamAction::ReplaceWithError(_)));
        assert_eq!(visible_affinity(&router, session_id), Some(hot));
    }

    #[tokio::test]
    async fn rebind_skips_session_lifecycle_actions() {
        let router = make_dp_test_router().await;
        let hot = WorkerWithDpRank::new(42, 0);
        router
            .chooser
            .add_request(
                "lifecycle-hot-load".to_string(),
                &vec![1; 8192],
                None,
                0,
                None,
                hot,
                None,
                None,
            )
            .await;

        for (index, action) in [
            SessionAction::Open,
            SessionAction::Bind,
            SessionAction::Close,
        ]
        .into_iter()
        .enumerate()
        {
            let session_id = format!("lifecycle-session-{index}");
            let bind_request = session_request(&session_id, Some(SessionAction::Bind), 1);
            router
                .sticky
                .on_routed(&bind_request, hot, &format!("bind-{session_id}"))
                .await
                .unwrap();

            let mut request_data = session_request(&session_id, Some(action), 128);
            let routing = request_data.routing.as_mut().unwrap();
            routing.allowed_worker_ids = Some(HashSet::from([999]));
            routing.prefill_worker_id = Some(999);
            routing.prefill_dp_rank = Some(0);
            let mut request = Context::with_id_and_metadata(
                request_data,
                format!("lifecycle-request-{index}"),
                Default::default(),
            );
            let expected = router
                .sticky
                .rebind_token_for_phase(&request, RequestPhase::Prefill);

            assert!(
                router
                    .check_rebind_decision(&request, RequestPhase::Prefill, expected)
                    .await
                    .is_none(),
                "session lifecycle action must not trigger a rebind"
            );
            let PrefillSelection {
                selection,
                pending_rebind,
                session_turn: _session_turn,
            } = router.select_prefill_request(&mut request).await.unwrap();
            assert!(pending_rebind.is_none());
            assert_eq!(selection.instance_id, hot.worker_id);
            assert_eq!(selection.dp_rank, hot.dp_rank);
            assert_eq!(
                router
                    .sticky
                    .worker_for_phase(&request, RequestPhase::Prefill),
                Some(hot)
            );
            router.chooser.free(request.context().id()).await.unwrap();
        }

        router.chooser.free("lifecycle-hot-load").await.unwrap();
    }

    #[tokio::test]
    async fn rebind_requires_worker_completion_ack_capability() {
        let router = make_dp_test_router_with_capability(&[42], false).await;
        let hot = WorkerWithDpRank::new(42, 0);
        router
            .chooser
            .add_request(
                "capability-hot-load".to_string(),
                &vec![1; 8192],
                None,
                0,
                None,
                hot,
                None,
                None,
            )
            .await;
        let bind_request = session_request("capability-session", Some(SessionAction::Bind), 1);
        router
            .sticky
            .on_routed(&bind_request, hot, "bind-capability-session")
            .await
            .unwrap();
        let request_data = session_request("capability-session", None, 128);
        let request = Context::with_id_and_metadata(
            request_data,
            "capability-request".to_string(),
            Default::default(),
        );
        let expected = router
            .sticky
            .rebind_token_for_phase(&request, RequestPhase::Prefill);

        assert!(
            router
                .check_rebind_decision(&request, RequestPhase::Prefill, expected)
                .await
                .is_none(),
            "a worker without the ACK capability must never enter shadow rebind"
        );
        router.chooser.free("capability-hot-load").await.unwrap();
    }

    #[tokio::test]
    async fn pinned_session_turn_is_rejected_while_rebind_is_pending_then_released() {
        pin_rebind_trigger_env_for_tests();
        let router = make_dp_test_router().await;
        let hot = WorkerWithDpRank::new(42, 0);
        router
            .chooser
            .add_request(
                "serialized-hot-load".to_string(),
                &vec![1; 8192],
                None,
                0,
                None,
                hot,
                None,
                None,
            )
            .await;
        let bind_request = session_request("serialized-session", Some(SessionAction::Bind), 1);
        router
            .sticky
            .on_routed(&bind_request, hot, "bind-serialized-session")
            .await
            .unwrap();

        let mut first = Context::with_id_and_metadata(
            session_request("serialized-session", None, 128),
            "serialized-first".to_string(),
            Default::default(),
        );
        let PrefillSelection {
            pending_rebind,
            session_turn,
            ..
        } = router.select_prefill_request(&mut first).await.unwrap();
        assert!(pending_rebind.is_some());

        let mut pinned_request = session_request("serialized-session", None, 64);
        let routing = pinned_request.routing.as_mut().unwrap();
        routing.prefill_worker_id = Some(hot.worker_id);
        routing.prefill_dp_rank = Some(hot.dp_rank);
        let mut second = Context::with_id_and_metadata(
            pinned_request,
            "serialized-second".to_string(),
            Default::default(),
        );
        let error = match router.select_prefill_request(&mut second).await {
            Ok(_) => panic!("concurrent turn unexpectedly acquired the session"),
            Err(error) => error.to_string(),
        };
        assert!(error.contains("already has an in-flight prefill turn"));

        let mut backend_pinned_request = session_request("serialized-session", None, 64);
        let routing = backend_pinned_request.routing.as_mut().unwrap();
        routing.backend_instance_id = Some(hot.worker_id);
        routing.dp_rank = Some(hot.dp_rank);
        let mut backend_pinned = Context::with_id_and_metadata(
            backend_pinned_request,
            "serialized-backend-pinned".to_string(),
            Default::default(),
        );
        let error = match router.select_prefill_request(&mut backend_pinned).await {
            Ok(_) => panic!("backend-pinned turn unexpectedly acquired the session"),
            Err(error) => error.to_string(),
        };
        assert!(error.contains("already has an in-flight prefill turn"));
        assert_eq!(
            router
                .sticky
                .worker_for_phase(&first, RequestPhase::Prefill),
            Some(hot)
        );

        drop(pending_rebind);
        drop(session_turn);
        router.chooser.free("serialized-first").await.unwrap();

        let PrefillSelection {
            selection,
            pending_rebind,
            session_turn,
        } = router.select_prefill_request(&mut second).await.unwrap();
        assert_eq!(selection.instance_id, hot.worker_id);
        assert_eq!(selection.dp_rank, hot.dp_rank);
        assert!(pending_rebind.is_none());
        assert!(session_turn.is_some());
        drop(session_turn);
        router.chooser.free("serialized-second").await.unwrap();
        router.chooser.free("serialized-hot-load").await.unwrap();
    }

    #[tokio::test]
    async fn actionless_session_turn_rejection_preserves_accounting_and_allows_reacquire() {
        pin_rebind_trigger_env_for_tests();
        let router = make_dp_test_router().await;
        let hot = WorkerWithDpRank::new(42, 0);
        router
            .chooser
            .add_request(
                "actionless-hot-load".to_string(),
                &vec![1; 8192],
                None,
                0,
                None,
                hot,
                None,
                None,
            )
            .await;
        let bind_request = session_request("actionless-session", Some(SessionAction::Bind), 1);
        router
            .sticky
            .on_routed(&bind_request, hot, "bind-actionless-session")
            .await
            .unwrap();

        let mut first = Context::with_id_and_metadata(
            session_request("actionless-session", None, 128),
            "actionless-first".to_string(),
            Default::default(),
        );
        let PrefillSelection {
            pending_rebind: first_rebind,
            session_turn: first_turn,
            ..
        } = router.select_prefill_request(&mut first).await.unwrap();
        assert!(first_rebind.is_some());
        assert!(first_turn.is_some());
        let held_snapshot = load_snapshot(&router).await;

        let mut second = Context::with_id_and_metadata(
            session_request("actionless-session", None, 64),
            "actionless-second".to_string(),
            Default::default(),
        );
        let error = match router.select_prefill_request(&mut second).await {
            Ok(_) => panic!("concurrent actionless turn unexpectedly acquired the session"),
            Err(error) => error,
        };
        let typed = error
            .downcast_ref::<dynamo_runtime::error::DynamoError>()
            .expect("concurrent session turn must return a typed client error");
        assert_eq!(
            typed.error_type(),
            dynamo_runtime::error::ErrorType::InvalidArgument
        );
        assert!(
            error
                .to_string()
                .contains("already has an in-flight prefill turn")
        );
        assert!(error.to_string().contains("actionless-first"));
        assert_eq!(
            load_snapshot(&router).await,
            held_snapshot,
            "a rejected turn must not add a second scheduler booking"
        );

        drop(first_rebind);
        drop(first_turn);
        router.chooser.free("actionless-first").await.unwrap();
        let released_snapshot = load_snapshot(&router).await;

        let PrefillSelection {
            pending_rebind: retry_rebind,
            session_turn: retry_turn,
            ..
        } = router.select_prefill_request(&mut second).await.unwrap();
        assert!(retry_rebind.is_some());
        assert!(retry_turn.is_some());
        drop(retry_rebind);
        drop(retry_turn);
        router.chooser.free("actionless-second").await.unwrap();
        assert_eq!(
            load_snapshot(&router).await,
            released_snapshot,
            "retry booking must be released exactly once"
        );

        router.chooser.free("actionless-hot-load").await.unwrap();
    }

    #[tokio::test]
    async fn rebind_chooses_coldest_eligible_target() {
        pin_rebind_trigger_env_for_tests();
        let router = make_dp_test_router_with_workers(&[42, 99]).await;
        let hot = WorkerWithDpRank::new(42, 0);
        let eligible_cold = WorkerWithDpRank::new(42, 1);
        router
            .chooser
            .add_request(
                "eligible-hot-load".to_string(),
                &vec![1; 8192],
                None,
                0,
                None,
                hot,
                None,
                None,
            )
            .await;
        router
            .chooser
            .add_request(
                "eligible-cold-load".to_string(),
                &vec![1; 1024],
                None,
                0,
                None,
                eligible_cold,
                None,
                None,
            )
            .await;

        let bind_request = session_request("eligible-session", Some(SessionAction::Bind), 1);
        router
            .sticky
            .on_routed(&bind_request, hot, "bind-eligible-session")
            .await
            .unwrap();

        let mut request_data = session_request("eligible-session", None, 128);
        request_data.routing.as_mut().unwrap().allowed_worker_ids = Some(HashSet::from([42]));
        let mut request = Context::with_id_and_metadata(
            request_data,
            "eligible-rebind-request".to_string(),
            Default::default(),
        );

        let PrefillSelection {
            selection,
            pending_rebind,
            session_turn: _session_turn,
        } = router.select_prefill_request(&mut request).await.unwrap();
        assert_eq!(selection.instance_id, eligible_cold.worker_id);
        assert_eq!(selection.dp_rank, eligible_cold.dp_rank);
        drop(pending_rebind);

        router
            .chooser
            .free("eligible-rebind-request")
            .await
            .unwrap();
        router.chooser.free("eligible-hot-load").await.unwrap();
        router.chooser.free("eligible-cold-load").await.unwrap();
    }

    #[tokio::test]
    async fn rebind_books_and_tracks_only_the_final_cold_target() {
        pin_rebind_trigger_env_for_tests();
        let router = make_dp_test_router().await;
        let hot = WorkerWithDpRank::new(42, 0);
        let cold = WorkerWithDpRank::new(42, 1);

        let hot_tokens = vec![1; 8192];
        router
            .chooser
            .add_request(
                "existing-hot-load".to_string(),
                &hot_tokens,
                None,
                0,
                None,
                hot,
                None,
                None,
            )
            .await;

        let bind_request = session_request("session-a", Some(SessionAction::Bind), 1);
        router
            .sticky
            .on_routed(&bind_request, hot, "bind-session-a")
            .await
            .unwrap();

        let before = load_snapshot(&router).await;
        assert_eq!(before[&hot], (8192, 1));
        assert_eq!(before[&cold], (0, 0));

        let tracker = Arc::new(RequestTracker::new());
        let phase_permit = tracker.set_phase(RequestPhase::Prefill).await;
        let mut request_data = session_request("session-a", None, 128);
        request_data.tracker = Some(tracker.clone());
        let mut request = Context::with_id_and_metadata(
            request_data,
            "rebind-request".to_string(),
            Default::default(),
        );

        let PrefillSelection {
            mut selection,
            pending_rebind,
            session_turn: _session_turn,
        } = router.select_prefill_request(&mut request).await.unwrap();
        assert_eq!(selection.instance_id, cold.worker_id);
        assert_eq!(selection.dp_rank, cold.dp_rank);
        assert_eq!(
            router
                .sticky
                .worker_for_phase(&request, RequestPhase::Prefill),
            Some(hot),
            "the cold target must remain shadowed until dispatch succeeds"
        );

        let after = load_snapshot(&router).await;
        assert_eq!(
            after[&hot], before[&hot],
            "rebind must not leave a stateful booking on the old hot rank"
        );
        assert_eq!(after[&cold], (128, 1));

        let mut guard = router
            .track_selection(&request, &mut selection)
            .await
            .unwrap();
        let worker_info = tracker.get_worker_info().unwrap();
        assert_eq!(worker_info.prefill_worker_id, Some(cold.worker_id));
        assert_eq!(worker_info.prefill_dp_rank, Some(cold.dp_rank));

        assert!(pending_rebind.unwrap().commit());
        assert_eq!(
            router
                .sticky
                .worker_for_phase(&request, RequestPhase::Prefill),
            Some(cold)
        );

        guard.abort().await;
        router.chooser.free("existing-hot-load").await.unwrap();
        drop(phase_permit);

        let cleaned = load_snapshot(&router).await;
        assert_eq!(cleaned[&hot], (0, 0));
        assert_eq!(cleaned[&cold], (0, 0));
    }

    #[tokio::test]
    async fn rebind_cas_does_not_restore_binding_removed_after_decision() {
        let router = make_dp_test_router().await;
        let hot = WorkerWithDpRank::new(42, 0);
        let cold = WorkerWithDpRank::new(42, 1);
        let request_data = session_request("session-race", None, 128);
        let request = Context::with_id_and_metadata(
            request_data,
            "stale-source-request".to_string(),
            Default::default(),
        );

        let bind_request = session_request("session-race", Some(SessionAction::Bind), 1);
        router
            .sticky
            .on_routed(&bind_request, hot, "bind-session-race")
            .await
            .unwrap();

        let expected = router
            .sticky
            .rebind_token_for_phase(&request, RequestPhase::Prefill)
            .unwrap();
        let decision = RebindDecision {
            session_id: "session-race".to_string(),
            expected,
            cold,
            ttl: Duration::from_secs(300),
        };
        let (_, removed) = router
            .sticky
            .unbind_for_phase(&request, RequestPhase::Prefill)
            .unwrap();
        assert_eq!(removed.unwrap().worker, hot);

        assert!(
            router
                .sticky
                .begin_rebind(
                    &decision.session_id,
                    decision.expected,
                    decision.cold,
                    decision.ttl,
                )
                .is_none()
        );
        assert_eq!(
            router
                .sticky
                .worker_for_phase(&request, RequestPhase::Prefill),
            None,
            "a removed binding must not be resurrected as cold by a stale decision"
        );
    }

    #[tokio::test]
    async fn rebind_prepare_failure_rolls_back_shadow_and_cold_booking() {
        let router = make_dp_test_router().await;
        let hot = WorkerWithDpRank::new(42, 0);
        let cold = WorkerWithDpRank::new(42, 1);
        router
            .chooser
            .add_request(
                "prepare-fail-hot-load".to_string(),
                &vec![1; 8192],
                None,
                0,
                None,
                hot,
                None,
                None,
            )
            .await;
        let bind_request = session_request("session-prepare-fail", Some(SessionAction::Bind), 1);
        router
            .sticky
            .on_routed(&bind_request, hot, "bind-session-prepare-fail")
            .await
            .unwrap();

        let tracker = Arc::new(RequestTracker::new());
        let phase_permit = tracker.set_phase(RequestPhase::Prefill).await;
        let mut request_data = session_request("session-prepare-fail", None, 128);
        request_data.tracker = Some(tracker);
        let request = Context::with_id_and_metadata(
            request_data,
            "prepare-fail-request".to_string(),
            Default::default(),
        );

        let result: Result<((), ManyOut<Annotated<LLMEngineOutput>>), Error> = router
            .select_and_dispatch_prefill(request, |_, _, _| {
                Err(anyhow::anyhow!("injected prepare failure").into())
            })
            .await;
        assert!(result.is_err());

        let probe = session_request("session-prepare-fail", None, 1);
        assert_eq!(
            router
                .sticky
                .worker_for_phase(&probe, RequestPhase::Prefill),
            Some(hot)
        );
        let loads = load_snapshot(&router).await;
        assert_eq!(loads[&hot], (8192, 1));
        assert_eq!(loads[&cold], (0, 0));

        router.chooser.free("prepare-fail-hot-load").await.unwrap();
        drop(phase_permit);
    }

    #[tokio::test]
    async fn rebind_dispatch_failure_rolls_back_shadow_and_cold_booking() {
        let router = make_dp_test_router().await;
        let hot = WorkerWithDpRank::new(42, 0);
        let cold = WorkerWithDpRank::new(42, 1);
        router
            .chooser
            .add_request(
                "dispatch-fail-hot-load".to_string(),
                &vec![1; 8192],
                None,
                0,
                None,
                hot,
                None,
                None,
            )
            .await;
        let bind_request = session_request("session-dispatch-fail", Some(SessionAction::Bind), 1);
        router
            .sticky
            .on_routed(&bind_request, hot, "bind-session-dispatch-fail")
            .await
            .unwrap();

        let tracker = Arc::new(RequestTracker::new());
        let phase_permit = tracker.set_phase(RequestPhase::Prefill).await;
        let mut request_data = session_request("session-dispatch-fail", None, 128);
        request_data.tracker = Some(tracker);
        let request = Context::with_id_and_metadata(
            request_data,
            "dispatch-fail-request".to_string(),
            Default::default(),
        );

        let result: Result<((), ManyOut<Annotated<LLMEngineOutput>>), Error> = router
            .select_and_dispatch_prefill(request, |_, _, _| Ok(()))
            .await;
        assert!(result.is_err(), "test router has no dispatchable backend");

        let probe = session_request("session-dispatch-fail", None, 1);
        assert_eq!(
            router
                .sticky
                .worker_for_phase(&probe, RequestPhase::Prefill),
            Some(hot)
        );
        let loads = load_snapshot(&router).await;
        assert_eq!(loads[&hot], (8192, 1));
        assert_eq!(loads[&cold], (0, 0));

        router.chooser.free("dispatch-fail-hot-load").await.unwrap();
        drop(phase_permit);
    }
}
