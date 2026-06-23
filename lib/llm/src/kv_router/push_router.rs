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
        KvRouter, metrics::RouterRequestMetrics, sticky::coordinator::StickySessionCoordinator,
    },
    preprocessor::PreprocessedRequest,
    protocols::common::{
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
const REBIND_HYSTERESIS_TOKENS: usize = 4096;
/// Minimum interval between rebinds of the same session, to avoid thrash.
const REBIND_COOLDOWN: Duration = Duration::from_secs(5);
/// Fallback TTL for a rebind when the request carries no session_control
/// timeout (in practice sticky gating guarantees one is present).
const REBIND_DEFAULT_TTL: Duration = Duration::from_secs(300);

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

    async fn select_request(
        &self,
        request: &SingleIn<PreprocessedRequest>,
        phase: RequestPhase,
        is_query_only: bool,
    ) -> Result<WorkerSelection, Error> {
        let context_id = request.context().id().to_string();
        let routing_parts = RoutingRequestParts::new(request);
        let sticky_worker = match self.sticky.worker_for_phase(request, phase) {
            Some(worker)
                if self.unbind_ineligible_sticky_worker_for_phase(
                    &context_id,
                    request,
                    phase,
                    worker,
                ) =>
            {
                None
            }
            worker => worker,
        };
        let request_context = request.context().clone();
        let mut selection_future = Box::pin(async {
            match self
                .select_worker(
                    &context_id,
                    request,
                    routing_parts,
                    phase,
                    is_query_only,
                    sticky_worker,
                )
                .instrument(tracing::info_span!("kv_router.select_worker"))
                .await
            {
                Ok(selection) => {
                    if sticky_worker.is_some() && !is_query_only {
                        self.sticky.refresh_worker_for_phase(request, phase);
                    }
                    Ok(selection)
                }
                Err(error) if sticky_worker.is_some() => {
                    if let Some(worker) = sticky_worker {
                        let unbound = self.unbind_ineligible_sticky_worker_for_phase(
                            &context_id,
                            request,
                            phase,
                            worker,
                        );
                        tracing::warn!(
                            request_id = %context_id,
                            worker_id = worker.worker_id,
                            dp_rank = worker.dp_rank,
                            error = %error,
                            unbound_due_to_ineligibility = unbound,
                            "Sticky worker routing failed; falling back to normal routing"
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
                guard.abort().await;
                return Err(error);
            }
        };

        guard.mark_dispatched();
        let stream_context = response_stream.context();
        let context_for_monitoring = stream_context.clone();
        let wrapped_stream = Box::pin(async_stream::stream! {
            let mut guard = guard;

            loop {
                tokio::select! {
                    biased;

                    _ = context_for_monitoring.stopped() => {
                        tracing::debug!("Request {context_id} cancelled, ending stream");
                        break;
                    }

                    item = response_stream.next() => {
                        let Some(item) = item else {
                            break;
                        };
                        guard.on_item(&item).await;
                        yield item;
                    }
                }
            }

            guard.finish().await;
        });
        Ok(ResponseStream::new(wrapped_stream, stream_context))
    }

    /// Layer-2 KV-migration: if the sticky-pinned (hot) rank is overloaded
    /// relative to the coldest eligible rank, rebind the session to the cold
    /// rank and return `(old_hot, cold)` so the caller can redirect the
    /// dispatch and inject `migrate_from`. Returns `None` when no rebind fires
    /// (non-prefill phase, no sticky session, within hysteresis, or in
    /// cooldown).
    ///
    /// Runs on the LIVE prefill path (`select_and_dispatch_prefill`), the only
    /// place with both `self.sticky` and `self.chooser` load data in scope.
    async fn check_and_trigger_rebind(
        &self,
        request: &SingleIn<PreprocessedRequest>,
        phase: RequestPhase,
        hot: WorkerWithDpRank,
    ) -> Option<(WorkerWithDpRank, WorkerWithDpRank)> {
        if phase != RequestPhase::Prefill {
            return None;
        }
        // Phase-gated sticky session id; also confirms session_control exists.
        let session_id = self.sticky.session_id_for_phase(request, phase)?.to_string();

        // F1: only rebind an EXISTING binding (affinity HIT). A brand-new
        // session's first turn carries session_control but has no prior home
        // rank — there is nothing to migrate FROM. `hot` is the freshly-selected
        // worker, not proof of a prior sticky home, so it cannot substitute for
        // this check. `worker_for_phase` -> `peek_session` is side-effect-free
        // (does NOT refresh TTL), so gating here is observationally pure.
        self.sticky.worker_for_phase(request, phase)?;

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

        let hot_load = loads
            .iter()
            .find(|l| l.worker_id == hot.worker_id && l.dp_rank == hot.dp_rank)?
            .potential_prefill_tokens;
        let cold = loads
            .iter()
            .filter(|l| !(l.worker_id == hot.worker_id && l.dp_rank == hot.dp_rank))
            .min_by_key(|l| l.potential_prefill_tokens)?;

        // Hysteresis: only rebind when the hot rank is meaningfully hotter.
        if hot_load.saturating_sub(cold.potential_prefill_tokens) <= REBIND_HYSTERESIS_TOKENS {
            return None;
        }
        // Cooldown: avoid thrash on a session we just moved.
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

        // F3: the store.put inside rebind() is immediately overwritten by the
        // identical on_routed() bind once `selection` is redirected to the cold
        // rank by the caller (dispatch_selection -> on_routed -> bind, same
        // store key). Only the last_rebind COOLDOWN STAMP set by rebind() is
        // load-bearing here. The double-put is intentional/harmless, not a bug;
        // do not "optimize" it away without first moving the cooldown stamp into
        // on_routed (rebind() has dedicated tests in sticky/router.rs that a
        // split would force-update).
        let cold_target = WorkerWithDpRank::new(cold.worker_id, cold.dp_rank);
        let old = self.sticky.rebind(&session_id, cold_target, ttl);
        let old_hot = old.unwrap_or(hot);

        // G2: confirm at runtime that this LIVE branch fires.
        tracing::info!(
            %session_id,
            hot_worker_id = old_hot.worker_id,
            hot_dp_rank = old_hot.dp_rank,
            hot_potential_prefill_tokens = hot_load,
            cold_worker_id = cold_target.worker_id,
            cold_dp_rank = cold_target.dp_rank,
            cold_potential_prefill_tokens = cold.potential_prefill_tokens,
            "Layer-2 rebind fired on live prefill path (select_and_dispatch_prefill)"
        );

        Some((old_hot, cold_target))
    }

    /// Resolve the source (hot) worker's deterministic SGLang migration peer
    /// endpoint, `tcp://{host}:{MIGRATION_PEER_PORT_BASE + dp_rank}`, for the
    /// `migrate_from` directive.
    ///
    /// The host comes from the source worker's registry entry. Only TCP
    /// request-plane addresses carry a routable IP; in NATS request-plane mode
    /// `transport.address()` is a subject, not an `ip:port`, so we return `None`
    /// (the caller skips the directive but keeps the cold-rank rebind). The TCP
    /// address string is `host:port[/endpoint]`, optionally `tcp://`-prefixed,
    /// and may be IPv6 (`[::1]:port`); we parse it as a `SocketAddr` to extract
    /// the host correctly rather than string-splitting on `:`.
    fn resolve_migration_endpoint(&self, src: WorkerWithDpRank) -> Option<String> {
        let inst = self
            .chooser
            .client()
            .instances()
            .into_iter()
            .find(|i| i.instance_id == src.worker_id)?;

        let addr_str = match &inst.transport {
            TransportType::Tcp(addr) => addr.as_str(),
            // NATS request plane: address is a subject, not a routable host.
            // Migration is unsupported here (documented limitation); skip.
            TransportType::Nats(_) => return None,
        };

        // Strip an optional `tcp://` scheme and any `/endpoint` suffix, then
        // parse the remaining `host:port` as a SocketAddr (IPv6-safe).
        let trimmed = addr_str.strip_prefix("tcp://").unwrap_or(addr_str);
        let socket_part = trimmed.split('/').next()?;
        let socket: std::net::SocketAddr = socket_part.parse().ok()?;
        let host = socket.ip();

        // Port = base + dp_rank, guarded against u16 overflow (dp_rank is small
        // in practice; this only trips on a misconfigured base or absurd rank).
        let port = u16::try_from(src.dp_rank)
            .ok()
            .and_then(|r| MIGRATION_PEER_PORT_BASE.checked_add(r))?;
        // SocketAddr's Display brackets IPv6 hosts; reconstruct so the migration
        // port (not the original request port) is used while keeping IPv6 valid.
        let migration_socket = std::net::SocketAddr::new(host, port);
        Some(format!("tcp://{migration_socket}"))
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
        let mut selection = self.select_request(&request, phase, false).await?;

        // --- Layer-2 KV-migration rebind hook (LIVE prefill path) ---
        // `selection` is the single value consumed by both `prepare` (below)
        // and `dispatch_selection`; overwriting it redirects the REAL forwarded
        // request, not a dead branch (G2). The OLD/hot source rides on
        // `request.migrate_from` because `prepare` only receives the cold rank.
        let hot = WorkerWithDpRank::new(selection.instance_id, selection.dp_rank);
        if let Some((old_hot, cold)) = self.check_and_trigger_rebind(&request, phase, hot).await {
            // Redirect the REAL forwarded request to the cold rank regardless of
            // whether we can build a migrate_from directive (the rebind itself
            // is the load-balancing action; migrate_from is the KV-reuse
            // optimization on top of it).
            selection.instance_id = cold.worker_id;
            selection.dp_rank = cold.dp_rank;

            // Resolve the source (hot) worker_id -> host via the registry, then
            // build the deterministic SGLang migration peer endpoint. May be
            // None in NATS request-plane mode or if the source worker is gone.
            match self.resolve_migration_endpoint(old_hot) {
                Some(source_endpoint) => {
                    let session_id = self
                        .sticky
                        .session_id_for_phase(&request, phase)
                        .map(str::to_string)
                        .unwrap_or_default();
                    request.migrate_from = Some(MigrateFrom {
                        source_endpoint,
                        source_dp_rank: old_hot.dp_rank,
                        session_id,
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
        // --- end rebind hook ---

        let mut guard = self.track_selection(&request, &mut selection).await?;
        let metadata = match prepare(&mut request, selection.instance_id, Some(selection.dp_rank)) {
            Ok(metadata) => metadata,
            Err(error) => {
                guard.abort().await;
                return Err(error);
            }
        };
        drop(route_guard);
        let stream = self
            .dispatch_selection(request, selection, guard, true)
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
        self.dispatch_selection(request, selection, guard, false)
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
