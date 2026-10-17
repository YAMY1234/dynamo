// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod coordinator;
mod replica_sync;

use std::time::Duration;

use dynamo_runtime::{component::Client, pipeline::Error};

#[cfg(test)]
pub(crate) use coordinator::to_table;
pub use coordinator::{AffinityCoordinator, AffinityTarget, explicit_target};
pub(crate) use coordinator::{affinity_id, from_table, invalid_argument};
pub use dynamo_kv_router::scheduling::policy_config::{
    PhaseAffinityConfig, PhaseAffinityMode, SessionAffinityBinding, SessionAffinityPhases,
};
pub(crate) use dynamo_kv_router::services::selection::affinity::{
    Hold, subagent_group_affinity_id,
};
pub use dynamo_kv_router::services::selection::affinity::{
    MAX_SESSION_AFFINITY_TTL_SECS, SessionAffinityMode,
};

pub type LlmResponse =
    crate::types::Annotated<crate::protocols::common::llm_backend::LLMEngineOutput>;

/// The session affinity one routing hop applies after its phase overrides.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PhaseAffinity {
    /// `None` when the hop routes without session affinity.
    pub mode: Option<SessionAffinityMode>,
    pub binding: SessionAffinityBinding,
}

impl PhaseAffinity {
    /// The router-wide `mode` with parent-group binding: a hop without overrides.
    pub fn inherit(mode: SessionAffinityMode) -> Self {
        Self::resolve(PhaseAffinityConfig::default(), mode)
    }

    /// Apply one phase's overrides to the router-wide `mode`.
    pub fn resolve(config: PhaseAffinityConfig, mode: SessionAffinityMode) -> Self {
        Self {
            mode: match config.mode {
                None => Some(mode),
                Some(PhaseAffinityMode::Hard) => Some(SessionAffinityMode::Hard),
                Some(PhaseAffinityMode::Soft) => Some(SessionAffinityMode::Soft),
                Some(PhaseAffinityMode::Off) => None,
            },
            binding: config.binding.unwrap_or_default(),
        }
    }
}

pub(crate) async fn create_affinity_coordinator(
    ttl: Option<Duration>,
    mode: Option<SessionAffinityMode>,
    client: Client,
) -> Result<Option<AffinityCoordinator>, Error> {
    let (Some(ttl), Some(mode)) = (ttl, mode) else {
        return Ok(None);
    };
    let coordinator = AffinityCoordinator::new(ttl, mode)?;
    coordinator.enable_replica_sync(client).await?;
    Ok(Some(coordinator))
}

#[cfg(test)]
mod tests;
