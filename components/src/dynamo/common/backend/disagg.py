# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tiny helpers shared across unified-backend engines for disagg dispatch.

Each engine's `generate()` consults `WorkerConfig.disaggregation_mode` and
branches on it. The functions below capture the two patterns that recur in
multiple engines so we don't reinvent them per backend:

* :func:`enforce_prefill_max_tokens` — clamp `stop_conditions.max_tokens`
  to 1 in PREFILL mode. Prefill workers only need the KV cache populated for
  the prompt; producing more than one token is wasted compute and breaks
  the prefill→decode handoff contract.

* :func:`extract_prefill_result` — pull the optional ``prefill_result`` dict
  out of the request.  Returns ``None`` when the request didn't pass through
  the frontend's prefill router (e.g. aggregated mode).

These are utilities, not abstractions. Backends are free to inline the
behavior if their generate path is shaped differently.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Optional

from dynamo.common.backend.engine import GenerateChunk, GenerateRequest
from dynamo.common.constants import DisaggregationMode


# Existing response metadata used as an internal prefill-router ACK. Backends
# must emit this only after the prefill stream has completed successfully.
PREFILL_COMPLETE_MARKER_KEY = "dynamo_prefill_complete"
PREFILL_COMPLETE_CAPABILITY_KEY = "kv_session_rebind_v1"
PREFILL_TERMINAL_PENDING = "pending"
PREFILL_TERMINAL_SUCCESS = "success"
PREFILL_TERMINAL_FAILURE = "failure"
PREFILL_TERMINAL_UNKNOWN = "unknown"

PREFILL_BEFORE_ACK_TEST_ENV = "DYN_SGLANG_TEST_PREFILL_BEFORE_ACK"
PREFILL_BEFORE_ACK_TEST_DELAY_MS_ENV = "DYN_SGLANG_TEST_PREFILL_BEFORE_ACK_DELAY_MS"
PREFILL_BEFORE_ACK_TEST_DELAY_MARKER = "_dynamo_test_prefill_before_ack_delay"
PREFILL_BEFORE_ACK_TEST_FAIL_MARKER = "_dynamo_test_prefill_before_ack_fail"
PREFILL_BEFORE_ACK_TEST_FAILURE = (
    "deterministic prefill failure after bootstrap before completion ACK"
)
PREFILL_BEFORE_ACK_TEST_CONFLICT = "conflicting prefill-before-ACK test hook markers"

_PREFILL_BEFORE_ACK_TEST_DELAY_MS_DEFAULT = 15_000
_PREFILL_BEFORE_ACK_TEST_DELAY_MS_MIN = 1
_PREFILL_BEFORE_ACK_TEST_DELAY_MS_MAX = 30_000

_PREFILL_SUCCESS_FINISH_REASONS = {"stop", "length"}
_PREFILL_FAILURE_FINISH_REASONS = {
    "abort",
    "aborted",
    "error",
    "cancel",
    "cancelled",
}


def prefill_complete_marker() -> GenerateChunk:
    """Return the internal ACK emitted after successful prefill completion."""
    return {
        "token_ids": [],
        "index": 0,
        "extra_args": {PREFILL_COMPLETE_MARKER_KEY: True},
    }


def _prefill_before_ack_test_fields(request: GenerateRequest) -> set[str]:
    sources: list[Any] = [request.get("nvext")]
    extra_args = request.get("extra_args")
    if isinstance(extra_args, dict):
        sources.append(extra_args.get("nvext"))

    fields: set[str] = set()
    for source in sources:
        if not isinstance(source, dict):
            continue
        extra_fields = source.get("extra_fields")
        if not isinstance(extra_fields, list):
            continue
        fields.update(field for field in extra_fields if isinstance(field, str))
    return fields


async def run_prefill_before_ack_test_hook(
    request: GenerateRequest, request_id: str | None
) -> None:
    """Run the opt-in SGLang prefill recovery hook immediately before ACK.

    The hook has two independent gates: an exact environment opt-in and a
    request marker carried by the existing ``nvext.extra_fields`` passthrough.
    With either gate absent it is a strict no-op.
    """
    if os.getenv(PREFILL_BEFORE_ACK_TEST_ENV) != "1":
        return

    fields = _prefill_before_ack_test_fields(request)
    delay = PREFILL_BEFORE_ACK_TEST_DELAY_MARKER in fields
    fail = PREFILL_BEFORE_ACK_TEST_FAIL_MARKER in fields
    if not delay and not fail:
        return
    if delay and fail:
        logging.warning(
            "[prefill-ack-test-hook] action=conflict state=triggered rid=%s",
            request_id,
        )
        raise RuntimeError(PREFILL_BEFORE_ACK_TEST_CONFLICT)
    if fail:
        logging.warning(
            "[prefill-ack-test-hook] action=fail state=triggered rid=%s",
            request_id,
        )
        raise RuntimeError(PREFILL_BEFORE_ACK_TEST_FAILURE)

    raw_delay_ms = os.getenv(
        PREFILL_BEFORE_ACK_TEST_DELAY_MS_ENV,
        str(_PREFILL_BEFORE_ACK_TEST_DELAY_MS_DEFAULT),
    )
    try:
        delay_ms = int(raw_delay_ms)
    except ValueError as exc:
        raise ValueError(
            f"{PREFILL_BEFORE_ACK_TEST_DELAY_MS_ENV} must be an integer"
        ) from exc
    if not (
        _PREFILL_BEFORE_ACK_TEST_DELAY_MS_MIN
        <= delay_ms
        <= _PREFILL_BEFORE_ACK_TEST_DELAY_MS_MAX
    ):
        raise ValueError(
            f"{PREFILL_BEFORE_ACK_TEST_DELAY_MS_ENV} must be between "
            f"{_PREFILL_BEFORE_ACK_TEST_DELAY_MS_MIN} and "
            f"{_PREFILL_BEFORE_ACK_TEST_DELAY_MS_MAX}"
        )

    logging.warning(
        "[prefill-ack-test-hook] action=delay state=entered rid=%s delay_ms=%d",
        request_id,
        delay_ms,
    )
    await asyncio.sleep(delay_ms / 1000)


def classify_prefill_terminal(payload: Any) -> str:
    """Classify one SGLang result without confusing normal stop with abort.

    SGLang normally serializes finish reasons as ``{"type": "stop"}``,
    ``{"type": "length"}``, or ``{"type": "abort", ...}``. Tests and
    internal adapters can also expose the finish-reason object or a string, so
    those shapes are normalized here. An unknown non-empty reason is not safe
    evidence of prefill completion and is classified separately for callers to
    reject fail-closed.
    """
    if not isinstance(payload, dict):
        return PREFILL_TERMINAL_UNKNOWN

    meta_info = payload.get("meta_info")
    if isinstance(meta_info, dict) and "finish_reason" in meta_info:
        reason = meta_info["finish_reason"]
    elif "finish_reason" in payload:
        reason = payload["finish_reason"]
    else:
        return PREFILL_TERMINAL_PENDING

    if reason is None:
        return PREFILL_TERMINAL_PENDING

    to_json = getattr(reason, "to_json", None)
    if callable(to_json):
        try:
            reason = to_json()
        except Exception:
            return PREFILL_TERMINAL_UNKNOWN

    if isinstance(reason, dict):
        reason = reason.get("type")
    if not isinstance(reason, str) or not reason.strip():
        return PREFILL_TERMINAL_UNKNOWN

    normalized = reason.strip().lower().replace("-", "_")
    normalized = normalized.split(":", 1)[0]
    if normalized.startswith("finish_"):
        normalized = normalized.removeprefix("finish_")

    if normalized in _PREFILL_SUCCESS_FINISH_REASONS:
        return PREFILL_TERMINAL_SUCCESS
    if normalized in _PREFILL_FAILURE_FINISH_REASONS:
        return PREFILL_TERMINAL_FAILURE
    return PREFILL_TERMINAL_UNKNOWN


def enforce_prefill_max_tokens(request: GenerateRequest) -> GenerateRequest:
    """In-place clamp ``stop_conditions.max_tokens`` to 1.

    Caller is responsible for only invoking this on prefill workers — the
    helper does not check the disaggregation mode itself.
    """
    stop = request.setdefault("stop_conditions", {})  # type: ignore[typeddict-item]
    stop["max_tokens"] = 1
    return request


def extract_prefill_result(request: GenerateRequest) -> Optional[dict[str, Any]]:
    """Return the request's ``prefill_result`` dict, or ``None`` if absent.

    The frontend's prefill router sets this on decode-bound requests; engines
    use the embedded ``disaggregated_params`` to resume from the KV cache the
    prefill peer already populated.
    """
    return request.get("prefill_result")  # type: ignore[return-value]


def require_prefill_result(
    request: GenerateRequest, mode: DisaggregationMode
) -> dict[str, Any]:
    """Like :func:`extract_prefill_result` but raises when the request is
    missing the prefill handoff payload — the canonical decode-mode
    pre-condition. Pass the mode in so the error message can name the role.
    """
    prefill_result = extract_prefill_result(request)
    if prefill_result is None:
        raise ValueError(
            f"{mode.value} worker received request with no prefill_result; "
            "expected the frontend's prefill router to forward "
            "disaggregated_params from a prefill peer"
        )
    return prefill_result
