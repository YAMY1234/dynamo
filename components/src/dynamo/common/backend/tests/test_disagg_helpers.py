# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the shared disagg request helpers."""

from __future__ import annotations

import pytest

import dynamo.common.backend.disagg as disagg
from dynamo.common.backend.disagg import (
    PREFILL_BEFORE_ACK_TEST_CONFLICT,
    PREFILL_BEFORE_ACK_TEST_DELAY_MARKER,
    PREFILL_BEFORE_ACK_TEST_DELAY_MS_ENV,
    PREFILL_BEFORE_ACK_TEST_ENV,
    PREFILL_BEFORE_ACK_TEST_FAIL_MARKER,
    PREFILL_BEFORE_ACK_TEST_FAILURE,
    PREFILL_COMPLETE_MARKER_KEY,
    PREFILL_TERMINAL_FAILURE,
    PREFILL_TERMINAL_PENDING,
    PREFILL_TERMINAL_SUCCESS,
    PREFILL_TERMINAL_UNKNOWN,
    classify_prefill_terminal,
    enforce_prefill_max_tokens,
    extract_prefill_result,
    prefill_complete_marker,
    require_prefill_result,
    run_prefill_before_ack_test_hook,
)
from dynamo.common.constants import DisaggregationMode

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]


def test_prefill_complete_marker_uses_internal_extra_args_field():
    assert prefill_complete_marker() == {
        "token_ids": [],
        "index": 0,
        "extra_args": {PREFILL_COMPLETE_MARKER_KEY: True},
    }


def _hook_request(*fields: str) -> dict:
    return {"extra_args": {"nvext": {"extra_fields": list(fields)}}}


@pytest.mark.parametrize("env_value", [None, "0", "true", "01"])
async def test_prefill_before_ack_hook_is_strict_noop_without_exact_env(
    monkeypatch, env_value
):
    if env_value is None:
        monkeypatch.delenv(PREFILL_BEFORE_ACK_TEST_ENV, raising=False)
    else:
        monkeypatch.setenv(PREFILL_BEFORE_ACK_TEST_ENV, env_value)

    await run_prefill_before_ack_test_hook(
        _hook_request(PREFILL_BEFORE_ACK_TEST_FAIL_MARKER), "env-off"
    )


@pytest.mark.parametrize(
    "payload",
    [
        {},
        _hook_request("worker_id"),
        {"extra_fields": [PREFILL_BEFORE_ACK_TEST_FAIL_MARKER]},
        {"extra_args": {"extra_fields": [PREFILL_BEFORE_ACK_TEST_FAIL_MARKER]}},
        {
            "extra_args": {
                "nvext": {"extra_fields": PREFILL_BEFORE_ACK_TEST_FAIL_MARKER}
            }
        },
    ],
)
async def test_prefill_before_ack_hook_is_noop_without_marker_list(
    monkeypatch, payload
):
    monkeypatch.setenv(PREFILL_BEFORE_ACK_TEST_ENV, "1")
    await run_prefill_before_ack_test_hook(payload, "marker-off")


@pytest.mark.parametrize("delay_ms", [1, 30_000])
async def test_prefill_before_ack_delay_hook_records_entry_and_waits(
    monkeypatch, caplog, delay_ms
):
    sleeps = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setenv(PREFILL_BEFORE_ACK_TEST_ENV, "1")
    monkeypatch.setenv(PREFILL_BEFORE_ACK_TEST_DELAY_MS_ENV, str(delay_ms))
    monkeypatch.setattr(disagg.asyncio, "sleep", fake_sleep)

    await run_prefill_before_ack_test_hook(
        _hook_request(PREFILL_BEFORE_ACK_TEST_DELAY_MARKER), "delay-rid"
    )

    assert sleeps == [delay_ms / 1000]
    assert (
        f"action=delay state=entered rid=delay-rid delay_ms={delay_ms}" in caplog.text
    )


@pytest.mark.parametrize("delay_ms", ["0", "30001", "not-an-int"])
async def test_prefill_before_ack_delay_hook_enforces_bounded_integer(
    monkeypatch, delay_ms
):
    monkeypatch.setenv(PREFILL_BEFORE_ACK_TEST_ENV, "1")
    monkeypatch.setenv(PREFILL_BEFORE_ACK_TEST_DELAY_MS_ENV, delay_ms)

    with pytest.raises(ValueError, match=PREFILL_BEFORE_ACK_TEST_DELAY_MS_ENV):
        await run_prefill_before_ack_test_hook(
            _hook_request(PREFILL_BEFORE_ACK_TEST_DELAY_MARKER), "bad-delay"
        )


async def test_prefill_before_ack_fail_hook_records_then_raises_fixed_error(
    monkeypatch, caplog
):
    monkeypatch.setenv(PREFILL_BEFORE_ACK_TEST_ENV, "1")

    with pytest.raises(RuntimeError, match=PREFILL_BEFORE_ACK_TEST_FAILURE):
        await run_prefill_before_ack_test_hook(
            _hook_request(PREFILL_BEFORE_ACK_TEST_FAIL_MARKER), "fail-rid"
        )

    assert "action=fail state=triggered rid=fail-rid" in caplog.text


async def test_prefill_before_ack_hook_rejects_conflicting_markers_deterministically(
    monkeypatch, caplog
):
    monkeypatch.setenv(PREFILL_BEFORE_ACK_TEST_ENV, "1")
    monkeypatch.setenv(PREFILL_BEFORE_ACK_TEST_DELAY_MS_ENV, "not-an-int")

    with pytest.raises(RuntimeError, match=PREFILL_BEFORE_ACK_TEST_CONFLICT):
        await run_prefill_before_ack_test_hook(
            _hook_request(
                PREFILL_BEFORE_ACK_TEST_DELAY_MARKER,
                PREFILL_BEFORE_ACK_TEST_FAIL_MARKER,
            ),
            "conflict-rid",
        )

    assert "action=conflict state=triggered rid=conflict-rid" in caplog.text


@pytest.mark.parametrize("reason", [{"type": "stop"}, {"type": "length"}, "stop"])
def test_prefill_terminal_classifier_accepts_normal_completion(reason):
    assert (
        classify_prefill_terminal({"meta_info": {"finish_reason": reason}})
        == PREFILL_TERMINAL_SUCCESS
    )


@pytest.mark.parametrize(
    "reason", [{"type": "abort"}, "abort", "error: transport", "cancelled"]
)
def test_prefill_terminal_classifier_rejects_failures(reason):
    assert (
        classify_prefill_terminal({"meta_info": {"finish_reason": reason}})
        == PREFILL_TERMINAL_FAILURE
    )


def test_prefill_terminal_classifier_handles_object_and_missing_reason():
    class FinishAbort:
        def to_json(self):
            return {"type": "abort", "message": "failed"}

    assert (
        classify_prefill_terminal({"meta_info": {"finish_reason": FinishAbort()}})
        == PREFILL_TERMINAL_FAILURE
    )
    assert classify_prefill_terminal({"meta_info": {}}) == PREFILL_TERMINAL_PENDING
    assert (
        classify_prefill_terminal({"meta_info": {"finish_reason": {}}})
        == PREFILL_TERMINAL_UNKNOWN
    )


def test_enforce_prefill_max_tokens_clamps_to_one():
    request = {
        "token_ids": [1, 2, 3],
        "stop_conditions": {"max_tokens": 64},
    }
    enforce_prefill_max_tokens(request)
    assert request["stop_conditions"]["max_tokens"] == 1


def test_enforce_prefill_max_tokens_creates_missing_section():
    # Some test fixtures omit `stop_conditions` entirely. The helper must
    # create it rather than KeyError-ing on the lookup, otherwise engines
    # would have to guard every call site.
    request = {"token_ids": [1, 2, 3]}
    enforce_prefill_max_tokens(request)
    assert request["stop_conditions"]["max_tokens"] == 1


def test_extract_prefill_result_returns_none_when_absent():
    assert extract_prefill_result({"token_ids": []}) is None


def test_extract_prefill_result_passes_through_value():
    payload = {"disaggregated_params": {"handle": "abc"}}
    out = extract_prefill_result({"token_ids": [], "prefill_result": payload})
    assert out == payload


def test_require_prefill_result_returns_value_when_present():
    payload = {"disaggregated_params": {"handle": "abc"}}
    out = require_prefill_result(
        {"token_ids": [], "prefill_result": payload},
        DisaggregationMode.DECODE,
    )
    assert out == payload


def test_require_prefill_result_raises_when_absent():
    # Decode workers can't proceed without the prefill peer's payload —
    # surface the misconfiguration eagerly. This pre-condition lives in
    # the shared helper so every backend's decode path can rely on it.
    with pytest.raises(ValueError, match="prefill_result"):
        require_prefill_result({"token_ids": []}, DisaggregationMode.DECODE)
