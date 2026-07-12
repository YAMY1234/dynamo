# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pin SGLang PREFILL probe, completion-ACK, and early-close contracts."""

from __future__ import annotations

import asyncio
import importlib.util
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any, cast

import pytest

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.gpu_1,
    pytest.mark.profiled_vram_gib(0),
    pytest.mark.pre_merge,
    pytest.mark.skipif(
        importlib.util.find_spec("sglang") is None,
        reason="sglang not installed in this container",
    ),
]


class _FakeContext:
    def __init__(self) -> None:
        self.stopped = False

    @property
    def trace_id(self) -> str:
        return "probe-1"

    def is_stopped(self) -> bool:
        return self.stopped

    def id(self) -> str:
        return self.trace_id


def _build_prefill_engine(
    stream_factory,
    *,
    captured_generate_kwargs: dict[str, Any] | None = None,
    supports_migrate_from: bool = True,
):
    from dynamo.common.constants import DisaggregationMode
    from dynamo.sglang.llm_engine import SglangLLMEngine

    server_args = SimpleNamespace(skip_tokenizer_init=True)
    dynamo_args = SimpleNamespace(use_sglang_tokenizer=False)
    engine = SglangLLMEngine(server_args, dynamo_args, DisaggregationMode.PREFILL)
    engine._input_param_manager = SimpleNamespace(
        get_input_param=lambda req, use_tokenizer: req.get("token_ids", [])
    )
    engine._bootstrap_host = "127.0.0.1"
    engine._bootstrap_port = 18000

    async def _async_generate(**kwargs):
        if captured_generate_kwargs is not None:
            captured_generate_kwargs.update(kwargs)
        return stream_factory()

    engine.engine = SimpleNamespace(
        async_generate=_async_generate,
        tokenizer_manager=SimpleNamespace(abort_request=lambda **_: None),
    )
    engine._engine_supports_migrate_from = supports_migrate_from
    return engine


def _build_legacy_prefill_handler(stream_factory):
    from dynamo.sglang.request_handlers.llm.prefill_handler import (
        PrefillWorkerHandler,
    )

    aborted: list[str | None] = []

    async def _async_generate(**_kwargs):
        return stream_factory()

    @asynccontextmanager
    async def _no_cancellation_monitor(*_args, **_kwargs):
        yield

    handler = PrefillWorkerHandler.__new__(PrefillWorkerHandler)
    handler.engine = SimpleNamespace(
        async_generate=_async_generate,
        tokenizer_manager=SimpleNamespace(
            abort_request=lambda *, rid, abort_all: aborted.append(rid)
        ),
    )
    handler.bootstrap_host = "127.0.0.1"
    handler.bootstrap_port = 18000
    handler.enable_trace = False
    handler._engine_supports_migrate_from = True
    handler._consume_tasks = set()
    handler._generate_bootstrap_room = lambda: 42
    handler._get_input_param = lambda request: {"input_ids": request["token_ids"]}
    handler._resolve_lora = lambda _request: None
    handler._session_kwargs = lambda _request: {}
    handler._priority_kwargs = lambda _priority: {}
    handler._migrate_from_kwargs = lambda _request: {}
    handler._cancellation_monitor = _no_cancellation_monitor
    return handler, aborted


async def _drain(gen):
    return [c async for c in gen]


def _probe_request() -> dict[str, Any]:
    return {
        "token_ids": [1],
        "_HEALTH_CHECK": True,
        "stop_conditions": {"max_tokens": 1},
        "sampling_options": {"temperature": 0.0},
    }


def _prefill_request() -> dict[str, Any]:
    return {
        "token_ids": [1, 2, 3],
        "stop_conditions": {"max_tokens": 1},
        "sampling_options": {"temperature": 0.0},
        "bootstrap_info": {
            "bootstrap_host": "127.0.0.1",
            "bootstrap_port": 18000,
            "bootstrap_room": 42,
        },
    }


def _legacy_prefill_request() -> dict[str, Any]:
    return {
        "request": {
            "token_ids": [1, 2, 3],
            "routing": {},
            "bootstrap_info": {
                "bootstrap_host": "127.0.0.1",
                "bootstrap_port": 18000,
                "bootstrap_room": 42,
            },
        },
        "sampling_params": {"max_new_tokens": 1},
    }


async def _successful_prefill_stream():
    yield {
        "meta_info": {"finish_reason": {"type": "stop"}},
        "output_ids": [],
    }


async def test_unified_prefill_forwards_migrate_from_as_sglang_tuple():
    captured: dict[str, Any] = {}
    engine = _build_prefill_engine(
        _successful_prefill_stream,
        captured_generate_kwargs=captured,
    )
    request = _prefill_request()
    request["migrate_from"] = ["tcp://10.0.0.5:20003", 3]

    await _drain(engine.generate(request, cast(Any, _FakeContext())))

    assert captured["migrate_from"] == ("tcp://10.0.0.5:20003", 3)


async def test_unified_prefill_without_migration_keeps_generate_kwargs_unchanged():
    captured: dict[str, Any] = {}
    engine = _build_prefill_engine(
        _successful_prefill_stream,
        captured_generate_kwargs=captured,
    )

    await _drain(engine.generate(_prefill_request(), cast(Any, _FakeContext())))

    assert "migrate_from" not in captured


async def test_unified_prefill_rejects_migration_when_engine_lacks_support():
    engine = _build_prefill_engine(
        _successful_prefill_stream,
        supports_migrate_from=False,
    )
    request = _prefill_request()
    request["migrate_from"] = ["tcp://10.0.0.5:20003", 3]

    with pytest.raises(RuntimeError, match="does not support migrate_from"):
        await anext(engine.generate(request, cast(Any, _FakeContext())))


async def test_prefill_probe_drains_stream_then_yields_single_terminal():
    consumed: list[dict] = []

    async def stream():
        for item in (
            {"meta_info": {"finish_reason": None}, "output_ids": []},
            {"meta_info": {"finish_reason": {"type": "stop"}}, "output_ids": []},
        ):
            consumed.append(item)
            yield item

    engine = _build_prefill_engine(stream)
    chunks = await _drain(engine.generate(_probe_request(), cast(Any, _FakeContext())))

    assert len(chunks) == 1, f"expected single terminal, got {chunks}"
    assert chunks[0]["finish_reason"] == "stop"
    assert "disaggregated_params" not in chunks[0]
    assert len(consumed) == 2


async def test_prefill_probe_yields_error_terminal_when_stream_raises():
    async def stream():
        yield {"meta_info": {"finish_reason": None}, "output_ids": []}
        raise RuntimeError("nixl transport down")

    engine = _build_prefill_engine(stream)
    chunks = await _drain(engine.generate(_probe_request(), cast(Any, _FakeContext())))

    assert len(chunks) == 1
    assert chunks[0]["finish_reason"].startswith("error:")
    assert "nixl transport down" in chunks[0]["finish_reason"]


@pytest.mark.parametrize("finish_type", ["stop", "length"])
async def test_prefill_completion_marker_is_emitted_only_after_inline_drain(
    finish_type: str,
):
    from dynamo.common.backend.disagg import PREFILL_COMPLETE_MARKER_KEY

    consumed: list[dict] = []

    async def stream():
        for item in (
            {"meta_info": {"finish_reason": None}, "output_ids": []},
            {
                "meta_info": {"finish_reason": {"type": finish_type}},
                "output_ids": [],
            },
        ):
            consumed.append(item)
            yield item

    engine = _build_prefill_engine(stream)
    response = engine.generate(_prefill_request(), cast(Any, _FakeContext()))

    bootstrap = await anext(response)
    assert bootstrap["disaggregated_params"]["bootstrap_room"] == 42
    assert consumed == []

    marker = await anext(response)
    assert marker["extra_args"] == {PREFILL_COMPLETE_MARKER_KEY: True}
    assert len(consumed) == 2
    with pytest.raises(StopAsyncIteration):
        await anext(response)


async def test_unified_prefill_test_fault_runs_after_bootstrap_and_suppresses_ack(
    monkeypatch,
):
    from dynamo.common.backend.disagg import (
        PREFILL_BEFORE_ACK_TEST_ENV,
        PREFILL_BEFORE_ACK_TEST_FAIL_MARKER,
        PREFILL_BEFORE_ACK_TEST_FAILURE,
    )

    monkeypatch.setenv(PREFILL_BEFORE_ACK_TEST_ENV, "1")
    engine = _build_prefill_engine(_successful_prefill_stream)
    request = _prefill_request()
    request["extra_args"] = {
        "nvext": {"extra_fields": [PREFILL_BEFORE_ACK_TEST_FAIL_MARKER]}
    }
    response = engine.generate(request, cast(Any, _FakeContext()))

    assert "disaggregated_params" in await anext(response)
    with pytest.raises(RuntimeError, match=PREFILL_BEFORE_ACK_TEST_FAILURE):
        await anext(response)
    with pytest.raises(StopAsyncIteration):
        await anext(response)


async def test_prefill_inline_drain_error_does_not_emit_completion_marker():
    async def stream():
        yield {"meta_info": {"finish_reason": None}, "output_ids": []}
        raise RuntimeError("prefill transport failed")

    engine = _build_prefill_engine(stream)
    aborted: list[str | None] = []
    engine._abort_sglang_request = aborted.append
    response = engine.generate(_prefill_request(), cast(Any, _FakeContext()))

    bootstrap = await anext(response)
    assert "disaggregated_params" in bootstrap
    with pytest.raises(RuntimeError, match="prefill transport failed"):
        await anext(response)
    assert aborted == ["probe-1"]


async def test_prefill_abort_terminal_does_not_emit_completion_marker():
    async def stream():
        yield {
            "meta_info": {
                "finish_reason": {
                    "type": "abort",
                    "message": "migration transport failed",
                }
            },
            "output_ids": [],
        }

    engine = _build_prefill_engine(stream)
    aborted: list[str | None] = []
    engine._abort_sglang_request = aborted.append
    response = engine.generate(_prefill_request(), cast(Any, _FakeContext()))

    assert "disaggregated_params" in await anext(response)
    with pytest.raises(RuntimeError, match="abort/error"):
        await anext(response)
    assert aborted == ["probe-1"]


async def test_prefill_empty_stream_does_not_emit_completion_marker():
    async def stream():
        if False:
            yield {}

    engine = _build_prefill_engine(stream)
    aborted: list[str | None] = []
    engine._abort_sglang_request = aborted.append
    response = engine.generate(_prefill_request(), cast(Any, _FakeContext()))

    assert "disaggregated_params" in await anext(response)
    with pytest.raises(RuntimeError, match="no results"):
        await anext(response)
    assert aborted == ["probe-1"]


async def test_prefill_inline_context_stop_does_not_emit_completion_marker():
    context = _FakeContext()
    aborted: list[str | None] = []

    async def stream():
        context.stopped = True
        yield {"meta_info": {"finish_reason": None}, "output_ids": []}

    engine = _build_prefill_engine(stream)
    engine._abort_sglang_request = aborted.append
    response = engine.generate(_prefill_request(), cast(Any, context))

    bootstrap = await anext(response)
    assert "disaggregated_params" in bootstrap
    with pytest.raises(asyncio.CancelledError):
        await anext(response)
    assert aborted == ["probe-1"]


async def test_direct_inline_drain_cancellation_aborts_bootstrap_room():
    started = asyncio.Event()

    async def stream():
        started.set()
        await asyncio.Event().wait()
        yield {}

    engine = _build_prefill_engine(stream)
    aborted: list[str | None] = []
    engine._abort_sglang_request = aborted.append
    engine._inflight_prefill_streams = 1
    task = asyncio.create_task(
        engine._consume_prefill_stream(
            stream(),
            cast(Any, _FakeContext()),
            "cancelled-rid",
            propagate_errors=True,
        )
    )
    await started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert aborted == ["cancelled-rid"]
    assert engine._inflight_prefill_streams == 0


async def test_prefill_immediate_aclose_aborts_and_closes_unclaimed_stream():
    class ClosableStream:
        def __init__(self) -> None:
            self.iterated = False
            self.closed = False

        def __aiter__(self):
            return self

        async def __anext__(self):
            self.iterated = True
            raise StopAsyncIteration

        async def aclose(self) -> None:
            self.closed = True

    underlying = ClosableStream()
    engine = _build_prefill_engine(lambda: underlying)
    aborted: list[str | None] = []
    engine._abort_sglang_request = aborted.append
    response = engine.generate(_prefill_request(), cast(Any, _FakeContext()))

    assert "disaggregated_params" in await anext(response)
    await response.aclose()

    assert aborted == ["probe-1"]
    assert underlying.closed
    assert not underlying.iterated
    assert engine._inflight_prefill_streams == 0
    with pytest.raises(StopAsyncIteration):
        await anext(response)


async def test_legacy_prefill_normal_stop_emits_completion_marker():
    from dynamo.common.backend.disagg import PREFILL_COMPLETE_MARKER_KEY

    async def stream():
        yield {
            "meta_info": {
                "id": "probe-1",
                "finish_reason": {"type": "stop"},
            },
            "output_ids": [],
        }

    handler, aborted = _build_legacy_prefill_handler(stream)
    response = handler.generate(_legacy_prefill_request(), cast(Any, _FakeContext()))

    assert "disaggregated_params" in await anext(response)
    marker = await anext(response)
    assert marker["extra_args"] == {PREFILL_COMPLETE_MARKER_KEY: True}
    assert aborted == []


async def test_legacy_prefill_test_fault_runs_after_bootstrap_and_suppresses_ack(
    monkeypatch,
):
    from dynamo.common.backend.disagg import (
        PREFILL_BEFORE_ACK_TEST_ENV,
        PREFILL_BEFORE_ACK_TEST_FAIL_MARKER,
        PREFILL_BEFORE_ACK_TEST_FAILURE,
    )

    async def stream():
        yield {
            "meta_info": {
                "id": "probe-1",
                "finish_reason": {"type": "stop"},
            },
            "output_ids": [],
        }

    monkeypatch.setenv(PREFILL_BEFORE_ACK_TEST_ENV, "1")
    handler, aborted = _build_legacy_prefill_handler(stream)
    request = _legacy_prefill_request()
    request["request"]["extra_args"] = {
        "nvext": {"extra_fields": [PREFILL_BEFORE_ACK_TEST_FAIL_MARKER]}
    }
    response = handler.generate(request, cast(Any, _FakeContext()))

    assert "disaggregated_params" in await anext(response)
    with pytest.raises(RuntimeError, match=PREFILL_BEFORE_ACK_TEST_FAILURE):
        await anext(response)
    with pytest.raises(StopAsyncIteration):
        await anext(response)
    assert aborted == []


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("abort", "abort/error"),
        ("empty", "no results"),
        ("exception", "legacy transport failed"),
    ],
)
async def test_legacy_prefill_failure_does_not_emit_completion_marker(
    case: str, message: str
):
    async def stream():
        if case == "abort":
            yield {
                "meta_info": {
                    "id": "probe-1",
                    "finish_reason": {"type": "abort", "message": "failed"},
                },
                "output_ids": [],
            }
        elif case == "exception":
            raise RuntimeError("legacy transport failed")

    handler, aborted = _build_legacy_prefill_handler(stream)
    response = handler.generate(_legacy_prefill_request(), cast(Any, _FakeContext()))

    assert "disaggregated_params" in await anext(response)
    with pytest.raises(RuntimeError, match=message):
        await anext(response)
    assert aborted == ["probe-1"]


async def test_legacy_prefill_context_stop_does_not_emit_completion_marker():
    context = _FakeContext()

    async def stream():
        context.stopped = True
        yield {
            "meta_info": {
                "id": "probe-1",
                "finish_reason": {"type": "stop"},
            },
            "output_ids": [],
        }

    handler, aborted = _build_legacy_prefill_handler(stream)
    response = handler.generate(_legacy_prefill_request(), cast(Any, context))

    assert "disaggregated_params" in await anext(response)
    with pytest.raises(asyncio.CancelledError):
        await anext(response)
    assert aborted == ["probe-1"]


async def test_legacy_prefill_immediate_aclose_cleans_registered_room():
    class ClosableStream:
        def __init__(self) -> None:
            self.iterated = False
            self.closed = False

        def __aiter__(self):
            return self

        async def __anext__(self):
            self.iterated = True
            raise StopAsyncIteration

        async def aclose(self) -> None:
            self.closed = True

    underlying = ClosableStream()
    handler, aborted = _build_legacy_prefill_handler(lambda: underlying)
    response = handler.generate(_legacy_prefill_request(), cast(Any, _FakeContext()))

    assert "disaggregated_params" in await anext(response)
    await response.aclose()

    assert aborted == ["probe-1"]
    assert underlying.closed
    assert not underlying.iterated
    with pytest.raises(StopAsyncIteration):
        await anext(response)
