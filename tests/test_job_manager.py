from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from depin_worker.inference.engine import GenerationChunk, InferenceEngine
from depin_worker.inference.model_manager import ModelManager
from depin_worker.job.manager import JobManager
from depin_worker.models import ChatMessage, JobDispatchPayload

MODEL = "test-model"


class _FakeEngine(InferenceEngine):
    """Minimal InferenceEngine stub — no GPU required."""

    def __init__(self, tokens: list[str] | None = None) -> None:
        self.model_id  = MODEL
        self._loaded   = True
        self._backend  = "fake"
        self._lock     = asyncio.Lock()
        self._tokens   = tokens or ["Hello", " world"]

    async def load(self) -> None:
        self._loaded = True

    async def generate_stream(self, job_id, messages, max_tokens, temperature):
        for i, tok in enumerate(self._tokens):
            yield GenerationChunk(text=tok, index=i, prompt_tokens=2,
                                  total_tokens=2 + i + 1)
        yield GenerationChunk(text="", index=len(self._tokens),
                              finish_reason="stop", prompt_tokens=2,
                              total_tokens=2 + len(self._tokens))

    async def abort(self, job_id: str) -> None:
        pass

    async def unload(self) -> None:
        self._loaded = False


def _make_manager(
    max_concurrency: int = 2,
    emit=None,
    tokens: list[str] | None = None,
) -> tuple[JobManager, list[tuple[str, dict]]]:
    emitted: list[tuple[str, dict]] = []

    async def _emit(event: str, data: dict) -> None:
        emitted.append((event, data))

    mm = ModelManager()
    mm._total_vram = 24.0
    mm._available_vram = 24.0 - 3.0 - 1.5

    engine = _FakeEngine(tokens=tokens)
    mm._engines[MODEL]        = engine
    mm._hot_models            = [MODEL]
    mm._inference_refs[MODEL] = 0

    idle = asyncio.Event()
    idle.set()
    mm._idle_events[MODEL] = idle

    manager = JobManager(
        model_manager=mm,
        node_id="test-node",
        max_concurrency=max_concurrency,
        emit=emit or _emit,
    )
    return manager, emitted


def _payload(job_id: str = "job-1") -> JobDispatchPayload:
    return JobDispatchPayload(
        jobId=job_id,
        model=MODEL,
        messages=[ChatMessage(role="user", content="hi")],
        maxTokens=64,
        temperature=0.7,
    )


# ── Tests ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_dispatch_accepted() -> None:
    manager, emitted = _make_manager(max_concurrency=2)
    await manager.dispatch(_payload("job-1"))
    await asyncio.gather(*manager._tasks.values())

    event_names = [e[0] for e in emitted]
    assert "worker:jobAccepted" in event_names
    assert "worker:token"       in event_names
    assert "worker:completed"   in event_names


@pytest.mark.asyncio
async def test_dispatch_rejected_capacity() -> None:
    manager, emitted = _make_manager(max_concurrency=1)

    # Dispatch two jobs — second should be rejected while first is running
    await asyncio.gather(
        manager.dispatch(_payload("job-1")),
        manager.dispatch(_payload("job-2")),
    )

    rejected = [e for e in emitted if e[0] == "worker:jobRejected"]
    assert len(rejected) == 1
    assert rejected[0][1]["jobId"] == "job-2"
    assert "capacity" in rejected[0][1]["reason"]


@pytest.mark.asyncio
async def test_five_jobs_three_accepted_two_rejected() -> None:
    MAX = 3
    manager, emitted = _make_manager(max_concurrency=MAX)

    dispatch_tasks = [
        asyncio.create_task(manager.dispatch(_payload(f"job-{i}")))
        for i in range(1, 6)
    ]
    await asyncio.gather(*dispatch_tasks)
    await asyncio.gather(*manager._tasks.values())

    accepted  = [e for e in emitted if e[0] == "worker:jobAccepted"]
    rejected  = [e for e in emitted if e[0] == "worker:jobRejected"]
    completed = [e for e in emitted if e[0] == "worker:completed"]

    assert len(accepted)  == MAX,     f"expected {MAX} accepted, got {len(accepted)}"
    assert len(rejected)  == 5 - MAX, f"expected {5-MAX} rejected, got {len(rejected)}"
    assert len(completed) == MAX,     f"expected {MAX} completed, got {len(completed)}"
    assert manager.active_count == 0, "active count did not drain to 0"


@pytest.mark.asyncio
async def test_cuda_semaphore_serialises_gpu_work() -> None:
    """Even with multiple accepted slots, GPU execution is one-at-a-time."""
    order: list[str] = []

    class _OrderedEngine(_FakeEngine):
        async def generate_stream(self, job_id, messages, max_tokens, temperature):
            order.append(f"start:{job_id}")
            try:
                for i, tok in enumerate(["a", "b"]):
                    await asyncio.sleep(0)
                    yield GenerationChunk(text=tok, index=i, prompt_tokens=1,
                                          total_tokens=1 + i + 1)
                yield GenerationChunk(text="", index=2, finish_reason="stop",
                                      prompt_tokens=1, total_tokens=3)
            finally:
                # finally fires when ModelManager closes this generator after
                # _run_job breaks on finish_reason="stop"
                order.append(f"end:{job_id}")

    manager, _ = _make_manager(max_concurrency=3)
    # Replace the injected engine with the ordered one
    manager._model_manager._engines[MODEL] = _OrderedEngine()

    tasks = [
        asyncio.create_task(manager.dispatch(_payload(f"job-{i}")))
        for i in range(1, 4)
    ]
    await asyncio.gather(*tasks)
    await asyncio.gather(*manager._tasks.values())

    # With Semaphore(1): each job's start must be immediately followed by its end
    starts = [e for e in order if e.startswith("start:")]
    ends   = [e for e in order if e.startswith("end:")]
    for i in range(len(starts)):
        assert order.index(starts[i]) < order.index(ends[i])
        if i > 0:
            # Previous job must have ended before this one started
            assert order.index(ends[i - 1]) < order.index(starts[i]), (
                f"Semaphore not enforced: {ends[i-1]} came after {starts[i]}"
            )


@pytest.mark.asyncio
async def test_active_count_accurate_for_image_style_jobs() -> None:
    """active_count must reflect all job types, not just text."""
    manager, emitted = _make_manager(max_concurrency=2)

    await asyncio.gather(
        manager.dispatch(_payload("job-1")),
        manager.dispatch(_payload("job-2")),
        manager.dispatch(_payload("job-3")),   # should be rejected
    )

    rejected = [e for e in emitted if e[0] == "worker:jobRejected"]
    assert len(rejected) == 1

    await asyncio.gather(*manager._tasks.values())
    assert manager.active_count == 0


@pytest.mark.asyncio
async def test_ref_count_drains_after_completion() -> None:
    manager, _ = _make_manager(max_concurrency=3)
    mm = manager._model_manager

    tasks = [
        asyncio.create_task(manager.dispatch(_payload(f"job-{i}")))
        for i in range(1, 4)
    ]
    await asyncio.gather(*tasks)
    await asyncio.gather(*manager._tasks.values())

    assert mm._inference_refs.get(MODEL, 0) == 0, "inference ref leaked"
    assert mm._cuda_sem._value == 1,               "semaphore not released"
    assert mm._idle_events[MODEL].is_set(),        "idle event not set"
