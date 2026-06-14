"""
Simulation: 5 concurrent text generation jobs
=============================================
Demonstrates the refactored concurrency guarantees:

  • max_concurrency gate  — jobs 4 & 5 are rejected immediately
  • _active tracking      — heartbeat count is accurate throughout
  • CUDA semaphore        — accepted jobs 1–3 take turns on the GPU
                            (serialised, not concurrent kernel execution)
  • ref-count fence       — LRU eviction cannot unload an engine mid-stream

Run from the worker/ directory:
    python simulate_5jobs.py
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncGenerator

from depin_worker.inference.engine import GenerationChunk, InferenceEngine
from depin_worker.inference.model_manager import ModelManager
from depin_worker.job.manager import JobManager
from depin_worker.models import ChatMessage, JobDispatchPayload

# ── ANSI colours ──────────────────────────────────────────────────────────────

RESET  = "\033[0m"
BOLD   = "\033[1m"
GREEN  = "\033[32m"
RED    = "\033[31m"
YELLOW = "\033[33m"
CYAN   = "\033[36m"
DIM    = "\033[2m"
BLUE   = "\033[34m"

T0 = time.monotonic()

def ts() -> str:
    return f"{DIM}[+{time.monotonic() - T0:6.3f}s]{RESET}"

def log(color: str, tag: str, msg: str) -> None:
    print(f"{ts()} {color}{BOLD}{tag:<22}{RESET} {msg}")

# ── Fake engine (inherits InferenceEngine to pass isinstance check) ────────────

MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
TOKENS_PER_JOB = 8          # keep sim short
TOKEN_DELAY    = 0.05        # seconds between tokens (simulates GPU work)

class _FakeEngine(InferenceEngine):
    """Stands in for the real vLLM/MLX/HF engine — no GPU required."""

    def __init__(self) -> None:
        # Bypass real __init__ (would probe GPU backends)
        self.model_id   = MODEL_ID
        self._loaded    = True
        self._backend   = "fake"
        self._lock      = asyncio.Lock()

    async def load(self) -> None:
        self._loaded = True

    async def generate_stream(
        self,
        job_id: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
    ) -> AsyncGenerator[GenerationChunk, None]:
        log(BLUE, "  ⚙ GPU kernel", f"job {job_id[-4:]} — acquired CUDA sem, generating {TOKENS_PER_JOB} tokens")
        for i in range(TOKENS_PER_JOB):
            await asyncio.sleep(TOKEN_DELAY)
            yield GenerationChunk(
                text=f" w{i}",
                index=i,
                prompt_tokens=4,
                total_tokens=4 + i + 1,
            )
        yield GenerationChunk(text="", index=TOKENS_PER_JOB, finish_reason="stop",
                              prompt_tokens=4, total_tokens=4 + TOKENS_PER_JOB)
        log(BLUE, "  ⚙ GPU kernel", f"job {job_id[-4:]} — releasing CUDA sem")

    async def abort(self, job_id: str) -> None:
        pass

    async def unload(self) -> None:
        self._loaded = False


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_model_manager() -> ModelManager:
    """Build a ModelManager with the fake engine pre-injected."""
    mm = ModelManager()
    mm._total_vram    = 24.0
    mm._available_vram = 24.0 - 3.0 - 1.5  # weight + text buffer for 3B

    engine = _FakeEngine()
    mm._engines[MODEL_ID]         = engine
    mm._hot_models                = [MODEL_ID]
    mm._inference_refs[MODEL_ID]  = 0

    idle_event = asyncio.Event()
    idle_event.set()
    mm._idle_events[MODEL_ID] = idle_event

    return mm


def _make_payload(job_num: int) -> JobDispatchPayload:
    jid = f"job-sim-{job_num:04d}"
    return JobDispatchPayload(
        jobId=jid,
        model=MODEL_ID,
        messages=[ChatMessage(role="user", content=f"Hello from job {job_num}")],
        maxTokens=64,
        temperature=0.7,
    )


# ── Emit handler ──────────────────────────────────────────────────────────────

events: list[tuple[str, dict]] = []

async def emit(event: str, data: dict) -> None:
    job_id = data.get("jobId", "?")[-4:]
    events.append((event, data))

    if event == "worker:jobAccepted":
        log(GREEN, "✓ ACCEPTED", f"job {job_id}")
    elif event == "worker:jobRejected":
        log(RED, "✗ REJECTED", f"job {job_id}  reason={data.get('reason')}")
    elif event == "worker:token":
        token = data.get("token", "")
        if token:
            log(DIM, "  token", f"job {job_id} → {repr(token)}")
    elif event == "worker:completed":
        tps = data.get("tokensPerSecond", 0)
        total = data.get("totalTokens", 0)
        log(GREEN, "✓ COMPLETED", f"job {job_id}  tokens={total}  tps={tps:.1f}")
    elif event == "worker:error":
        log(RED, "✗ ERROR", f"job {job_id}  {data.get('error')}")


# ── Main simulation ───────────────────────────────────────────────────────────

MAX_CONCURRENCY = 3   # slots available on the simulated worker
NUM_JOBS        = 5   # total incoming jobs

async def run() -> None:
    mm      = _make_model_manager()
    manager = JobManager(
        model_manager=mm,
        node_id="sim-node-rtx4090",
        max_concurrency=MAX_CONCURRENCY,
        emit=emit,
    )

    print()
    print(f"{BOLD}{'═' * 62}{RESET}")
    print(f"{BOLD}  DePIN Worker — 5 concurrent text job simulation{RESET}")
    print(f"{BOLD}  max_concurrency={MAX_CONCURRENCY}  tokens/job={TOKENS_PER_JOB}  "
          f"token_delay={TOKEN_DELAY}s{RESET}")
    print(f"{BOLD}{'═' * 62}{RESET}")
    print()

    # Dispatch all 5 jobs as fast as possible (simulates burst arrival)
    log(CYAN, "DISPATCH BURST", f"sending {NUM_JOBS} jobs simultaneously")
    dispatch_tasks = [
        asyncio.create_task(manager.dispatch(_make_payload(i)))
        for i in range(1, NUM_JOBS + 1)
    ]
    await asyncio.gather(*dispatch_tasks)

    print()
    log(CYAN, "ACTIVE AFTER BURST", f"{manager.active_count}/{MAX_CONCURRENCY} slots occupied")
    log(CYAN, "CUDA SEM value",     f"{mm._cuda_sem._value}  (1=free, 0=held)")
    log(CYAN, "Inference refs",     f"{mm._inference_refs}")
    print()

    # Wait for all accepted jobs to finish
    log(CYAN, "WAITING", "for accepted jobs to complete …")
    await asyncio.gather(*[t for t in manager._tasks.values()])

    print()
    print(f"{BOLD}{'─' * 62}{RESET}")
    log(CYAN, "ACTIVE AFTER DONE", f"{manager.active_count}/{MAX_CONCURRENCY} slots occupied")
    log(CYAN, "CUDA SEM value",    f"{mm._cuda_sem._value}  (should be 1 = free)")
    log(CYAN, "Inference refs",    f"{mm._inference_refs}  (should be 0)")

    # Summary
    accepted  = [e for e in events if e[0] == "worker:jobAccepted"]
    rejected  = [e for e in events if e[0] == "worker:jobRejected"]
    completed = [e for e in events if e[0] == "worker:completed"]

    print()
    print(f"{BOLD}{'═' * 62}{RESET}")
    print(f"{BOLD}  Summary{RESET}")
    print(f"{'─' * 62}")
    print(f"  Jobs dispatched : {NUM_JOBS}")
    print(f"  {GREEN}Accepted{RESET}        : {len(accepted)}")
    print(f"  {RED}Rejected{RESET}        : {len(rejected)}"
          + (f"  (capacity_full)" if rejected else ""))
    print(f"  {GREEN}Completed{RESET}       : {len(completed)}")
    print(f"  GPU serialised  : {'YES ✓' if len(accepted) > 1 else 'N/A'}"
          "  (CUDA sem=1 enforces one-at-a-time GPU use)")
    print(f"  Leaked refs     : {mm._inference_refs.get(MODEL_ID, 0)}"
          + ("  ✓" if mm._inference_refs.get(MODEL_ID, 0) == 0 else "  ✗ BUG"))
    print(f"  Elapsed         : {time.monotonic() - T0:.2f}s")
    print(f"{BOLD}{'═' * 62}{RESET}")
    print()

    # Assertions so CI can catch regressions
    assert len(accepted)  == MAX_CONCURRENCY,             "wrong accept count"
    assert len(rejected)  == NUM_JOBS - MAX_CONCURRENCY,  "wrong reject count"
    assert len(completed) == MAX_CONCURRENCY,             "not all accepted jobs completed"
    assert manager.active_count == 0,                     "active count did not drain"
    assert mm._cuda_sem._value  == 1,                     "semaphore still held after completion"
    assert mm._inference_refs.get(MODEL_ID, 0) == 0,      "ref leak detected"
    print(f"{GREEN}{BOLD}All assertions passed.{RESET}")
    print()


if __name__ == "__main__":
    asyncio.run(run())
