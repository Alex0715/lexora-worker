# Multi-GPU Cluster Support — Future Plan

Current state: one worker process = one GPU. On a machine with N GPUs, you
run N separate worker processes with `CUDA_VISIBLE_DEVICES=0`, `=1`, etc.
Each registers as an independent node. This works but wastes RAM (N copies of
Python runtime, N socket.io connections) and has no cross-GPU coordination.

---

## What needs to change

### 1. GPU Discovery at startup

At launch, scan all available CUDA devices and build a per-GPU profile:

```
GPU 0 — RTX 3060 12 GB  → text  (Llama 8B GGUF)
GPU 1 — RTX 3060 12 GB  → image (FLUX schnell)
GPU 2 — RTX A4000 16 GB → text + image (co-loaded)
```

```python
# hardware/profiler.py  — extend build_hardware_profile()
def discover_gpus() -> list[GpuInfo]:
    """Return one GpuInfo per visible CUDA device."""
    ...
```

### 2. Per-GPU Model Assignment

Let the resolver decide which model fits best on each GPU, given the full
inventory of available VRAM across all devices.

Two assignment strategies:

**Greedy (default):** assign each model to the first GPU where it fits.
Good for mixed fleets (different VRAM per card).

**Balanced:** spread models evenly so no GPU is bottlenecked while another
idles. Better for homogeneous clusters (same card × N).

```python
# inference/gpu_scheduler.py  (new file)
def assign_models(
    models: list[str],
    gpus: list[GpuInfo],
    strategy: Literal["greedy", "balanced"] = "greedy",
) -> dict[int, list[str]]:   # gpu_index → [model_ids]
    ...
```

### 3. Per-GPU Engine Pool

`ModelManager` currently targets a single implicit CUDA device. Extend it to
hold one engine pool per GPU device index:

```python
class ModelManager:
    # Current
    _engines: dict[str, InferenceEngine | ImageInferenceEngine]

    # Future
    _gpu_pools: dict[int, GpuEnginePool]   # device_index → pool
```

`GpuEnginePool` wraps the existing LRU + CUDA semaphore logic, scoped to one
device. `InferenceEngine` and `ImageInferenceEngine` gain a `device` parameter
(`cuda:0`, `cuda:1`, …).

### 4. Job Routing within the Worker

When a job arrives the worker must pick the right GPU:

```
Text job  → find a GPU pool that has the text model hot
Image job → find a GPU pool that has the image model hot
Fallback  → LRU-evict from the least-loaded GPU and load there
```

`JobManager.dispatch()` asks a new `GpuRouter` for the target pool before
handing the job to `ModelManager.generate_stream()`.

### 5. Registration & Heartbeat changes

A multi-GPU node registers once but advertises all its GPU capabilities:

```json
{
  "capabilities": {
    "gpus": [
      { "index": 0, "model": "RTX 3060", "vram": 12, "loadedModels": ["llama-3.1-8b"] },
      { "index": 1, "model": "RTX 3060", "vram": 12, "loadedModels": ["flux.1-schnell"] }
    ],
    "totalVram": 24,
    "maxConcurrentJobs": 6
  }
}
```

Heartbeat includes per-GPU utilisation so the orchestrator can make smarter
routing decisions (e.g. prefer the GPU that already has the right model loaded).

### 6. CLI changes

```bash
# Auto-assign: worker figures out which model goes on which GPU
lexora-worker start --model llama-3.1-8b --model flux.1-schnell

# Explicit assignment (power users / testing)
lexora-worker start \
  --model llama-3.1-8b:gpu=0 \
  --model flux.1-schnell:gpu=1
```

`--max-concurrency` becomes per-GPU (or a total shared across all GPUs with
automatic per-GPU caps based on VRAM).

---

## Implementation order

1. `discover_gpus()` — enumerate devices, return `list[GpuInfo]`
2. `assign_models()` — greedy assignment; add `device` param to engines
3. `GpuEnginePool` — wrap existing `ModelManager` logic per device
4. `GpuRouter` — job → pool routing in `JobManager`
5. Updated registration payload — `gpus[]` array
6. CLI auto-assignment — parse `--model id:gpu=N` syntax
7. Orchestrator changes — route by GPU capability, not just node

---

## What the orchestrator needs

- Accept `gpus[]` array in `worker:register`
- Track per-GPU loaded models (not just per-node)
- Route `job:dispatch` with a `preferredGpu` hint the worker can use
- Heartbeat dashboard: per-GPU utilisation graphs

---

## Notes

- vLLM tensor parallelism (`tensor_parallel_size=N`) is a separate concept —
  it splits one large model across multiple GPUs. That's for 70B+ models.
  This plan is about running *different* models on *different* GPUs.
- The CUDA semaphore per `GpuEnginePool` keeps text/image serialised per GPU,
  same as today. Cross-GPU jobs can genuinely run in parallel.
- Windows multi-GPU: `CUDA_VISIBLE_DEVICES` works the same way.
