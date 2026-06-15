from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator
from typing import Any

from lexora_worker.inference.capability import MODEL_VRAM_COST
from lexora_worker.inference.engine import GenerationChunk, InferenceEngine
from lexora_worker.inference.image_engine import ImageInferenceEngine

logger = logging.getLogger(__name__)

# Runtime execution overhead added on top of static model weight cost.
# Prevents OOM when concurrent text + image inference activations accumulate.
_TEXT_RUNTIME_BUFFER = 1.5        # GB — KV cache + activation tensors
_IMAGE_RUNTIME_BUFFER = 4.5       # GB — denoising intermediate tensors at 1024px
_IMAGE_RUNTIME_BUFFER_GGUF = 2.5  # GB — GGUF variant: T5 encoder runs on CPU


def _is_image_model(model_id: str) -> bool:
    lower = model_id.lower()
    return (
        "flux" in lower
        or "sdxl" in lower
        or "stable-video" in lower
        or "wan" in lower
    )


def _is_video_model(model_id: str) -> bool:
    lower = model_id.lower()
    return "video" in lower or "wan" in lower or "svd" in lower


def _runtime_vram_cost(model_id: str) -> float:
    """Static weight cost + execution buffer used for all internal VRAM planning."""
    import os
    base = MODEL_VRAM_COST.get(model_id, 8.0)
    if _is_image_model(model_id):
        # FLUX GGUF variant offloads the T5 encoder to CPU, so GPU intermediate
        # tensors are smaller — use the reduced buffer.
        buffer = (
            _IMAGE_RUNTIME_BUFFER_GGUF
            if os.environ.get("LEXORA_FLUX_GGUF_REPO")
            else _IMAGE_RUNTIME_BUFFER
        )
    else:
        buffer = _TEXT_RUNTIME_BUFFER
    return base + buffer


class ModelManager:
    """
    Holds multiple inference engines simultaneously, subject to a VRAM budget.

    Concurrency guarantees
    ----------------------
    _lock          – serialises engine load/unload and ref-count mutations.
    _cuda_sem      – Semaphore(1) around every GPU kernel dispatch so text and
                     image generation take turns on the CUDA cores instead of
                     racing each other into an OOM.
    _inference_refs – per-model active-inference counter; LRU eviction waits
                     until a model's counter reaches 0 before unloading it.
    _idle_events   – per-model asyncio.Event, set when _inference_refs[model]
                     drops to 0, used by the eviction fence.
    """

    def __init__(self, model_cache_dir: str | None = None) -> None:
        self._engines: dict[str, InferenceEngine | ImageInferenceEngine] = {}
        self._hot_models: list[str] = []          # LRU front, MRU back
        self._total_vram: float = 0.0
        self._available_vram: float = 0.0
        self._model_cache_dir = model_cache_dir
        self._lock = asyncio.Lock()
        self._swap_in_progress: bool = False
        # Concurrency primitives
        self._cuda_sem = asyncio.Semaphore(1)
        self._inference_refs: dict[str, int] = {}
        self._idle_events: dict[str, asyncio.Event] = {}

    # ── Lifecycle ───────────────────────────────────────────────────────────

    async def initialize(self, default_model: str, total_vram: float) -> None:
        self._total_vram = total_vram
        self._available_vram = total_vram
        await self.ensure_model_loaded(default_model)

    async def ensure_model_loaded(self, model_id: str) -> None:
        # Manual acquire/release so we can drop the lock while waiting for
        # an idle engine during LRU eviction (prevents deadlock with
        # in-flight inference that still needs _inc_ref / _dec_ref calls).
        await self._lock.acquire()
        try:
            if model_id in self._engines:
                if model_id in self._hot_models:
                    self._hot_models.remove(model_id)
                self._hot_models.append(model_id)
                return

            cost = _runtime_vram_cost(model_id)
            if self._available_vram >= cost:
                await self._load_engine(model_id)
                return

            # Need to free VRAM via LRU eviction.
            self._swap_in_progress = True
            try:
                while self._available_vram < cost and self._hot_models:
                    lru = self._hot_models[0]
                    # Release the main lock so in-flight inference on `lru`
                    # can finish and decrement its ref to 0.
                    self._lock.release()
                    try:
                        await self._wait_for_idle(lru)
                    finally:
                        await self._lock.acquire()
                    # Verify engine is still present after re-acquiring.
                    if lru in self._engines:
                        await self._unload_engine(lru)
                await self._load_engine(model_id)
            finally:
                self._swap_in_progress = False
        finally:
            self._lock.release()

    async def _load_engine(self, model_id: str) -> None:
        cost = _runtime_vram_cost(model_id)
        engine: ImageInferenceEngine | InferenceEngine
        if _is_image_model(model_id):
            engine = ImageInferenceEngine(model_id=model_id)
            await engine.load()
        else:
            engine = InferenceEngine(model_id=model_id)
            await engine.load()

        self._engines[model_id] = engine
        self._hot_models.append(model_id)
        self._available_vram = max(0.0, self._available_vram - cost)
        # Initialise idle state for this model.
        event = asyncio.Event()
        event.set()   # idle by default — no jobs running yet
        self._idle_events[model_id] = event
        self._inference_refs[model_id] = 0
        logger.info(
            "Loaded %s (cost=%.1f GB incl. buffer) — available VRAM: %.2f / %.2f GB",
            model_id,
            cost,
            self._available_vram,
            self._total_vram,
        )

    async def _unload_engine(self, model_id: str) -> None:
        engine = self._engines.pop(model_id, None)
        if engine is None:
            return
        try:
            await engine.unload()
        except Exception as exc:
            logger.warning("Failed to cleanly unload %s: %s", model_id, exc)
        if model_id in self._hot_models:
            self._hot_models.remove(model_id)
        cost = _runtime_vram_cost(model_id)
        self._available_vram = min(self._total_vram, self._available_vram + cost)
        self._inference_refs.pop(model_id, None)
        self._idle_events.pop(model_id, None)
        logger.info(
            "Unloaded %s — available VRAM: %.2f / %.2f GB",
            model_id,
            self._available_vram,
            self._total_vram,
        )

    # ── Ref-count helpers ───────────────────────────────────────────────────

    def _inc_ref(self, model_id: str) -> None:
        self._inference_refs[model_id] = self._inference_refs.get(model_id, 0) + 1
        event = self._idle_events.get(model_id)
        if event:
            event.clear()

    def _dec_ref(self, model_id: str) -> None:
        count = max(0, self._inference_refs.get(model_id, 0) - 1)
        self._inference_refs[model_id] = count
        if count == 0:
            event = self._idle_events.get(model_id)
            if event:
                event.set()

    async def _wait_for_idle(self, model_id: str) -> None:
        """Block until no inference is running on model_id."""
        if self._inference_refs.get(model_id, 0) == 0:
            return
        event = self._idle_events.get(model_id)
        if event:
            await event.wait()

    # ── Introspection ───────────────────────────────────────────────────────

    def is_image_model(self, model_id: str) -> bool:
        return _is_image_model(model_id)

    def is_video_model(self, model_id: str) -> bool:
        return _is_video_model(model_id)

    @property
    def hot_models(self) -> list[str]:
        return list(self._hot_models)

    @property
    def available_vram(self) -> float:
        return self._available_vram

    @property
    def total_vram(self) -> float:
        return self._total_vram

    @property
    def swap_in_progress(self) -> bool:
        return self._swap_in_progress

    def has_text_engine(self, model_id: str) -> bool:
        return isinstance(self._engines.get(model_id), InferenceEngine)

    def has_image_engine(self, model_id: str) -> bool:
        return isinstance(self._engines.get(model_id), ImageInferenceEngine)

    # ── Generation routing ──────────────────────────────────────────────────

    async def generate_stream(
        self,
        model_id: str,
        job_id: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
    ) -> AsyncGenerator[GenerationChunk, None]:
        # Atomically capture the engine reference and increment the ref-count
        # under the main lock so LRU eviction cannot unload the engine between
        # the lookup and the first token.
        async with self._lock:
            engine = self._engines.get(model_id)
            if not isinstance(engine, InferenceEngine):
                raise RuntimeError(f"No text engine loaded for model {model_id}")
            self._inc_ref(model_id)

        # Serialise GPU kernel execution with image generation.
        await self._cuda_sem.acquire()
        # Hold an explicit reference to the engine generator so we can
        # await aclose() on it BEFORE releasing the semaphore.  This ensures
        # the engine's own finally/cleanup runs (and GPU resources are freed)
        # before the next job is allowed to acquire the semaphore.
        engine_gen = engine.generate_stream(
            job_id=job_id,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        try:
            async for chunk in engine_gen:
                yield chunk
        finally:
            await engine_gen.aclose()   # drain / close before releasing sem
            self._cuda_sem.release()
            self._dec_ref(model_id)

    async def generate_image(
        self,
        model_id: str,
        prompt: str,
        width: int,
        height: int,
        num_steps: int,
        guidance: float,
    ) -> str:
        async with self._lock:
            engine = self._engines.get(model_id)
            if not isinstance(engine, ImageInferenceEngine):
                raise RuntimeError(f"No image engine loaded for model {model_id}")
            self._inc_ref(model_id)

        await self._cuda_sem.acquire()
        try:
            return await engine.generate(
                prompt=prompt,
                width=width,
                height=height,
                num_steps=num_steps,
                guidance_scale=guidance,
            )
        finally:
            self._cuda_sem.release()
            self._dec_ref(model_id)

    async def abort_text(self, model_id: str, job_id: str) -> None:
        engine = self._engines.get(model_id)
        if isinstance(engine, InferenceEngine):
            await engine.abort(job_id)

    async def unload_all(self) -> None:
        for model_id in list(self._engines.keys()):
            await self._unload_engine(model_id)
