from __future__ import annotations

import asyncio
import logging
from functools import partial
from typing import Any

logger = logging.getLogger(__name__)

try:
    from fastembed import TextEmbedding
    _FASTEMBED_AVAILABLE = True
except ImportError:
    _FASTEMBED_AVAILABLE = False
    logger.warning("fastembed not installed — BGE-M3 unavailable. Run: pip install fastembed")


class EmbedEngine:
    """CPU-based BGE-M3 embedding engine using fastembed (ONNX backend).

    fastembed runs BGE-M3 via ONNX with no transformers dependency, avoiding
    transformers version compatibility issues. Runs in a thread-pool executor
    so the asyncio event loop stays unblocked during embedding.
    """

    MODEL_ID = "BAAI/bge-m3"

    def __init__(self, model_cache_dir: str | None = None) -> None:
        self._model: Any | None = None
        self._model_cache_dir = model_cache_dir

    async def load(self) -> None:
        if not _FASTEMBED_AVAILABLE:
            raise RuntimeError(
                "fastembed is not installed. Run: pip install fastembed"
            )
        loop = asyncio.get_event_loop()
        self._model = await loop.run_in_executor(None, self._load_sync)
        logger.info("BGE-M3 embed engine loaded (CPU via fastembed/ONNX)")

    def _load_sync(self) -> Any:
        kwargs: dict[str, Any] = {}
        if self._model_cache_dir:
            kwargs["cache_dir"] = self._model_cache_dir
        return TextEmbedding(self.MODEL_ID, **kwargs)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if self._model is None:
            raise RuntimeError("EmbedEngine not loaded — call load() first")
        loop = asyncio.get_event_loop()

        def _encode() -> list[list[float]]:
            return [emb.tolist() for emb in self._model.embed(texts, batch_size=12)]

        return await loop.run_in_executor(None, _encode)

    async def unload(self) -> None:
        self._model = None
        logger.info("BGE-M3 embed engine unloaded")
