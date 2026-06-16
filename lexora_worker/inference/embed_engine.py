from __future__ import annotations

import asyncio
import logging
from functools import partial
from typing import Any

logger = logging.getLogger(__name__)

try:
    from FlagEmbedding import BGEM3FlagModel
    _FLAG_AVAILABLE = True
except ImportError:
    _FLAG_AVAILABLE = False
    logger.warning("FlagEmbedding not installed — BGE-M3 unavailable. Run: pip install FlagEmbedding")


class EmbedEngine:
    """CPU-based BGE-M3 embedding engine using BAAI's FlagEmbedding library.

    FlagEmbedding is the official BAAI library for BGE-M3 and handles the
    model's custom architecture correctly. Runs in a thread-pool executor so
    the asyncio event loop stays unblocked during embedding.
    """

    MODEL_ID = "BAAI/bge-m3"

    def __init__(self, model_cache_dir: str | None = None) -> None:
        self._model: Any | None = None
        self._model_cache_dir = model_cache_dir

    async def load(self) -> None:
        if not _FLAG_AVAILABLE:
            raise RuntimeError(
                "FlagEmbedding is not installed. Run: pip install FlagEmbedding"
            )
        loop = asyncio.get_event_loop()
        self._model = await loop.run_in_executor(None, self._load_sync)
        logger.info("BGE-M3 embed engine loaded (CPU via FlagEmbedding)")

    def _load_sync(self) -> Any:
        return BGEM3FlagModel(
            self.MODEL_ID,
            use_fp16=False,   # CPU: fp16 is not beneficial, fp32 is correct
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if self._model is None:
            raise RuntimeError("EmbedEngine not loaded — call load() first")
        loop = asyncio.get_event_loop()
        output = await loop.run_in_executor(
            None,
            partial(self._model.encode, texts, batch_size=12, max_length=8192),
        )
        # FlagEmbedding returns a dict; dense_vecs is the standard embedding
        return output["dense_vecs"].tolist()

    async def unload(self) -> None:
        self._model = None
        logger.info("BGE-M3 embed engine unloaded")
