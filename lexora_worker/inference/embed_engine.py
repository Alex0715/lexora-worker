from __future__ import annotations

import asyncio
import logging
from functools import partial
from typing import Any

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False
    logger.warning("sentence-transformers not installed — BGE-M3 unavailable. Run: pip install sentence-transformers")


class EmbedEngine:
    """CPU-based BGE-M3 embedding engine using sentence-transformers.

    sentence-transformers tracks transformers compatibility and handles
    BGE-M3's dense embeddings correctly. Runs in a thread-pool executor so
    the asyncio event loop stays unblocked during embedding.
    """

    MODEL_ID = "BAAI/bge-m3"

    def __init__(self, model_cache_dir: str | None = None) -> None:
        self._model: Any | None = None
        self._model_cache_dir = model_cache_dir

    async def load(self) -> None:
        if not _ST_AVAILABLE:
            raise RuntimeError(
                "sentence-transformers is not installed. Run: pip install sentence-transformers"
            )
        loop = asyncio.get_event_loop()
        self._model = await loop.run_in_executor(None, self._load_sync)
        logger.info("BGE-M3 embed engine loaded (CPU via sentence-transformers)")

    def _load_sync(self) -> Any:
        kwargs: dict[str, Any] = {"device": "cpu"}
        if self._model_cache_dir:
            kwargs["cache_folder"] = self._model_cache_dir
        return SentenceTransformer(self.MODEL_ID, **kwargs)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if self._model is None:
            raise RuntimeError("EmbedEngine not loaded — call load() first")
        loop = asyncio.get_event_loop()
        output = await loop.run_in_executor(
            None,
            partial(
                self._model.encode,
                texts,
                batch_size=12,
                show_progress_bar=False,
                convert_to_numpy=True,
            ),
        )
        return output.tolist()

    async def unload(self) -> None:
        self._model = None
        logger.info("BGE-M3 embed engine unloaded")
