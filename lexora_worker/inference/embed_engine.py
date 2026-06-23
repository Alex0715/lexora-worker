from __future__ import annotations

import asyncio
import logging
from typing import Any

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# BGE-M3 is XLM-RoBERTa under the hood. We load it with explicit classes to
# bypass AutoModel's model_type check, which fails on transformers 5.x because
# BGE-M3's config.json doesn't carry a model_type field.
try:
    from transformers import XLMRobertaModel, XLMRobertaTokenizerFast
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False
    logger.warning("transformers not installed — BGE-M3 unavailable")


def _mean_pool(last_hidden_state: "torch.Tensor", attention_mask: "torch.Tensor") -> "torch.Tensor":
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    return torch.sum(last_hidden_state * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)


class EmbedEngine:
    """CPU-based BGE-M3 embedding engine.

    Loads BAAI/bge-m3 directly as XLMRobertaModel + XLMRobertaTokenizerFast
    to avoid the AutoModel model_type resolution that fails on transformers 5.x.
    Mean-pools the last hidden state and L2-normalises, matching BGE-M3 dense
    embedding semantics. Runs in a thread-pool executor so the event loop stays
    unblocked.
    """

    MODEL_ID = "BAAI/bge-m3"
    _BATCH_SIZE = 12
    _MAX_LENGTH = 8192

    def __init__(self, model_cache_dir: str | None = None) -> None:
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._model_cache_dir = model_cache_dir

    async def load(self) -> None:
        if not _TRANSFORMERS_AVAILABLE:
            raise RuntimeError("transformers is not installed")
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_sync)
        logger.info("BGE-M3 embed engine loaded (CPU, explicit XLMRoberta)")

    def _load_sync(self) -> None:
        import glob
        import os

        cache_dir = self._model_cache_dir or os.path.expanduser("~/.cache/huggingface/hub")
        slug = self.MODEL_ID.replace("/", "--")
        snapshots = glob.glob(os.path.join(cache_dir, f"models--{slug}", "snapshots", "*/"))
        if snapshots:
            local_path = snapshots[0].rstrip("/")
            logger.info("Loading BGE-M3 from local snapshot: %s", local_path)
        else:
            local_path = self.MODEL_ID  # fall back to Hub download

        self._tokenizer = XLMRobertaTokenizerFast.from_pretrained(local_path)
        self._model = XLMRobertaModel.from_pretrained(local_path)
        self._model.eval()

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("EmbedEngine not loaded — call load() first")
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._encode_sync, texts)

    def _encode_sync(self, texts: list[str]) -> list[list[float]]:
        results: list[list[float]] = []
        for i in range(0, len(texts), self._BATCH_SIZE):
            batch = texts[i : i + self._BATCH_SIZE]
            inputs = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self._MAX_LENGTH,
                return_tensors="pt",
            )
            with torch.no_grad():
                outputs = self._model(**inputs)
            embeddings = _mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
            embeddings = F.normalize(embeddings, p=2, dim=1)
            results.extend(embeddings.tolist())
        return results

    async def unload(self) -> None:
        self._model = None
        self._tokenizer = None
        logger.info("BGE-M3 embed engine unloaded")
