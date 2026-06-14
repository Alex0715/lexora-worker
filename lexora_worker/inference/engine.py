from __future__ import annotations

import asyncio
import logging
import os
import platform
import threading
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

logger = logging.getLogger(__name__)

_IS_MAC = platform.system() == "Darwin"

# ── Backend availability ──────────────────────────────────────────────────────

try:
    import mlx_lm as _mlx_lm_module

    _MLX_AVAILABLE = _IS_MAC
except Exception:
    _MLX_AVAILABLE = False
    if _IS_MAC:
        logger.warning("mlx-lm not installed — Mac inference unavailable. pip install mlx-lm")

if not _IS_MAC:
    try:
        from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
        from vllm.outputs import RequestOutput

        _VLLM_AVAILABLE = True
    except Exception:
        _VLLM_AVAILABLE = False
        logger.warning("vLLM not available — falling back to HuggingFace transformers")
else:
    _VLLM_AVAILABLE = False

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
    import torch

    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False


def _active_backend() -> str:
    if _MLX_AVAILABLE:
        return "mlx"
    if _VLLM_AVAILABLE:
        return "vllm"
    if _TRANSFORMERS_AVAILABLE:
        return "transformers"
    return "none"


# ── Data classes ──────────────────────────────────────────────────────────────


@dataclass
class GenerationChunk:
    text: str
    index: int
    finish_reason: str | None = None
    prompt_tokens: int = 0
    total_tokens: int = 0


@dataclass
class EngineStats:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    tokens_per_second: float = 0.0


# ── Engine ────────────────────────────────────────────────────────────────────


class InferenceEngine:
    """
    Inference backend selected automatically by platform and available packages:
      - macOS              → mlx-lm  (Metal / Apple Silicon)
      - Linux/Windows GPU  → vLLM    (CUDA)
      - fallback           → HuggingFace Transformers
    """

    def __init__(self, model_id: str, max_model_len: int = 4096) -> None:
        self.model_id = model_id
        self.max_model_len = max_model_len

        # vLLM
        self._engine: Any | None = None
        # HuggingFace
        self._hf_model: Any | None = None
        self._hf_tokenizer: Any | None = None
        # MLX
        self._mlx_model: Any | None = None
        self._mlx_tokenizer: Any | None = None

        self._loaded = False
        self._lock = asyncio.Lock()
        self._backend = _active_backend()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def load(self) -> None:
        async with self._lock:
            if self._loaded:
                return

            if self._backend == "mlx":
                await self._load_mlx()
            elif self._backend == "vllm":
                await self._load_vllm()
            elif self._backend == "transformers":
                await self._load_hf()
            else:
                raise RuntimeError(
                    "No inference backend available. "
                    "On Mac: pip install mlx-lm. "
                    "On Linux/Windows: pip install -e ./worker[inference]"
                )

            self._loaded = True
            logger.info("Model loaded via %s: %s", self._backend, self.model_id)

    async def _load_mlx(self) -> None:
        import mlx_lm

        loop = asyncio.get_event_loop()

        def _load() -> tuple[Any, Any]:
            return mlx_lm.load(self.model_id)

        self._mlx_model, self._mlx_tokenizer = await loop.run_in_executor(None, _load)

    async def _load_vllm(self) -> None:
        # The hardware profiler (and other earlier startup code) calls
        # torch.cuda.* in this process, initializing a CUDA context. vLLM's
        # default "fork" multiprocessing start method can't re-init CUDA in
        # the forked EngineCore subprocess ("Cannot re-initialize CUDA in
        # forked subprocess"). Force "spawn" so the subprocess gets a fresh
        # interpreter instead of a fork of this CUDA-initialized one.
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

        args = AsyncEngineArgs(
            model=self.model_id,
            max_model_len=self.max_model_len,
            dtype="auto",
            gpu_memory_utilization=0.90,
            trust_remote_code=True,
        )
        loop = asyncio.get_event_loop()
        self._engine = await loop.run_in_executor(
            None, AsyncLLMEngine.from_engine_args, args
        )

    async def _load_hf(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        loop = asyncio.get_event_loop()

        def _load() -> tuple[Any, Any]:
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_id, trust_remote_code=True
            )
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
                trust_remote_code=True,
            )
            return model, tokenizer

        self._hf_model, self._hf_tokenizer = await loop.run_in_executor(None, _load)

    def is_loaded(self) -> bool:
        return self._loaded

    # ── Generation ────────────────────────────────────────────────────────────

    async def generate_stream(
        self,
        job_id: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
    ) -> AsyncGenerator[GenerationChunk, None]:
        if not self._loaded:
            raise RuntimeError("Engine not loaded. Call load() first.")

        if self._backend == "mlx":
            async for chunk in self._mlx_stream(job_id, messages, max_tokens, temperature):
                yield chunk
        elif self._backend == "vllm":
            async for chunk in self._vllm_stream(job_id, messages, max_tokens, temperature):
                yield chunk
        else:
            async for chunk in self._hf_stream(job_id, messages, max_tokens, temperature):
                yield chunk

    async def _mlx_stream(
        self,
        job_id: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
    ) -> AsyncGenerator[GenerationChunk, None]:
        import mlx_lm

        assert self._mlx_model is not None and self._mlx_tokenizer is not None

        prompt = self._build_prompt(messages, self._mlx_tokenizer)
        prompt_tokens = len(self._mlx_tokenizer.encode(prompt))
        model = self._mlx_model
        tokenizer = self._mlx_tokenizer
        loop = asyncio.get_event_loop()
        queue: asyncio.Queue[tuple[str, str | None] | None] = asyncio.Queue()

        def _generate() -> None:
            try:
                from mlx_lm.sample_utils import make_sampler
                sampler = make_sampler(temperature)
                for response in mlx_lm.stream_generate(
                    model,
                    tokenizer,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    sampler=sampler,
                ):
                    text = response.text if hasattr(response, "text") else str(response)
                    finish = getattr(response, "finish_reason", None)
                    loop.call_soon_threadsafe(queue.put_nowait, (text, finish))
                    if finish in ("stop", "length"):
                        break
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        thread = threading.Thread(target=_generate, daemon=True)
        thread.start()

        index = 0
        completion_tokens = 0
        while True:
            item = await queue.get()
            if item is None:
                break
            token_text, finish_reason = item
            if token_text:
                completion_tokens += 1
                yield GenerationChunk(
                    text=token_text,
                    index=index,
                    finish_reason=None,
                    prompt_tokens=prompt_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                )
                index += 1
            if finish_reason in ("stop", "length"):
                yield GenerationChunk(
                    text="",
                    index=index,
                    finish_reason=finish_reason,
                    prompt_tokens=prompt_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                )
                return

        yield GenerationChunk(
            text="",
            index=index,
            finish_reason="stop",
            prompt_tokens=prompt_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )

    async def _vllm_stream(
        self,
        job_id: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
    ) -> AsyncGenerator[GenerationChunk, None]:
        assert self._engine is not None

        prompt = self._build_prompt(messages, None)
        sampling = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            stop=["</s>", "<|im_end|>", "<|eot_id|>"],
        )

        start = time.monotonic()
        index = 0
        last_text = ""
        prompt_tokens = 0
        total_tokens = 0

        async for output in self._engine.generate(prompt, sampling, request_id=job_id):
            output: RequestOutput
            if not output.outputs:
                continue

            completion = output.outputs[0]
            new_text = completion.text[len(last_text):]
            last_text = completion.text

            if output.prompt_token_ids:
                prompt_tokens = len(output.prompt_token_ids)
            total_tokens = prompt_tokens + len(
                completion.token_ids if hasattr(completion, "token_ids") else []
            )

            finish_reason: str | None = completion.finish_reason

            if new_text:
                yield GenerationChunk(
                    text=new_text,
                    index=index,
                    finish_reason=finish_reason if finish_reason else None,
                    prompt_tokens=prompt_tokens,
                    total_tokens=total_tokens,
                )
                index += 1

            if finish_reason in ("stop", "length"):
                yield GenerationChunk(
                    text="",
                    index=index,
                    finish_reason=finish_reason,
                    prompt_tokens=prompt_tokens,
                    total_tokens=total_tokens,
                )
                return

    async def _hf_stream(
        self,
        job_id: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
    ) -> AsyncGenerator[GenerationChunk, None]:
        assert self._hf_model is not None and self._hf_tokenizer is not None
        import torch
        from transformers import TextIteratorStreamer

        loop = asyncio.get_event_loop()
        prompt = self._build_prompt(messages, self._hf_tokenizer)
        tokenizer = self._hf_tokenizer
        model = self._hf_model

        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]
        if torch.cuda.is_available():
            input_ids = input_ids.cuda()

        prompt_tokens = int(input_ids.shape[-1])
        streamer = TextIteratorStreamer(
            tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        gen_kwargs = {
            "input_ids": input_ids,
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "do_sample": temperature > 0,
            "streamer": streamer,
        }

        gen_thread = threading.Thread(
            target=model.generate, kwargs=gen_kwargs, daemon=True
        )
        gen_thread.start()

        index = 0
        total_tokens = prompt_tokens
        queue: asyncio.Queue[str | None] = asyncio.Queue()

        def _drain_streamer() -> None:
            for text in streamer:
                loop.call_soon_threadsafe(queue.put_nowait, text)
            loop.call_soon_threadsafe(queue.put_nowait, None)

        drain_thread = threading.Thread(target=_drain_streamer, daemon=True)
        drain_thread.start()

        finish_reason = "stop"
        while True:
            token_text = await queue.get()
            if token_text is None:
                break

            total_tokens += 1
            if total_tokens >= prompt_tokens + max_tokens:
                finish_reason = "length"

            yield GenerationChunk(
                text=token_text,
                index=index,
                finish_reason=None,
                prompt_tokens=prompt_tokens,
                total_tokens=total_tokens,
            )
            index += 1

        yield GenerationChunk(
            text="",
            index=index,
            finish_reason=finish_reason,
            prompt_tokens=prompt_tokens,
            total_tokens=total_tokens,
        )

    # ── Abort / unload ────────────────────────────────────────────────────────

    async def abort(self, job_id: str) -> None:
        if self._backend == "vllm" and self._engine is not None:
            try:
                await self._engine.abort(job_id)
            except Exception as exc:
                logger.debug("abort %s: %s", job_id, exc)

    async def unload(self) -> None:
        async with self._lock:
            if self._backend == "vllm" and self._engine is not None:
                del self._engine
                self._engine = None

            if self._backend == "transformers" and self._hf_model is not None:
                import torch

                del self._hf_model
                self._hf_model = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if self._backend == "mlx" and self._mlx_model is not None:
                del self._mlx_model
                self._mlx_model = None
                self._mlx_tokenizer = None

            self._loaded = False
            logger.info("Model unloaded: %s", self.model_id)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _build_prompt(messages: list[dict[str, str]], tokenizer: Any | None) -> str:
        """Use the tokenizer's chat template when available, else fall back to manual format."""
        if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
            try:
                return tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                pass
        return InferenceEngine._format_messages(messages)

    @staticmethod
    def _format_messages(messages: list[dict[str, str]]) -> str:
        parts: list[str] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                parts.append(f"<|system|>\n{content}")
            elif role == "user":
                parts.append(f"<|user|>\n{content}")
            elif role == "assistant":
                parts.append(f"<|assistant|>\n{content}")
        parts.append("<|assistant|>")
        return "\n".join(parts)
