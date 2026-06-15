from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# Output dir — workers save PNGs here; orchestrator will upload to S3/R2 later
_OUTPUT_DIR = Path(os.environ.get("LEXORA_IMAGE_OUTPUT_DIR", tempfile.gettempdir())) / "lexora_images"

# Pre-quantized FLUX transformer checkpoints (safetensors only). These are
# quantized offline — loading them is a pure mmap + deserialize, with no
# on-the-fly bitsandbytes quantization pass.
#
# Override per-deployment via env vars (useful for models without a known
# community NF4 checkpoint, e.g. FLUX.1-schnell):
#   LEXORA_FLUX_NF4_REPO=<repo_id>
#   LEXORA_FLUX_NF4_SUBFOLDER=<subfolder>   (default: "transformer")
_FLUX_NF4_TRANSFORMERS: dict[str, tuple[str, str]] = {
    "black-forest-labs/FLUX.1-dev": ("hf-internal-testing/flux.1-dev-nf4-pkg", "transformer"),
}

# Local cache root for "quantize-once" FLUX NF4 transformers — bitsandbytes
# quantization (~5 min CPU-bound pass) runs at most once per model; every
# subsequent boot mmap-loads the cached safetensors directly.
#   LEXORA_NF4_CACHE_DIR overrides the root (default: ~/.lexora/cache/models)
_NF4_CACHE_ROOT = Path(
    os.environ.get("LEXORA_NF4_CACHE_DIR", str(Path.home() / ".lexora" / "cache" / "models"))
)

# <12GB VRAM nodes: model_id -> local cache subdirectory name for the
# quantize-once NF4 transformer.
_FLUX_NF4_CACHE_NAMES: dict[str, str] = {
    "black-forest-labs/FLUX.1-schnell": "flux-schnell-nf4",
}


def _get_vram_gb() -> float:
    """Return total VRAM in GB of the primary CUDA device, or 0 on CPU."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    except Exception:
        pass
    return 0.0


def _require_safetensors(repo_id: str, subfolder: str | None = None, revision: str | None = None) -> None:
    """Fail fast unless the repo/component ships .safetensors weights.

    Protects the network's SLA by refusing to fall back to slow, CPU-bound
    .bin checkpoints or pull weights that require on-the-fly quantization.
    """
    from huggingface_hub import HfApi
    from huggingface_hub.errors import RepositoryNotFoundError, RevisionNotFoundError

    api = HfApi()
    try:
        files = api.list_repo_files(repo_id, revision=revision)
    except (RepositoryNotFoundError, RevisionNotFoundError) as exc:
        raise RuntimeError(
            f"Repo '{repo_id}' not found on the Hugging Face Hub. "
            f"Set LEXORA_FLUX_NF4_REPO / LEXORA_FLUX_NF4_SUBFOLDER to a valid "
            f"pre-quantized safetensors checkpoint for this model."
        ) from exc

    prefix = f"{subfolder}/" if subfolder else ""
    component_files = [f for f in files if f.startswith(prefix)]
    component_files = [f for f in component_files if f.count("/") == prefix.count("/")]

    has_safetensors = any(f.endswith(".safetensors") for f in component_files)
    if not has_safetensors:
        raise RuntimeError(
            f"Refusing to load {repo_id}/{subfolder or ''}: no .safetensors weights found. "
            f"On-the-fly bitsandbytes quantization and .bin checkpoints are disallowed."
        )


class ImageInferenceEngine:
    """
    Image generation backend using diffusers.

    Supports:
      - FLUX.1-schnell  (4-step, no CFG, pre-quantized NF4 safetensors)
      - FLUX.1-dev      (guidance-distilled, pre-quantized NF4 safetensors)
      - SDXL variants   (fp16, negative prompt support)

    All weights are loaded as memory-mapped .safetensors via
    use_safetensors=True + low_cpu_mem_usage=True. No quantization happens
    during load — quantized checkpoints are pulled pre-quantized.

    Output: absolute path to a saved PNG file.
    """

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self._pipe: object | None = None
        self._loaded = False
        self._lock = asyncio.Lock()
        self._variant = self._detect_variant(model_id)
        self._t5_on_cpu = False
        self._t5_can_gpu = False  # set True in _load_flux_gguf when ≥10 GB free

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    async def load(self) -> None:
        async with self._lock:
            if self._loaded:
                return

            try:
                import diffusers  # noqa: F401
                import torch  # noqa: F401
            except ImportError as exc:
                raise RuntimeError(
                    f"Image inference requires diffusers and torch, and failed "
                    f"to import: {exc}. If they're already installed, this is "
                    f"likely a broken/incompatible dependency in that import "
                    f"chain — run `pip install --force-reinstall diffusers "
                    f"transformers accelerate torch` in the worker venv to see "
                    f"the full traceback."
                ) from exc

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._load_sync)
            self._loaded = True
            _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            logger.info("ImageInferenceEngine loaded %s (%s)", self.model_id, self._variant)

    def _load_sync(self) -> None:
        import torch
        from diffusers import FluxPipeline, StableDiffusionXLPipeline, DiffusionPipeline

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        if self._variant == "flux":
            pipe = self._load_flux(dtype, device)
        elif self._variant == "sdxl":
            _require_safetensors(self.model_id)
            pipe = StableDiffusionXLPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                variant="fp16" if device == "cuda" else None,
                use_safetensors=True,
                low_cpu_mem_usage=True,
            )
            pipe = pipe.to(device)
        else:
            _require_safetensors(self.model_id)
            pipe = DiffusionPipeline.from_pretrained(
                self.model_id,
                torch_dtype=dtype,
                use_safetensors=True,
                low_cpu_mem_usage=True,
            )
            pipe = pipe.to(device)

        self._pipe = pipe

    def _load_flux_gguf(self, dtype: object, device: str) -> object:
        """Load FLUX via a GGUF-quantized transformer (Q4_0).
        Transformer + CLIP + VAE run on GPU; T5-XXL stays on CPU by default.
        If ≥10 GB VRAM is free after loading the static pipeline, T5 is moved
        to GPU temporarily per-job (1-2s encoding vs 50-60s on CPU).
        Used on cards with 7–39 GB VRAM when LEXORA_FLUX_GGUF_REPO is set."""
        import torch
        from diffusers import FluxPipeline, FluxTransformer2DModel, GGUFQuantizationConfig
        from huggingface_hub import hf_hub_download

        # Reduce CUDA memory fragmentation at high resolutions
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        gguf_repo = os.environ["LEXORA_FLUX_GGUF_REPO"]
        gguf_filename = os.environ.get("LEXORA_FLUX_GGUF_FILENAME", "flux1-schnell-Q4_0.gguf")

        logger.info("FLUX GGUF: downloading %s / %s", gguf_repo, gguf_filename)
        gguf_path = hf_hub_download(repo_id=gguf_repo, filename=gguf_filename)

        quantization_config = GGUFQuantizationConfig(compute_dtype=torch.bfloat16)
        transformer = FluxTransformer2DModel.from_single_file(
            gguf_path,
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
        )

        pipe = FluxPipeline.from_pretrained(
            self.model_id,
            transformer=transformer,
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
            low_cpu_mem_usage=True,
        )

        # CLIP-L + VAE + transformer go to GPU.
        pipe.text_encoder.to(device)
        pipe.vae.to(device)
        pipe.transformer.to(device)

        # VAE tiling + slicing prevent the ~300 MB spike during high-res decode.
        pipe.vae.enable_tiling()
        pipe.vae.enable_slicing()

        # TF32 matmuls on Ampere — faster with no meaningful quality cost.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

        torch.cuda.empty_cache()
        free_vram_gb = torch.cuda.mem_get_info()[0] / (1024 ** 3)

        # T5-XXL placement — three tiers based on free VRAM after the static pipeline:
        #   ≥ 10 GB free  → bf16 on GPU permanently   (~9.4 GB, encoding ~1s)
        #   ≥  5 GB free  → int8 on GPU permanently   (~4.7 GB, encoding ~1s)
        #   <  5 GB free  → stays on CPU              (encoding ~60s, unavoidable)
        if free_vram_gb >= 10.0:
            pipe.text_encoder_2.to(device)
            self._t5_on_cpu = False
            self._t5_can_gpu = False
            logger.info("FLUX GGUF: T5 on GPU bf16 (%.1f GB free)", free_vram_gb)
        elif free_vram_gb >= 5.0:
            try:
                from transformers import BitsAndBytesConfig, T5EncoderModel
                t5_int8 = T5EncoderModel.from_pretrained(
                    self.model_id,
                    subfolder="text_encoder_2",
                    quantization_config=BitsAndBytesConfig(load_in_8bit=True),
                    torch_dtype=torch.float16,
                )
                del pipe.text_encoder_2
                torch.cuda.empty_cache()
                pipe.text_encoder_2 = t5_int8
                self._t5_on_cpu = False
                self._t5_can_gpu = False
                logger.info("FLUX GGUF: T5 on GPU int8 (~4.7 GB, %.1f GB was free)", free_vram_gb)
            except Exception as exc:
                logger.warning("FLUX GGUF: T5 int8 failed (%s) — CPU fallback", exc)
                pipe.text_encoder_2.to("cpu")
                self._t5_on_cpu = True
                self._t5_can_gpu = False
        else:
            pipe.text_encoder_2.to("cpu")
            self._t5_on_cpu = True
            self._t5_can_gpu = False
            logger.warning(
                "FLUX GGUF: T5 on CPU (only %.1f GB free — image encoding will take ~60s). "
                "Run without co-loaded text models to get GPU-speed encoding.",
                free_vram_gb,
            )
        return pipe

    def _load_flux(self, dtype: object, device: str) -> object:
        """
        Load a FLUX pipeline, choosing the right path based on VRAM and env:

          LEXORA_FLUX_GGUF_REPO set (7–39 GB)
                      →  GGUF Q4_0 transformer on GPU, T5-XXL on CPU
          ≥ 40 GB     →  full bf16 pipeline on GPU (A100 80 GB class)
          6–40 GB     →  pre-quantized NF4 transformer + CLIP-L/VAE on GPU,
                         T5-XXL on CPU
          < 12 GB, schnell only
                      →  NF4 transformer + model CPU offload + VAE tiling
          < 6 GB      →  NF4 transformer + sequential cpu offload
        """
        import torch
        from diffusers import FluxPipeline, FluxTransformer2DModel

        vram_gb = _get_vram_gb()
        logger.info("FLUX load — VRAM: %.1f GB", vram_gb)

        if vram_gb == 0:
            raise RuntimeError(
                "No CUDA GPU is visible to PyTorch (torch.cuda.is_available() "
                "is False), so FLUX cannot be loaded. This usually means your "
                "NVIDIA driver is too old for the installed PyTorch/CUDA build. "
                "Update your GPU driver from https://www.nvidia.com/Download/index.aspx"
            )

        # GGUF path — model_resolver sets this when the GGUF variant was chosen
        if os.environ.get("LEXORA_FLUX_GGUF_REPO"):
            return self._load_flux_gguf(dtype, device)

        if vram_gb >= 40:
            _require_safetensors(self.model_id)
            pipe = FluxPipeline.from_pretrained(
                self.model_id,
                torch_dtype=dtype,
                use_safetensors=True,
                low_cpu_mem_usage=True,
            )
            pipe.to(device)
            logger.info("FLUX: full bf16 pipeline on GPU (mmap'd safetensors)")

        elif vram_gb < 12 and "schnell" in self.model_id:
            # 8 GB tier. enable_model_cpu_offload() keeps only the *active*
            # pipeline stage resident in VRAM — the text encoders, transformer
            # and VAE are swapped in and out per phase — so peak VRAM stays
            # well under the 8 GB cap and Windows never spills to shared system
            # RAM over PCIe. The NF4 transformer must load WITHOUT a fixed
            # device_map so the offload hooks can own its placement (a pinned
            # device_map={"":0} makes enable_model_cpu_offload raise).
            transformer = self._load_or_quantize_flux_transformer_nf4(
                offload_managed=True
            )
            pipe = FluxPipeline.from_pretrained(
                self.model_id,
                transformer=transformer,
                torch_dtype=torch.bfloat16,
                use_safetensors=True,
                low_cpu_mem_usage=True,
            )

            # TF32 matmuls on Ampere (RTX 3060 Ti) — faster with no meaningful
            # quality cost for diffusion. Global backend flags.
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

            # VAE tiling + slicing eliminate the large latent→pixel decode
            # spike, the single worst VRAM offender at 1024px.
            pipe.vae.enable_tiling()
            pipe.vae.enable_slicing()

            # Memory-efficient attention. FLUX already uses PyTorch SDPA; prefer
            # xformers' kernel when present (lower peak activation VRAM), else
            # the SDPA fallback is already memory-efficient.
            try:
                pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                pass

            # Phase-based CPU offload: each component is moved into VRAM only
            # while its stage runs, then evicted. Offload also runs T5-XXL on
            # GPU for its single prompt-encode pass and immediately offloads it
            # — so there is no slow CPU-bf16 T5 path and no manual prompt-
            # embedding handling needed (hence _t5_on_cpu stays False).
            # Do NOT call pipe.to(cuda) after this — the hooks manage devices.
            pipe.enable_model_cpu_offload()
            self._t5_on_cpu = False

            torch.cuda.empty_cache()
            logger.info(
                "FLUX: NF4 transformer + model CPU offload, VAE tiling/slicing, "
                "TF32 (%.1f GB VRAM — peak kept under ~7.5 GB)",
                vram_gb,
            )

        elif vram_gb >= 6:
            transformer = self._load_flux_transformer_nf4()
            pipe = FluxPipeline.from_pretrained(
                self.model_id,
                transformer=transformer,
                torch_dtype=torch.float16,
                use_safetensors=True,
                low_cpu_mem_usage=True,
            )

            # Move VAE and CLIP-L to GPU (small, ~650 MB total)
            # T5-XXL (text_encoder_2, ~9 GB float16) stays on CPU — it runs
            # only during prompt encoding, not during the denoising loop.
            # This means the denoising steps are pure CUDA with no CPU traffic
            # → GIL is released → asyncio heartbeats run normally.
            pipe.vae.to(device)
            pipe.text_encoder.to(device)
            self._t5_on_cpu = True
            logger.info(
                "FLUX: pre-quantized NF4 transformer + CLIP/VAE on GPU, T5 on CPU"
            )

        else:
            transformer = self._load_flux_transformer_nf4()
            pipe = FluxPipeline.from_pretrained(
                self.model_id,
                transformer=transformer,
                torch_dtype=torch.float16,
                use_safetensors=True,
                low_cpu_mem_usage=True,
            )
            pipe.enable_sequential_cpu_offload()
            logger.info("FLUX: pre-quantized NF4 + sequential cpu offload (%.1f GB VRAM)", vram_gb)

        return pipe

    def _load_flux_transformer_nf4(self) -> object:
        """Load a pre-quantized NF4 FLUX transformer (mmap'd safetensors, no
        on-the-fly quantization). The BitsAndBytesConfig here only describes
        the on-disk packing format so the quantized tensors deserialize into
        the correct module layout — no quantization compute happens."""
        import torch
        from diffusers import FluxTransformer2DModel
        from transformers import BitsAndBytesConfig

        env_repo = os.environ.get("LEXORA_FLUX_NF4_REPO")
        if env_repo:
            repo_id = env_repo
            subfolder = os.environ.get("LEXORA_FLUX_NF4_SUBFOLDER", "transformer")
        else:
            entry = _FLUX_NF4_TRANSFORMERS.get(self.model_id)
            if entry is None:
                raise RuntimeError(
                    f"No pre-quantized NF4 transformer registered for {self.model_id}. "
                    f"On-the-fly bitsandbytes quantization has been removed; set "
                    f"LEXORA_FLUX_NF4_REPO (and optionally LEXORA_FLUX_NF4_SUBFOLDER) "
                    f"to a pre-quantized safetensors checkpoint for this model."
                )
            repo_id, subfolder = entry
        _require_safetensors(repo_id, subfolder)

        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        transformer = FluxTransformer2DModel.from_pretrained(
            repo_id,
            subfolder=subfolder,
            quantization_config=nf4_config,
            torch_dtype=torch.float16,
            use_safetensors=True,
            low_cpu_mem_usage=True,
        )
        return transformer

    def _load_or_quantize_flux_transformer_nf4(
        self, offload_managed: bool = True
    ) -> object:
        """Load the NF4 FLUX transformer, quantizing at most once.

        Cache hit:  mmap the previously-quantized safetensors directly from
                    the local cache dir (local_files_only=True) — instant.
        Cache miss: download the full-precision transformer, run
                    bitsandbytes NF4 quantization (the ~5 min CPU-bound
                    pass), then save_pretrained() the quantized result to
                    the cache dir so every later boot hits the fast path.

        When offload_managed is True (the 8 GB tier, governed by
        enable_model_cpu_offload), the transformer is loaded WITHOUT a fixed
        device_map so the offload hooks can move it between CPU and GPU per
        phase. A pinned device_map={"":0} would make enable_model_cpu_offload
        raise.
        """
        import torch
        from diffusers import FluxTransformer2DModel
        from transformers import BitsAndBytesConfig

        cache_name = _FLUX_NF4_CACHE_NAMES.get(self.model_id)
        if cache_name is None:
            raise RuntimeError(
                f"No NF4 cache mapping registered for {self.model_id}."
            )

        cache_dir = _NF4_CACHE_ROOT / cache_name
        cached = (cache_dir / "config.json").exists() and (
            (cache_dir / "diffusion_pytorch_model.safetensors").exists()
            or (cache_dir / "diffusion_pytorch_model.safetensors.index.json").exists()
        )

        if cached:
            logger.info("FLUX NF4 transformer cache hit — mmap loading from %s", cache_dir)
            return FluxTransformer2DModel.from_pretrained(
                str(cache_dir),
                torch_dtype=torch.bfloat16,
                use_safetensors=True,
                low_cpu_mem_usage=True,
                local_files_only=True,
                device_map=None if offload_managed else {"": 0},
            )

        logger.info(
            "FLUX NF4 transformer cache miss — quantizing %s once "
            "(one-time CPU-bound pass, subsequent boots will mmap the cache)",
            self.model_id,
        )
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        # bitsandbytes 4-bit quantization needs the weights on GPU, so the
        # one-time quantize pass uses device_map={"":0} regardless of mode.
        transformer = FluxTransformer2DModel.from_pretrained(
            self.model_id,
            subfolder="transformer",
            quantization_config=nf4_config,
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
            low_cpu_mem_usage=True,
            device_map={"": 0},
        )

        cache_dir.mkdir(parents=True, exist_ok=True)
        transformer.save_pretrained(str(cache_dir))
        logger.info("FLUX NF4 transformer cached to %s for future boots", cache_dir)

        # Reload from the freshly-written cache in the offload-compatible
        # layout (device_map=None) so enable_model_cpu_offload can manage it.
        if offload_managed:
            del transformer
            torch.cuda.empty_cache()
            return FluxTransformer2DModel.from_pretrained(
                str(cache_dir),
                torch_dtype=torch.bfloat16,
                use_safetensors=True,
                low_cpu_mem_usage=True,
                local_files_only=True,
                device_map=None,
            )

        return transformer

    def is_loaded(self) -> bool:
        return self._loaded

    # ── Generation ─────────────────────────────────────────────────────────────

    async def generate(
        self,
        prompt: str,
        width: int = 1024,
        height: int = 1024,
        num_steps: int | None = None,
        guidance_scale: float | None = None,
    ) -> str:
        """Run inference and return the path to the saved PNG."""
        if not self._loaded or self._pipe is None:
            raise RuntimeError("Engine not loaded. Call load() first.")

        # Sensible per-model defaults
        if num_steps is None:
            num_steps = 4 if self._variant == "flux" and "schnell" in self.model_id else 20
        if guidance_scale is None:
            # schnell uses guidance_scale=0 (it's guidance-free)
            guidance_scale = 0.0 if "schnell" in self.model_id else 7.5

        # Run the blocking diffusion pipeline in a worker thread so the asyncio
        # event-loop thread stays free to service the socket.io client's
        # ping/pong frames for the full duration of the generation. PyTorch
        # releases the GIL during its heavy native ops, so a thread (rather than
        # a separate process) is sufficient — provided OpenMP doesn't spin-wait
        # and starve the loop thread of CPU (see _configure_torch_runtime() at
        # worker startup, which sets OMP_WAIT_POLICY=PASSIVE and reserves a core).
        output_path = await asyncio.to_thread(
            self._generate_sync,
            prompt,
            width,
            height,
            num_steps,
            guidance_scale,
        )
        return output_path

    def _generate_sync(
        self,
        prompt: str,
        width: int,
        height: int,
        num_steps: int,
        guidance_scale: float,
    ) -> str:
        import torch

        start = time.monotonic()

        kwargs: dict = dict(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=num_steps,
            output_type="pil",
        )

        if self._variant == "flux":
            # FLUX doesn't use guidance_scale in the traditional sense
            if "schnell" not in self.model_id:
                kwargs["guidance_scale"] = guidance_scale
        else:
            kwargs["guidance_scale"] = guidance_scale

        if self._t5_on_cpu:
            import torch
            pipe = self._pipe
            prompt_2 = kwargs.pop("prompt")

            if self._t5_can_gpu:
                # Enough free VRAM to encode with T5 on GPU (~1-2s vs 50-60s
                # on CPU). Move T5 to GPU, encode, then immediately offload it
                # before the denoising steps so the transformer has full headroom.
                pipe.text_encoder_2.to("cuda")
                torch.cuda.empty_cache()
                try:
                    pooled_prompt_embeds = pipe._get_clip_prompt_embeds(
                        prompt=[prompt_2], device="cuda"
                    )
                    prompt_embeds = pipe._get_t5_prompt_embeds(
                        prompt=[prompt_2], device="cuda"
                    ).to("cuda", dtype=pipe.transformer.dtype)
                finally:
                    pipe.text_encoder_2.to("cpu")
                    torch.cuda.empty_cache()
            else:
                # T5 stays on CPU — encode there and move embeddings to CUDA.
                pooled_prompt_embeds = pipe._get_clip_prompt_embeds(
                    prompt=[prompt_2], device="cuda"
                )
                prompt_embeds = pipe._get_t5_prompt_embeds(
                    prompt=[prompt_2], device="cpu"
                ).to("cuda", dtype=pipe.transformer.dtype)

            kwargs["prompt"] = None
            kwargs["prompt_embeds"] = prompt_embeds
            kwargs["pooled_prompt_embeds"] = pooled_prompt_embeds

        with torch.inference_mode():
            result = self._pipe(**kwargs)  # type: ignore[operator]

        image = result.images[0]

        fname = f"{int(time.time() * 1000)}.png"
        out_path = str(_OUTPUT_DIR / fname)
        image.save(out_path)

        elapsed = (time.monotonic() - start) * 1000
        logger.info(
            "Generated image in %.0f ms → %s",
            elapsed,
            out_path,
        )
        return out_path

    # ── Unload ─────────────────────────────────────────────────────────────────

    async def unload(self) -> None:
        async with self._lock:
            if self._pipe is not None:
                try:
                    import torch
                    del self._pipe
                    self._pipe = None
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception as exc:
                    logger.warning("Error unloading image engine: %s", exc)
            self._loaded = False
            logger.info("ImageInferenceEngine unloaded %s", self.model_id)

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _detect_variant(model_id: str) -> str:
        lower = model_id.lower()
        if "flux" in lower:
            return "flux"
        if "sdxl" in lower or "stable-diffusion-xl" in lower:
            return "sdxl"
        return "generic"
