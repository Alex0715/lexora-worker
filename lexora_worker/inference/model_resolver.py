from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelVariant:
    """A single hardware-tier variant of a model alias."""

    min_vram_gb: float
    hf_repo: str
    format: str          # "safetensors" | "nf4" | "gguf"
    quant_repo: str | None = None
    filename: str | None = None
    subfolder: str | None = None
    label: str = ""
    backend: str | None = None   # "mlx" | "cuda" | None (any)


@dataclass(frozen=True)
class ResolvedModel:
    """The variant chosen for the current host, plus its alias."""

    alias: str
    variant: ModelVariant

    @property
    def model_id(self) -> str:
        return self.variant.hf_repo

    def apply_env(self) -> None:
        """Set LEXORA_* env vars consumed by inference engines so the chosen
        variant is what actually gets downloaded and loaded."""
        variant = self.variant
        is_image = any(k in self.alias for k in ("flux", "sdxl", "stable-diffusion"))

        if variant.format == "gguf":
            if is_image:
                if variant.quant_repo:
                    os.environ["LEXORA_FLUX_GGUF_REPO"] = variant.quant_repo
                if variant.filename:
                    os.environ["LEXORA_FLUX_GGUF_FILENAME"] = variant.filename
            else:
                # Text GGUF: store repo + filename for InferenceEngine._load_vllm
                os.environ["LEXORA_TEXT_GGUF_REPO"] = variant.hf_repo
                if variant.filename:
                    os.environ["LEXORA_TEXT_GGUF_FILENAME"] = variant.filename
                # quant_repo holds the original model repo used as tokenizer source
                if variant.quant_repo:
                    os.environ["LEXORA_TEXT_GGUF_TOKENIZER"] = variant.quant_repo
        elif variant.format == "nf4":
            if variant.quant_repo:
                os.environ["LEXORA_FLUX_NF4_REPO"] = variant.quant_repo
            if variant.subfolder:
                os.environ["LEXORA_FLUX_NF4_SUBFOLDER"] = variant.subfolder


# Generic model aliases → hardware-specific variants, ordered by quality
# (highest min_vram_gb first is enforced by resolve_model, not by this
# declaration order).
MODEL_MANIFEST: dict[str, list[ModelVariant]] = {
    # ── Text: Llama 3.2 3B ───────────────────────────────────────────────────
    "llama-3.2-3b": [
        ModelVariant(
            min_vram_gb=4.0,
            hf_repo="mlx-community/Llama-3.2-3B-Instruct-4bit",
            format="safetensors",
            label="4-bit MLX (Mac)",
            backend="mlx",
        ),
        ModelVariant(
            min_vram_gb=9.0,
            hf_repo="meta-llama/Llama-3.2-3B-Instruct",
            format="safetensors",
            label="bf16 (CUDA, ≥9 GB)",
            backend="cuda",
        ),
        ModelVariant(
            min_vram_gb=2.0,
            hf_repo="bartowski/Llama-3.2-3B-Instruct-GGUF",
            format="gguf",
            filename="Llama-3.2-3B-Instruct-Q4_K_M.gguf",
            quant_repo="meta-llama/Llama-3.2-3B-Instruct",
            label="GGUF Q4_K_M (CUDA, 2 GB)",
            backend="cuda",
        ),
    ],
    # ── Text: Llama 3.1 8B ───────────────────────────────────────────────────
    "llama-3.1-8b": [
        ModelVariant(
            min_vram_gb=18.0,
            hf_repo="meta-llama/Llama-3.1-8B-Instruct",
            format="safetensors",
            label="fp16 (CUDA, ≥18 GB)",
            backend="cuda",
        ),
        ModelVariant(
            min_vram_gb=8.0,
            hf_repo="mlx-community/Llama-3.1-8B-Instruct-4bit",
            format="safetensors",
            label="4-bit MLX (Mac)",
            backend="mlx",
        ),
        ModelVariant(
            min_vram_gb=6.0,
            hf_repo="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
            format="gguf",
            filename="Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
            label="GGUF Q4_K_M (CUDA, 6–15 GB)",
            backend="cuda",
        ),
    ],
    # ── Image: FLUX.1-schnell ─────────────────────────────────────────────────
    "flux.1-schnell": [
        ModelVariant(
            min_vram_gb=40.0,
            hf_repo="black-forest-labs/FLUX.1-schnell",
            format="safetensors",
            label="full bf16 pipeline (≥40 GB)",
        ),
        ModelVariant(
            min_vram_gb=7.0,
            hf_repo="black-forest-labs/FLUX.1-schnell",
            format="gguf",
            quant_repo="city96/FLUX.1-schnell-gguf",
            filename="flux1-schnell-Q4_0.gguf",
            label="GGUF Q4_0 transformer, T5 on CPU",
        ),
    ],
    # ── Image: FLUX.1-dev ────────────────────────────────────────────────────
    "flux.1-dev": [
        ModelVariant(
            min_vram_gb=60.0,
            hf_repo="black-forest-labs/FLUX.1-dev",
            format="safetensors",
            label="full bf16 pipeline (≥60 GB)",
        ),
        ModelVariant(
            min_vram_gb=12.0,
            hf_repo="black-forest-labs/FLUX.1-dev",
            format="nf4",
            quant_repo="hf-internal-testing/flux.1-dev-nf4-pkg",
            subfolder="transformer",
            label="NF4 transformer, T5 on CPU",
        ),
    ],
}

# Reverse lookup so passing a full HF repo id (e.g. via --model) also resolves.
_REPO_TO_ALIAS: dict[str, str] = {}
for _alias, _variants in MODEL_MANIFEST.items():
    for _variant in _variants:
        _REPO_TO_ALIAS.setdefault(_variant.hf_repo.lower(), _alias)


def normalize_alias(model: str) -> str:
    """Map a user-supplied model string (alias or full HF repo id) to a
    MODEL_MANIFEST key. Falls through unchanged if unrecognized."""
    lower = model.lower()
    if lower in MODEL_MANIFEST:
        return lower
    return _REPO_TO_ALIAS.get(lower, lower)


def _active_backend() -> str:
    """Return 'mlx', 'cuda', or 'cpu' for the current host."""
    import platform as _platform
    if _platform.system() == "Darwin":
        try:
            import mlx  # noqa: F401
            return "mlx"
        except ImportError:
            pass
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def cheapest_vram_for(alias: str) -> float:
    """Return the minimum VRAM (GB) of the cheapest compatible variant.
    Used to reserve budget for co-loaded models before resolving each one."""
    import platform as _platform
    key = normalize_alias(alias)
    variants = MODEL_MANIFEST.get(key, [])
    if not variants:
        return 8.0
    is_mac = _platform.system() == "Darwin"
    backend = _active_backend()
    platform_variants = [
        v for v in variants
        if (v.backend != "mlx" or is_mac) and (v.backend != "cuda" or not is_mac)
    ] or list(variants)
    compatible = [v for v in platform_variants if v.backend is None or v.backend == backend] or platform_variants
    return min(v.min_vram_gb for v in compatible)


def resolve_model(alias: str, available_vram_gb: float) -> ResolvedModel:
    """Resolve a generic model alias to the best-fitting hardware variant.

    Filters by the current backend (mlx / cuda / cpu) first, then sorts
    remaining candidates by min_vram_gb descending so the highest-quality
    variant that fits available VRAM wins.

    Raises ValueError if no variant fits the VRAM budget.
    """
    key = normalize_alias(alias)
    variants = MODEL_MANIFEST.get(key)
    if not variants:
        raise ValueError(
            f"Unknown model alias '{alias}' — no hardware variants registered "
            f"in MODEL_MANIFEST."
        )

    import platform as _platform

    backend = _active_backend()
    is_mac = _platform.system() == "Darwin"

    # MLX variants only ever run on Mac, and "cuda"-tagged variants are
    # PyTorch/CUDA checkpoints that won't load correctly via MLX. Exclude
    # whichever family can never run on this OS *before* falling back —
    # otherwise a transient `torch.cuda.is_available() == False` on a
    # Windows/Linux CUDA box (driver issue, etc.) would fall through to an
    # MLX-quantized checkpoint that transformers can't load.
    platform_variants = [
        v for v in variants
        if (v.backend != "mlx" or is_mac) and (v.backend != "cuda" or not is_mac)
    ]
    if not platform_variants:
        platform_variants = list(variants)

    # Keep variants that match the active backend (or have no backend restriction).
    compatible = [v for v in platform_variants if v.backend is None or v.backend == backend]

    # Fall back to all platform-compatible variants if nothing matches (e.g.
    # CUDA reported unavailable but this is still the only sane option).
    if not compatible:
        compatible = platform_variants

    floor = min(v.min_vram_gb for v in compatible)
    if available_vram_gb < floor:
        raise ValueError(
            f"'{alias}' requires at least {floor:.1f} GB VRAM on this platform "
            f"({backend}), but this node has {available_vram_gb:.1f} GB available."
        )

    ordered = sorted(compatible, key=lambda v: v.min_vram_gb, reverse=True)
    for variant in ordered:
        if available_vram_gb >= variant.min_vram_gb:
            return ResolvedModel(alias=key, variant=variant)

    raise ValueError(
        f"No variant of '{alias}' fits {available_vram_gb:.1f} GB VRAM on {backend}."
    )
