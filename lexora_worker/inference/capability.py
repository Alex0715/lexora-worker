from __future__ import annotations

import platform

# Model IDs (same as the orchestrator's capability.config.ts)
TEXT_3B = "mlx-community/Llama-3.2-3B-Instruct-4bit"  # Mac
TEXT_3B_CUDA = "meta-llama/Llama-3.2-3B-Instruct"
TEXT_8B = "mlx-community/Llama-3.1-8B-Instruct-4bit"  # Mac
TEXT_8B_CUDA = "meta-llama/Llama-3.1-8B-Instruct"
TEXT_70B = "mlx-community/Llama-3.1-70B-Instruct-4bit"
TEXT_70B_CUDA = "meta-llama/Llama-3.1-70B-Instruct"
IMG_SCHNELL = "black-forest-labs/FLUX.1-schnell"
IMG_DEV = "black-forest-labs/FLUX.1-dev"
VID_WAN = "Wan-AI/Wan2.1-T2V-14B"
VID_SVD = "stabilityai/stable-video-diffusion-img2vid-xt"


MODEL_VRAM_COST: dict[str, float] = {
    TEXT_3B: 3.0,
    TEXT_3B_CUDA: 3.0,
    "bartowski/Llama-3.2-3B-Instruct-GGUF": 2.0,   # Q4_K_M
    TEXT_8B: 6.0,
    TEXT_8B_CUDA: 6.0,
    "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF": 5.0,  # Q4_K_M
    TEXT_70B: 42.0,
    TEXT_70B_CUDA: 42.0,
    IMG_SCHNELL: 7.5,   # Q4_0 transformer ~6.8 GB + CLIP/VAE ~0.7 GB (T5 stays on CPU when VRAM is tight)
    IMG_DEV: 20.0,
    VID_WAN: 16.0,
    VID_SVD: 14.0,
}


def _is_mac(gpu_model: str) -> bool:
    lower = (gpu_model or "").lower()
    return (
        platform.system() == "Darwin"
        or lower.startswith("apple")
        or lower.startswith("m1")
        or lower.startswith("m2")
        or lower.startswith("m3")
        or lower.startswith("m4")
        or "metal" in lower
    )


def get_capable_models(gpu_model: str, vram_gb: float) -> list[str]:
    mac = _is_mac(gpu_model)

    if mac:
        if vram_gb >= 64:
            return [TEXT_3B, TEXT_8B, TEXT_70B, IMG_SCHNELL]
        if vram_gb >= 32:
            return [TEXT_3B, TEXT_8B, IMG_SCHNELL]
        if vram_gb >= 16:
            return [TEXT_3B, TEXT_8B, IMG_SCHNELL]
        return [TEXT_3B]

    if vram_gb >= 80:
        return [
            TEXT_3B_CUDA,
            TEXT_8B_CUDA,
            TEXT_70B_CUDA,
            IMG_SCHNELL,
            IMG_DEV,
            VID_WAN,
            VID_SVD,
        ]
    if vram_gb >= 40:
        return [
            TEXT_3B_CUDA,
            TEXT_8B_CUDA,
            TEXT_70B_CUDA,
            IMG_SCHNELL,
            IMG_DEV,
            VID_WAN,
            VID_SVD,
        ]
    if vram_gb >= 24:
        return [TEXT_3B_CUDA, TEXT_8B_CUDA, IMG_SCHNELL, IMG_DEV, VID_WAN]
    if vram_gb >= 16:
        return [TEXT_3B_CUDA, TEXT_8B_CUDA, IMG_SCHNELL]
    if vram_gb >= 8:
        return [TEXT_3B_CUDA, IMG_SCHNELL]
    return [TEXT_3B_CUDA]


def get_default_model(gpu_model: str, vram_gb: float) -> str:
    mac = _is_mac(gpu_model)
    if mac:
        if vram_gb >= 64:
            return TEXT_70B
        if vram_gb >= 16:
            return TEXT_8B
        return TEXT_3B
    if vram_gb >= 40:
        return TEXT_70B_CUDA
    if vram_gb >= 12:
        return TEXT_8B_CUDA
    return TEXT_3B_CUDA
