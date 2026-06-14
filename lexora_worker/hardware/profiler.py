from __future__ import annotations

import asyncio
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING

import psutil

from lexora_worker.models import GpuInfo, HardwareProfile

_IS_MAC = platform.system() == "Darwin"

try:
    import pynvml

    if _IS_MAC:
        _NVML_AVAILABLE = False
    else:
        pynvml.nvmlInit()
        _NVML_AVAILABLE = True
except Exception:
    _NVML_AVAILABLE = False

try:
    import torch

    _TORCH_AVAILABLE = not _IS_MAC and torch.cuda.is_available()
except ImportError:
    _TORCH_AVAILABLE = False


def _apple_chip_name() -> str:
    try:
        out = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
        if out:
            return out
    except Exception:
        pass
    return platform.processor() or "Apple Silicon"


def _get_gpu_info_mac() -> list[GpuInfo]:
    """Report Apple Silicon unified memory as GPU info (no discrete VRAM on Mac)."""
    mem = psutil.virtual_memory()
    chip = _apple_chip_name()
    total_gb = mem.total / 1024**3
    free_gb = mem.available / 1024**3
    used_gb = (mem.total - mem.available) / 1024**3
    return [
        GpuInfo(
            index=0,
            name=f"{chip} (Unified Memory)",
            vram_total_gb=round(total_gb, 2),
            vram_free_gb=round(free_gb, 2),
            vram_used_gb=round(used_gb, 2),
            gpu_utilization_pct=0.0,
            temperature_c=0.0,
        )
    ]


# ─── GPU detection ────────────────────────────────────────────────────────────


def _get_gpu_info_nvml() -> list[GpuInfo]:
    gpus: list[GpuInfo] = []
    device_count: int = pynvml.nvmlDeviceGetCount()

    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        name: str = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode()

        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)

        try:
            temp: float = float(
                pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            )
        except pynvml.NVMLError:
            temp = 0.0

        gpus.append(
            GpuInfo(
                index=i,
                name=name,
                vram_total_gb=mem.total / 1024**3,
                vram_free_gb=mem.free / 1024**3,
                vram_used_gb=mem.used / 1024**3,
                gpu_utilization_pct=float(util.gpu),
                temperature_c=temp,
            )
        )

    return gpus


def _get_gpu_info_torch() -> list[GpuInfo]:
    gpus: list[GpuInfo] = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        total = props.total_memory / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        free = total - reserved

        gpus.append(
            GpuInfo(
                index=i,
                name=props.name,
                vram_total_gb=total,
                vram_free_gb=free,
                vram_used_gb=allocated,
                gpu_utilization_pct=0.0,
                temperature_c=0.0,
            )
        )
    return gpus


def _get_gpu_info_fallback() -> list[GpuInfo]:
    return []


def get_gpu_info() -> list[GpuInfo]:
    if _IS_MAC:
        return _get_gpu_info_mac()
    if _NVML_AVAILABLE:
        return _get_gpu_info_nvml()
    if _TORCH_AVAILABLE and torch.cuda.is_available():
        return _get_gpu_info_torch()
    return _get_gpu_info_fallback()


# ─── Model cache detection ────────────────────────────────────────────────────


def detect_cached_models(cache_dir: str | None = None) -> list[str]:
    resolved = Path(cache_dir or os.path.expanduser("~/.cache/huggingface/hub"))
    if not resolved.exists():
        return []

    models: list[str] = []
    for entry in resolved.iterdir():
        if entry.is_dir() and entry.name.startswith("models--"):
            model_name = entry.name[len("models--") :].replace("--", "/")
            models.append(model_name)

    return sorted(models)


# ─── Network speed ────────────────────────────────────────────────────────────


async def measure_network_speed() -> float:
    """
    Returns estimated download speed in Mbps via a lightweight
    HTTP range request. Falls back to 0 on any failure.
    """
    try:
        import aiohttp

        url = "https://speed.cloudflare.com/__down?bytes=2000000"
        start = time.monotonic()
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=8)) as resp:
                data = await resp.read()
        elapsed = time.monotonic() - start
        if elapsed == 0:
            return 0.0
        mbps = (len(data) * 8) / elapsed / 1_000_000
        return round(mbps, 2)
    except Exception:
        return 0.0


# ─── Full profile ─────────────────────────────────────────────────────────────


async def build_hardware_profile(cache_dir: str | None = None) -> HardwareProfile:
    gpus = get_gpu_info()
    ram_gb = psutil.virtual_memory().total / 1024**3
    cpu_count_logical = psutil.cpu_count(logical=True) or 1
    cpu_count_physical = psutil.cpu_count(logical=False) or 1

    cpu_brand = platform.processor() or "unknown"
    if not cpu_brand and platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        cpu_brand = line.split(":")[1].strip()
                        break
        except OSError:
            pass

    network_speed = await measure_network_speed()
    cached_models = detect_cached_models(cache_dir)

    primary = gpus[0] if gpus else None

    return HardwareProfile(
        gpu_model=primary.name if primary else "cpu-only",
        vram=primary.vram_total_gb if primary else 0.0,
        vram_free=primary.vram_free_gb if primary else 0.0,
        ram=round(ram_gb, 2),
        cpu=cpu_brand,
        cpu_cores=cpu_count_physical,
        cpu_threads=cpu_count_logical,
        loaded_models=cached_models,
        network_speed=network_speed,
        gpus=gpus,
    )


def get_live_vram_free() -> float:
    gpus = get_gpu_info()
    if not gpus:
        return 0.0
    return gpus[0].vram_free_gb


def get_live_gpu_utilization() -> float:
    gpus = get_gpu_info()
    if not gpus:
        return 0.0
    return gpus[0].gpu_utilization_pct
