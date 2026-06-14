from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from depin_worker.hardware.profiler import build_hardware_profile, detect_cached_models


@pytest.mark.asyncio
async def test_build_hardware_profile_cpu_only() -> None:
    """Profile builds successfully even when no GPU is present."""
    with (
        patch("depin_worker.hardware.profiler._NVML_AVAILABLE", False),
        patch("depin_worker.hardware.profiler._TORCH_AVAILABLE", False),
        patch(
            "depin_worker.hardware.profiler.measure_network_speed",
            new=AsyncMock(return_value=100.0),
        ),
    ):
        profile = await build_hardware_profile()

    assert profile.gpu_model == "cpu-only"
    assert profile.vram == 0.0
    assert profile.ram > 0


def test_detect_cached_models_empty(tmp_path: object) -> None:
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as d:
        models = detect_cached_models(d)
    assert models == []


def test_detect_cached_models_found(tmp_path: object) -> None:
    import tempfile
    import os
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "models--mistralai--Mistral-7B-v0.1").mkdir()
        (Path(d) / "models--meta-llama--Llama-2-7b-chat-hf").mkdir()
        models = detect_cached_models(d)

    assert "mistralai/Mistral-7B-v0.1" in models
    assert "meta-llama/Llama-2-7b-chat-hf" in models
