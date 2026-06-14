from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
from pathlib import Path
from typing import Optional

# ── Keep the asyncio event loop schedulable during heavy CPU inference ──────────
# PyTorch's OpenMP/MKL backend spin-waits by default (busy-loops on the CPU even
# between ops), pinning every core at 100%. That starves the single asyncio
# event-loop thread of OS scheduling, so the socket.io client misses ping/pong
# frames and the orchestrator drops the worker mid-image-generation. Making idle
# OpenMP threads sleep instead frees the loop thread to keep the connection
# alive. These MUST be set before torch (and thus the OpenMP runtime) is first
# imported anywhere — hence module top, before any heavy import. setdefault so an
# operator can still override from the environment.
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")  # libgomp (Linux)
os.environ.setdefault("KMP_BLOCKTIME", "0")          # libiomp / Intel MKL (Windows)

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table

from lexora_worker.config.settings import (
    _apply_hf_token,
    clear_token,
    get_hardware_fingerprint,
    load_config,
    save_config,
    save_token,
)
from lexora_worker.models import WorkerConfig

# Heavy imports (torch, vllm, socketio) are deferred to the commands that
# need them so that `login`, `logout`, and `info` start instantly.

app = typer.Typer(
    name="lexora-worker",
    help="Lexora Network — Worker Node CLI",
    add_completion=False,
)
console = Console()


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[
            RichHandler(
                console=Console(stderr=True),
                show_time=True,
                show_path=verbose,
                markup=True,
                # rich_tracebacks dumps ALL local variables in every frame.
                # Only enable in --verbose so user prompts, messages, and
                # API payloads never appear in production logs.
                rich_tracebacks=verbose,
                tracebacks_suppress=[],
            )
        ],
    )
    for noisy in ("engineio", "socketio", "urllib3", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# ─── login ────────────────────────────────────────────────────────────────────


@app.command()
def login(
    token: str = typer.Option(..., "--token", "-t", help="JWT issued by the orchestrator"),
    orchestrator_url: Optional[str] = typer.Option(
        None,
        "--url",
        help="Orchestrator base URL (defaults to DEPIN_ORCHESTRATOR_URL from .env)",
    ),
    hf_token: Optional[str] = typer.Option(
        None,
        "--hf-token",
        help="HuggingFace access token (needed to download gated models like Llama/FLUX)",
    ),
) -> None:
    """Save worker JWT token to system keychain."""
    cfg = load_config()
    cfg.token = token
    if orchestrator_url:
        cfg.orchestrator_url = orchestrator_url
    if hf_token:
        cfg.hf_token = hf_token
        _apply_hf_token(hf_token)
    save_config(cfg)

    fingerprint = get_hardware_fingerprint()
    console.print(
        Panel.fit(
            f"[green]Token saved[/green]\n"
            f"Orchestrator: [cyan]{cfg.orchestrator_url}[/cyan]\n"
            f"Hardware fingerprint: [dim]{fingerprint}[/dim]",
            title="[bold]Lexora Worker — Login",
        )
    )


# ─── logout ───────────────────────────────────────────────────────────────────


@app.command()
def logout() -> None:
    """Remove stored JWT token from system keychain."""
    clear_token()
    console.print("[yellow]Token removed.[/yellow]")


# ─── info ─────────────────────────────────────────────────────────────────────


@app.command()
def info() -> None:
    """Print detected hardware profile."""
    from lexora_worker.hardware.profiler import build_hardware_profile

    async def _run() -> None:
        profile = await build_hardware_profile()
        table = Table(title="Hardware Profile", show_header=True)
        table.add_column("Field", style="cyan")
        table.add_column("Value", style="white")

        table.add_row("GPU", profile.gpu_model)
        table.add_row("VRAM Total", f"{profile.vram:.1f} GB")
        table.add_row("VRAM Free", f"{profile.vram_free:.1f} GB")
        table.add_row("RAM", f"{profile.ram:.1f} GB")
        table.add_row("CPU", profile.cpu)
        table.add_row("CPU Cores / Threads", f"{profile.cpu_cores} / {profile.cpu_threads}")
        table.add_row("Network Speed", f"{profile.network_speed:.1f} Mbps")
        table.add_row("Cached Models", str(len(profile.loaded_models)))
        for m in profile.loaded_models:
            table.add_row("  └", m)

        console.print(table)

    asyncio.run(_run())


# ─── start ────────────────────────────────────────────────────────────────────


@app.command()
def start(
    model: Optional[list[str]] = typer.Option(None, "--model", "-m", help="HuggingFace model ID (auto-detected if omitted). Repeat to preload multiple models."),
    max_concurrency: int = typer.Option(
        1, "--max-concurrency", "-c", min=1, max=16, help="Max parallel jobs"
    ),
    max_model_len: int = typer.Option(
        4096, "--max-model-len", help="Max sequence length for vLLM"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Start the worker node and connect to the orchestrator."""
    _setup_logging(verbose)

    cfg = load_config()
    if not cfg.token:
        console.print(
            "[red]No token found. Run [bold]lexora-worker login --token <JWT>[/bold] first.[/red]"
        )
        raise typer.Exit(1)

    if not model:
        from lexora_worker.hardware.profiler import build_hardware_profile
        profile = asyncio.run(build_hardware_profile())
        recommendations = _recommend_models(profile.gpu_model, profile.vram)
        model = [recommendations[0][0]]
        console.print(f"[dim]Auto-selected model: [bold]{model[0]}[/bold][/dim]")

    try:
        asyncio.run(
            _worker_main(
                cfg=cfg,
                model_id=model,
                max_concurrency=max_concurrency,
                max_model_len=max_model_len,
            )
        )
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass


async def _worker_main(
    cfg: WorkerConfig,
    model_id: str | list[str],
    max_concurrency: int,
    max_model_len: int,
) -> None:
    # Deferred imports — only loaded when `start` is actually invoked
    from lexora_worker.hardware.profiler import build_hardware_profile
    from lexora_worker.inference.model_manager import ModelManager
    from lexora_worker.inference.model_resolver import resolve_model
    from lexora_worker.job.manager import JobManager
    from lexora_worker.websocket.client import WorkerSocketClient

    logger = logging.getLogger("lexora.worker")

    profile = await build_hardware_profile(cfg.model_cache_dir)

    model_ids = [model_id] if isinstance(model_id, str) else list(model_id)

    # Resolve each requested alias/repo to the hardware variant that fits this
    # node's VRAM *before* any weights are downloaded or loaded. A ValueError
    # here means no variant fits — exit cleanly instead of risking an OOM.
    resolved_ids: list[str] = []
    resolved_labels: list[str] = []
    for mid in model_ids:
        try:
            resolved = resolve_model(mid, profile.vram)
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(1) from exc
        resolved.apply_env()
        resolved_ids.append(resolved.model_id)
        resolved_labels.append(f"{resolved.model_id} [dim]({resolved.variant.label})[/dim]")

    console.print(
        Panel.fit(
            f"Models: [cyan]{', '.join(resolved_labels)}[/cyan]\n"
            f"Concurrency: [cyan]{max_concurrency}[/cyan]\n"
            f"Orchestrator: [cyan]{cfg.orchestrator_url}[/cyan]",
            title="[bold green]Lexora Worker — Starting",
        )
    )

    model_manager = ModelManager(model_cache_dir=cfg.model_cache_dir)

    console.print(f"[dim]Loading model [bold]{resolved_ids[0]}[/bold]...[/dim]")
    try:
        await model_manager.initialize(resolved_ids[0], total_vram=profile.vram)
    except Exception as exc:
        console.print(f"[red]Failed to load model: {exc}[/red]")
        raise typer.Exit(1) from exc
    console.print(f"[green]Model loaded[/green] ✓")

    for extra_id in resolved_ids[1:]:
        console.print(f"[dim]Loading model [bold]{extra_id}[/bold]...[/dim]")
        try:
            await model_manager.ensure_model_loaded(extra_id)
            console.print(f"[green]Model loaded[/green] ✓")
        except Exception as exc:
            console.print(f"[yellow]⚠[/yellow] Failed to preload {extra_id}: {exc}")

    loop = asyncio.get_running_loop()
    node_id = cfg.node_id or get_hardware_fingerprint()[:16]

    socket_ref: list[WorkerSocketClient] = []

    async def emit(event: str, data: dict) -> None:  # type: ignore[type-arg]
        if socket_ref:
            await socket_ref[0].emit(event, data)

    job_manager = JobManager(
        model_manager=model_manager,
        node_id=node_id,
        max_concurrency=max_concurrency,
        emit=emit,
    )

    socket_client = WorkerSocketClient(
        orchestrator_url=cfg.orchestrator_url,
        token=cfg.token,
        node_id=node_id,
        job_manager=job_manager,
        model_cache_dir=cfg.model_cache_dir,
    )
    socket_ref.append(socket_client)

    shutdown_event = asyncio.Event()

    def _handle_signal(sig: signal.Signals) -> None:
        logger.info("Received %s — shutting down gracefully...", sig.name)
        shutdown_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _handle_signal, sig)
        except NotImplementedError:
            # add_signal_handler is unsupported on Windows event loops.
            signal.signal(sig, lambda s, _frame, sig=sig: _handle_signal(sig))

    connect_task = asyncio.create_task(
        socket_client.connect_with_retry(), name="socket-connect"
    )

    try:
        await shutdown_event.wait()
    finally:
        connect_task.cancel()
        try:
            await connect_task
        except asyncio.CancelledError:
            pass

        await socket_client.shutdown()
        await model_manager.unload_all()

        console.print("[bold yellow]Worker stopped.[/bold yellow]")


# ─── setup wizard ─────────────────────────────────────────────────────────────

_MODEL_RECOMMENDATIONS = {
    # (platform, vram_gb) -> (model_id, label)
    "mac_small":   ("mlx-community/Llama-3.2-3B-Instruct-4bit",  "Llama 3.2 3B  (fast, 4-bit, MLX)"),
    "mac_medium":  ("mlx-community/Llama-3.1-8B-Instruct-4bit",  "Llama 3.1 8B  (balanced, 4-bit, MLX)"),
    "mac_large":   ("mlx-community/Llama-3.1-70B-Instruct-4bit", "Llama 3.1 70B (powerful, 4-bit, MLX)"),
    "gpu_small":   ("meta-llama/Llama-3.2-3B-Instruct",          "Llama 3.2 3B  (fast, NVIDIA)"),
    "gpu_medium":  ("meta-llama/Llama-3.1-8B-Instruct",          "Llama 3.1 8B  (balanced, NVIDIA)"),
    "gpu_large":   ("meta-llama/Llama-3.1-70B-Instruct",         "Llama 3.1 70B (powerful, NVIDIA)"),
    "gpu_image":   ("black-forest-labs/FLUX.1-schnell",          "FLUX.1 Schnell (image generation, NVIDIA)"),
    "cpu":         ("microsoft/Phi-3-mini-4k-instruct",           "Phi-3 Mini    (CPU-friendly)"),
}


def _recommend_models(gpu_model: str, vram_gb: float) -> list[tuple[str, str]]:
    import platform as _platform
    is_mac = _platform.system() == "Darwin"
    is_cpu = gpu_model == "cpu-only"

    if is_cpu:
        return [_MODEL_RECOMMENDATIONS["cpu"]]

    if is_mac:
        opts = [_MODEL_RECOMMENDATIONS["mac_small"]]
        if vram_gb >= 12:
            opts.append(_MODEL_RECOMMENDATIONS["mac_medium"])
        if vram_gb >= 48:
            opts.append(_MODEL_RECOMMENDATIONS["mac_large"])
        return opts

    opts = [_MODEL_RECOMMENDATIONS["gpu_small"]]
    if vram_gb >= 8:
        opts.append(_MODEL_RECOMMENDATIONS["gpu_image"])
    if vram_gb >= 12:
        opts.append(_MODEL_RECOMMENDATIONS["gpu_medium"])
    if vram_gb >= 40:
        opts.append(_MODEL_RECOMMENDATIONS["gpu_large"])
    return opts


def _install_service_mac(models: list[str], orchestrator_url: str) -> bool:
    """Install a launchd plist so the worker starts on login."""
    import shutil
    import subprocess
    plist_dir = Path.home() / "Library" / "LaunchAgents"
    plist_dir.mkdir(parents=True, exist_ok=True)
    plist_path = plist_dir / "network.lexora.worker.plist"
    worker_bin = shutil.which("lexora-worker") or "lexora-worker"
    log_path = Path.home() / ".config" / "lexora-worker" / "worker.log"

    model_args = "".join(f"<string>--model</string><string>{m}</string>\n    " for m in models)

    plist = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>network.lexora.worker</string>
  <key>ProgramArguments</key>
  <array>
    <string>{worker_bin}</string>
    <string>start</string>
    {model_args}</array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>{log_path}</string>
  <key>StandardErrorPath</key><string>{log_path}</string>
</dict>
</plist>"""

    plist_path.write_text(plist)
    result = subprocess.run(["launchctl", "load", str(plist_path)], capture_output=True)
    return result.returncode == 0


def _install_service_linux(models: list[str], orchestrator_url: str) -> bool:
    """Install a systemd user service so the worker starts on login."""
    import shutil
    import subprocess
    service_dir = Path.home() / ".config" / "systemd" / "user"
    service_dir.mkdir(parents=True, exist_ok=True)
    service_path = service_dir / "lexora-worker.service"
    worker_bin = shutil.which("lexora-worker") or "lexora-worker"
    log_path = Path.home() / ".config" / "lexora-worker" / "worker.log"

    model_args = " ".join(f"--model {m}" for m in models)

    unit = f"""[Unit]
Description=Lexora Worker Node
After=network.target

[Service]
ExecStart={worker_bin} start {model_args}
Restart=always
RestartSec=10
StandardOutput=append:{log_path}
StandardError=append:{log_path}

[Install]
WantedBy=default.target
"""
    service_path.write_text(unit)
    subprocess.run(["systemctl", "--user", "daemon-reload"], capture_output=True)
    result = subprocess.run(
        ["systemctl", "--user", "enable", "--now", "lexora-worker"],
        capture_output=True,
    )
    return result.returncode == 0


@app.command()
def setup() -> None:
    """Interactive setup wizard — configure and start your worker node."""
    import platform as _platform
    from rich.prompt import Confirm, Prompt

    console.rule("[bold green]Lexora Worker — Setup Wizard")
    console.print()
    console.print("This wizard will configure your node in under 2 minutes.\n")

    # ── Step 1: detect hardware ───────────────────────────────────────────────
    from lexora_worker.hardware.profiler import build_hardware_profile
    from lexora_worker.inference.engine import _active_backend

    console.print("[dim]Detecting hardware...[/dim]")
    profile = asyncio.run(build_hardware_profile())

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="cyan")
    table.add_column(style="white")
    table.add_row("GPU / Chip", profile.gpu_model)
    table.add_row("Memory", f"{profile.vram:.0f} GB")
    table.add_row("RAM", f"{profile.ram:.0f} GB")
    table.add_row("Backend", _active_backend().upper())
    table.add_row("Network", f"{profile.network_speed:.0f} Mbps")
    console.print(Panel(table, title="[bold]Your Hardware", border_style="green"))
    console.print()

    # ── Step 2: orchestrator URL ──────────────────────────────────────────────
    cfg = load_config()
    orchestrator_url = cfg.orchestrator_url or "https://api.lexora.network"

    # ── Step 3: worker token ──────────────────────────────────────────────────
    console.print()
    web_url = os.environ.get("LEXORA_WEB_URL", orchestrator_url)
    console.print(
        Panel(
            f"Get your worker token from:\n"
            f"[bold cyan]{web_url}/provider/nodes[/bold cyan]\n\n"
            f"Click [bold]Generate Worker Token[/bold] and paste it below.",
            title="[bold yellow]Worker Token Required",
            border_style="yellow",
        )
    )
    token = Prompt.ask("[cyan]Paste your worker token[/cyan]", password=True)
    if not token.strip():
        console.print("[red]No token provided. Exiting.[/red]")
        raise typer.Exit(1)

    # ── Step 3b: HuggingFace token (for gated models like Llama/FLUX) ─────────
    console.print()
    if cfg.hf_token:
        console.print("[green]✓[/green] Using saved HuggingFace token")
        hf_token = cfg.hf_token
        _apply_hf_token(hf_token)
    else:
        console.print(
            Panel(
                "Some models (Llama, FLUX) are gated on HuggingFace and require an\n"
                "access token with the model's license accepted.\n\n"
                "Get one from: [bold cyan]https://huggingface.co/settings/tokens[/bold cyan]\n"
                "Leave blank to skip (only ungated models will work).",
                title="[bold yellow]HuggingFace Token (optional)",
                border_style="yellow",
            )
        )
        hf_token = Prompt.ask(
            "[cyan]Paste your HuggingFace token[/cyan]",
            password=True,
            default="",
            show_default=False,
        )
        if hf_token.strip():
            _apply_hf_token(hf_token.strip())

    # ── Step 4: model selection ───────────────────────────────────────────────
    console.print()
    recommendations = _recommend_models(profile.gpu_model, profile.vram)
    custom_idx = len(recommendations) + 1
    console.print("[bold]Recommended models for your hardware:[/bold]")
    for i, (model_id, label) in enumerate(recommendations, 1):
        console.print(f"  [cyan]{i}[/cyan]. {label}")
        console.print(f"     [dim]{model_id}[/dim]")
    console.print(f"  [cyan]{custom_idx}[/cyan]. Enter custom model ID")
    console.print()
    console.print("[dim]Select one or more (e.g. \"1,2\") if your VRAM can fit multiple models.[/dim]")

    valid_choices = {str(i) for i in range(1, custom_idx + 1)}
    while True:
        raw_choice = Prompt.ask("Select model(s)", default="1")
        picks = [p.strip() for p in raw_choice.split(",") if p.strip()]
        if picks and all(p in valid_choices for p in picks):
            break
        console.print(f"[red]Enter a comma-separated list of numbers between 1 and {custom_idx}.[/red]")

    selected_models: list[str] = []
    for pick in picks:
        idx = int(pick) - 1
        if idx < len(recommendations):
            selected_models.append(recommendations[idx][0])
        else:
            selected_models.append(Prompt.ask("[cyan]Enter HuggingFace model ID[/cyan]"))
    # De-duplicate while preserving order.
    selected_models = list(dict.fromkeys(selected_models))

    # ── Step 5: save config & login ───────────────────────────────────────────
    console.print()
    cfg.orchestrator_url = orchestrator_url
    cfg.token = token.strip()
    if hf_token.strip():
        cfg.hf_token = hf_token.strip()
    save_config(cfg)
    console.print("[green]✓[/green] Token saved to system keychain")

    # ── Step 6: auto-start as service ─────────────────────────────────────────
    console.print()
    install_service = Confirm.ask(
        "Start worker automatically on login? (recommended)",
        default=True,
    )

    if install_service:
        sys_platform = _platform.system()
        if sys_platform == "Darwin":
            ok = _install_service_mac(selected_models, orchestrator_url)
            if ok:
                console.print("[green]✓[/green] Installed as launchd service (auto-starts on login)")
            else:
                console.print("[yellow]⚠[/yellow] Service install failed — you can start manually with [bold]lexora-worker start[/bold]")
        elif sys_platform == "Linux":
            ok = _install_service_linux(selected_models, orchestrator_url)
            if ok:
                console.print("[green]✓[/green] Installed as systemd user service (auto-starts on login)")
            else:
                console.print("[yellow]⚠[/yellow] Service install failed — you can start manually with [bold]lexora-worker start[/bold]")
        else:
            console.print("[yellow]Auto-start not supported on Windows yet — run [bold]lexora-worker start[/bold] manually.[/yellow]")

    # ── Step 7: start now ─────────────────────────────────────────────────────
    console.print()
    start_now = Confirm.ask("Start the worker now?", default=True)
    if start_now:
        console.print()
        try:
            asyncio.run(
                _worker_main(
                    cfg=cfg,
                    model_id=selected_models,
                    max_concurrency=1,
                    max_model_len=4096,
                )
            )
        except (KeyboardInterrupt, asyncio.CancelledError):
            pass
    else:
        console.print()
        start_cmd = "lexora-worker start " + " ".join(f"--model {m}" for m in selected_models)
        console.print(Panel.fit(
            f"[green]Setup complete![/green]\n\n"
            f"Start your node anytime:\n"
            f"[bold cyan]{start_cmd}[/bold cyan]",
            title="[bold]Lexora Worker — Ready",
            border_style="green",
        ))


if __name__ == "__main__":
    app()
