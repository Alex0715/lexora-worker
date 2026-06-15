from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import socketio
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_never,
    wait_exponential,
    before_sleep_log,
    RetryCallState,
)

from lexora_worker.config.settings import get_hardware_fingerprint
from lexora_worker.hardware.profiler import build_hardware_profile, get_live_vram_free, get_live_gpu_utilization
from lexora_worker.job.manager import JobManager
from lexora_worker.models import (
    ImageDispatchPayload,
    JobCancelPayload,
    JobDispatchPayload,
    LoadModelPayload,
    UpdateConfigPayload,
    WorkerHeartbeatPayload,
    WorkerModelLoadErrorPayload,
    WorkerModelReadyPayload,
    WorkerRegisterPayload,
)

logger = logging.getLogger(__name__)

HEARTBEAT_INTERVAL = 10.0
RECONNECT_MIN_WAIT = 1
RECONNECT_MAX_WAIT = 60


class WorkerSocketClient:
    """
    Manages the Socket.IO connection lifecycle:
    - Connects to /workers namespace with JWT auth.
    - Handles all server-emitted events.
    - Runs the heartbeat loop.
    - Exposes a clean shutdown path.
    """

    def __init__(
        self,
        orchestrator_url: str,
        token: str,
        node_id: str,
        job_manager: JobManager,
        model_cache_dir: str | None = None,
    ) -> None:
        self._url = orchestrator_url
        self._token = token
        self._node_id = node_id
        self._job_manager = job_manager
        self._model_cache_dir = model_cache_dir
        self._connected = False
        self._shutdown = asyncio.Event()
        self._heartbeat_task: asyncio.Task[None] | None = None
        # Results that couldn't be sent because the socket was down — flushed on reconnect
        self._queued_results: list[tuple[str, dict[str, Any]]] = []

        self._sio = socketio.AsyncClient(
            reconnection=False,  # we handle reconnect ourselves
            logger=False,
            engineio_logger=False,
        )

        self._register_handlers()

    def _register_handlers(self) -> None:
        sio = self._sio

        @sio.event(namespace="/workers")
        async def connect() -> None:
            self._connected = True
            logger.info("Connected to orchestrator")
            await self._on_connect()

        @sio.event(namespace="/workers")
        async def disconnect() -> None:
            self._connected = False
            logger.warning("Disconnected from orchestrator")

        @sio.event(namespace="/workers")
        async def connect_error(data: Any) -> None:
            logger.error("Connection error: %s", data)

        @sio.on("job:dispatch", namespace="/workers")
        async def on_job_dispatch(data: dict[str, Any]) -> None:
            try:
                payload = JobDispatchPayload(**data)
                await self._job_manager.dispatch(payload)
            except Exception as exc:
                # Use logger.error (not .exception) — tracebacks here include
                # the full message payload which must not appear in logs.
                logger.error("job:dispatch handler error: %s", exc)

        @sio.on("job:cancel", namespace="/workers")
        async def on_job_cancel(data: dict[str, Any]) -> None:
            try:
                payload = JobCancelPayload(**data)
                await self._job_manager.cancel(payload.jobId)
            except Exception as exc:
                logger.error("job:cancel handler error: %s", exc)

        @sio.on("job:imageDispatch", namespace="/workers")
        async def on_image_dispatch(data: dict[str, Any]) -> None:
            try:
                payload = ImageDispatchPayload(**data)
                asyncio.create_task(
                    self._job_manager.dispatch_image(payload),
                    name=f"img-{payload.jobId}",
                )
            except Exception as exc:
                # Use logger.error (not .exception) — traceback includes the
                # full image prompt which must not appear in logs.
                logger.error("job:imageDispatch handler error: %s", exc)

        @sio.on("node:updateConfig", namespace="/workers")
        async def on_update_config(data: dict[str, Any]) -> None:
            try:
                payload = UpdateConfigPayload(**data)
                if payload.maxConcurrentJobs is not None:
                    self._job_manager.update_max_concurrency(payload.maxConcurrentJobs)
                logger.info("Config updated: %s", payload)
            except Exception as exc:
                logger.exception("node:updateConfig handler error: %s", exc)

        @sio.on("node:loadModel", namespace="/workers")
        async def on_load_model(data: dict[str, Any]) -> None:
            try:
                payload = LoadModelPayload(**data)
                asyncio.create_task(
                    self._handle_load_model(payload.model),
                    name=f"load-model-{payload.model}",
                )
            except Exception as exc:
                logger.exception("node:loadModel handler error: %s", exc)

        @sio.on("node:evicted", namespace="/workers")
        async def on_evicted(data: dict[str, Any]) -> None:
            logger.warning("Evicted by orchestrator: %s", data)
            self._shutdown.set()

    async def _handle_load_model(self, model: str) -> None:
        """Load a model on demand (cold-swap path). Emits modelReady or modelLoadError."""
        logger.info("node:loadModel received — loading model '%s'", model)
        try:
            await self._job_manager.ensure_model_loaded(model)
            payload = WorkerModelReadyPayload(nodeId=self._node_id, model=model)
            await self.emit("worker:modelReady", payload.model_dump())
            logger.info("Model '%s' ready — emitted worker:modelReady", model)
        except Exception as exc:
            logger.exception("Failed to load model '%s': %s", model, exc)
            err_payload = WorkerModelLoadErrorPayload(
                nodeId=self._node_id,
                model=model,
                error=str(exc),
            )
            await self.emit("worker:modelLoadError", err_payload.model_dump())

    async def _on_connect(self) -> None:
        profile = await build_hardware_profile(self._model_cache_dir)

        # Merge disk-cached models with the engine's actively loaded model so
        # the scheduler can route to this node immediately on connect.
        loaded = list({*profile.loaded_models, *self._job_manager.loaded_models})

        payload = WorkerRegisterPayload(
            capabilities={
                "gpuModel": profile.gpu_model,
                "vram": profile.vram,
                "ram": profile.ram,
                "cpu": profile.cpu,
                "loadedModels": loaded,
                "networkSpeed": profile.network_speed,
                "maxConcurrentJobs": self._job_manager._max_concurrency,
            },
            hardwareFingerprint=get_hardware_fingerprint(),
        )

        await self.emit("worker:register", payload.model_dump())
        logger.info(
            "Registered — GPU: %s | VRAM: %.1f GB | Models: %s",
            profile.gpu_model,
            profile.vram,
            loaded,
        )

        if self._heartbeat_task is None or self._heartbeat_task.done():
            self._heartbeat_task = asyncio.create_task(
                self._heartbeat_loop(), name="heartbeat"
            )

        # Flush any results that were generated while the socket was down
        if self._queued_results:
            pending = list(self._queued_results)
            self._queued_results.clear()
            for event, data in pending:
                logger.info(
                    "Flushing queued result: %s job_id=%s",
                    event,
                    data.get("jobId", "?"),
                )
                await self.emit(event, data)

    async def _heartbeat_loop(self) -> None:
        while not self._shutdown.is_set():
            await asyncio.sleep(HEARTBEAT_INTERVAL)

            if not self._connected:
                continue

            try:
                import psutil

                cpu_pct = psutil.cpu_percent(interval=None)
                gpu_pct = get_live_gpu_utilization()
                vram_used = 0.0

                gpus = []
                try:
                    from lexora_worker.hardware.profiler import get_gpu_info
                    gpus = get_gpu_info()
                    if gpus:
                        vram_used = gpus[0].vram_used_gb
                except Exception:
                    pass

                hb = WorkerHeartbeatPayload(
                    nodeId=self._node_id,
                    activeJobs=self._job_manager.active_count,
                    loadedModels=self._job_manager.loaded_models,
                    cpuUsage=cpu_pct,
                    gpuUsage=gpu_pct,
                    vramUsed=vram_used,
                )
                await self.emit("worker:heartbeat", hb.model_dump())

            except Exception as exc:
                logger.warning("Heartbeat failed: %s", exc)

    # Events whose results must not be lost if the socket is temporarily down
    _CRITICAL_EVENTS = frozenset({"worker:imageCompleted", "worker:imageError"})

    async def emit(self, event: str, data: dict[str, Any]) -> None:
        critical = event in self._CRITICAL_EVENTS

        if self._connected:
            try:
                if critical:
                    # During a long image generation the asyncio loop is GIL-
                    # starved, so engine.io pings stall and the orchestrator may
                    # have already half-closed this socket. A plain emit() would
                    # be written into that dead socket and silently lost. Use a
                    # request/ack round-trip (call) so we only treat the result
                    # as delivered once the server actually acknowledges it.
                    await self._sio.call(
                        event, data, namespace="/workers", timeout=30
                    )
                else:
                    await self._sio.emit(event, data, namespace="/workers")
                return
            except Exception as exc:
                if not critical:
                    logger.warning("Emit %s failed: %s", event, exc)
                    return
                logger.warning(
                    "Delivery of %s not acked (%s) — queuing for reconnect",
                    event,
                    exc,
                )
                # fall through to queue below

        if critical:
            # Socket is down (or the ack never came) during a long-running job
            # such as image generation. Queue the result and flush it the
            # moment we reconnect. Re-delivery is idempotent server-side: if the
            # original actually landed, the pending job is already resolved and
            # the replay is a no-op.
            self._queued_results.append((event, data))
            logger.warning(
                "Socket down — queued %s for job %s (will send on reconnect)",
                event,
                data.get("jobId", "?"),
            )

    async def connect_with_retry(self) -> None:
        """
        Connects and stays connected, using exponential backoff on failures.
        Exits only when self._shutdown is set.
        """
        attempt = 0
        while not self._shutdown.is_set():
            wait_time = min(RECONNECT_MIN_WAIT * (2**attempt), RECONNECT_MAX_WAIT)
            try:
                await self._sio.connect(
                    self._url,
                    namespaces=["/workers"],
                    auth={"token": self._token},
                    transports=["websocket"],
                    wait_timeout=15,
                )
                attempt = 0  # reset on success

                # Block until disconnected or shutdown
                while self._connected and not self._shutdown.is_set():
                    await asyncio.sleep(1)

            except socketio.exceptions.ConnectionError as exc:
                attempt += 1
                logger.warning(
                    "Connection failed (attempt %d): %s — retrying in %.1fs",
                    attempt,
                    exc,
                    wait_time,
                )
            except Exception as exc:
                attempt += 1
                logger.error(
                    "Unexpected connection error (attempt %d): %s — retrying in %.1fs",
                    attempt,
                    exc,
                    wait_time,
                )

            if not self._shutdown.is_set():
                await asyncio.sleep(wait_time)

    async def shutdown(self) -> None:
        self._shutdown.set()

        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

        await self._job_manager.cancel_all()

        if self._connected:
            try:
                await self._sio.disconnect()
            except Exception:
                pass

        logger.info("Worker shutdown complete")
