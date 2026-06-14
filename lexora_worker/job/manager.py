from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable, Coroutine
from typing import Any

import base64
from lexora_worker.inference.model_manager import ModelManager
from lexora_worker.models import (
    ActiveJob,
    ImageDispatchPayload,
    JobDispatchPayload,
    JobStatus,
    WorkerCompletedPayload,
    WorkerErrorPayload,
    WorkerImageCompletedPayload,
    WorkerImageErrorPayload,
    WorkerJobAcceptedPayload,
    WorkerJobRejectedPayload,
    WorkerTokenPayload,
)

logger = logging.getLogger(__name__)

EmitCb = Callable[[str, dict[str, Any]], Coroutine[Any, Any, None]]


class JobManager:
    """
    Manages the lifecycle of inference jobs — both text and image.

    Both job types share the same _active registry, _max_concurrency gate,
    and asyncio.Task tracking so that:
      - worker:heartbeat accurately reports all active load.
      - Capacity checks are unified regardless of modality.
    """

    def __init__(
        self,
        model_manager: ModelManager,
        node_id: str,
        max_concurrency: int,
        emit: EmitCb,
    ) -> None:
        self._model_manager = model_manager
        self._node_id = node_id
        self._max_concurrency = max_concurrency
        self._emit = emit
        self._active: dict[str, ActiveJob] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._lock = asyncio.Lock()

    @property
    def active_count(self) -> int:
        return len(self._active)

    @property
    def loaded_models(self) -> list[str]:
        return self._model_manager.hot_models

    def update_max_concurrency(self, value: int) -> None:
        self._max_concurrency = value
        logger.info("max_concurrency updated to %d", value)

    async def ensure_model_loaded(self, model_id: str) -> None:
        await self._model_manager.ensure_model_loaded(model_id)

    # ── Text dispatch ───────────────────────────────────────────────────────

    async def dispatch(self, payload: JobDispatchPayload) -> None:
        """Entry point called by the WebSocket layer on job:dispatch."""
        async with self._lock:
            if len(self._active) >= self._max_concurrency:
                await self._emit(
                    "worker:jobRejected",
                    WorkerJobRejectedPayload(
                        jobId=payload.jobId,
                        nodeId=self._node_id,
                        reason="capacity_full",
                    ).model_dump(),
                )
                logger.warning("Rejected text job %s — capacity full", payload.jobId)
                return

            if payload.model not in self._model_manager.hot_models:
                try:
                    await self._model_manager.ensure_model_loaded(payload.model)
                except Exception as exc:
                    await self._emit(
                        "worker:jobRejected",
                        WorkerJobRejectedPayload(
                            jobId=payload.jobId,
                            nodeId=self._node_id,
                            reason=f"model_load_failed:{exc}",
                        ).model_dump(),
                    )
                    logger.error(
                        "Rejected text job %s — could not load model %s: %s",
                        payload.jobId, payload.model, exc,
                    )
                    return

            job = ActiveJob(
                job_id=payload.jobId,
                model=payload.model,
                messages=payload.messages,
                max_tokens=payload.maxTokens,
                temperature=payload.temperature,
                status=JobStatus.PENDING,
                start_time=time.monotonic(),
            )
            self._active[payload.jobId] = job

        await self._emit(
            "worker:jobAccepted",
            WorkerJobAcceptedPayload(
                jobId=payload.jobId, nodeId=self._node_id
            ).model_dump(),
        )
        logger.info(
            "▶ Text job accepted  job_id=%s  model=%s  max_tokens=%d  active=%d/%d",
            payload.jobId, payload.model, payload.maxTokens,
            len(self._active), self._max_concurrency,
        )

        task = asyncio.create_task(
            self._run_job(job), name=f"job-{payload.jobId}"
        )
        self._tasks[payload.jobId] = task
        task.add_done_callback(lambda t: self._on_task_done(payload.jobId, t))

    # ── Image dispatch ──────────────────────────────────────────────────────

    async def dispatch_image(self, payload: ImageDispatchPayload) -> None:
        """Entry point called by the WebSocket layer on job:imageDispatch.

        Uses the same _active registry and _max_concurrency gate as text jobs
        so that worker:heartbeat accurately reflects total load and the
        orchestrator's load-balancing decisions are correct.
        """
        async with self._lock:
            if len(self._active) >= self._max_concurrency:
                await self._emit(
                    "worker:jobRejected",
                    WorkerJobRejectedPayload(
                        jobId=payload.jobId,
                        nodeId=self._node_id,
                        reason="capacity_full",
                    ).model_dump(),
                )
                logger.warning("Rejected image job %s — capacity full", payload.jobId)
                return

            if payload.model not in self._model_manager.hot_models:
                try:
                    await self._model_manager.ensure_model_loaded(payload.model)
                except Exception as exc:
                    await self._emit(
                        "worker:jobRejected",
                        WorkerJobRejectedPayload(
                            jobId=payload.jobId,
                            nodeId=self._node_id,
                            reason=f"model_load_failed:{exc}",
                        ).model_dump(),
                    )
                    logger.error(
                        "Rejected image job %s — could not load model %s: %s",
                        payload.jobId, payload.model, exc,
                    )
                    return

            job = ActiveJob(
                job_id=payload.jobId,
                model=payload.model,
                messages=[],   # not used for image jobs
                max_tokens=0,
                temperature=0.0,
                status=JobStatus.PENDING,
                start_time=time.monotonic(),
            )
            self._active[payload.jobId] = job

        await self._emit(
            "worker:jobAccepted",
            WorkerJobAcceptedPayload(
                jobId=payload.jobId, nodeId=self._node_id
            ).model_dump(),
        )
        logger.info(
            "▶ Image job accepted  job_id=%s  model=%s  size=%dx%d  active=%d/%d",
            payload.jobId, payload.model, payload.width, payload.height,
            len(self._active), self._max_concurrency,
        )

        task = asyncio.create_task(
            self._run_image_job(payload, job), name=f"img-{payload.jobId}"
        )
        self._tasks[payload.jobId] = task
        task.add_done_callback(lambda t: self._on_task_done(payload.jobId, t))

    # ── Cancel ──────────────────────────────────────────────────────────────

    async def cancel(self, job_id: str) -> None:
        job = self._active.get(job_id)
        task = self._tasks.get(job_id)
        if task and not task.done():
            task.cancel()
        if job:
            await self._model_manager.abort_text(job.model, job_id)
        logger.info("Cancelled job %s", job_id)
        self._cleanup_job(job_id)

    async def cancel_all(self) -> None:
        for job_id in list(self._tasks.keys()):
            await self.cancel(job_id)

    # ── Internal runners ────────────────────────────────────────────────────

    async def _run_job(self, job: ActiveJob) -> None:
        job.status = JobStatus.RUNNING
        start = time.monotonic()

        try:
            messages = [m.model_dump() for m in job.messages]
            index = 0
            last_chunk = None

            async for chunk in self._model_manager.generate_stream(
                model_id=job.model,
                job_id=job.job_id,
                messages=messages,
                max_tokens=job.max_tokens,
                temperature=job.temperature,
            ):
                last_chunk = chunk

                if chunk.text:
                    job.status = JobStatus.STREAMING
                    job.tokens_emitted += 1
                    await self._emit(
                        "worker:token",
                        WorkerTokenPayload(
                            jobId=job.job_id,
                            token=chunk.text,
                            index=index,
                            finishReason=None,
                        ).model_dump(),
                    )
                    index += 1

                if chunk.finish_reason in ("stop", "length"):
                    await self._emit(
                        "worker:token",
                        WorkerTokenPayload(
                            jobId=job.job_id,
                            token="",
                            index=index,
                            finishReason=chunk.finish_reason,
                        ).model_dump(),
                    )
                    break

            elapsed_ms = (time.monotonic() - start) * 1000
            prompt_tokens = last_chunk.prompt_tokens if last_chunk else 0
            completion_tokens = job.tokens_emitted
            total_tokens = prompt_tokens + completion_tokens
            tps = completion_tokens / max((time.monotonic() - start), 1e-6)

            job.status = JobStatus.COMPLETED
            logger.info(
                "✓ Text job finished  job_id=%s  total_tokens=%d  prompt=%d"
                "  completion=%d  tps=%.1f  latency=%.0fms",
                job.job_id, total_tokens, prompt_tokens,
                completion_tokens, round(tps, 2), elapsed_ms,
            )
            await self._emit(
                "worker:completed",
                WorkerCompletedPayload(
                    jobId=job.job_id,
                    totalTokens=total_tokens,
                    promptTokens=prompt_tokens,
                    completionTokens=completion_tokens,
                    latencyMs=elapsed_ms,
                    tokensPerSecond=round(tps, 2),
                ).model_dump(),
            )

        except asyncio.CancelledError:
            job.status = JobStatus.CANCELLED
            logger.info("Text job %s cancelled during execution", job.job_id)
            raise

        except _torch_oom_exception() as exc:
            job.status = JobStatus.FAILED
            logger.error("OOM on text job %s: %s", job.job_id, exc)
            await self._emit(
                "worker:error",
                WorkerErrorPayload(
                    jobId=job.job_id,
                    error="GPU out of memory",
                    code="OOM",
                ).model_dump(),
            )

        except Exception as exc:
            job.status = JobStatus.FAILED
            logger.error("Text job %s failed: %s", job.job_id, exc)
            await self._emit(
                "worker:error",
                WorkerErrorPayload(
                    jobId=job.job_id,
                    error=str(exc),
                    code="INFERENCE_ERROR",
                ).model_dump(),
            )

        finally:
            self._cleanup_job(job.job_id)

    async def _run_image_job(
        self, payload: ImageDispatchPayload, job: ActiveJob
    ) -> None:
        job.status = JobStatus.RUNNING
        start = time.monotonic()

        try:
            output_path = await self._model_manager.generate_image(
                model_id=payload.model,
                prompt=payload.prompt,
                width=payload.width,
                height=payload.height,
                num_steps=payload.numSteps,
                guidance=payload.guidanceScale,
            )
            with open(output_path, "rb") as f:
                image_b64 = base64.b64encode(f.read()).decode("utf-8")

            elapsed_ms = (time.monotonic() - start) * 1000
            job.status = JobStatus.COMPLETED
            logger.info(
                "✓ Image job finished  job_id=%s  size=%dx%d  latency=%.0fms",
                payload.jobId, payload.width, payload.height, elapsed_ms,
            )
            await self._emit(
                "worker:imageCompleted",
                WorkerImageCompletedPayload(
                    jobId=payload.jobId,
                    imageBase64=image_b64,
                    latencyMs=round(elapsed_ms, 1),
                ).model_dump(),
            )

        except asyncio.CancelledError:
            job.status = JobStatus.CANCELLED
            logger.info("Image job %s cancelled during execution", payload.jobId)
            raise

        except _torch_oom_exception() as exc:
            job.status = JobStatus.FAILED
            logger.error("OOM on image job %s: %s", payload.jobId, exc)
            await self._emit(
                "worker:imageError",
                WorkerImageErrorPayload(
                    jobId=payload.jobId,
                    error="GPU out of memory",
                ).model_dump(),
            )

        except Exception as exc:
            job.status = JobStatus.FAILED
            logger.error("Image job %s failed: %s", payload.jobId, exc)
            await self._emit(
                "worker:imageError",
                WorkerImageErrorPayload(
                    jobId=payload.jobId,
                    error=str(exc),
                ).model_dump(),
            )

        finally:
            self._cleanup_job(payload.jobId)

    # ── Shared helpers ──────────────────────────────────────────────────────

    def _cleanup_job(self, job_id: str) -> None:
        self._active.pop(job_id, None)
        self._tasks.pop(job_id, None)

    def _on_task_done(self, job_id: str, task: asyncio.Task[None]) -> None:
        if task.cancelled():
            return
        exc = task.exception()
        if exc and not isinstance(exc, asyncio.CancelledError):
            logger.error("Unhandled task exception for job %s: %s", job_id, exc)


def _torch_oom_exception() -> type[Exception]:
    try:
        import torch
        return torch.cuda.OutOfMemoryError  # type: ignore[return-value]
    except (ImportError, AttributeError):
        return MemoryError
