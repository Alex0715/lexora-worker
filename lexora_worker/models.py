from __future__ import annotations

import os
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


# ─── Hardware ────────────────────────────────────────────────────────────────


class GpuInfo(BaseModel):
    index: int
    name: str
    vram_total_gb: float
    vram_free_gb: float
    vram_used_gb: float
    gpu_utilization_pct: float
    temperature_c: float


class HardwareProfile(BaseModel):
    gpu_model: str
    vram: float = Field(description="Total VRAM in GB of primary GPU")
    vram_free: float = Field(description="Free VRAM in GB")
    ram: float = Field(description="Total system RAM in GB")
    cpu: str
    cpu_cores: int
    cpu_threads: int
    loaded_models: list[str]
    network_speed: float = Field(description="Estimated bandwidth in Mbps")
    gpus: list[GpuInfo]


# ─── Job states ───────────────────────────────────────────────────────────────


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    STREAMING = "streaming"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# ─── Server → Worker events ───────────────────────────────────────────────────


class ChatMessage(BaseModel):
    role: str
    content: str


class JobDispatchPayload(BaseModel):
    jobId: str
    model: str
    messages: list[ChatMessage]
    maxTokens: int = Field(default=512, ge=1, le=32768)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    stream: bool = True


class JobCancelPayload(BaseModel):
    jobId: str
    reason: str = ""


class UpdateConfigPayload(BaseModel):
    maxConcurrentJobs: int | None = None
    loadedModels: list[str] | None = None


class LoadModelPayload(BaseModel):
    model: str


class ImageDispatchPayload(BaseModel):
    jobId: str
    model: str
    prompt: str
    width: int = 1024
    height: int = 1024
    numSteps: int = 4
    guidanceScale: float = 0.0


class WorkerImageCompletedPayload(BaseModel):
    jobId: str
    imageBase64: str
    latencyMs: float


class WorkerImageErrorPayload(BaseModel):
    jobId: str
    error: str


# ─── Worker → Server events ───────────────────────────────────────────────────


class WorkerRegisterPayload(BaseModel):
    capabilities: dict[str, Any]
    hardwareFingerprint: str


class WorkerHeartbeatPayload(BaseModel):
    nodeId: str
    activeJobs: int
    loadedModels: list[str]
    cpuUsage: float
    gpuUsage: float
    vramUsed: float


class WorkerJobAcceptedPayload(BaseModel):
    jobId: str
    nodeId: str


class WorkerJobRejectedPayload(BaseModel):
    jobId: str
    nodeId: str
    reason: str


class WorkerTokenPayload(BaseModel):
    jobId: str
    token: str
    index: int
    finishReason: str | None = None


class WorkerCompletedPayload(BaseModel):
    jobId: str
    totalTokens: int
    promptTokens: int
    completionTokens: int
    latencyMs: float
    tokensPerSecond: float


class WorkerErrorPayload(BaseModel):
    jobId: str
    error: str
    code: str = "INFERENCE_ERROR"


class WorkerModelReadyPayload(BaseModel):
    nodeId: str
    model: str


class WorkerModelLoadErrorPayload(BaseModel):
    nodeId: str
    model: str
    error: str


# ─── Active job record ────────────────────────────────────────────────────────


class ActiveJob(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    job_id: str
    model: str
    messages: list[ChatMessage]
    max_tokens: int
    temperature: float
    status: JobStatus = JobStatus.PENDING
    start_time: float = 0.0
    tokens_emitted: int = 0
    prompt_tokens: int = 0


# ─── Worker config (persisted) ────────────────────────────────────────────────


class WorkerConfig(BaseModel):
    token: str = ""
    hf_token: str = ""
    node_id: str = ""
    orchestrator_url: str = Field(
        default_factory=lambda: os.environ.get("LEXORA_ORCHESTRATOR_URL", "https://api.lexora.network")
    )
    model_cache_dir: str = Field(
        default_factory=lambda: os.environ.get("LEXORA_MODEL_CACHE_DIR", "~/.cache/huggingface/hub")
    )

    @field_validator("orchestrator_url")
    @classmethod
    def strip_trailing_slash(cls, v: str) -> str:
        return v.rstrip("/")
