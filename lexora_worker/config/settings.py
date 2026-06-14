from __future__ import annotations

import hashlib
import json
import os
import platform
import uuid
from pathlib import Path

import keyring
from dotenv import load_dotenv

from lexora_worker.models import WorkerConfig

_SERVICE_NAME = "lexora-worker"
_CONFIG_DIR = Path.home() / ".config" / "lexora-worker"
_CONFIG_FILE = _CONFIG_DIR / "config.json"

# Pick up `worker/.env` (editable install) as well as a `.env` in the
# current working directory, without overriding already-set env vars.
load_dotenv(Path(__file__).resolve().parents[2] / ".env")
load_dotenv()


def _config_dir() -> Path:
    _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    return _CONFIG_DIR


def _keyring_get_token() -> str | None:
    try:
        return keyring.get_password(_SERVICE_NAME, "jwt_token")
    except keyring.errors.KeyringError:
        # Headless Linux boxes often have no usable keyring backend
        # (no Secret Service / KWallet). Fall back to the config file.
        if _CONFIG_FILE.exists():
            raw = json.loads(_CONFIG_FILE.read_text())
            return raw.get("token")
        return None


def _keyring_set_token(token: str) -> None:
    try:
        keyring.set_password(_SERVICE_NAME, "jwt_token", token)
    except keyring.errors.KeyringError:
        raw = {}
        if _CONFIG_FILE.exists():
            raw = json.loads(_CONFIG_FILE.read_text())
        raw["token"] = token
        _config_dir()
        _CONFIG_FILE.write_text(json.dumps(raw, indent=2))


def _keyring_delete_token() -> None:
    try:
        keyring.delete_password(_SERVICE_NAME, "jwt_token")
    except keyring.errors.PasswordDeleteError:
        pass
    except keyring.errors.KeyringError:
        pass

    if _CONFIG_FILE.exists():
        raw = json.loads(_CONFIG_FILE.read_text())
        if raw.pop("token", None) is not None:
            _CONFIG_FILE.write_text(json.dumps(raw, indent=2))


def load_config() -> WorkerConfig:
    if _CONFIG_FILE.exists():
        raw = json.loads(_CONFIG_FILE.read_text())
        cfg = WorkerConfig(**{k: v for k, v in raw.items() if k != "token"})
    else:
        cfg = WorkerConfig()

    # Env vars (from `.env` or the shell) override the persisted config.
    if env_url := os.environ.get("LEXORA_ORCHESTRATOR_URL"):
        cfg.orchestrator_url = env_url
    if env_cache_dir := os.environ.get("LEXORA_MODEL_CACHE_DIR"):
        cfg.model_cache_dir = env_cache_dir

    token = _keyring_get_token()
    if token:
        cfg.token = token

    return cfg


def save_config(cfg: WorkerConfig) -> None:
    _config_dir()
    # Never persist the raw token to disk if a keyring backend is available
    safe = cfg.model_dump(exclude={"token"})
    _CONFIG_FILE.write_text(json.dumps(safe, indent=2))

    if cfg.token:
        _keyring_set_token(cfg.token)


def save_token(token: str) -> None:
    _keyring_set_token(token)


def clear_token() -> None:
    _keyring_delete_token()


def get_hardware_fingerprint() -> str:
    """
    Derives a stable hardware fingerprint from machine-level identifiers.
    Falls back gracefully on non-Linux platforms.
    """
    components: list[str] = [
        platform.node(),
        platform.machine(),
        platform.processor(),
        str(uuid.getnode()),  # MAC address-derived
    ]

    try:
        machine_id_path = Path("/etc/machine-id")
        if machine_id_path.exists():
            components.append(machine_id_path.read_text().strip())
    except OSError:
        pass

    raw = "|".join(components)
    return hashlib.sha256(raw.encode()).hexdigest()
