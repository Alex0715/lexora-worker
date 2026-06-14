# Lexora Worker

The GPU worker node CLI for the [Lexora Network](https://lexora.network) — a distributed inference network that routes AI workloads to consumer GPUs.

Run this on your machine to serve inference jobs and earn rewards.

---

## Requirements

| Hardware | Minimum |
|---|---|
| NVIDIA GPU | 8 GB VRAM (RTX 3060 or better) |
| Apple Silicon | 8 GB unified memory (M1 or newer) |
| CPU-only | Supported, not recommended |

**Software:** Python 3.9+, CUDA 11.8+ (NVIDIA)

---

## Install

**macOS / Linux**
```bash
bash -c "$(curl -fsSL https://raw.githubusercontent.com/Alex0715/lexora-worker/main/install.sh)"
```

**Windows (PowerShell)**
```powershell
powershell -ExecutionPolicy Bypass -Command "& { iwr -useb https://raw.githubusercontent.com/Alex0715/lexora-worker/main/install.ps1 | iex }"
```

The installer detects your GPU, installs the right backend (MLX / vLLM / Transformers), and launches the setup wizard automatically.

---

## Manual Install

```bash
git clone https://github.com/Alex0715/lexora-worker
cd lexora-worker

python3 -m venv ~/.lexora-worker/venv
source ~/.lexora-worker/venv/bin/activate

# macOS / Apple Silicon
pip install -e ".[mac]"

# Linux / Windows — NVIDIA GPU
pip install -e ".[inference,image]"

# CPU-only
pip install -e .
```

---

## Setup

```bash
lexora-worker setup
```

The wizard will:
1. Detect your GPU and VRAM
2. Ask for your orchestrator URL (`https://api.lexora.network`)
3. Ask for your worker token (get one at [lexora.network/provider/nodes](https://lexora.network/provider/nodes))
4. Recommend the best model for your hardware
5. Install as a background service (launchd on macOS, systemd on Linux)

---

## CLI Reference

```bash
lexora-worker setup                          # First-time setup wizard
lexora-worker start                          # Start with auto-selected model
lexora-worker start --model <hf-repo-id>    # Start with a specific model
lexora-worker start --max-concurrency 2     # Allow 2 parallel jobs
lexora-worker start --verbose               # Debug logging
lexora-worker info                          # Show detected hardware
lexora-worker login --token <jwt>           # Save worker token
lexora-worker logout                        # Remove stored token
```

---

## Supported Models

| Model | VRAM Required | Backend |
|---|---|---|
| Llama 3.2 3B Instruct | 4 GB | MLX / vLLM / Transformers |
| Llama 3.1 8B Instruct | 6 GB+ | MLX / vLLM (GGUF Q4) |
| FLUX.1 schnell | 7 GB | diffusers (GGUF Q4 / bf16) |

The worker auto-selects the right variant for your VRAM — no manual configuration needed.

---

## Environment Variables

| Variable | Description |
|---|---|
| `LEXORA_ORCHESTRATOR_URL` | Override the orchestrator URL |
| `LEXORA_MODEL_CACHE_DIR` | Override the model cache directory |
| `LEXORA_NF4_CACHE_DIR` | Override the NF4 quantised model cache |

---

## Provider Onboarding

Provider sign-ups are currently **invite-only**. Email your GPU specs to:

**chirantan@lexoratechnologies.com**

Include your GPU model, VRAM, and approximate location.

---

## License

MIT — see [LICENSE](LICENSE)
