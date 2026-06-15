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

**1. Clone the repo**

```bash
git clone https://github.com/Alex0715/AI_Inference
cd AI_Inference
```

**2. Install the worker**

```bash
# macOS / Apple Silicon
pip install -e "./worker[mac]"

# Linux / Windows — NVIDIA GPU
pip install -e "./worker[inference,image]"

# CPU-only
pip install -e ./worker
```

**3. Log in with your worker token**

Get a token at [lexora.network/provider/nodes](https://lexora.network/provider/nodes), then:

```bash
lexora-worker login --token <your-token>
```

**4. Start the worker**

```bash
lexora-worker start
```

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
