#!/usr/bin/env bash
# Serve base Qwen3.6 + LoRA adapter with vLLM (OpenAI-compatible API).
#
# On the pod (after: pip install -U vllm):
#   bash scripts/06_serve_vllm.sh
#   bash scripts/06_serve_vllm.sh --port 8000 --max-model-len 8192
#
# Then:
#   curl http://127.0.0.1:8000/v1/models
#   # use model name "agent" (LoRA) in /v1/chat/completions
#
# Env overrides:
#   BASE_MODEL   default: Qwen/Qwen3.6-35B-A3B
#   ADAPTER      default: outputs/qwen36-agent-lora/lora_adapter
#   LORA_NAME    default: agent
#   HF_TOKEN     optional, for HF downloads
#   PORT         default: 8000
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3.6-35B-A3B}"
# Prefer unsloth mirror if you trained against it and have it cached:
# BASE_MODEL="${BASE_MODEL:-unsloth/Qwen3.6-35B-A3B}"
ADAPTER="${ADAPTER:-$ROOT/outputs/qwen36-agent-lora/lora_adapter}"
LORA_NAME="${LORA_NAME:-agent}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_LORA_RANK="${MAX_LORA_RANK:-16}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
DTYPE="${DTYPE:-bfloat16}"

# Optional: use merged weights instead of dynamic LoRA
#   MERGE_PATH=outputs/qwen36-agent-lora/merged_16bit bash scripts/06_serve_vllm.sh
MERGE_PATH="${MERGE_PATH:-}"

extra_args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --port) PORT="$2"; shift 2 ;;
    --host) HOST="$2"; shift 2 ;;
    --max-model-len) MAX_MODEL_LEN="$2"; shift 2 ;;
    --base) BASE_MODEL="$2"; shift 2 ;;
    --adapter) ADAPTER="$2"; shift 2 ;;
    --lora-name) LORA_NAME="$2"; shift 2 ;;
    --merge) MERGE_PATH="$2"; shift 2 ;;
    --gpu-memory-utilization) GPU_MEM_UTIL="$2"; shift 2 ;;
    --dtype) DTYPE="$2"; shift 2 ;;
    --) shift; extra_args+=("$@"); break ;;
    *) extra_args+=("$1"); shift ;;
  esac
done

if [[ -n "${HF_TOKEN:-}" ]]; then
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
elif [[ -f "$ROOT/.secrets/hf_token" ]]; then
  export HF_TOKEN
  HF_TOKEN="$(cat "$ROOT/.secrets/hf_token")"
  export HF_TOKEN
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

if ! command -v vllm >/dev/null 2>&1; then
  echo "vllm not found on PATH. Activate your env and: pip install -U vllm" >&2
  exit 1
fi

echo "=== vLLM serve ==="
echo "  host/port:     ${HOST}:${PORT}"
echo "  max_model_len: ${MAX_MODEL_LEN}"
echo "  dtype:         ${DTYPE}"
echo "  gpu_mem_util:  ${GPU_MEM_UTIL}"

if [[ -n "$MERGE_PATH" ]]; then
  if [[ ! -d "$MERGE_PATH" ]]; then
    echo "MERGE_PATH not a directory: $MERGE_PATH" >&2
    exit 1
  fi
  echo "  mode:          MERGED (no dynamic LoRA)"
  echo "  model:         $MERGE_PATH"
  exec vllm serve "$MERGE_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --dtype "$DTYPE" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --trust-remote-code \
    "${extra_args[@]}"
fi

if [[ ! -d "$ADAPTER" ]]; then
  echo "Adapter not found: $ADAPTER" >&2
  echo "Expected PEFT folder with adapter_config.json + adapter_model.safetensors" >&2
  exit 1
fi
if [[ ! -f "$ADAPTER/adapter_config.json" ]]; then
  echo "Missing $ADAPTER/adapter_config.json" >&2
  exit 1
fi

echo "  mode:          BASE + LoRA"
echo "  base:          $BASE_MODEL"
echo "  adapter:       $ADAPTER"
echo "  lora name:     $LORA_NAME  (use this as OpenAI 'model')"
echo "  max_lora_rank: $MAX_LORA_RANK"
echo ""
echo "Test:"
echo "  curl -s http://127.0.0.1:${PORT}/v1/models | head"
echo "  curl -s http://127.0.0.1:${PORT}/v1/chat/completions \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"model\":\"${LORA_NAME}\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}'"
echo ""

# Attention-only PEFT LoRA from Unsloth — standard enable-lora path.
# If load fails on MoE weight layout, try MERGE_PATH=... or add:
#   --enable-mixed-moe-lora-format
exec vllm serve "$BASE_MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  --dtype "$DTYPE" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --trust-remote-code \
  --enable-lora \
  --max-loras 1 \
  --max-lora-rank "$MAX_LORA_RANK" \
  --lora-modules "${LORA_NAME}=${ADAPTER}" \
  "${extra_args[@]}"
