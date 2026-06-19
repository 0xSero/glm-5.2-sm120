#!/usr/bin/env bash
set -euo pipefail

# Load local overrides from .env next to this script (see .env.example).
cd "$(dirname "$0")"
RECIPE_DIR="$(pwd)"
[ -f .env ] && { set -a; . ./.env; set +a; }

# GLM-5.2-NVFP4-REAP-469B on 4x RTX PRO 6000 Blackwell (SM_120)  —  one command.
#
# Uses the voipmonitor Black Benediction vLLM image (public on Docker Hub) — the
# only image that supports GlmMoeDsaForCausalLM + Glm4MoeMTPModel + B12X_MLA_SPARSE
# (SM120-native sparse MLA decode) + NVFP4 (modelopt_fp4) MoE together.
#
# This script launches the server, WAITS until it is healthy, and runs a smoke
# test, so a green "READY" means it actually works.

IMAGE="${IMAGE:-voipmonitor/vllm:black-benediction-b12xpr11-vllmbb6c5b7-b12xd90d89c-fi3395b41aa8d-dg324aced12c-cu132-20260608}"
NAME="${NAME:-glm52-vllm}"
PORT="${PORT:-8000}"
MODEL="${MODEL:-/mnt/llm_models/GLM-5.2-NVFP4-REAP-469B}"
MODEL_DIR="$(cd "$(dirname "${MODEL}")" && pwd)"  # bind-mounted so any MODEL path works (not just /mnt)
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-glm-5.2}"
MODEL_ALIASES="${MODEL_ALIASES:-GLM-5.2-NVFP4-REAP-469B GLM-5.2}"  # extra ids that also resolve

# --- Reasoning: this is a reasoning model. Thinking is OFF by default so a normal
# request (even with a small max_tokens) returns a direct answer instead of an
# empty `content` while the model is still "thinking". Set ENABLE_THINKING=1 for
# full chain-of-thought (then give requests a generous max_tokens, >=2000). ---
ENABLE_THINKING="${ENABLE_THINKING:-1}"
if [[ "${ENABLE_THINKING}" == "1" ]]; then
  CHAT_TEMPLATE="/recipe/chat_template.think.jinja"  # thinking on, reasoning effort defaults to high
  REASONING_PARSER="glm45"                          # split reasoning vs content
else
  CHAT_TEMPLATE="/recipe/chat_template.nothink.jinja"  # thinking off by default
  REASONING_PARSER=""                               # no parser: the direct answer goes to content
fi

# --- Hardware: 4x RTX PRO 6000 ---
TP_SIZE="${TP_SIZE:-4}"
DCP_SIZE="${DCP_SIZE:-4}"  # shard MLA KV across the 4 GPUs (seq dim) -> 250k ctx; DCP=1 caps ~177k.
                           # For a faster <=128k endpoint: DCP_SIZE=1 + MAX_MODEL_LEN=131072.

# --- MTP speculative decoding ---
MTP="${MTP:-1}"
NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-3}"

# --- Memory ---
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-2}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-250000}"

# --- PCIe allreduce (no NVLink on RTX PRO 6000) ---
VLLM_ENABLE_PCIE_ALLREDUCE="${VLLM_ENABLE_PCIE_ALLREDUCE:-0}"

# --- Coherence: GLM-5.2 DSA per-layer indexer pattern (F=full, S=share/skip) ---
# vLLM reads index_topk_pattern (NOT the config's indexer_types). Without it, all
# 78 layers build full indexers and the 57 "S" layers corrupt long-context
# attention -> garbage. 78 chars, 21 F / 57 S.
INDEX_TOPK_PATTERN="${INDEX_TOPK_PATTERN:-FFFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSS}"
HF_OVERRIDES=$(printf '{"index_topk_pattern":"%s"}' "${INDEX_TOPK_PATTERN}")

echo "==> Launching GLM-5.2 on Black Benediction (TP=$TP_SIZE DCP=$DCP_SIZE MTP=$MTP thinking=$([[ $ENABLE_THINKING == 1 ]] && echo on || echo off))"
echo "    Image: $IMAGE"
echo "    Model: $MODEL"
echo "    Port:  $PORT"

docker rm -f "${NAME}" >/dev/null 2>&1 || true

# Build speculative config if MTP enabled (passed as env var to avoid quoting hell)
if [[ "${MTP}" == "1" ]]; then
  MAX_CUDAGRAPH_CAPTURE_SIZE=$((MAX_NUM_SEQS * (NUM_SPECULATIVE_TOKENS + 1)))
  SPEC_CONFIG=$(printf '{"model":"%s","method":"mtp","num_speculative_tokens":%s,"moe_backend":"b12x","draft_sample_method":"probabilistic"}' \
    "${MODEL}" "${NUM_SPECULATIVE_TOKENS}")
else
  MAX_CUDAGRAPH_CAPTURE_SIZE="${MAX_NUM_SEQS}"
  SPEC_CONFIG=""
fi

docker run -d \
  --restart unless-stopped \
  --log-opt max-size=100m \
  --log-opt max-file=5 \
  --gpus all \
  --ipc=host \
  --shm-size=32g \
  --network host \
  --privileged \
  --name "${NAME}" \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3 \
  -e OMP_NUM_THREADS=16 \
  -e CUTE_DSL_ARCH=sm_120a \
  -e CUDA_DEVICE_MAX_CONNECTIONS=32 \
  -e NCCL_P2P_LEVEL=SYS \
  -e NCCL_P2P_DISABLE=1 \
  -e NCCL_SHM_DISABLE=0 \
  -e NCCL_PROTO=LL,LL128,Simple \
  -e NCCL_IB_DISABLE=1 \
  -e USE_NCCL_XML=0 \
  -e NCCL_GRAPH_FILE=/dev/null \
  -e NCCL_GRAPH_DUMP_FILE= \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e SAFETENSORS_FAST_GPU=1 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e PORT="${PORT}" \
  -e MODEL="${MODEL}" \
  -e SERVED_MODEL_NAME="${SERVED_MODEL_NAME}" \
  -e MODEL_ALIASES="${MODEL_ALIASES}" \
  -e TP_SIZE="${TP_SIZE}" \
  -e DCP_SIZE="${DCP_SIZE}" \
  -e MTP="${MTP}" \
  -e GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION}" \
  -e MAX_NUM_SEQS="${MAX_NUM_SEQS}" \
  -e MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS}" \
  -e MAX_MODEL_LEN="${MAX_MODEL_LEN}" \
  -e NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS}" \
  -e MAX_CUDAGRAPH_CAPTURE_SIZE="${MAX_CUDAGRAPH_CAPTURE_SIZE}" \
  -e SPEC_CONFIG="${SPEC_CONFIG:-}" \
  -e CHAT_TEMPLATE="${CHAT_TEMPLATE}" \
  -e REASONING_PARSER="${REASONING_PARSER}" \
  -e VLLM_USE_B12X_SPARSE_INDEXER=1 \
  -e VLLM_USE_B12X_MOE=1 \
  -e VLLM_USE_V2_MODEL_RUNNER=1 \
  -e VLLM_USE_FLASHINFER_SAMPLER=1 \
  -e VLLM_USE_B12X_FP8_GEMM=1 \
  -e VLLM_ENABLE_PCIE_ALLREDUCE="${VLLM_ENABLE_PCIE_ALLREDUCE}" \
  -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0 \
  -e VLLM_DISABLED_KERNELS=MarlinFP8ScaledMMLinearKernel \
  -e USES_B12X=True \
  -e B12X_DENSE_SPLITK_TURBO=1 \
  -e B12X_W4A16_TC_DECODE=1 \
  -e B12X_MOE_FORCE_A16=1 \
  -e HF_OVERRIDES="${HF_OVERRIDES}" \
  -e XDG_CACHE_HOME=/cache/jit \
  -e CUDA_CACHE_PATH=/cache/jit \
  -e VLLM_CACHE_DIR=/cache/jit/vllm \
  -e TVM_FFI_CACHE_DIR=/cache/jit/tvm-ffi \
  -e FLASHINFER_WORKSPACE_BASE=/cache/jit/flashinfer \
  -e VLLM_CACHE_ROOT=/root/.cache/vllm \
  -e TRITON_CACHE_DIR=/root/.cache/triton \
  -e TORCHINDUCTOR_CACHE_DIR=/root/.cache/torchinductor \
  -e TORCH_EXTENSIONS_DIR=/cache/jit/torch_extensions \
  -e CUTE_DSL_CACHE_DIR=/root/.cache/cutlass_dsl \
  -v "${MODEL_DIR}:${MODEL_DIR}:ro" \
  -v "${RECIPE_DIR}:/recipe:ro" \
  -v "jit-glm52:/cache/jit" \
  -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
  --entrypoint /bin/bash \
  "${IMAGE}" \
  -lc "set -euo pipefail; unset NCCL_GRAPH_FILE NCCL_GRAPH_DUMP_FILE VLLM_B12X_MLA_EXTEND_MAX_CHUNKS; \
       cd /; \
       SPEC_ARGS=(); \
       if [ -n \"\${SPEC_CONFIG:-}\" ]; then SPEC_ARGS=(--speculative-config \"\${SPEC_CONFIG}\"); fi; \
       CT_ARGS=(); \
       if [ -n \"\${CHAT_TEMPLATE:-}\" ]; then CT_ARGS=(--chat-template \"\${CHAT_TEMPLATE}\"); fi; \
       RP_ARGS=(); \
       if [ -n \"\${REASONING_PARSER:-}\" ]; then RP_ARGS=(--reasoning-parser \"\${REASONING_PARSER}\"); fi; \
       exec /opt/venv/bin/vllm serve '${MODEL}' \
         --served-model-name '${SERVED_MODEL_NAME}' ${MODEL_ALIASES} \
         --trust-remote-code \
         --host 0.0.0.0 \
         --port '${PORT}' \
         --tensor-parallel-size '${TP_SIZE}' \
         --pipeline-parallel-size 1 \
         --decode-context-parallel-size '${DCP_SIZE}' \
         --enable-chunked-prefill \
         --enable-prefix-caching \
         --load-format fastsafetensors \
         --async-scheduling \
         -cc.pass_config.fuse_allreduce_rms=True \
         --gpu-memory-utilization '${GPU_MEMORY_UTILIZATION}' \
         --max-num-batched-tokens '${MAX_NUM_BATCHED_TOKENS}' \
         --max-num-seqs '${MAX_NUM_SEQS}' \
         --max-cudagraph-capture-size '${MAX_CUDAGRAPH_CAPTURE_SIZE}' \
         --max-model-len '${MAX_MODEL_LEN}' \
         --quantization modelopt_fp4 \
         --attention-backend B12X_MLA_SPARSE \
         --moe-backend b12x \
         --kv-cache-dtype fp8 \
         --tool-call-parser glm47 \
         --enable-auto-tool-choice \
         \"\${RP_ARGS[@]}\" \
         --hf-overrides '${HF_OVERRIDES}' \
         \"\${CT_ARGS[@]}\" \
         \"\${SPEC_ARGS[@]}\""

# ---- Wait until healthy, then smoke-test, so READY means it actually works ----
BASE="http://localhost:${PORT}"
echo "==> Waiting for the server to come up (first boot compiles kernels + captures CUDA graphs, ~6 min)..."
DEADLINE=$(( $(date +%s) + ${BOOT_TIMEOUT:-1200} ))
FAILS=0
until [ "$(curl -s -o /dev/null -w '%{http_code}' "${BASE}/health" 2>/dev/null)" = "200" ]; do
  # Restart-tolerant: --restart unless-stopped may briefly bounce the container
  # during boot. Only fail if it stays exited/dead/gone for ~30s (not on a blip).
  STATE="$(docker inspect -f '{{.State.Status}}' "${NAME}" 2>/dev/null || echo missing)"
  case "${STATE}" in
    running|created|restarting) FAILS=0 ;;
    *) FAILS=$(( ${FAILS:-0} + 1 ))
       if [ "${FAILS}" -ge 6 ]; then
         echo "!! Container is ${STATE} and not recovering. Last 40 log lines:"
         docker logs --tail 40 "${NAME}" 2>&1 || true
         exit 1
       fi ;;
  esac
  if [ "$(date +%s)" -ge "${DEADLINE}" ]; then
    echo "!! Timed out waiting for /health. Last 40 log lines:"; docker logs --tail 40 "${NAME}" 2>&1 || true
    exit 1
  fi
  sleep 5
done
echo "==> /health is 200. Running smoke test..."

SMOKE=$(curl -s "${BASE}/v1/chat/completions" -H 'Content-Type: application/json' -d "{
  \"model\": \"${SERVED_MODEL_NAME}\",
  \"messages\": [{\"role\":\"user\",\"content\":\"Reply with exactly: READY\"}],
  \"max_tokens\": 64, \"temperature\": 0
}")
ANSWER=$(printf '%s' "${SMOKE}" | python3 -c 'import sys,json
try:
    m=json.load(sys.stdin)["choices"][0]["message"]; print((m.get("content") or "").strip())
except Exception as e:
    print("")' 2>/dev/null)

if [ -n "${ANSWER}" ]; then
  # Forward serve logs to a file + persist peak prefill/decode (survives restarts).
  LOG_DIR="${LOG_DIR:-${RECIPE_DIR}/logs}"; mkdir -p "${LOG_DIR}"
  [ -f "${LOG_DIR}/monitor.pid" ] && kill "$(cat "${LOG_DIR}/monitor.pid")" 2>/dev/null || true
  LOG_DIR="${LOG_DIR}" nohup ./monitor.sh >/dev/null 2>&1 &
  echo $! > "${LOG_DIR}/monitor.pid"
  echo ""
  echo "  ============================================================"
  echo "  ✅ READY — GLM-5.2 is serving and answered: ${ANSWER}"
  echo "  Endpoint : ${BASE}/v1   (model: ${SERVED_MODEL_NAME})"
  echo "  Try it   : ./chat.sh \"write a haiku about GPUs\""
  echo "  Reasoning: thinking is $([[ ${ENABLE_THINKING} == 1 ]] && echo ON || echo OFF) (set ENABLE_THINKING=1 for chain-of-thought)"
  echo "  Logs     : ${LOG_DIR}/serve.log  |  Peaks: ${LOG_DIR}/peak.json"
  echo "  ============================================================"
else
  echo "!! Server is up but the smoke test returned empty content. Raw response:"
  printf '%s\n' "${SMOKE}" | head -c 800; echo
  exit 1
fi
