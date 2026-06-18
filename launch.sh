#!/usr/bin/env bash
set -euo pipefail

# Load local overrides from .env next to this script (see .env.example).
cd "$(dirname "$0")"
[ -f .env ] && { set -a; . ./.env; set +a; }

# GLM-5.2-NVFP4-REAP-469B on 4x RTX PRO 6000 Blackwell (SM_120)
#
# Uses voipmonitor Black Benediction vLLM image — the only image that supports
# GlmMoeDsaForCausalLM + Glm4MoeMTPModel + B12X_MLA_SPARSE (SM120-native sparse
# MLA decode) together. Same pattern as the working GLM-5.1 v10 recipe.
#
# GLM-5.2 uses DSA (DeepSeek Sparse Attention). vLLM reads index_topk_pattern,
# NOT config.json's indexer_types, so we inject the pattern via --hf-overrides
# below (see INDEX_TOPK_PATTERN). Without it, all 78 layers build full indexers
# and the 57 "S" layers corrupt long-context attention -> garbage.

IMAGE="${IMAGE:-voipmonitor/vllm:black-benediction-b12xpr11-vllmbb6c5b7-b12xd90d89c-fi3395b41aa8d-dg324aced12c-cu132-20260608}"
NAME="${NAME:-glm52-vllm}"
PORT="${PORT:-5329}"
MODEL="${MODEL:-/mnt/llm_models/GLM-5.2-NVFP4-REAP-469B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-GLM-5.2-NVFP4-REAP-469B}"

# --- Hardware: 4x RTX PRO 6000 ---
TP_SIZE="${TP_SIZE:-4}"
DCP_SIZE="${DCP_SIZE:-4}"  # shard MLA KV across the 4 GPUs (seq dim) -> ~4x ctx; REQUIRED for 250k.
                           # Runs as a TP*DCP subgroup on this 4-GPU box (no extra GPUs needed);
                           # per-step DCP collective rides NCCL SHM/SYS since P2P is disabled.

# --- MTP speculative decoding ---
MTP="${MTP:-1}"
NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-3}"

# --- Memory ---
# 4x RTX PRO 6000 = 384GB total. Model takes ~78GB/worker (~313GB total).
# Leaves ~10GB/worker for KV at 0.95 util. With DCP=1 the MLA KV is replicated
# per TP rank, so 250k needs 14.5GB > 10.3GB free -> OOM (vLLM est max ~177k).
# DCP=4 shards the KV across the 4 GPUs along the sequence dim -> 710,593-token
# pool = 2.84x concurrency at 250k (measured). MTP on (3 draft tokens).
# Confirm via the startup "Maximum concurrency for N tokens per request" log.
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-2}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-250000}"

# --- PCIe allreduce (no NVLink on RTX PRO 6000) ---
# Disabled for first bring-up: b12x PCIe allreduce deadlocked in NCCL
# init_model_parallel_group on 4-GPU PCIe. Re-enable once base load works.
VLLM_ENABLE_PCIE_ALLREDUCE="${VLLM_ENABLE_PCIE_ALLREDUCE:-0}"

# --- Coherence: GLM-5.2 DSA per-layer indexer pattern (F=full, S=share/skip) ---
# vLLM reads index_topk_pattern (NOT the config's indexer_types). Without it, all
# 78 layers build full indexers and the 57 "S" layers corrupt long-context
# attention -> garbage. This 78-char string is derived from config.indexer_types
# (21 F / 57 S) and matches lukealonso's GLM-5.2-NVFP4 reference exactly.
INDEX_TOPK_PATTERN="${INDEX_TOPK_PATTERN:-FFFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSS}"
HF_OVERRIDES=$(printf '{"index_topk_pattern":"%s"}' "${INDEX_TOPK_PATTERN}")

echo "==> Launching GLM-5.2 on Black Benediction (TP=$TP_SIZE DCP=$DCP_SIZE MTP=$MTP)"
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

exec docker run -d \
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
  -v /mnt:/mnt \
  -v "jit-glm52:/cache/jit" \
  -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
  --entrypoint /bin/bash \
  "${IMAGE}" \
  -lc "set -euo pipefail; unset NCCL_GRAPH_FILE NCCL_GRAPH_DUMP_FILE VLLM_B12X_MLA_EXTEND_MAX_CHUNKS; \
       cd /; \
       SPEC_ARGS=(); \
       if [ -n \"\${SPEC_CONFIG:-}\" ]; then SPEC_ARGS=(--speculative-config \"\${SPEC_CONFIG}\"); fi; \
       exec /opt/venv/bin/python -m vllm.entrypoints.cli.main serve '${MODEL}' \
         --served-model-name '${SERVED_MODEL_NAME}' \
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
         --reasoning-parser glm45 \
         --hf-overrides '${HF_OVERRIDES}' \
         \"\${SPEC_ARGS[@]}\""
