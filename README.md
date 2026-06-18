# GLM-5.2-NVFP4-REAP-469B — vLLM serving (4× RTX PRO 6000 Blackwell)

A turnkey Docker setup to serve **GLM-5.2-NVFP4-REAP-469B** (REAP-pruned, NVFP4,
DeepSeek-Sparse-Attention) on **4× NVIDIA RTX PRO 6000 Blackwell (SM120, 96 GB each)**
with the `voipmonitor` b12x vLLM image.

Validated config: **250k context · concurrency 2 · fp8 KV cache · MTP speculative
decode · tool-calling + reasoning parsers**.

## Hardware target

| | |
|---|---|
| GPUs | 4× RTX PRO 6000 Blackwell (SM120), 96 GB each, **no NVLink** (PCIe) |
| Model on disk | ~313 GB (NVFP4), ~78.6 GB/GPU resident |
| Interconnect | PCIe — requires `NCCL_P2P_DISABLE=1` (see below) |

## Prerequisites

- Docker + the **NVIDIA Container Toolkit**.
- Access to the b12x image (`voipmonitor/vllm:black-benediction-…`). It is the only
  image that bundles `GlmMoeDsaForCausalLM` + `Glm4MoeMTPModel` + the SM120 sparse-MLA
  kernel (`B12X_MLA_SPARSE`) + the ModelOpt NVFP4 MoE loader.
- The model weights on a local path (default `/mnt/llm_models/GLM-5.2-NVFP4-REAP-469B`).

## Quick start

```bash
cp .env.example .env
# edit .env: set MODEL to your weights path (and IMAGE if your tag differs)

# Option A — script (mounts /mnt, serves the absolute MODEL path)
./launch.sh

# Option B — compose (mounts $MODEL at /model)
docker compose up -d

# First boot compiles kernels + captures CUDA graphs (~6 min). Watch:
curl -s -o /dev/null -w '%{http_code}\n' http://localhost:5329/health   # 200 = ready
```

The OpenAI-compatible API is then at `http://localhost:5329/v1`.

## Validated configuration

| Setting | Value | Why |
|---|---|---|
| `--tensor-parallel-size` | 4 | one shard per GPU |
| `--decode-context-parallel-size` | **4** | shards MLA KV across the 4 GPUs → 250k fits |
| `--max-model-len` | 250000 | 710,593-token pool → **2.84× concurrency at 250k** |
| `--max-num-seqs` | 2 | target concurrency |
| `--kv-cache-dtype` | **fp8** | `fp8_ds_mla`; **required** on SM120 (bf16 = garbage) |
| `--quantization` | modelopt_fp4 | NVFP4 weights |
| `--attention-backend` | B12X_MLA_SPARSE | SM120-native sparse MLA decode |
| `--moe-backend` | b12x | NVFP4 MoE |
| `--speculative-config` | mtp, 3 tokens | MTP speculative decode |
| `--hf-overrides` | `index_topk_pattern` | **coherence-critical** (see below) |
| `--tool-call-parser` / `--reasoning-parser` | glm47 / glm45 | tool calls + thinking |

### Why `index_topk_pattern` (coherence-critical)

GLM-5.2 uses DeepSeek Sparse Attention. vLLM reads `index_topk_pattern`, **not** the
checkpoint's `indexer_types` array. Without the pattern, **all 78 layers build full
indexers** and the 57 "share/skip" (`S`) layers corrupt long-context attention →
garbage output. The 78-char `F`/`S` string (21 `F`, 57 `S`) is derived from the
model's `indexer_types` and injected via `--hf-overrides`. On boot you should see
**57** log lines: `Using index_topk_pattern/index_topk_freq to skip sparse MLA indexer …`.

### Why `DCP_SIZE=4`

With DCP=1 the MLA KV cache is replicated per TP rank, so a single 250k request needs
~14.5 GB but only ~10.3 GB/GPU is free → OOM (max ~177k). `decode-context-parallel-size=4`
shards the KV across the 4 GPUs along the sequence dim, yielding a 710,593-token pool.

### Why `NCCL_P2P_DISABLE=1`

These RTX PRO 6000 are PCIe (no NVLink); the b12x PCIe allreduce path hangs at NCCL
init without P2P disabled.

## Performance (measured, warm)

| Metric | Value |
|---|---|
| Decode | ~50–54 tok/s (short ctx), ~40 tok/s @ 64k–100k |
| Prefill | ~5,100 tok/s @ 64k (warm); ~45k–65k tok/s on prefix-cache hits |
| TTFT | sub-second (short ctx); ~12 s for a fresh uncached 64k prefill |
| Concurrency | 2.84× at 250k |

> First touch of a brand-new long prefix incurs a one-time compile of that size
> bucket (e.g. ~195 s for a fresh 99.5k prompt). Subsequent same-size prefills and
> prefix-cache hits are fast.

## Testing

```bash
python3 test/coherence_test.py core         # logic, math, code, philosophy, ascii, multi-turn
python3 test/coherence_test.py long         # 64k needle-in-haystack recall
python3 test/longctx_multiturn_test.py      # ~100k-token, 6-turn reasoning battery
```

All harnesses stream with **no `max_tokens`** and report TTFT, prefill tok/s, and
decode tok/s. The reasoning model emits its chain-of-thought in the `reasoning` field
and the final answer in `content`.

## fp8 / fp4 KV cache on SM120

- **fp8** KV (`fp8_ds_mla`) works and is the practical floor. The checkpoint ships no
  `k/v_scale`, so fp8 runs at scale 1.0 (a one-line startup warning).
- **fp4** KV is **hardware-blocked** on SM120: the DSA fp4 indexer cache asserts SM100
  (datacenter Blackwell, B200/GB200). Not available on the RTX PRO 6000.

## Troubleshooting

| Symptom | Fix |
|---|---|
| OOM at 250k, "estimated maximum model length ~177728" | set `DCP_SIZE=4` |
| Garbage / incoherent long-context output | ensure `INDEX_TOPK_PATTERN` is set (57 skip lines on boot) |
| Hang at NCCL init | keep `NCCL_P2P_DISABLE=1` |
| Garbage at all lengths | `--kv-cache-dtype fp8` is mandatory on SM120 |
