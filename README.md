# GLM-5.2-NVFP4-REAP-469B — vLLM serving (4× RTX PRO 6000 Blackwell)

A turnkey Docker setup to serve **[0xSero/GLM-5.2-NVFP4-REAP-469B](https://huggingface.co/0xSero/GLM-5.2-NVFP4-REAP-469B)**
(REAP-pruned, NVFP4, DeepSeek-Sparse-Attention) on **4× NVIDIA RTX PRO 6000 Blackwell
(SM120, 96 GB each)** with the `voipmonitor` b12x vLLM image.

> **Model:** [huggingface.co/0xSero/GLM-5.2-NVFP4-REAP-469B](https://huggingface.co/0xSero/GLM-5.2-NVFP4-REAP-469B) · ~313 GB on disk (NVFP4) · REAP-pruned 469B MoE · DeepSeek Sparse Attention + MTP.

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
# 1. Download the weights (~313 GB NVFP4) — needs the hf CLI: pip install -U huggingface_hub
hf download 0xSero/GLM-5.2-NVFP4-REAP-469B --local-dir /mnt/llm_models/GLM-5.2-NVFP4-REAP-469B

# 2. Configure
cp .env.example .env
# edit .env: set MODEL to your weights path (and IMAGE if your tag differs)

# Option A — script (mounts /mnt, serves the absolute MODEL path)
./launch.sh

# Option B — compose (mounts $MODEL at /model)
docker compose up -d

# First boot compiles kernels + captures CUDA graphs (~6 min). Watch:
curl -s -o /dev/null -w '%{http_code}\n' http://localhost:8000/health   # 200 = ready
```

The OpenAI-compatible API is then at `http://localhost:8000/v1`.

## Talking to the model — it's a reasoning model (read this first)

GLM-5.2 **thinks before it answers.** With `--reasoning-parser glm45`, the
chain-of-thought goes into `message.reasoning` (`delta.reasoning` when streaming)
and the **final answer appears in `message.content` only after the thinking
finishes.**

> ⚠️ **The #1 "it returns nothing" gotcha.** A small `max_tokens` is consumed
> *entirely by the thinking phase*, so the response comes back with
> `content: null` and `finish_reason: "length"` — the model simply never reached
> its answer. It looks broken but isn't. **Give it room: set a large
> `max_tokens` (≥ 2000) or omit it**, and `content` populates as expected.

Verified against the running server:

```bash
curl -s http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "GLM-5.2-NVFP4-REAP-469B",
  "messages": [{"role": "user", "content": "Reply with exactly: PONG"}],
  "max_tokens": 2000
}' | python3 -c 'import sys,json; m=json.load(sys.stdin)["choices"][0]["message"]; print("reasoning:", (m.get("reasoning") or "")[:60], "...\ncontent  :", m.get("content"))'
# reasoning: The user wants me to reply with exactly: PONG ...
# content  : PONG
```

| If you see… | Cause | Fix |
|---|---|---|
| `content: null`, `finish_reason: "length"` | thinking consumed the whole budget | raise `max_tokens` (≥ 2000) or omit it |
| empty `content`, you wanted the thinking | it's in `message.reasoning` | read `reasoning` / stream `delta.reasoning` |
| function calling | `--tool-call-parser glm47` is enabled | parse `message.tool_calls` as usual |

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

### Choosing DCP — context vs speed (measured on this box)

`DCP=4` is the default because it unlocks **250k context + 2.84× concurrency**.
But sharding the MLA KV across 4 GPUs has a real throughput cost — **decode is
~1.6× slower than DCP=1.** If you don't need long context, drop DCP.

| `DCP_SIZE` | Max context | KV pool @ max | Decode (single) | Concurrency | When to use |
|---|---|---|---|---|---|
| **1** | ~131k (OOM > ~177k) | ~178k tok | **~81 tok/s** | 1.36× | ≤128k context, want max speed |
| 2 | 250k | ~355k tok | ~49 tok/s | 1.42× | middle ground |
| **4** *(default)* | 250k | 710,593 tok | ~50 tok/s | **2.84×** | long context / high concurrency |

> Measured at temp 0, b12x, MTP=3, fp8 KV. Decode tok/s is steady-state (reliable);
> **cold TTFT is compile-dominated** — the first request at a new prompt length
> JIT-compiles that size bucket (tens of seconds), then warm/prefix-cache hits are
> fast (see [Performance](#performance-measured-warm)). So a slow *first* request is
> the kernel cache warming up, **not** a hang.

**For a fast ≤128k endpoint**, set in `.env`:

```bash
DCP_SIZE=1
MAX_MODEL_LEN=131072
```

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
| Empty / `null` `content` (`finish_reason: length`) | reasoning model ate the token budget | raise `max_tokens` (≥2000) or omit — see [Talking to the model](#talking-to-the-model--its-a-reasoning-model-read-this-first) |
