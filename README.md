# sglang-qwen38fn-sm120-turbo

Serving stack for **Qwen3.8-Flash-Next** on 96 GiB VRAM (1x RTX Pro 6000, might also work on DGX Spark):
- https://qwen.ai/blog?id=qwen3.8-flash-next
- https://huggingface.co/Qwen/Qwen3.8-Flash-Next
- https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/main/tech_report.pdf

Qwen3.8-Flash-Next is a highly performant LLM that serves as a preview for the future Qwen4 family.
Despite being undertrained, and having very few active parameters and a small size (125B + 6B active + 51B of offloadable embedding table), benchmarks show performance comparable to closed-source LLMs from just 3 months ago (e.g. Opus 4.8).

> [!TIP]
> *Monitor your server with [`sgtop`](https://github.com/mratsim/sgtop), my sglang dedicated monitoring tool.*

## Numbers

On the NVFP4 checkpoint [`RadixArk/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4), served by the day-0 SGLang image at TP=1:
- 1x RTX Pro 6000, power-limited to 360W, memory overclocked by +3000MT/s (+6000 in LACT)
- 939456 tokens in KV cache for 4 max concurrent requests @ 0.98 GPU memory utilization
- 856256 tokens in KV cache for 8 max concurrent requests @ 0.98 GPU memory utilization
- 11k~13k prefill tok/s

  <details><summary>prefill trace</summary>

  ![prefill 11k-13k tok/s](images/prefill-11k-13k.png)

  </details>
- Aggregate 800 decode tok/s (~200 tok/s per stream) on hard-to-predict `--profile estonia` reasoning benchmark from https://github.com/local-inference-lab/llm-inference-bench

  <details><summary>decode trace, 4 concurrent requests</summary>

  ![decode aggregate ~800 tok/s at 4 concurrent requests](images/decode-agg4-800.png)

  </details>
- Aggregate 1170 decode tok/s aggregate at 8 concurrent requests

  <details><summary>decode trace, 8 concurrent requests</summary>

  ![decode aggregate ~1170 tok/s at 8 concurrent requests](images/decode-agg8-1170.png)

  </details>
- up to 355 decode tok/s single request on easy to predict code/compaction. `--profile lavd`  reasoning benchmark from https://github.com/local-inference-lab/llm-inference-bench

  <details><summary>decode trace, single request</summary>

  ![decode single request ~355 tok/s on easy-to-predict code/compaction](images/decode-single-355.png)

  </details>
- 206~234 tok/s single request on hard to predict concurrency benchmark from https://github.com/local-inference-lab/llm-inference-bench

![single request 206~234 tok/s on the hard concurrency benchmark](images/llm-inference-bench.png)

On the local-inference-lab QAD checkpoint
[`local-inference-lab/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/local-inference-lab/Qwen3.8-Flash-Next-NVFP4),
served by `serve_sglang_qwen3.8-flash-next-lil-tp2-example.sh` at TP=2 on 2x RTX Pro 6000:
- Aggregate 1600~1800 decode tok/s for 16 max concurrent requests (MTP accept length 3.2~3.5)

  <details><summary>decode trace, 16 concurrent requests</summary>

  ![decode aggregate 1600~1800 tok/s at 16 concurrent requests on TP=2](images/decode-tp2-agg16-1800.png)

  </details>
- 228 decode tok/s single request on the same hard-to-predict concurrency benchmark

## Behind-the-scenes

This builds on the pinned `lmsysorg/sglang:dev-cu13-qwen38-next-local`
image at SGLang commit `4ccff141dbe992794f9da6c3aa23535b4f72000d`.
The resulting image serves all three released Qwen3.8-Flash-Next checkpoints:

- **0001** makes fp8 KV cache work on sm_120.
- **0002** linear-attention layers don't cache MTP drafts, they are recomputed. Saves ~2 GB of KV budget. (And it's surprisingly not slower)
- **0003** quantizes at load whatever the checkpoint left in bf16 (attention, MLP, lm_head, hyperconnection mix) to MXFP8 to reduce memory bandwidth at close to zero-accuracy cost. On FlashInfer 0.6.18, this should be even faster as FlashInfer 0.6.18 integrates [`local-inference-lab/b12x`](https://github.com/local-inference-lab/b12x) and its hardware-accelerated block-scaled GEMM kernel.
- **0005** keeps abandoned runs from eating the machine: an aborted or timed-out client now really evicts its request, and no longer starves the queue behind it.
- **0006** stops the sampler from using NNCL when the server has a single GPU. This was a bug or an oversight in structured JSON decoding, that led to extra GPU memory utilization.
- **0007** Preload triton kernels at boot via long prefill warmup and structure decoding warmup to ensure reserved GPU memory is sufficient and server doesn't crash in the middle of queries.
- **0008** removes a self-reference in the PLE shard loader. Its closure retained
  the loading-time parameter dictionary, keeping replaced MoE scale buffers and
  the old BF16 `lm_head` alive until cyclic GC. On the tested NVIDIA checkpoint
  with online MXFP8, those stale tensors total 8.215 GiB. They can now be
  released during loading.

Earlier release changes when memory becomes available; it does not remove
additional resident model weights. Automatic KV sizing can use the reclaimed
space. To retain GPU headroom, set an explicit total KV token cap, for example
cap: append `--max-total-tokens 570048` to the NVIDIA launcher. It caps the shared cache
capacity, not the per-request context length. MTP also needs its own model, KV, and graphs.

> [!IMPORTANT]
> **What's new since r22.**
>
> **r23 — the NVIDIA release.** Support for the ModelOpt checkpoint
> `nvidia/Qwen3.8-Flash-Next-NVFP4`, served by
> `serve_sglang_qwen3.8-flash-next-nvidia-tp1-example.sh`.
>
> **r24 — the local-inference-lab release, on two GPUs.** The third released checkpoint,
> `local-inference-lab/Qwen3.8-Flash-Next-NVFP4`, whose layers are quantized three different ways
> inside a single checkpoint. It runs across two GPUs with a **327,680-token** context window.
>
> r24 also stamps the build revision into the log lines and the image label, so a log can
> identify which build produced it. See `ai.sglang.revision`.

## Build and serve

Build the shared image from the repository root:

```bash
podman build -t localhost/sglang-qwen38fn-sm120-turbo:r24 .
```

### Support matrix

All three run on an RTX PRO 6000 (SM120). Pick the row that matches your checkpoint.

| Checkpoint | Launcher | Quantization | GPUs | Context | Status |
| --- | --- | --- | --- | --- | --- |
| [RadixArk/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4) | `serve_sglang_qwen3.8-flash-next-radixark-tp1-example.sh` | NVFP4, MXFP8 added at load | 1 | 262144 | benchmarked above |
| [nvidia/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4) | `serve_sglang_qwen3.8-flash-next-nvidia-tp1-example.sh` | mixed, MXFP8 added at load | 1 | 262144 | measured below |
| [local-inference-lab/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/local-inference-lab/Qwen3.8-Flash-Next-NVFP4) | `serve_sglang_qwen3.8-flash-next-lil-tp2-example.sh` | mixed, prequantized | 2 | 327680 | new in r24 |

All three are ready to run; edit their image tag, cache paths and tuning knobs for your
machine. The third needs a second GPU and more system RAM, and takes `MODEL_SOURCE` and
`MODEL_REVISION` because its checkpoint is a branch rather than the default revision.

### RadixArk checkpoint

The launcher exposes:
- `IMAGE`: tag you built below
- `MODEL_SOURCE`: `hf` (downloads to `HF_CACHE`) or `local` (reads `LOCAL_MODELS/<dir>` offline)
- `GPU_UTIL`, `MAX_RUNNING`, `HICACHE_SIZE`: KV fraction, concurrency, KVcache RAM offloading
- `PODNAME`, `SGLANG_PORT`: container name and port

```bash
./serve_sglang_qwen3.8-flash-next-radixark-tp1-example.sh    # recap of the full config goes to stderr
curl -s localhost:30000/health
```

### NVIDIA checkpoint

The launcher uses [`nvidia/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4) with
`modelopt_mixed`. It explicitly selects `flashinfer_cutlass` for both target
and speculative NVFP4 MoE; leaving these runners on `auto` selects an
unsupported backend on this base.

```bash
./serve_sglang_qwen3.8-flash-next-nvidia-tp1-example.sh
curl -s localhost:8000/health
```

Its RTX PRO 6000 defaults include:

```bash
ONLINE_MXFP8=true
MTP=true
GDN_MTP_CACHE_MODE=none
MAX_RUNNING=12
MAMBA_CACHE=$(( 4 * MAX_RUNNING + 3 ))
HICACHE_SIZE=30
```

The NVIDIA configuration loaded target and MTP draft weights, replaced 194
otherwise-unquantized weights with online MXFP8, captured target verify plus
draft decode/extend CUDA graphs, and completed a 231-token generation with a
speculative accept length of 3.17 and accept rate of 0.72. The 30 GB
hierarchical cache allocated 23.68 GB for KV and 6.35 GB for Mamba.

With FP8 KV cache, this checkpoint currently logs that no KV scaling factors
were provided and defaults them to 1.0. This does not prevent startup, but
accuracy-sensitive deployments should compare it with the default KV dtype.

### The local-inference-lab release

The third supported checkpoint, and the only one that needs two GPUs. Its layers arrive already
quantized three different ways — MXFP8 for the dense linears, NVFP4 for the experts, and a
narrower NVFP4 for a few projections — so unlike the launchers above it runs with the at-load
quantization switched off: there is nothing left for it to do.

Served by `serve_sglang_qwen3.8-flash-next-lil-tp2-example.sh`, which reads the checkpoint
from either source: `MODEL_SOURCE=hf` lets sglang resolve the hub id itself, and
`MODEL_SOURCE=local` (the default) reads a snapshot you fetched once with
`hf download --revision qad-step-4000 --local-dir …`, no network at boot. The QAD snapshot is
a *branch* of that repo, so `MODEL_REVISION` has to name it on both paths.

```bash
./serve_sglang_qwen3.8-flash-next-lil-tp2-example.sh
curl -s localhost:30000/health
```

- **Hardware:** 2x RTX PRO 6000.
- **System RAM:** 192 GiB. The prompt-embedding table (27 GiB) and the KV offload pool both
  live in pinned host memory, so budget about 86 GiB of it as untouchable.
- **Context:** 327,680 tokens, 1.25x YaRN over the trained 262,144. The launcher injects the
  rope settings, because this checkpoint ships none of its own.
- **Quality:** `qad-step-4000` is the recommended checkpoint for output quality. Against the
  published NVFP4 revision it scored 79.4% vs 77.5% on AA-LCR v1.1 over ten generations
  ([AA-LCR report](https://github.com/local-inference-lab/rtx6kpro/blob/e8e23de/models/qwen38-flash-next/aa-lcr-nvfp4-vs-qad.md))
  and 78.69% vs 77.17% on exact arithmetic with reasoning disabled
  ([arithmetic report](https://github.com/local-inference-lab/rtx6kpro/blob/e8e23de/models/qwen38-flash-next/direct-arithmetic-stability-nvfp4-vs-qad.md)).

Two settings the server will not start without:

- `--quantization modelopt_mixed`, because the checkpoint is mixed-precision rather than one
  quantization scheme.
- `--mm-enable-dp-encoder`, which replicates the image encoder across the two GPUs instead of
  splitting it: at this width the encoder's 4304-wide projections do not divide.

### Your own variants

Keep your own variants in `internal/`: the folder ships empty and everything in it is git-ignored, so a customized launcher (`internal/serve_my.sh`, host paths, bench settings) lives beside the stack without ever being committed or published.

## Acknowledgements

While I choose different solutions, a significant amount of work has gone into both quant and tuning inference engines.

Thanks to:
- [SGLang](https://github.com/sgl-project/sglang) / [RadixArk](https://huggingface.co/RadixArk) for the day-0 image support ([`lmsysorg/sglang:qwen38flashnext`](https://hub.docker.com/r/lmsysorg/sglang)) and the [NVFP4 quant](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4)
- local-inference-lab ([GitHub](https://github.com/local-inference-lab), [Hugging Face](https://huggingface.co/local-inference-lab), incl. the [NVFP4 release](https://huggingface.co/local-inference-lab/Qwen3.8-Flash-Next-NVFP4) and the [llm-inference-bench](https://github.com/local-inference-lab/llm-inference-bench) the Numbers section measures on), voipmonitor, lukealonso and the community for supporting RTX Pro 6000 development with kernels, docker images, quants and benchmark tooling.
- [kanadaj/sglang](https://github.com/kanadaj/sglang) — the local-inference-lab checkpoint support in this image is ported from their fork: `0020`–`0025` re-anchor their `0004`, `0008`, `0009`, `0010`, `0011` and `0018` onto this base.
- Other serving stacks for Qwen3.8-Flash-Next:
  - [jpezzulli/sglang-rtxpro6000](https://github.com/jpezzulli/sglang-rtxpro6000)
  - [gabrielolympie/sglang-flashnext-sm120](https://github.com/gabrielolympie/sglang-flashnext-sm120)
  - [lovedheart/sglang](https://github.com/lovedheart/sglang) and their [NVFP4-FP8 checkpoint](https://huggingface.co/lovedheart/Qwen3.8-Flash-Next-NVFP4-FP8)
  - [ormandj/sglang-qwen38-flash-next-sm120](https://github.com/ormandj/sglang-qwen38-flash-next-sm120)
  - [yepapa-nest/qwen38-flashnext-rtx6000](https://github.com/yepapa-nest/qwen38-flashnext-rtx6000)

## License

SGLang and the patches are under the Apache-2.0 License.

The SGLang docker image builds on top of NVIDIA's Ubuntu+CUDA image, which is under the NVIDIA Deep Learning Container License.
