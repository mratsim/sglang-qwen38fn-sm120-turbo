# sglang-qwen38fn-sm120-turbo

Serving stack for **Qwen3.8-Flash-Next** on 96 GiB VRAM (1x RTX Pro 6000, might also work on DGX Spark):
- https://qwen.ai/blog?id=qwen3.8-flash-next
- https://huggingface.co/Qwen/Qwen3.8-Flash-Next
- https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/main/tech_report.pdf

Qwen3.8-Flash-Next is a highly performant LLM that serves as a preview for the future Qwen4 family.
Despite being undertrained, and having very few active parameters and a small size (125B + 6B active + 51B of offloadable embedding table), benchmarks show performance comparable to closed-source LLMs from just 3 months ago (e.g. Opus 4.8).

## Numbers

On the NVFP4 checkpoint [`RadixArk/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4), served by the day-0 SGLang image:
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

## Behind-the-scenes

This builds on the pinned `lmsysorg/sglang:dev-cu13-qwen38-next-local`
image at SGLang commit `4ccff141dbe992794f9da6c3aa23535b4f72000d`.
The resulting image supports both the RadixArk and NVIDIA ModelOpt checkpoints:

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
`./serve_nv.sh --max-total-tokens 570048`. This caps the shared cache capacity,
not the per-request context length. MTP also needs its own model, KV, and graphs.

## Build and serve

Build the shared image from the repository root:

```bash
podman build -t localhost/sglang-qwen38fn-sm120-turbo:next-local .
```

Use `serve_sglang_qwen3.8-flash-next-example.sh` for the RadixArk checkpoint,
or `serve_nv.sh` for the NVIDIA checkpoint. Modify the selected script's image
tag, cache paths, and tuning knobs for your machine.

The RadixArk launcher exposes:
- `IMAGE`: tag you built below
- `MODEL_SOURCE`: `hf` (downloads to `HF_CACHE`) or `local` (reads `LOCAL_MODELS/<dir>` offline)
- `GPU_UTIL`, `MAX_RUNNING`, `HICACHE_SIZE`: KV fraction, concurrency, KVcache RAM offloading
- `PODNAME`, `SGLANG_PORT`: container name and port

```bash
./serve_sglang_qwen3.8-flash-next-example.sh    # recap of the full config goes to stderr
curl -s localhost:30000/health
```

The NVIDIA launcher uses `nv-community/Qwen3.8-Flash-Next-NVFP4` with
`modelopt_mixed`. It explicitly selects `flashinfer_cutlass` for both target
and speculative NVFP4 MoE; leaving these runners on `auto` selects an
unsupported backend on this base.

```bash
./serve_nv.sh
curl -s localhost:8000/health
```

Its validated RTX PRO 6000 defaults include:

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

Keep your own variants in `internal/`: the folder ships empty and everything in it is git-ignored, so a customized launcher (`internal/serve_my.sh`, host paths, bench settings) lives beside the stack without ever being committed or published.

## Acknowledgements

While I choose different solutions, a significant amount of work has gone into both quant and tuning inference engines.

Thanks to:
- [SGLang](https://github.com/sgl-project/sglang) / [RadixArk](https://huggingface.co/RadixArk) for the day-0 image support ([`lmsysorg/sglang:qwen38flashnext`](https://hub.docker.com/r/lmsysorg/sglang)) and the [NVFP4 quant](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4)
- local-inference-lab ([GitHub](https://github.com/local-inference-lab), [Hugging Face](https://huggingface.co/local-inference-lab), incl. the [NVFP4 release](https://huggingface.co/local-inference-lab/Qwen3.8-Flash-Next-NVFP4) and the [llm-inference-bench](https://github.com/local-inference-lab/llm-inference-bench) the Numbers section measures on), voipmonitor, lukealonso and the community for supporting RTX Pro 6000 development with kernels, docker images, quants and benchmark tooling.
- Other serving stacks for Qwen3.8-Flash-Next:
  - [jpezzulli/sglang-rtxpro6000](https://github.com/jpezzulli/sglang-rtxpro6000)
  - [gabrielolympie/sglang-flashnext-sm120](https://github.com/gabrielolympie/sglang-flashnext-sm120)
  - [lovedheart/sglang](https://github.com/lovedheart/sglang) and their [NVFP4-FP8 checkpoint](https://huggingface.co/lovedheart/Qwen3.8-Flash-Next-NVFP4-FP8)
  - [ormandj/sglang-qwen38-flash-next-sm120](https://github.com/ormandj/sglang-qwen38-flash-next-sm120)
  - [yepapa-nest/qwen38-flashnext-rtx6000](https://github.com/yepapa-nest/qwen38-flashnext-rtx6000)

## License

SGLang and the patches are under the Apache-2.0 License.

The SGLang docker image builds on top of NVIDIA's Ubuntu+CUDA image, which is under the NVIDIA Deep Learning Container License.
