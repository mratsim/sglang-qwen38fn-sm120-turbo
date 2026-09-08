# NVIDIA next-local variant

This variant targets `nv-community/Qwen3.8-Flash-Next-NVFP4` on one RTX PRO
6000 using SGLang's `dev-cu13-qwen38-next-local` image at commit
`4ccff141dbe992794f9da6c3aa23535b4f72000d`.

It keeps the default stack's `0002`, `0005`, `0006`, and `0007` patches. Two
patches are base-specific:

- `0001-sm120-fp8-kv-cache-trimmed.patch` omits the TRT-LLM sparse-decode hunk,
  because the next-local base already enables that path on SM120.
- `0003-sm120-online-fp8.patch` rebases online MXFP8 onto the next-local source
  layout, including the current HC-mix kernel signature and moved Qwen4Exp code.

Patch `0004` is not used: it adapts the local-inference-lab checkpoint, while the
NVIDIA checkpoint is already registered as `qwen4_exp` with ModelOpt mixed
precision metadata.

## Build

Run from the repository root so the Dockerfile can copy the shared patches:

```bash
docker build \
  -f variants/nvidia-next-local/Dockerfile \
  -t localhost/sglang-qwen38fn-sm120-nvidia:next-local \
  .
```

## Serve

The image supports both the RadixArk and NVIDIA checkpoints. Keep the existing
RadixArk launcher for RadixArk, and use this separate NVIDIA launcher for the
ModelOpt mixed checkpoint. Edit the image tag and cache paths at the top of
the root `serve_nv.sh`, then run:

```bash
./serve_nv.sh
```

The NVIDIA launcher follows the same layout as the root example and prints its fully
assembled environment and server command before starting the container. Its
validated defaults are:

```bash
GPU_UTIL=0.975
CONTEXT_SIZE=262144
KVFP8=true
ONLINE_MXFP8=true

MTP=true
GDN_MTP_CACHE_MODE=none

MAX_RUNNING=12
MAMBA_CACHE=$(( 4 * MAX_RUNNING + 3 ))

HICACHE=true
HICACHE_SIZE=30
```

`ONLINE_MXFP8` controls only the load-time conversion supplied by patch `0003`:

```bash
SGLANG_SM120_ONLINE_MXFP8="${ONLINE_MXFP8}"
```

It does not disable the checkpoint's native ModelOpt NVFP4/FP8 weights or FP8
KV cache. Set `ONLINE_MXFP8=false` to retain the checkpoint's otherwise
unquantized linear weights in their original dtype.

With `MTP=true`, the launcher selects NEXTN and passes
`--gdn-mtp-cache-mode none`. SGLang normalizes NEXTN to EAGLE internally, so
`speculative_algorithm='EAGLE'` in `server_args` is expected. Set `MTP=false`
to omit both speculation and the RecoverSSM argument.

The Mamba/Radix cache uses `extra_buffer_lazy`. Its state pool is sized with
`4 * MAX_RUNNING + 3`, which gives 51 slots at the default concurrency of 12.

## NVIDIA NVFP4 MoE backend

Both target and draft models must use FlashInfer CUTLASS:

```bash
--moe-runner-backend flashinfer_cutlass
--speculative-moe-runner-backend flashinfer_cutlass
```

Leaving these on `auto` selects `flashinfer_trtllm` on this base and fails
during the FlashInfer warmup with:

```text
NotImplementedError: Unsupported moe_runner_backend for NVFP4 MoE ...
Use --moe-runner-backend flashinfer_cutlass instead.
```

## RTX PRO 6000 validation

The default launcher was tested on one 96 GiB RTX PRO 6000 with the NVIDIA
checkpoint, MTP, RecoverSSM, online MXFP8, FP8 KV cache, and a 30 GB
hierarchical cache:

- online MXFP8 replaced 194 otherwise-unquantized weights;
- target and MTP draft ModelOpt weights loaded successfully;
- RecoverSSM used `gdn_mtp_cache_mode=none` with 51 Mamba slots;
- target and draft KV pools each held 968,512 tokens; the target K/V tensors
  used 5.54 GB each and the one-layer draft K/V tensors used 0.46 GB each;
- target verify, draft decode, and draft extend CUDA graphs were captured;
- hierarchical cache allocation was 23.68 GB for KV plus 6.35 GB for Mamba;
- structured-output warmup completed and `/health` returned HTTP 200;
- a 231-token test generation used CUDA graphs with speculative accept length
  3.17 and accept rate 0.72.

After graph capture, approximately 2.41 GB of device memory remained available
in this configuration. Lower `GPU_UTIL` or cap `--max-total-tokens` if the
deployment needs more runtime headroom.

SGLang currently logs the following warning with `KVFP8=true` because this
checkpoint does not provide KV-cache scaling factors:

```text
Using FP8 KV cache but no scaling factors provided. Defaulting to scaling
factors of 1.0. This may lead to less accurate results!
```

This is not a startup failure, but accuracy-sensitive deployments should test
FP8 KV cache against the default KV dtype before enabling it in production.
