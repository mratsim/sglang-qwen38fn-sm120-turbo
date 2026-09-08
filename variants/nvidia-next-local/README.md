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

Edit the cache paths at the top of `serve.sh`, then run:

```bash
./variants/nvidia-next-local/serve.sh
```

The example enables online MXFP8, FP8 KV cache, NEXTN, RecoverSSM
(`--gdn-mtp-cache-mode none`), and Mamba/Radix tracking. Set `MTP=false` to
run without speculative decoding; the RecoverSSM argument is then omitted. It
uses `extra_buffer_lazy` with `MAMBA_CACHE=$((4 * MAX_RUNNING + 3))`.
