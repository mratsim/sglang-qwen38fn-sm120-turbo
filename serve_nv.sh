#!/bin/bash
# Qwen3.8-Flash-Next — NVIDIA NVFP4 on RTX PRO 6000 (sm_120), TP=1.
# NVIDIA checkpoint launcher for the shared next-local image.
# Image: ./Dockerfile.
# SGLANG_SM120_ONLINE_MXFP8 is read by the patched image only; stock SGLang ignores it.

set -euo pipefail

# ============================================================
# Container setup
# ============================================================
IMAGE="localhost/sglang-qwen38fn-sm120-turbo:next-local"
PODNAME="sglang-qwen38fn-nv"
SGLANG_PORT=8000

# ============================================================
# Paths
# ============================================================
MODELSCOPE_CACHE="/mnt/data/gpustack-data/cache/model_scope"
HF_CACHE="${HOME}/.cache/huggingface"

DIR=$(realpath "$(dirname "${BASH_SOURCE[0]}")")
SGL_CACHE="${DIR}/cache-sglang-nv"

mkdir -p "${SGL_CACHE}"/{sglang-generated,triton,tilelang}

# ============================================================
# Checkpoint
# ============================================================
MODELNAME="Qwen3.8-Flash-Next"
MODEL_DIR="nv-community/Qwen3.8-Flash-Next-NVFP4"
QUANTIZATION=modelopt_mixed
OFFLINE_MODE=true

# ============================================================
# Tuning knobs
# ============================================================
TP_SIZE=1

GPU_UTIL=0.975              # --mem-fraction-static, ~76K KV tokens per 0.01.
                            # Image preprocessing and JIT kernels need spare VRAM.
CONTEXT_SIZE=262144
KVFP8=true

ONLINE_MXFP8=true           # Quantize otherwise-unquantized linears at load on SM120.

MTP=true
GDN_MTP_CACHE_MODE=none     # none: RecoverSSM recomputes the accepted state after verify.
                            # full: retain one state copy per draft token.

MAX_RUNNING=12
CHUNKED_PREFILL=4096

HICACHE=true
HICACHE_SIZE=30             # GB of pinned host RAM, in addition to the host PLE table.

DEFAULT_REASONING_EFFORT="medium" # xhigh | medium | low; per-request kwargs win.

# ============================================================
# Guards
# ============================================================
for knob in MTP KVFP8 HICACHE ONLINE_MXFP8 OFFLINE_MODE; do
  case "${!knob}" in true|false) ;; *) echo "${knob} must be true or false" >&2; exit 2 ;; esac
done
case "${GDN_MTP_CACHE_MODE}" in
  none|full) ;;
  *) echo "GDN_MTP_CACHE_MODE must be none or full" >&2; exit 2 ;;
esac
case "${DEFAULT_REASONING_EFFORT}" in
  xhigh|medium|low) ;;
  *) echo 'DEFAULT_REASONING_EFFORT must be xhigh, medium, or low' >&2; exit 2 ;;
esac

# ============================================================
# Assemble env & args
# ============================================================
ENV_VARS=(
    SAFETENSORS_FAST_GPU=1
    SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=1
    OMP_NUM_THREADS=1
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True # Avoid allocator fragmentation.
    SGLANG_AUTO_NUMA_BIND=false
    SGLANG_SM120_ONLINE_MXFP8="${ONLINE_MXFP8}"
    XDG_CACHE_HOME=/root/.cache
    SGLANG_CACHE_DIR=/root/.cache/sglang-generated
)
if [[ "${OFFLINE_MODE}" == true ]]; then
    ENV_VARS+=(TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1)
fi

# Exact sizing for extra_buffer_lazy; stock autoderivation overallocates slots.
MAMBA_CACHE=$(( 4 * MAX_RUNNING + 3 ))

SPEC_ARGS=()
KV_ARGS=()
HICACHE_ARGS=()
if [[ "${MTP}" == true ]]; then
    SPEC_ARGS+=(--speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --gdn-mtp-cache-mode "${GDN_MTP_CACHE_MODE}")
fi
if [[ "${KVFP8}" == true ]]; then
    KV_ARGS+=(--kv-cache-dtype fp8_e4m3)
fi
if [[ "${HICACHE}" == true ]]; then
    HICACHE_ARGS+=(--enable-hierarchical-cache --hicache-size "${HICACHE_SIZE}" --hicache-write-policy write_through)
fi

# Preflight: fail before starting a container if the offline snapshot is absent.
if ! find "${MODELSCOPE_CACHE}/${MODEL_DIR}" -maxdepth 1 -name "*.safetensors" -print -quit 2>/dev/null | grep -q .; then
    echo "no .safetensors under ${MODELSCOPE_CACHE}/${MODEL_DIR}" >&2
    exit 2
fi
MODEL_PATH="/workspace/local_models/${MODEL_DIR}"

# ============================================================
# Server command (single source: the recap prints it, Docker runs it)
# ============================================================

SERVER_ARGS=(
    sglang serve
        # Networking
        --host 0.0.0.0
        --port "${SGLANG_PORT}"
        # Model identity
        --served-model-name "${MODELNAME}"
        --model-path "${MODEL_PATH}"
        # Tokenizer / tools / reasoning
        --reasoning-parser auto
        --tool-call-parser auto
        --default-chat-template-kwargs '{"reasoning_effort":"'"${DEFAULT_REASONING_EFFORT}"'"}'
        # Startup warmup
        --warmups sm120_turbo_structured_output
        # PLE 51B n-gram table -> pinned host RAM
        --ple-offload-embedding
        # Attention backends
        --linear-attn-prefill-backend flashinfer
        --linear-attn-decode-backend flashinfer
        # Mamba (GDN linear attention)
        --max-mamba-cache-size "${MAMBA_CACHE}"
        --mamba-radix-cache-strategy extra_buffer_lazy
        --mamba-track-interval 64
        --mamba-ssm-dtype bfloat16
        # Parallelism
        --tp "${TP_SIZE}"
        # Quantization
        --quantization "${QUANTIZATION}"
        --moe-runner-backend flashinfer_cutlass
        --speculative-moe-runner-backend flashinfer_cutlass
        "${KV_ARGS[@]}"
        "${HICACHE_ARGS[@]}"
        # Context / memory
        --context-length "${CONTEXT_SIZE}"
        --mem-fraction-static "${GPU_UTIL}"
        --page-size 64
        --chunked-prefill-size "${CHUNKED_PREFILL}"
        # Batching
        --max-running-requests "${MAX_RUNNING}"
        # Speculation
        "${SPEC_ARGS[@]}"
        # Serving statistics
        --enable-metrics
        --enable-cache-report
        # Idle behavior
        --sleep-on-idle
        # Multimodal limits
        --limit-mm-data-per-request '{"image":4}'
        "$@"
)

# ============================================================
# Launch recap
# ============================================================
{
  printf 'launch %s as %s\n' "${MODEL_PATH}" "${MODELNAME}"
  printf '  checkpoint       nvidia (modelscope, local)\n'
  printf '  offline          %s\n' "$([ "${OFFLINE_MODE}" == true ] && echo on || echo off)"
  printf '  image            %s\n' "${IMAGE}"
  printf '  pod / port       %s / %s\n' "${PODNAME}" "${SGLANG_PORT}"
  printf '  parallel         tp=%s\n' "${TP_SIZE}"
  printf '  context          %s tokens, mem-fraction=%s\n' "${CONTEXT_SIZE}" "${GPU_UTIL}"
  printf '  batching         max-running=%s, chunked-prefill=%s, state-slots=%s\n' "${MAX_RUNNING}" "${CHUNKED_PREFILL}" "${MAMBA_CACHE}"
  printf '  speculation      MTP=%s, gdn-cache-mode=%s\n' "$([ "${MTP}" == true ] && echo on || echo off)" "${GDN_MTP_CACHE_MODE}"
  printf '  kv / hicache     fp8-kv=%s, hicache=%s (%s GB)\n' "$([ "${KVFP8}" == true ] && echo on || echo off)" "$([ "${HICACHE}" == true ] && echo on || echo off)" "${HICACHE_SIZE}"
  printf '  online mxfp8     %s\n' "$([ "${ONLINE_MXFP8}" == true ] && echo on || echo off)"
  printf '  reasoning        %s\n' "${DEFAULT_REASONING_EFFORT}"
  printf '  container env    %s\n' "${ENV_VARS[*]}"
  printf '  server command\n    '
  printf '%s ' "${SERVER_ARGS[@]}"
  printf '\n'
} >&2

# ============================================================
# Container
# ============================================================
ENV_ARGS=()
for kv in "${ENV_VARS[@]}"; do
    ENV_ARGS+=(-e "${kv}")
done

docker run --detach --restart always \
    --health-cmd="curl -f http://localhost:${SGLANG_PORT}/health || exit 1" \
    --health-start-period=600s \
    --health-interval=30s \
    --health-retries=20 \
    --name "${PODNAME}" \
    --device nvidia.com/gpu=all \
    --network=host \
    --ipc=host \
    "${ENV_ARGS[@]}" \
    -v "${SGL_CACHE}/sglang-generated:/root/.cache/sglang-generated" \
    -v "${SGL_CACHE}/triton:/root/.triton" \
    -v "${SGL_CACHE}/tilelang:/root/.cache/tilelang" \
    -v "${MODELSCOPE_CACHE}":/workspace/local_models:ro \
    -v "${HF_CACHE}":/root/.cache/huggingface:rw \
    "${IMAGE}" \
    "${SERVER_ARGS[@]}"

echo "started ${PODNAME}; follow with: docker logs -f ${PODNAME}; health: curl -s localhost:${SGLANG_PORT}/health"
