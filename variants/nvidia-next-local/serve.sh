#!/bin/bash
# Qwen3.8-Flash-Next — NVIDIA NVFP4 on RTX PRO 6000 (sm_120), TP=1.
# Image: localhost/sglang-qwen38fn-sm120-nvidia:next-local (base
#        lmsysorg/sglang:dev-cu13-qwen38-next-local + 0001/0002/0003/0005/0006/0007).
# MTP defaults on with RecoverSSM and Mamba/Radix tracking.
# No RadixArk: serves the NVIDIA release from the ModelScope cache.

set -euo pipefail

IMAGE="localhost/sglang-qwen38fn-sm120-nvidia:next-local"
PODNAME="sglang-nv"
SGLANG_PORT=30001

# ============================================================
# Paths (match the running turbo container's mounts)
# ============================================================
MODELSCOPE_CACHE=/mnt/data/gpustack-data/cache/model_scope
HF_CACHE="${HOME}"/.cache/huggingface
DIR=$(realpath "$(dirname "${BASH_SOURCE[0]}")")
SGL_CACHE="${DIR}/cache-sglang-nv"
mkdir -p "${SGL_CACHE}"/{sglang-generated,triton,tilelang}

# ============================================================
# Checkpoint: NVIDIA release (modelopt_mixed, 10 shards + fp8 MTP/PLE shard)
# ============================================================
MODELNAME="Qwen3.8-Flash-Next"
MODEL_DIR="nv-community/Qwen3.8-Flash-Next-NVFP4"
QUANTIZATION=modelopt_mixed
OFFLINE_MODE=true

# ============================================================
# Tuning knobs (inherited from the validated turbo run on this box)
# ============================================================
TP_SIZE=1
# 0003 converts otherwise-unquantized linear-attn/shared-expert/
# hyper-connection/lm_head weights online to MXFP8 on SM120.
GPU_UTIL=0.975              # ~76K KV tokens per 0.01 of this fraction
CONTEXT_SIZE=262144
KVFP8=true
MTP=true
GDN_MTP_CACHE_MODE=none
MAX_RUNNING=12
CHUNKED_PREFILL=4096
HICACHE=true
HICACHE_SIZE=90             # GB of pinned host RAM, on top of the ~51 GB host PLE table
DEFAULT_REASONING_EFFORT="medium"

# ============================================================
# Guards
# ============================================================
for knob in KVFP8 MTP HICACHE OFFLINE_MODE; do
    case "${!knob}" in true|false) ;; *) echo "${knob} must be true or false" >&2; exit 2 ;; esac
done
case "${GDN_MTP_CACHE_MODE}" in
    none|full) ;;
    *) echo "GDN_MTP_CACHE_MODE must be none or full" >&2; exit 2 ;;
esac

# Preflight: the NVIDIA snapshot must be in the ModelScope cache.
if ! find "${MODELSCOPE_CACHE}/${MODEL_DIR}" -maxdepth 1 -name "*.safetensors" -print -quit 2>/dev/null | grep -q .; then
    echo "no .safetensors under ${MODELSCOPE_CACHE}/${MODEL_DIR}" >&2
    exit 2
fi
MODEL_PATH="/workspace/local_models/${MODEL_DIR}"

# ============================================================
# Env & args
# ============================================================
ENV_VARS=(
    SAFETENSORS_FAST_GPU=1
    SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=1
    OMP_NUM_THREADS=1
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    SGLANG_AUTO_NUMA_BIND=false
    SGLANG_SM120_ONLINE_MXFP8=true
    XDG_CACHE_HOME=/root/.cache
    SGLANG_CACHE_DIR=/root/.cache/sglang-generated
)
if [[ "${OFFLINE_MODE}" == true ]]; then
    ENV_VARS+=(TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1)
fi

# extra_buffer_lazy uses four request slots plus three startup/recovery slots.
MAMBA_CACHE=$((4 * MAX_RUNNING + 3))

KV_ARGS=()
SPEC_ARGS=()
HICACHE_ARGS=()
if [[ "${MTP}" == true ]]; then
    SPEC_ARGS+=(
        --speculative-algorithm NEXTN
        --speculative-num-steps 3
        --speculative-eagle-topk 1
        --speculative-num-draft-tokens 4
        --gdn-mtp-cache-mode "${GDN_MTP_CACHE_MODE}"
    )
fi
if [[ "${KVFP8}" == true ]]; then
    KV_ARGS+=(--kv-cache-dtype fp8_e4m3)
fi
if [[ "${HICACHE}" == true ]]; then
    HICACHE_ARGS+=(--enable-hierarchical-cache --hicache-size "${HICACHE_SIZE}" --hicache-write-policy write_through)
fi

SERVER_ARGS=(
    sglang serve
        --host 0.0.0.0
        --port "${SGLANG_PORT}"
        --served-model-name "${MODELNAME}"
        --model-path "${MODEL_PATH}"
        --reasoning-parser auto
        --tool-call-parser auto
        --default-chat-template-kwargs '{"reasoning_effort":"'"${DEFAULT_REASONING_EFFORT}"'"}'
        # 0007 only registers this custom warmup; it is inert unless selected.
        --warmups sm120_turbo_structured_output
        # PLE 51B n-gram table -> pinned host RAM
        --ple-offload-embedding
        --linear-attn-prefill-backend flashinfer
        --linear-attn-decode-backend flashinfer
        --max-mamba-cache-size "${MAMBA_CACHE}"
        --mamba-radix-cache-strategy extra_buffer_lazy
        --mamba-track-interval 64
        --mamba-ssm-dtype bfloat16
        --tp "${TP_SIZE}"
        --quantization "${QUANTIZATION}"
        "${SPEC_ARGS[@]}"
        "${KV_ARGS[@]}"
        "${HICACHE_ARGS[@]}"
        --context-length "${CONTEXT_SIZE}"
        --mem-fraction-static "${GPU_UTIL}"
        --page-size 64
        --chunked-prefill-size "${CHUNKED_PREFILL}"
        --max-running-requests "${MAX_RUNNING}"
        --enable-metrics
        --enable-cache-report
        --sleep-on-idle
        --limit-mm-data-per-request '{"image":4}'
        "$@"
)

{
  printf 'launch %s as %s\n' "${MODEL_PATH}" "${MODELNAME}"
  printf '  image            %s\n' "${IMAGE}"
  printf '  pod / port       %s / %s\n' "${PODNAME}" "${SGLANG_PORT}"
  printf '  parallel         tp=%s\n' "${TP_SIZE}"
  printf '  context          %s tokens, mem-fraction=%s\n' "${CONTEXT_SIZE}" "${GPU_UTIL}"
  printf '  batching         max-running=%s, chunked-prefill=%s, state-slots=%s\n' "${MAX_RUNNING}" "${CHUNKED_PREFILL}" "${MAMBA_CACHE}"
  printf '  speculation      mtp=%s, gdn-cache=%s%s\n' "${MTP}" "${GDN_MTP_CACHE_MODE}" "$([ "${MTP}" == true ] && echo ' (RecoverSSM/Radix active)' || echo ' (inactive)')"
  printf '  kv / hicache     fp8-kv=%s, hicache=%s (%s GB)\n' "$([ "${KVFP8}" == true ] && echo on || echo off)" "$([ "${HICACHE}" == true ] && echo on || echo off)" "${HICACHE_SIZE}"
  printf '  offline          %s\n' "$([ "${OFFLINE_MODE}" == true ] && echo on || echo off)"
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
