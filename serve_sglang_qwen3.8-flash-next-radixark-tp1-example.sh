#!/bin/bash
# Qwen3.8-Flash-Next NVFP4 on RTX PRO 6000 (sm_120, 96 GB), TP=1.
# Image: ./Dockerfile (extends the official day-0 image with patches/).
# SGLANG_SM120_ONLINE_MXFP8 is read by this image only, stock sglang ignores it.

set -euo pipefail

# ============================================================
# Container setup
# ============================================================
IMAGE="localhost/sglang-qwen38fn-sm120-turbo:r24"
PODNAME="sglang"
SGLANG_PORT=30000

# ============================================================
# Paths
# ============================================================
HF_CACHE="${HOME}"/.cache/huggingface
LOCAL_MODELS="${HOME}"/models

DIR=$(realpath "$(dirname "${BASH_SOURCE[0]}")")
SGL_CACHE="${DIR}/cache-sglang"

mkdir -p "${HF_CACHE}" "${LOCAL_MODELS}" "${SGL_CACHE}"/{sglang-generated,triton,tilelang}

# ============================================================
# Checkpoint
# ============================================================
MODELNAME="Qwen3.8-Flash-Next"
MODEL_SOURCE=local             # hf: resolve the id on first boot (downloads to ${HF_CACHE})
                               # local: read ${LOCAL_MODELS}/<dir>, no downloads
OFFLINE_MODE=true              # TRANSFORMERS_OFFLINE=1 and HF_HUB_OFFLINE=1 inside the container, nothing reaches the Hub
                               # an hf boot with a cold cache needs it false to download

CHECKPOINT=radixark
MODEL_HF="RadixArk/Qwen3.8-Flash-Next-NVFP4"
MODEL_DIR="RadixArk-Qwen3.8-Flash-Next-NVFP4"
QUANTIZATION=modelopt_fp4
ONLINE_MXFP8=true              # This checkpoint ships its linears in bf16, quantized at load.

case "${MODEL_SOURCE}" in
  hf)    MODEL="${MODEL_HF}" ;;
  local) MODEL="${LOCAL_MODELS}/${MODEL_DIR}" ;;
  *) echo 'MODEL_SOURCE must be hf or local' >&2; exit 2 ;;
esac

# ============================================================
# Tuning knobs
# ============================================================
TP_SIZE=1

GPU_UTIL=0.975              # --mem-fraction-static, ~76K KV tokens per 0.01.
                            # 0.975 leaves about ~475MB spare out of 96GiB
                            # Images preprocess on the device and need ~100 MB each.
                            # JIT-ed Triton kernels also use spare VRAM
CONTEXT_SIZE=262144
KVFP8=true

MAX_RUNNING=4
CHUNKED_PREFILL=4096

MTP=true                          # MTP-3 speculation (3 steps / topk 1 / 4 drafts)
GDN_MTP_CACHE_MODE=none           # none: saves no state copies during MTP verification (~2 GB freed). The accepted state is recomputed after verify.
                                  # full: sglang's stock behavior, copies the state per draft token so verification can restore any accepted prefix.

HICACHE=true
HICACHE_SIZE=32                   # GB of pinned host RAM for evicted prefixes, on top of the ~51 GB host PLE table (192 GiB host)

DEFAULT_REASONING_EFFORT="medium" # xhigh | medium | low, per-request chat_template_kwargs wins

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
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True # Required: fragmentation without it costs up to ~373K KV tokens
    SGLANG_SM120_ONLINE_MXFP8="${ONLINE_MXFP8}"
    XDG_CACHE_HOME=/root/.cache
    SGLANG_CACHE_DIR=/root/.cache/sglang-generated
)
if [[ "${MODEL_SOURCE}" == local ]]; then
    OFFLINE_MODE=true
fi
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

# Preflight: decide from the host whether the weights are there before a container
# can freeze on a download and get killed by the health check into a restart loop.
if [[ "${MODEL_SOURCE}" == local ]]; then
    if ! find "${LOCAL_MODELS}/${MODEL_DIR}" -maxdepth 1 -name "*.safetensors" -print -quit 2>/dev/null | grep -q .; then
        echo "MODEL_SOURCE=local: no .safetensors under ${LOCAL_MODELS}/${MODEL_DIR}" >&2
        echo "place the snapshot there first, or boot with MODEL_SOURCE=hf to download it" >&2
        exit 2
    fi
    MODEL_PATH="/workspace/local_models/${MODEL_DIR}"
else
    if ! find "${HF_CACHE}/hub" -path "*--$(tr '[:lower:]' '[:upper:]' <<< "${MODEL_HF%%/*}")--$(tr '[:lower:]' '[:upper:]' <<< "${MODEL_HF##*/}")" -prune -name "*.safetensors" -print -quit 2>/dev/null | grep -q .; then
        echo "note: ${MODEL_HF} is not in ${HF_CACHE}, first boot downloads the full checkpoint" >&2
        [[ "${OFFLINE_MODE}" == true ]] && echo "OFFLINE_MODE=true blocks that download, set it false for a cold-cache hf boot" >&2
    fi
    MODEL_PATH="${MODEL}"
fi

# ============================================================
# Server command (single source: the recap prints it, podman runs it)
# ============================================================
shift || true  # discard the script name before the "$@" passthrough

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
        # Backends
        --linear-attn-prefill-backend flashinfer
        --linear-attn-decode-backend flashinfer
        --moe-runner-backend flashinfer_cutlass
        # Mamba (GDN linear attention)
        --max-mamba-cache-size "${MAMBA_CACHE}"
        --mamba-radix-cache-strategy extra_buffer_lazy
        --mamba-track-interval 64
        --mamba-ssm-dtype bfloat16
        # Parallelism
        --tp "${TP_SIZE}"
        # Quantization
        --quantization "${QUANTIZATION}"
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
        "$@"
)

# ============================================================
# Launch recap
# ============================================================
{
  printf 'launch %s as %s\n' "${MODEL}" "${MODELNAME}"
  printf '  checkpoint       %s (%s)\n' "${CHECKPOINT}" "${MODEL_SOURCE}"
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

podman run --replace --detach --restart always \
    --health-cmd="curl -f http://localhost:${SGLANG_PORT}/health || exit 1" \
    --health-start-period=300s \
    --health-interval=30s \
    --health-retries=10 \
    --health-on-failure=kill \
    --name "${PODNAME}" \
    --device nvidia.com/gpu=all \
    --network=host \
    --ipc=host \
    "${ENV_ARGS[@]}" \
    -v "${SGL_CACHE}/sglang-generated:/root/.cache/sglang-generated" \
    -v "${SGL_CACHE}/triton:/root/.triton" \
    -v "${SGL_CACHE}/tilelang:/root/.cache/tilelang" \
    -v "${LOCAL_MODELS}":/workspace/local_models:ro \
    -v "${HF_CACHE}":/root/.cache/huggingface:rw \
    "${IMAGE}" \
    "${SERVER_ARGS[@]}"
