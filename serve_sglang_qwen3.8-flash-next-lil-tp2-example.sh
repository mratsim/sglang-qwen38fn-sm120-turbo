#!/bin/bash
# Qwen3.8-Flash-Next — local-inference-lab NVFP4 on 2x RTX PRO 6000 (sm_120), TP=2.
# LIL checkpoint launcher for the shared next-local image; needs the 0020-0025 patchset.
# Image: ./Dockerfile.
# QAD snapshot is a branch: hf download local-inference-lab/Qwen3.8-Flash-Next-NVFP4 \
#   --revision qad-step-4000 --local-dir ~/models/Qwen3.8-Flash-Next-NVFP4-QAD4000

set -euo pipefail

# ============================================================
# Container setup
# ============================================================
IMAGE="localhost/sglang-qwen38fn-sm120-turbo:r24"
PODNAME="sglang-qwen38fn-lil"
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
MODEL_HF="local-inference-lab/Qwen3.8-Flash-Next-NVFP4"
MODEL_REVISION=qad-step-4000       # the QAD snapshot only exists on this branch
MODEL_DIR="Qwen3.8-Flash-Next-NVFP4-QAD4000"   # where hf download --local-dir put it
MODEL_SOURCE=local                 # hf: resolve MODEL_HF@MODEL_REVISION on first boot
                                   # local: read ${LOCAL_MODELS}/${MODEL_DIR}, no downloads
OFFLINE_MODE=true                  # TRANSFORMERS_OFFLINE=1 and HF_HUB_OFFLINE=1 inside the container

QUANTIZATION=modelopt_mixed
ONLINE_MXFP8=false                 # prequantized mxfp8/NVFP4/W4A16, the online path has nothing to do

# ============================================================
# Tuning knobs
# ============================================================
TP_SIZE=2

GPU_UTIL=0.85                      # --mem-fraction-static; ~76K KV tokens per 0.01.
CONTEXT_SIZE=327680                # = YARN_FACTOR x YARN_ORIGINAL_MAX
YARN_ORIGINAL_MAX=262144           # pre-YaRN trained length; 4.0 reaches the 1M rung
YARN_FACTOR=1.25                   # 1.0 serves the checkpoint's native length instead
KVFP8=true

MAX_RUNNING=16
CHUNKED_PREFILL=8192

MTP=true                           # NEXTN on the checkpoint's own mtp.* module
MTP_STEPS=3                        # the mtp block declares num_hidden_layers 1, the same
MTP_TOPK=1                         # layer is walked MTP_STEPS times; the drafts are the
MTP_DRAFT_TOKENS=4                 # num_draft_tokens.
GDN_MTP_CACHE_MODE=none            # none frees ~2 GB by not copying GDN state per draft token; full is stock.

# 51.2B-element n-gram table, too big for the device either way. 0025 keeps it packed in
# pinned RAM (13.4 GiB/rank, 26.8 GiB total) and dequantizes each row in the gather
# kernel, so nothing expands it at boot. The file backend stays on the fp8 expansion
# (47.7 GiB of SSD) because packed is pinned-only. pinned wins: host RAM covers this and HiCache.
PLE_BACKEND=pinned
# Only used by file; the model mount is read-only, so sglang cannot derive the path.
PLE_HOST_DIR="${SGL_CACHE}/ple-tables"
PLE_DIR="${PLE_DIR:-/workspace/ple-tables}"

HICACHE=true
# PER RANK: one host pool per TP worker, so the host pays HICACHE_SIZE * TP_SIZE.
# hicache-size is decimal GB: 32 x 2 = 64 GB (60 GiB) here, plus 26.8 GiB of packed
# PLE, so ~86 of the 192 GiB sits pinned and unreclaimable.
HICACHE_SIZE=32

LANGUAGE_MODEL_ONLY=false          # true: skip the vision tower, free its VRAM, reject image requests.

DEFAULT_REASONING_EFFORT="medium"  # xhigh | medium | low, per-request chat_template_kwargs wins

# ============================================================
# Guards
# ============================================================
for knob in MTP KVFP8 HICACHE ONLINE_MXFP8 OFFLINE_MODE LANGUAGE_MODEL_ONLY; do
  case "${!knob}" in true|false) ;; *) echo "${knob} must be true or false" >&2; exit 2 ;; esac
done
case "${PLE_BACKEND}" in
  file|pinned) ;;
  *) echo "PLE_BACKEND must be file or pinned" >&2; exit 2 ;;
esac
case "${GDN_MTP_CACHE_MODE}" in
  none|full) ;;
  *) echo "GDN_MTP_CACHE_MODE must be none or full" >&2; exit 2 ;;
esac
case "${DEFAULT_REASONING_EFFORT}" in
  xhigh|medium|low) ;;
  *) echo 'DEFAULT_REASONING_EFFORT must be xhigh, medium, or low' >&2; exit 2 ;;
esac
case "${MODEL_SOURCE}" in
  hf)    OFFLINE_MODE=false ;;
  local) OFFLINE_MODE=true ;;
  *) echo 'MODEL_SOURCE must be hf or local' >&2; exit 2 ;;
esac

# ============================================================
# Context extension
# ============================================================
# The checkpoint ships no rope_parameters, and --json-model-override-args replaces a
# text_config that is still a dict rather than merging into it, so the merged dict is
# built from the checkpoint's own config.json. Send every rope key: a dropped
# mrope_section loses the mrope path, a dropped rope_theta silently means 10000
# instead of this family's 1e7.
CONFIG_JSON="${LOCAL_MODELS}/${MODEL_DIR}/config.json"
if [[ "${MODEL_SOURCE}" == hf ]]; then
    # Only this 12 KB file is needed to build the override; the weights themselves are
    # resolved by sglang from the hub cache.
    CONFIG_JSON="${SGL_CACHE}/config-${MODEL_REVISION}.json"
    if [[ ! -s "${CONFIG_JSON}" ]] || ! grep -q text_config "${CONFIG_JSON}"; then
        curl -fsSL --retry 3 -o "${CONFIG_JSON}.tmp" \
            "https://huggingface.co/${MODEL_HF}/resolve/${MODEL_REVISION}/config.json" \
            || { echo "cannot fetch ${MODEL_HF}@${MODEL_REVISION}/config.json; use MODEL_SOURCE=local" >&2; rm -f "${CONFIG_JSON}.tmp"; exit 2; }
        mv "${CONFIG_JSON}.tmp" "${CONFIG_JSON}"
    fi
fi

# CONTEXT_SIZE has to equal the length this merge writes. If it does not, sglang derives
# the checkpoint's 262144 and the run is silently rope-truncated at the top of the range.
EXTENDED=$(python3 -c "print(int(round(${YARN_FACTOR} * ${YARN_ORIGINAL_MAX})))")
if [[ "${CONTEXT_SIZE}" != "${EXTENDED}" ]]; then
  echo "CONTEXT_SIZE=${CONTEXT_SIZE} but YaRN ${YARN_FACTOR} x ${YARN_ORIGINAL_MAX} extends to ${EXTENDED}" >&2
  exit 2
fi

YARN_JSON=""
if [[ "${YARN_FACTOR}" != "1.0" ]]; then
if ! YARN_JSON=$(python3 - "${CONFIG_JSON}" "${YARN_FACTOR}" "${YARN_ORIGINAL_MAX}" <<'PY'
import json, os, sys
path, factor, original = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
if not os.path.exists(path):
    raise SystemExit("cannot read %s" % path)
text = dict(json.load(open(path))["text_config"])
text["rope_parameters"] = {
    "mrope_interleaved": True, "mrope_section": [11, 11, 10], "rope_type": "yarn",
    "rope_theta": 10000000, "partial_rotary_factor": 0.25,
    "factor": factor, "original_max_position_embeddings": original,
}
ext = int(round(factor * original))
text["max_position_embeddings"] = ext
# The parent mirrors the child's rope dict (qwen4_exp.py:172) and runs the yarn validator
# on itself, and this checkpoint declares no top-level max_position_embeddings for it.
print(json.dumps({"text_config": text, "max_position_embeddings": ext}))
PY
); then
  # Stop here rather than launch a container that dies on the config.
  echo "cannot merge YaRN into ${CONFIG_JSON}: MODEL_DIR / MODEL_REVISION must name a readable snapshot, or set YARN_FACTOR=1.0" >&2
  exit 2
fi
fi

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
if [[ "${OFFLINE_MODE}" == true ]]; then
    ENV_VARS+=(TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1)
fi

# Exact derivation, SGLang autoderivation overallocates and wastes KV-cache.
MAMBA_CACHE=$(( 5 * MAX_RUNNING + 4 ))

MODEL_ARGS=()
if [[ "${MODEL_SOURCE}" == hf ]]; then
    # local reads the read-only mount below; hf hands sglang the hub id and the branch.
    # The MTP draft inherits this revision (speculative_hook.py:600), so it cannot resolve
    # a different snapshot of the same repo than its target.
    MODEL_ARGS=(--model-path "${MODEL_HF}" --revision "${MODEL_REVISION}")
else
    MODEL_ARGS=(--model-path "/workspace/local_models/${MODEL_DIR}")
fi

SPEC_ARGS=()
KV_ARGS=()
HICACHE_ARGS=()
OVERRIDE_ARGS=()
[[ -n "${YARN_JSON}" ]] && OVERRIDE_ARGS+=(--json-model-override-args "${YARN_JSON}")
PLE_ARGS=(--ple-offload-embedding --ple-offload-backend "${PLE_BACKEND}")
PLE_MOUNT=()
if [[ "${PLE_BACKEND}" == file ]]; then
    # The table is written into the container path, so the directory only means
    # something alongside its mount. pinned keeps everything in RAM and wants neither.
    mkdir -p "${PLE_HOST_DIR}"
    PLE_ARGS+=(--ple-offload-dir "${PLE_DIR}")
    PLE_MOUNT+=(-v "${PLE_HOST_DIR}:${PLE_DIR}:rw")
fi
LM_ONLY_ARGS=()
[[ "${LANGUAGE_MODEL_ONLY}" == true ]] && LM_ONLY_ARGS+=(--language-model-only)
if [[ "${MTP}" == true ]]; then
    SPEC_ARGS+=(--speculative-algorithm NEXTN --speculative-num-steps "${MTP_STEPS}" --speculative-eagle-topk "${MTP_TOPK}" --speculative-num-draft-tokens "${MTP_DRAFT_TOKENS}")
fi
if [[ "${KVFP8}" == true ]]; then
    KV_ARGS+=(--kv-cache-dtype fp8_e4m3)
fi
if [[ "${HICACHE}" == true ]]; then
    HICACHE_ARGS+=(--enable-hierarchical-cache --hicache-size "${HICACHE_SIZE}" --hicache-write-policy write_through)
fi

# Keys of sglang.srt.entrypoints.warmup. prefill_shapes walks the prefill ladder to 32K;
# sm120_turbo_structured_output (0007) pays the xgrammar FSM compile here rather than
# leaving it to the first structured request of the boot.
WARMUPS="prefill_shapes,sm120_turbo_structured_output"

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
        "${MODEL_ARGS[@]}"
        # Tokenizer / tools / reasoning
        --reasoning-parser auto
        --tool-call-parser auto
        --default-chat-template-kwargs '{"reasoning_effort":"'"${DEFAULT_REASONING_EFFORT}"'"}'
        # PLE table kept off the device. --cpu-offload-gb must NOT appear here:
        # generic layer offload would stage it back to the device and raises.
        "${PLE_ARGS[@]}"
        # Vision tower
        "${LM_ONLY_ARGS[@]}"
        # The g16 vision fc2 has K=4304, which at TP=2 splits to 2152:
        # not a multiple of 16, and its 269 scale groups cannot divide by 2. Replicating
        # keeps K whole, which is also the geometry patch 0023's fc1 padding assumes.
        --mm-enable-dp-encoder
        # Backends
        --linear-attn-prefill-backend flashinfer
        --linear-attn-decode-backend flashinfer
        --moe-runner-backend flashinfer_cutlass
        # Mamba (GDN linear attention)
        --max-mamba-cache-size "${MAMBA_CACHE}"
        --mamba-radix-cache-strategy extra_buffer
        --mamba-track-interval 64
        --mamba-ssm-dtype bfloat16
        --gdn-mtp-cache-mode "${GDN_MTP_CACHE_MODE}"
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
        # Startup warmups
        --warmups "${WARMUPS}"
        # Idle behavior
        --sleep-on-idle
        "${OVERRIDE_ARGS[@]}"
        "$@"
)

# ============================================================
# Launch recap
# ============================================================
{
  printf 'launch %s as %s\n' "${MODEL_ARGS[1]}" "${MODELNAME}"
  printf '  checkpoint       %s @ %s (%s)\n' "${MODEL_HF}" "${MODEL_REVISION}" "${MODEL_SOURCE}"
  printf '  offline          %s\n' "$([ "${OFFLINE_MODE}" == true ] && echo on || echo off)"
  printf '  image            %s\n' "${IMAGE}"
  printf '  pod / port       %s / %s\n' "${PODNAME}" "${SGLANG_PORT}"
  printf '  parallel         tp=%s\n' "${TP_SIZE}"
  printf '  quantization     %s, online mxfp8=%s\n' "${QUANTIZATION}" "$([ "${ONLINE_MXFP8}" == true ] && echo on || echo off)"
  printf '  context          %s tokens (YaRN %s x %s), mem-fraction=%s\n' "${CONTEXT_SIZE}" "${YARN_FACTOR}" "${YARN_ORIGINAL_MAX}" "${GPU_UTIL}"
  printf '  batching         max-running=%s, chunked-prefill=%s, state-slots=%s\n' "${MAX_RUNNING}" "${CHUNKED_PREFILL}" "${MAMBA_CACHE}"
  printf '  speculation      MTP=%s %s/%s/%s, gdn-cache-mode=%s\n' "$([ "${MTP}" == true ] && echo on || echo off)" "${MTP_STEPS}" "${MTP_TOPK}" "${MTP_DRAFT_TOKENS}" "${GDN_MTP_CACHE_MODE}"
  if [[ "${PLE_BACKEND}" == file ]]; then
    printf '  ple table        offloaded to file, dir=%s (host %s)\n' "${PLE_DIR}" "${PLE_HOST_DIR}"
  else
    printf '  ple table        offloaded to pinned host RAM, ~13.4 GiB packed per rank\n'
  fi
  printf '  vision           %s\n' "$([ "${LANGUAGE_MODEL_ONLY}" == true ] && echo 'language-model-only, encoder skipped' || echo 'encoder built, quantized, replicated per rank (required by the g16 fc2 at TP=2)')"
  printf '  kv / hicache     fp8-kv=%s, hicache=%s (%s GB/rank = %s GB host)\n' "$([ "${KVFP8}" == true ] && echo on || echo off)" "$([ "${HICACHE}" == true ] && echo on || echo off)" "${HICACHE_SIZE}" "$(( HICACHE_SIZE * TP_SIZE ))"
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

# The health start period is ~5x the TP=1 budget: 98 GiB of weights must be read and the
# draft loaded before /health answers, and a kill mid-load restarts everything. A cold
# MODEL_SOURCE=hf boot also pays the download inside this window.
podman run --replace --detach --restart always \
    --health-cmd="curl -f http://localhost:${SGLANG_PORT}/health || exit 1" \
    --health-start-period=900s \
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
    "${PLE_MOUNT[@]}" \
    -v "${LOCAL_MODELS}":/workspace/local_models:ro \
    -v "${HF_CACHE}":/root/.cache/huggingface:rw \
    "${IMAGE}" \
    "${SERVER_ARGS[@]}"
