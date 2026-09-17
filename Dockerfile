# Qwen3.8-Flash-Next — RadixArk and NVIDIA NVFP4 checkpoints on RTX PRO 6000 (sm_120).
# Base: day-0 dev image of the qwen4-main-squashed branch (sglang 0.0.0.dev1+g4ccff141d).
# The image's sglang is a source install of /sgl-workspace/sglang, so the build
# patches that tree. Every patch must apply cleanly or the build fails.
#
#   0001            fp8_e4m3 KV cache on the QSA path: fp8 tile dequant casts in the
#                   sparse prefill Triton kernels + exact chunked-prefill rereads.
#                   The trtllm-decode hunk of upstream 0001 is dropped: this base
#                   already enables the FlashInfer trtllm paged decode on SM120
#                   natively (qwen_sparse_attn_backend.py `is_sm100_supported() or is_sm120()`).
#   0002            SM120 FlashInfer GDN dtype compatibility plus RecoverSSM.
#                   With MTP and --gdn-mtp-cache-mode none, accepted GDN state is
#                   reconstructed after verify and Radix boundary state tracking
#                   remains enabled without the intermediate SSM pool.
#   0003            SM120 online MXFP8 for otherwise-unquantized linear weights,
#                   including HC mix, shared experts, linear attention and lm_head.
#   0005            evict abandoned requests: abort reaches the scheduler even when
#                   the tokenizer state is already gone; prefill loop stops latching
#                   the queue behind a budget it consumed itself.
#   0006            skip the sampler's token-id reduction on a single-GPU group, so
#                   no NCCL communicator is created during serving (TP=1).
#   0007            sample one grammar-bound request during startup warmup so the
#                   xgrammar FSM compile and bitmask kernel load land in the boot
#                   window, not inside the first structured request.
#   0008            break the Qwen4 weight-loader closure cycle so replaced
#                   lm_head and initial MoE scale buffers are released promptly.
#
# RecoverSSM is present but only active with MTP/NEXTN and
# --gdn-mtp-cache-mode none.
#
# LIL (local-inference-lab) support is 0020, not 0004. Kanadaj's 0004 was written
# against the old qwen38flashnext base; this image base refactored the PLE dtype
# selection into _ple_table_is_fp8() and absorbed the fp8 auto-switch, so 0020 is
# that patch re-anchored onto 4ccff141db. It applies after 0001-0008, not between
# 0003 and 0005, because its PLE loader context includes 0008's lifetime fix.

FROM docker.io/lmsysorg/sglang:dev-cu13-qwen38-next-local@sha256:9d2a843c706c74bc259c0d9abf360551eb2734e1e7d255ab012a6965f10480b6

# Log-prefix revision. Patches carry the SGLANG_PATCHES_REVISION marker rather than a
# literal, so the string is stamped once here instead of in 26 places across the patch
# set, and a future upstream drop cannot carry a stale revision. The stamp below also
# catches a bare r<digits> a patch may still bring. Declared before LABEL so that
# ai.sglang.revision expands instead of going empty. Keep it in step with the tag.
ARG SGLANG_REVISION=r24

LABEL ai.sglang.base.commit="4ccff141dbe992794f9da6c3aa23535b4f72000d" \
      ai.sglang.patchset="0001-sm120-fp8-kv-cache,0002-sm120-gdn-recover-ssm,0003-sm120-online-fp8,0005-abort-ghost-fixes,0006-single-gpu-token-sync,0007-warm-grammar-path-before-serving,0008-qwen4-loader-release-refs,0020-serve-lil-checkpoint,0021-lil-loader-tp2-corrections,0022-lil-language-model-only,0023-vision-mxfp8-padding-marlin-bias,0024-qwen-flash-next-multimodal-alias,0025-packed-ple-host-storage" \
      ai.sglang.revision="${SGLANG_REVISION}" \
      ai.sglang.target.checkpoint="RadixArk/Qwen3.8-Flash-Next-NVFP4,nv-community/Qwen3.8-Flash-Next-NVFP4,local-inference-lab/Qwen3.8-Flash-Next-NVFP4" \
      ai.sglang.mtp="RecoverSSM-capable"

WORKDIR /sgl-workspace/sglang

# 0020-0025 are the ported local-inference-lab layers, numbered past 0008 on
# purpose: the sorted apply loop must run them after the upstream loader fixes.
# Kanadaj 0008/0011/0018 applied unchanged against this base; 0020 and part of
# 0022 were re-anchored. Two of kanadaj's hunks were dropped as already upstream:
# the MTP mixed-precision gate (qwen3_5_mtp.py answers from quantized_layers now)
# and the draft NVFP4 A16 support (W4A16_NVFP4 is a native algo here).
# 0025 is kanadaj 0010 re-anchored onto 0020's loader: the nvfp4 table stays packed in
# pinned host RAM and the gather kernel dequantizes a row at lookup, so boot never
# expands 51.2B elements. On by default for nvfp4 + pinned backend at TP1/TP2; the fp8
# expansion stays as the fallback for the file backend, TP>2, or SGLANG_PLE_PACKED_NVFP4=0.
# Not ported: kanadaj 0012 (loader-resolved checkpoint source for the PLE scale pre-read).
COPY patches/0001-sm120-fp8-kv-cache.patch /opt/qwen38-patches/0001-sm120-fp8-kv-cache.patch
COPY patches/0002-sm120-gdn-recover-ssm.patch /opt/qwen38-patches/0002-sm120-gdn-recover-ssm.patch
COPY patches/0003-sm120-online-fp8.patch /opt/qwen38-patches/0003-sm120-online-fp8.patch
COPY patches/0005-abort-ghost-fixes.patch /opt/qwen38-patches/0005-abort-ghost-fixes.patch
COPY patches/0006-single-gpu-token-sync.patch /opt/qwen38-patches/0006-single-gpu-token-sync.patch
COPY patches/0007-warm-grammar-path-before-serving.patch /opt/qwen38-patches/0007-warm-grammar-path-before-serving.patch
COPY patches/0008-qwen4-loader-release-refs.patch /opt/qwen38-patches/0008-qwen4-loader-release-refs.patch
COPY patches/0020-serve-lil-checkpoint.patch /opt/qwen38-patches/0020-serve-lil-checkpoint.patch
COPY patches/0021-lil-loader-tp2-corrections.patch /opt/qwen38-patches/0021-lil-loader-tp2-corrections.patch
COPY patches/0022-lil-language-model-only.patch /opt/qwen38-patches/0022-lil-language-model-only.patch
COPY patches/0023-vision-mxfp8-padding-marlin-bias.patch /opt/qwen38-patches/0023-vision-mxfp8-padding-marlin-bias.patch
COPY patches/0024-qwen-flash-next-multimodal-alias.patch /opt/qwen38-patches/0024-qwen-flash-next-multimodal-alias.patch
COPY patches/0025-packed-ple-host-storage.patch /opt/qwen38-patches/0025-packed-ple-host-storage.patch
COPY tests/check_loader_lifetime.py /opt/qwen38-tests/check_loader_lifetime.py

RUN set -eux; \
    cd /sgl-workspace/sglang; \
    for p in /opt/qwen38-patches/*.patch; do sed -i 's/\r$//' "$p"; done; \
    for p in $(ls /opt/qwen38-patches/*.patch | sort); do \
        echo "=== applying $(basename $p) ==="; \
        git apply --check "$p" || { echo "ERROR: $(basename $p) does not apply cleanly to the image tree"; exit 1; }; \
        git apply "$p"; \
    done; \
    grep -rl 'sm120-turbo' python/sglang | xargs -r sed -i -e "s/sm120-turbo SGLANG_PATCHES_REVISION/sm120-turbo ${SGLANG_REVISION}/g" -e "s/sm120-turbo r[0-9][0-9]*/sm120-turbo ${SGLANG_REVISION}/g"; \
    rm -rf /opt/qwen38-patches; \
    python3 /opt/qwen38-tests/check_loader_lifetime.py python/sglang/srt/models/qwen4_exp.py

# Load-time assertions that every patch landed. No GPU needed.
RUN python3 -c "import inspect, pathlib; \
from sglang.srt.layers.attention import qwen_sparse_attn_backend as q; \
src = pathlib.Path('python/sglang/srt/layers/attention/qsa/sparse_attn.py').read_text(); \
assert src.count('.to(q_values.dtype)') >= 4, 'patch 0001 fp8 prefill casts missing'; \
assert '_kv_cast' in inspect.getsource(q), 'patch 0001 chunked-prefill reread cast missing'; \
from sglang.srt.layers.attention.linear.kernels import gdn_flashinfer as g; \
gsrc = inspect.getsource(g.FlashInferGDNKernel); \
assert '_prefill_needs_fp32_state' in gsrc and 'query_start_loc.to(torch.int64)' in gsrc, 'patch 0002 SM120 GDN dtype fix missing'; \
from sglang.srt.layers.attention.linear import gdn_backend as gb; \
assert '_recover_ssm' in inspect.getsource(gb.GDNAttnBackend), 'patch 0002 RecoverSSM missing'; \
from sglang.kernels.ops.gemm import sm120_online_fp8 as ofp8; \
assert hasattr(ofp8, 'online_fp8_enabled'), 'patch 0003 online MXFP8 missing'; \
from sglang.srt.models.qwen4_exp import Qwen4ExpForConditionalGeneration as q4; \
assert 'nonlocal warned_ple_downcast' in inspect.getsource(q4.load_weights), 'patch 0008 loader lifetime fix missing'; \
from sglang.srt.layers import sampler as s; \
assert 'tp_sync_world_size' in inspect.getsource(s.Sampler), 'patch 0006 did not apply'; \
from sglang.srt.entrypoints.warmup import _warmup_registry as r; \
assert 'sm120_turbo_structured_output' in r, 'patch 0007 did not register the grammar warmup'; \
import sglang.srt.managers.scheduler as sc; \
assert 'self.abort_request(AbortReq(rid=req.rid))' in inspect.getsource(sc.Scheduler), 'patch 0005 did not apply'; \
from sglang.srt.utils.hf_transformers.common import _CONFIG_REGISTRY as lil; \
assert 'qwen3_8_flash_next' in lil and 'qwen3_8_flash_next_text' in lil, 'patch 0020 LIL config aliases missing'; \
msrc = pathlib.Path('python/sglang/srt/layers/quantization/modelopt_quant.py').read_text(); \
assert '_ple_source_packed' in inspect.getsource(q4.load_weights), 'patch 0020 packed nvfp4 PLE loader missing'; \
assert '_ple_resolve_global_scale' in inspect.getsource(q4.load_weights), 'patch 0020 nvfp4 global-scale pre-read missing'; \
assert getattr(q4, 'supports_visual_quantization', False) is True, 'patch 0021 visual quantization not enabled'; \
from sglang.srt.models.qwen3_vl import Qwen3VLForConditionalGeneration as vl; \
assert getattr(vl, 'supports_visual_quantization', True) is False, 'patch 0021 VL opt-in default missing'; \
assert 'quant_config if self.supports_visual_quantization else None' in inspect.getsource(vl.__init__), 'patch 0021 vision quant_config gate missing'; \
from sglang.srt.server_args import ServerArgs as sa; \
assert 'Qwen4ExpForConditionalGeneration' in sa.LANGUAGE_MODEL_ONLY_ARCHITECTURES, 'patch 0022 language-model-only allowlist missing'; \
assert 'qwen3_8_flash_next' in pathlib.Path('python/sglang/srt/layers/rotary_embedding/mrope_rope_index.py').read_text(), 'patch 0022 mrope model_type alias missing'; \
from sglang.srt.layers.quantization import vision_mxfp8 as vm; \
assert hasattr(vm, 'VisionMxfp8PaddedLinearMethod') and hasattr(vm, 'VisionNvFp4A16LinearMethod'), 'patch 0023 vision adapters missing'; \
assert 'VisionMxfp8PaddedLinearMethod(self.mxfp8_config)' in msrc and 'VisionNvFp4A16LinearMethod(self.nvfp4a16_config)' in msrc, 'patch 0023 vision branches not wired into get_quant_method'; \
assert pathlib.Path('python/sglang/srt/multimodal/processors/qwen_vl.py').read_text().count('qwen3_8_flash_next') >= 4, 'patch 0024 multimodal alias missing'; \
from sglang.srt.models import packed_ple as pp; \
assert hasattr(pp, 'PackedPLEStorage') and hasattr(pp, 'gather_packed_kernel'), 'patch 0025 packed PLE module missing'; \
from sglang.srt.models.qwen4_exp import Qwen4ExpPinnedHostEmbedding as phe, _ple_packed_host_wanted as pw; \
assert 'gather_packed_kernel' in inspect.getsource(phe.gather), 'patch 0025 gather does not dispatch to the packed kernel'; \
assert 'storage.finalize(' in inspect.getsource(q4.load_weights), 'patch 0025 loader does not retain packed shards'; \
assert pw(type('C', (), {'ple_embedding_dtype': 'nvfp4', 'ple_offload_embedding': True, 'ple_offload_backend': 'pinned'})()) is True, 'patch 0025 packed storage is not the default for nvfp4 + pinned'; \
assert not [q for q in pathlib.Path('python/sglang').rglob('*.py') if 'SGLANG_PATCHES_REVISION' in q.read_text()], 'revision marker survived the stamp'"
