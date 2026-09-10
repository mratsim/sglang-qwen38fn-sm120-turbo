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
# 0004 is not applied: this image supports the RadixArk and NVIDIA checkpoints;
# the local-inference-lab compatibility aliases are outside this stack.

FROM docker.io/lmsysorg/sglang:dev-cu13-qwen38-next-local@sha256:9d2a843c706c74bc259c0d9abf360551eb2734e1e7d255ab012a6965f10480b6

LABEL ai.sglang.base.commit="4ccff141dbe992794f9da6c3aa23535b4f72000d" \
      ai.sglang.patchset="0001-sm120-fp8-kv-cache,0002-sm120-gdn-recover-ssm,0003-sm120-online-fp8,0005-abort-ghost-fixes,0006-single-gpu-token-sync,0007-warm-grammar-path-before-serving,0008-qwen4-loader-release-refs" \
      ai.sglang.target.checkpoint="RadixArk/Qwen3.8-Flash-Next-NVFP4,nv-community/Qwen3.8-Flash-Next-NVFP4" \
      ai.sglang.mtp="RecoverSSM-capable"

WORKDIR /sgl-workspace/sglang

# 0004 is intentionally omitted because it only adapts the
# local-inference-lab checkpoint.
COPY patches/0001-sm120-fp8-kv-cache.patch /opt/qwen38-patches/0001-sm120-fp8-kv-cache.patch
COPY patches/0002-sm120-gdn-recover-ssm.patch /opt/qwen38-patches/0002-sm120-gdn-recover-ssm.patch
COPY patches/0003-sm120-online-fp8.patch /opt/qwen38-patches/0003-sm120-online-fp8.patch
COPY patches/0005-abort-ghost-fixes.patch /opt/qwen38-patches/0005-abort-ghost-fixes.patch
COPY patches/0006-single-gpu-token-sync.patch /opt/qwen38-patches/0006-single-gpu-token-sync.patch
COPY patches/0007-warm-grammar-path-before-serving.patch /opt/qwen38-patches/0007-warm-grammar-path-before-serving.patch
COPY patches/0008-qwen4-loader-release-refs.patch /opt/qwen38-patches/0008-qwen4-loader-release-refs.patch
COPY tests/check_loader_lifetime.py /opt/qwen38-tests/check_loader_lifetime.py

RUN set -eux; \
    cd /sgl-workspace/sglang; \
    for p in /opt/qwen38-patches/*.patch; do sed -i 's/\r$//' "$p"; done; \
    for p in $(ls /opt/qwen38-patches/*.patch | sort); do \
        echo "=== applying $(basename $p) ==="; \
        git apply --check "$p" || { echo "ERROR: $(basename $p) does not apply cleanly to the image tree"; exit 1; }; \
        git apply "$p"; \
    done; \
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
 print('next-local: all patches verified')"
