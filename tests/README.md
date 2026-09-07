# QSA prefill allocation comparison

## Qualification update: production-geometry exact equality failed

The later September 6 test uses 4096 query rows, 24 query heads, 2 KV heads,
head dimension 256, and 2048 sparse-index slots (four masked), matching the
configured head geometry and indexer budget. It covers 131072- and 262144-token
cached histories. These remain synthetic cache/query tensors and index choices,
not the model's learned indexer outputs. Run the reproduction command below
with `--production-geometry` to select these cases.

The [production result](results/qsa-prefill-production-20260906.json) is a
**failed qualification**, with process exit status 1. Both FP8 cases fail exact
baseline/candidate equality in both execution orders. Do not treat the earlier
small-fixture exact-equality result as proof for production geometry.
This retained September 6 JSON predates the `complete` field and records the
previous runner hash. Its four selected production cases and `qualified: false`
remain historical numerical evidence; it does not validate the current
incremental-report state contract.

| History | Differing values / 25165824 | Maximum absolute difference | FP8 peak saving |
|---|---:|---:|---:|
| 131072 | 2690 | 0.00006103515625 | 268435456 bytes (256 MiB) |
| 262144 | 2867 | 0.00006103515625 | 536870912 bytes (512 MiB) |

Both methods passed the unchanged independent FP32 reference tolerance; maximum
reference errors were 0.0001220703125 at 128k and 0.000244140625 at 262k for FP8.
BF16 cases remained exactly equal and had unchanged allocation peaks. This is
evidence of small numerical differences specific to the changed FP8 path, not
proof of semantic harm or full-model correctness. The precise kernel/compiler
mechanism remains unresolved. The exact-equality gate is **not relaxed**.

The runner now records failed numerical assertions and continues through the
remaining cases so diagnostics are retained, then exits nonzero if any numerical
assertion failed. Partial reports retain `complete: false` and `qualified: false`
until every selected case finishes. Memory-saving assertions still fail
immediately. Previous output files are refused. The grouped FP32 reference avoids
duplicating KV heads but evaluates every query row. All six original small cases
were rerun and passed after this generalization. No assertion tolerance, expected
value, or case was weakened or removed.

A separate candidate server replay retrieved all four expected six-digit values
over a growing 261722–274143-token chat-formatted context, with prefix reuse and
no workload errors or retractions. That bounded retrieval pass does not supersede
this failed numerical gate. Detailed server artifacts and rollback state are in
the workstation bundle's `OPERATIONS-NOTES.md`. Keep the PR draft pending resolution.

Patch 0008 removes the full-history cache-to-query-dtype casts introduced by
patch 0001. The existing sparse kernel already converts loaded tiles before
arithmetic. Gathered and concatenated FP8 histories can therefore remain FP8.
The existing scale-1 KV assumption is unchanged. This does not remove history
gathering, alter decoding, or change the configured token pool.

## Recorded result

The September 6, 2026 comparison ran on an RTX PRO 6000 Blackwell using the
deployed r22 image with its existing Mamba correctness overlay:

`sha256:50e9346836250c97105ac9cc5bd6dc22642acc1ae797715f0c822da719b132a6`

The source base was `b3a0fbbb859408a82171aec4f61eb1e8c5786d93`. The candidate
backend was copied from that image and modified only by patch 0008. Both
methods used the installed, already-patched sparse kernel. Exact source hashes,
Torch version, and measurements are in
[the result JSON](results/qsa-prefill-20260906.json).

All six cases passed an independent FP32 attention reference comparison and
exact baseline/candidate output equality. Each case ran in both orders after
kernel warmup. BF16 allocation peaks were unchanged.
As a negative control, supplying the baseline file for both sides failed the
memory-saving assertion (33,560,064 bytes on each side), as expected.

| Cached histories | FP8 baseline peak increment | FP8 candidate peak increment |
|---|---:|---:|
| 8192 | 33,560,064 bytes | 16,782,848 bytes |
| 6144 + 8192 | 58,729,984 bytes | 29,369,856 bytes |
| 131072 | 536,876,544 bytes | 268,441,088 bytes |

These are process-local PyTorch peak allocated bytes above the pre-call
allocation baseline, not total physical VRAM or a full model's serving peak.
The fixture uses two query tokens per history, four query heads, two KV heads,
head dimension 256, noncontiguous physical cache slots, and masked sparse
indices. Minimal scheduler/cache metadata is supplied; actual backend gathering,
concatenation, wrapper and Triton kernel execute. Cache writes, model layers,
MTP, cold startup, and server scheduling are outside this test. The serving
process was left running; no throughput conclusion is drawn from this test.

## Reproduce

Use the same pinned image for dependencies and kernel implementation, with
the old and candidate backend files mounted separately. `BASELINE_IMAGE` must
be a compatible r22 image ID, and the candidate file must have patch 0008
applied. Do not use a different kernel version between the two methods.

```bash
docker run --rm --gpus 'device=0' --network none \
  --entrypoint python3 \
  -v "$PWD/tests:/qsa-tests:ro" \
  -v "$QSA_EVIDENCE_DIR:/qsa-evidence" \
  "$BASELINE_IMAGE" \
  /qsa-tests/compare_qsa_prefill.py \
  --baseline-backend /qsa-evidence/baseline.py \
  --candidate-backend /qsa-evidence/candidate.py \
  --production-geometry \
  --output /qsa-evidence/comparison.json
```

The script fails when CUDA is unavailable. GPU/JIT overhead requires spare
VRAM even though it does not load model weights. A failed assertion is not a
successful qualification; do not relax exact output equality merely to make
the candidate pass.

## Remaining server qualification

Build a separately tagged image from this branch and preserve any existing
deployment-specific correctness overlay. Do not change production defaults.
Compare baseline and candidate at identical launch settings, fixed benchmark
revision, matching actual prompts and cache conditions. Start with 128k/262k
requests if the baseline pool is only 332k. Use llm-inference-bench cold-prefill
mode plus an explicit repeated long-prefix/tool-turn continuation replay.
Record actual request lengths, prefix reuse, correctness, errors/retractions,
server info, and memory observations. Hardware polling can miss brief peaks.

Only after that comparison should memory fraction or concurrency be tuned for
400k/500k requests. The focused result does not establish safe operation at
0.96 or 0.97, recover a measured 800k pool, or qualify 786k contexts.
