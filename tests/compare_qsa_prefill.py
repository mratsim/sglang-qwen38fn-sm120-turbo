"""Compare actual baseline/candidate QSA prefill methods without loading weights.

Run inside the pinned SGLang image with a GPU. Backend files must come from
the same base; only the candidate should include patch 0008. No server calls.
"""

import argparse
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import torch


def load_backend(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_case(lengths, dtype, query_tokens=2, query_heads=4, topk=16):
    # Current-chunk queries, 256-wide K/V, noncontiguous physical cache slots.
    assert query_tokens > 0 and all(length > query_tokens for length in lengths)
    assert query_heads % 2 == 0 and topk > 4
    torch.manual_seed(117)
    total = sum(lengths)
    slots = torch.randperm(total, device="cuda")
    k = (torch.randn(total, 2, 256, device="cuda", dtype=torch.bfloat16) / 4).to(dtype)
    v = (torch.randn(total, 2, 256, device="cuda", dtype=torch.bfloat16) / 4).to(dtype)
    mapping = torch.zeros(len(lengths), max(lengths), dtype=torch.int32, device="cuda")
    rows = []
    offset = 0
    for batch, length in enumerate(lengths):
        mapping[batch, :length] = slots[offset:offset + length].to(torch.int32)
        offset += length
        for query in range(query_tokens):
            # Include beginning, interior and last visible history positions,
            # plus masked entries; never select future tokens.
            selected = torch.linspace(
                0, length - query_tokens + query, topk - 4, device="cuda"
            ).long()
            rows.append(torch.cat((selected, selected.new_full((4,), -1))))
    indices = torch.stack(rows).to(torch.int32)
    q = torch.randn(len(lengths) * query_tokens, query_heads, 256,
                    dtype=torch.bfloat16, device="cuda")
    batch = SimpleNamespace(
        forward_mode=None,
        extend_seq_lens_cpu=[query_tokens] * len(lengths),
        seq_lens_cpu=lengths,
        extend_seq_lens=torch.tensor([query_tokens] * len(lengths), device="cuda"),
        req_pool_indices=torch.arange(len(lengths), device="cuda"),
    )
    layer = SimpleNamespace(layer_id=0, tp_q_head_num=query_heads,
                            head_dim=256, scaling=256 ** -0.5)
    # Supply only scheduler/cache metadata; the real gather, wrapper and GPU
    # attention kernel execute. Current-chunk K/V is already in this cache.
    backend = SimpleNamespace(
        token_to_kv_pool=SimpleNamespace(get_key_buffer=lambda _: k, get_value_buffer=lambda _: v),
        req_to_token_pool=SimpleNamespace(req_to_token=mapping),
        _is_speculative_paged_mode=lambda _: False,
        _pad_extend_output=lambda output, _: output,
    )
    return backend, q, k, v, layer, batch, indices, mapping


def reference(case):
    _, q, k, v, layer, batch, indices, mapping = case
    outputs = []
    query_tokens = batch.extend_seq_lens_cpu[0]
    groups = q.shape[1] // k.shape[1]
    for row in range(q.shape[0]):
        logical = indices[row][indices[row] >= 0].long()
        physical = mapping[row // query_tokens, logical].long()
        # Grouped FP32 reference avoids materializing repeated KV heads.
        keys = k.index_select(0, physical).float()
        values = v.index_select(0, physical).float()
        query = q[row].float().reshape(k.shape[1], groups, q.shape[2])
        scores = torch.einsum("hgd,khd->hgk", query, keys) * layer.scaling
        outputs.append(torch.einsum("hgk,khd->hgd", scores.softmax(-1), values)
                       .reshape(q.shape[1], q.shape[2]))
    return torch.stack(outputs).to(q.dtype)


def run(module, case):
    backend, q, k, v, layer, batch, indices, _ = case
    return module.QwenSparseAttnBackend.forward_extend(
        backend, q, k, v, layer, batch,
        save_kv_cache=False, topk_indices=indices,
    )


def measured(module, case):
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    initial = torch.cuda.memory_allocated()
    output = run(module, case)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - initial
    return output, peak


def compare(baseline, candidate, lengths, dtype, **geometry):
    case = make_case(lengths, dtype, **geometry)
    expected = reference(case)
    # Compile/warm each specialization before resetting allocator peaks.
    for module in (baseline, candidate):
        warm = run(module, case)
        torch.cuda.synchronize()
        del warm
    results = []
    failures = []
    # Reverse ordering to detect order-dependent allocator/measurement effects.
    for order in (("baseline", "candidate"), ("candidate", "baseline")):
        outputs, peaks, errors = {}, {}, {}
        for name in order:
            module = baseline if name == "baseline" else candidate
            output, peaks[name] = measured(module, case)
            try:
                torch.testing.assert_close(output, expected, rtol=3e-2, atol=3e-2)
            except AssertionError as error:
                failures.append(f"{order}: {name} reference: {error}")
            errors[name] = (output.float() - expected.float()).abs().max().item()
            outputs[name] = output.cpu()
            del output
        # Moving FP8->BF16 conversion should preserve arithmetic exactly on
        # this pinned kernel. Do not widen this merely to make a failure pass.
        try:
            torch.testing.assert_close(outputs["baseline"], outputs["candidate"], rtol=0, atol=0)
        except AssertionError as error:
            failures.append(f"{order}: exact equality: {error}")
        difference = (outputs["baseline"].float() - outputs["candidate"].float()).abs()
        if dtype == torch.float8_e4m3fn:
            # At least one full FP8 K+V history's bytes should be recovered.
            # This is a conservative bound, not a claim of half total VRAM.
            minimum_saving = sum(lengths) * 2 * 2 * 256
            assert peaks["baseline"] - peaks["candidate"] >= minimum_saving, peaks
        else:
            assert peaks["candidate"] <= peaks["baseline"], peaks
        results.append({"order": order, "peak_increment_bytes": peaks,
                        "max_abs_reference_error": errors,
                        "max_abs_baseline_candidate_error": difference.max().item(),
                        "mismatched_elements": int(torch.count_nonzero(difference).item()),
                        "total_elements": difference.numel()})
    return {"lengths": lengths, "cache_dtype": str(dtype), "geometry": geometry,
            "measurements": results, "failures": failures}


def write_report(result_file, report):
    result_file.seek(0)
    json.dump(report, result_file, indent=2)
    result_file.truncate()
    result_file.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-backend", type=Path, required=True)
    parser.add_argument("--candidate-backend", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--production-geometry", action="store_true",
                        help="4096 queries, 24 query heads, 2048 index slots; 128k/262k histories")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required; comparison was not run")
    if args.output.exists():
        raise SystemExit("Output already exists; preserve previous results")
    torch.backends.cuda.matmul.allow_tf32 = False
    baseline = load_backend(args.baseline_backend, "qsa_baseline")
    candidate = load_backend(args.candidate_backend, "qsa_candidate")
    report = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "test_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_sha256": {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in (("baseline", args.baseline_backend), ("candidate", args.candidate_backend))
        },
        "cases": [],
        "complete": False,
        "qualified": False,
        "scope": "Prefill gather/kernel only; no model, cache writes, MTP or server qualification",
    }
    geometry = ({"query_tokens": 4096, "query_heads": 24, "topk": 2048}
                if args.production_geometry else {})
    lengths_cases = ([131072], [262144]) if args.production_geometry else ([8192], [6144, 8192], [131072])
    # Persist progress even if a later case fails; never overwrite prior runs.
    with args.output.open("x") as result_file, torch.inference_mode():
        write_report(result_file, report)
        for dtype in (torch.float8_e4m3fn, torch.bfloat16):
            for lengths in lengths_cases:
                result = compare(baseline, candidate, lengths, dtype, **geometry)
                report["cases"].append(result)
                print(json.dumps(result), flush=True)
                write_report(result_file, report)
        report["complete"] = True
        report["qualified"] = not any(case["failures"] for case in report["cases"])
        write_report(result_file, report)
    if not report["qualified"]:
        raise SystemExit("FAIL: numerical assertions failed; diagnostics retained without relaxing tolerances")
    print("PASS: output equivalence, reference accuracy and allocation checks")


if __name__ == "__main__":
    main()
