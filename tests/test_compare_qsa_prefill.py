"""Control-flow tests for QSA prefill evidence persistence without CUDA."""

import contextlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


SCRIPT = Path(__file__).with_name("compare_qsa_prefill.py")


def load_script():
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True,
                             get_device_name=lambda: "mock-gpu"),
        backends=SimpleNamespace(cuda=SimpleNamespace(
            matmul=SimpleNamespace(allow_tf32=True))),
        float8_e4m3fn="float8",
        bfloat16="bfloat16",
        __version__="mock",
        inference_mode=contextlib.nullcontext,
    )
    with patch.dict(sys.modules, {"torch": fake_torch}):
        spec = importlib.util.spec_from_file_location("compare_qsa_prefill_test", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


class ReportPersistenceTests(unittest.TestCase):
    def setUp(self):
        self.module = load_script()
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.baseline = self.root / "baseline.py"
        self.candidate = self.root / "candidate.py"
        self.baseline.write_text("")
        self.candidate.write_text("")

    def argv(self, output):
        return ["compare_qsa_prefill.py", "--baseline-backend", str(self.baseline),
                "--candidate-backend", str(self.candidate), "--output", str(output)]

    def test_later_abort_keeps_partial_report_unqualified(self):
        output = self.root / "partial.json"
        with patch.object(sys, "argv", self.argv(output)), \
             patch.object(self.module, "load_backend", return_value=object()), \
             patch.object(self.module, "compare", side_effect=[
                 {"failures": []}, RuntimeError("later case aborts")]):
            with self.assertRaisesRegex(RuntimeError, "later case aborts"):
                self.module.main()
        report = json.loads(output.read_text())
        self.assertFalse(report["complete"])
        self.assertFalse(report["qualified"])
        self.assertEqual(len(report["cases"]), 1)

    def test_all_selected_cases_complete_before_qualification(self):
        output = self.root / "complete.json"
        with patch.object(sys, "argv", self.argv(output)), \
             patch.object(self.module, "load_backend", return_value=object()), \
             patch.object(self.module, "compare", return_value={"failures": []}):
            self.module.main()
        report = json.loads(output.read_text())
        self.assertTrue(report["complete"])
        self.assertTrue(report["qualified"])
        self.assertEqual(len(report["cases"]), 6)


if __name__ == "__main__":
    unittest.main()
