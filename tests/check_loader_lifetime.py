"""Exercise the real load_weights body with a minimal parameter-swap fixture.

No GPU or checkpoint is required: this test checks Python reference lifetime,
not quantization numerics. The method body is compiled unchanged from the
provided SGLang source; only the surrounding model is a small fixture.
"""

import ast
import gc
import pathlib
import sys
import types
import typing
import weakref


class Parameter:
    pass


class OtherModule:
    pass


class Model:
    config = types.SimpleNamespace(num_experts=None)

    def __init__(self):
        self.weight = Parameter()

    def named_parameters(self, **kwargs):
        return [("lm_head.weight", self.weight)]

    def named_buffers(self):
        return []

    def named_modules(self):
        return [("", self)]

    def modules(self):
        return [self]

    def post_load_weights(self):
        self.weight = Parameter()

    def _log_weight_dtype_census(self):
        pass


source = pathlib.Path(sys.argv[1])
tree = ast.parse(source.read_text(encoding="utf-8"))
cls = next(
    n
    for n in tree.body
    if isinstance(n, ast.ClassDef) and n.name == "Qwen4ExpForConditionalGeneration"
)
method = next(
    n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "load_weights"
)
namespace = dict(
    torch=types.SimpleNamespace(Tensor=object),
    Iterable=typing.Iterable,
    Tuple=typing.Tuple,
    Set=typing.Set,
    Qwen4ExpNGramEmbedding=OtherModule,
    Qwen3_5GatedDeltaNet=OtherModule,
)
exec(
    compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace
)
gc.collect()
was_enabled = gc.isenabled()
gc.disable()
try:
    model = Model()
    old_weight = weakref.ref(model.weight)
    namespace["load_weights"](model, [])
    retained = old_weight() is not None
    gc.collect()
    retained_after_gc = old_weight() is not None
    print(dict(retained_after_load=retained, retained_after_gc=retained_after_gc))
    assert not retained, "load_weights retains the replaced parameter until cyclic GC"
finally:
    if was_enabled:
        gc.enable()
