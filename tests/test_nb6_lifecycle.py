"""Execute NB6's actual model lifecycle with a one-model memory budget."""
import ast
import gc
from pathlib import Path
from types import SimpleNamespace
import weakref


def test_nb6_releases_the_merged_base_before_loading_another(tmp_path):
    source = Path(__file__).resolve().parents[1] / "notebooks/06_merge_and_serve.py"
    tree = ast.parse(source.read_text())
    loads = [i for i, node in enumerate(tree.body)
             if isinstance(node, ast.Assign)
             and isinstance(node.value, ast.Call)
             and ast.unparse(node.value.func) == "generate.load_base"]
    assert len(loads) == 2, "the fixture must exercise both actual model loads"
    # Execute the full real call site from first load through adapter switching.
    fragment = ast.Module(body=tree.body[loads[0]:], type_ignores=[])
    live = weakref.WeakSet()
    calls = []

    class Base:
        def save_pretrained(self, path):
            pass

    class Adapter:
        def __init__(self, base, name="correct"):
            self.base = base

        @classmethod
        def from_pretrained(cls, base, path, adapter_name="correct"):
            return cls(base, adapter_name)

        def merge_and_unload(self):
            # Like PEFT, the old wrapper still refers to its underlying base.
            return self.base

        def load_adapter(self, path, adapter_name):
            pass

        def set_adapter(self, name):
            calls.append(name)

    def load_base(tier):
        gc.collect()
        assert not live, "NB6 loads a second base while the merged base is still alive"
        base = Base()
        live.add(base)
        return base, SimpleNamespace(save_pretrained=lambda path: None)

    for name in ["correct", "attn_only", "qlora"]:
        (tmp_path / "adapters" / name).mkdir(parents=True)
    namespace = {
        "ROOT": tmp_path, "TIER": object(), "target": [{"input": "ticket", "label": {}}],
        "PeftModel": Adapter,
        "generate": SimpleNamespace(load_base=load_base, free_memory=gc.collect,
                                    NAIVE_PROMPT="classify", generate_batch=lambda *a, **k: (["{}"], 1)),
        "ev": SimpleNamespace(triage_field_accuracy=lambda pred, label: 1),
        "report": SimpleNamespace(write_json=lambda *a, **k: None),
    }
    exec(compile(fragment, str(source), "exec"), namespace)
    assert calls == ["correct", "attn_only", "qlora"]
