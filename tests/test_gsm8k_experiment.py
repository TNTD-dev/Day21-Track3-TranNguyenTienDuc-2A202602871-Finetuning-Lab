"""Grade final answers without counting intermediate thinking numbers as answers."""
import importlib.util
from pathlib import Path

from labkit.evaluate import valid_reasoning_trace
import pytest

ROOT = Path(__file__).resolve().parents[1]
def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

experiment = load("gsm8k_bonus_experiment")
prep = load("prepare_gsm8k_bonus")


def test_known_prompt_opening_is_reconstructed_but_empty_trace_is_not_valid():
    raw = 'First add 4 and 5 to get 9.\n</think>\n{"answer":"9"}'
    assert experiment.final_answer(raw, True, prep.canonical_number) == ("9", True)
    assert valid_reasoning_trace(experiment.effective_completion(raw, True)) == 1
    empty = '</think>\n{"answer":"9"}'
    assert valid_reasoning_trace(experiment.effective_completion(empty, True)) == 0
    # No closing tag: do not score an intermediate JSON as a final answer.
    truncated = 'Maybe the answer is {"answer":"9"}. Let me reconsider.'
    assert experiment.final_answer(truncated, True, prep.canonical_number) == (None, False)


def test_target_and_format_are_separate_for_correct_prose_and_boxed_fraction():
    assert experiment.final_answer('Answer: {"answer":"0.5"}', False, prep.canonical_number) == ("1/2", False)
    assert experiment.final_answer('Final answer: \\boxed{\\frac{1}{2}}', False, prep.canonical_number) == ("1/2", False)
    assert experiment.final_answer('{"answer":"1/2","extra":42}', False, prep.canonical_number) == ("1/2", False)
    assert experiment.final_answer('{"answer":0.5}', False, prep.canonical_number) == ("1/2", False)
    assert experiment.final_answer('{"answer":true}', False, prep.canonical_number) == (None, False)


def test_invalid_dataset_checksum_is_rejected_before_gpu_work(tmp_path):
    (tmp_path / "results").mkdir()
    (tmp_path / "data").mkdir()
    (tmp_path / "data/train.jsonl").write_text('changed')
    experiment.write_json(tmp_path / "results/dataset_manifest.json",
                          {"checksums": {"train.jsonl": "wrong"}})
    with pytest.raises(RuntimeError, match="Frozen dataset changed"):
        experiment.validate(tmp_path)


def test_completed_baseline_is_not_recomputed_after_training(tmp_path, monkeypatch):
    path = tmp_path / "results/math_baselines_frozen.json"
    experiment.write_json(path, {"original": "before training"})
    monkeypatch.setattr(experiment, "load_base", lambda *a: pytest.fail("baseline must not reload a model"))
    experiment.baseline(tmp_path, {})
    assert experiment.read_json(path) == {"original": "before training"}
