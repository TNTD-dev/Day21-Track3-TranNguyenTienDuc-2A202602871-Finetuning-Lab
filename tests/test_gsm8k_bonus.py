"""Corpus preparation must reject bad labels and separate number-swapped templates."""
import importlib.util
from pathlib import Path

import pytest

script = Path(__file__).resolve().parents[1] / "scripts/prepare_gsm8k_bonus.py"
spec = importlib.util.spec_from_file_location("prepare_gsm8k_bonus", script)
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)


def test_equivalent_numeric_labels_share_one_exact_value():
    assert {prep.canonical_number(v) for v in ["0.5", "1/2", "5/10", "5e-1"]} == {"1/2"}
    assert prep.canonical_number("1,234") == "1234"
    with pytest.raises(ValueError):
        prep.canonical_number("__import__('os').system('anything')")
    with pytest.raises(ValueError):
        prep.canonical_number("3; 4")


def test_converting_human_solution_preserves_gold_without_executing_annotations():
    row = {"question": "How many clips were sold?", "answer":
           "April had 48 clips. May had half as many, <<48/2=24>>24 clips. "
           "Together, there were <<48+24=72>>72 clips.\n#### 72"}
    record = prep.convert(row, "train", 42)
    assert record["label"] == {"answer": "72"}
    assert "<<" not in record["reasoning"]
    assert record["output"].startswith("<think>\n")
    assert record["output"].endswith('{"answer": "72"}')
    assert record["source_row"] == 42
    assert record["original_answer"] == row["answer"]


def test_number_swapped_templates_cannot_cross_splits():
    assert prep.family_key("Ann buys 24 apples at $2.50 each.") == prep.family_key(
        "Ann buys 96 apples at $5.00 each.")
    # Different questions are not silently collapsed just because numbers match.
    assert prep.family_key("Ann buys 24 apples.") != prep.family_key("Ann sells 24 oranges.")


def test_bad_source_answer_is_rejected_instead_of_guessed():
    with pytest.raises(ValueError, match="delimiter"):
        prep.convert({"question": "What is 3+4?", "answer": "The answer may be seven."}, "test", 1)
