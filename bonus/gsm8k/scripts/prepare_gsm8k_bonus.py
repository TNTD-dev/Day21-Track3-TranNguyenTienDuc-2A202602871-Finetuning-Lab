"""Curate an isolated GSM8K reasoning corpus for B2/B3; no model training.

Only tokenizer files are loaded. Never change the core CSKH dataset/results.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import random
import re
import shutil
import sys
import unicodedata
import urllib.request

DATASET = "openai/gsm8k"
REVISION = "740312add88f781978c0658806c59bc2815b9866"
MODEL = "unsloth/Qwen3.5-4B"
SYSTEM = 'Solve the math problem. Give your reasoning in the thinking block, then return one JSON object: {"answer": "number"}. No other text after the JSON.'
MAX_LENGTH = 512
# Exclusions were chosen by source/solution review before any model evaluation.
EXCLUSIONS = {"train": {3341, 7373, 5032}, "test": {389, 1313}}


def canonical_number(text):
    text = str(text).strip().replace(",", "").replace(" ", "")
    scalar = r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?"
    if not re.fullmatch(scalar + r"(?:/" + scalar + r")?", text):
        raise ValueError("Expected one numeric answer, not executable code or multiple answers.")
    parts = text.split("/")
    number = Fraction(parts[0])
    if len(parts) == 2:
        number /= Fraction(parts[1])
    return str(number.numerator) if number.denominator == 1 else f"{number.numerator}/{number.denominator}"


def normalized_problem(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).lower()).strip()


def family_key(text):
    return re.sub(r"\d+(?:[,.]\d+)*", "#", normalized_problem(text))


def convert(row, split, index):
    pieces = row["answer"].rsplit("####", 1)
    if len(pieces) != 2:
        raise ValueError("Missing GSM8K final-answer delimiter.")
    answer = canonical_number(pieces[1])
    reasoning = re.sub(r"<<[^<>]*>>", "", pieces[0]).strip()
    if len(reasoning.split()) < 12:
        raise ValueError("Reasoning is too short for the trace contrast.")
    question = row["question"].strip()
    if any(token in question + reasoning for token in ("<think>", "</think>", "<|im_start|>", "<|im_end|>")):
        raise ValueError("Unexpected chat control tags in source data.")
    output = f"<think>\n{reasoning}\n</think>\n" + json.dumps({"answer": answer})
    return {"id": hashlib.sha256(normalized_problem(question).encode()).hexdigest(),
            "input": question, "output": output, "label": {"answer": answer},
            "source_dataset": DATASET, "source_revision": REVISION,
            "source_split": split, "source_row": index,
            "reasoning": reasoning, "original_answer": row["answer"]}


def messages(record):
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": record["input"]},
            {"role": "assistant", "content": record["output"]}]


def select(rows, split, n, tokenizer, lab_data, used_ids, used_families, seed):
    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    selected = []
    for index in indices:
        if index in EXCLUSIONS.get(split, set()):
            continue
        try:
            record = convert(rows[index], split, index)
        except ValueError:
            continue
        family = family_key(record["input"])
        if record["id"] in used_ids or family in used_families:
            continue
        example = lab_data.build_example(tokenizer, messages(record), max_length=8192,
                                         mask_mode="assistant-only", enable_thinking=True)
        if len(example.input_ids) > MAX_LENGTH:
            continue
        response = lab_data.build_example(tokenizer, messages(record), max_length=8192,
                                          mask_mode="response-only", enable_thinking=True)
        if not 0 < response.n_supervised < example.n_supervised < len(example.input_ids):
            raise RuntimeError("The two masks do not form the intended contrast on this tokenizer.")
        record["n_tokens"] = len(example.input_ids)
        used_ids.add(record["id"])
        used_families.add(family)
        selected.append(record)
        if len(selected) == n:
            return selected
    raise RuntimeError(f"Only {len(selected)} suitable {split} examples found; expected {n}.")


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def prepare(root, output):
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    sys.path.insert(0, str(root / "src"))
    from labkit import data

    if (output / "results/dataset_manifest.json").exists():
        raise RuntimeError("Dataset already frozen here. Keep it; do not overwrite a completed experiment.")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    raw = {}
    for split in ("train", "test"):
        filename = f"main/{split}-00000-of-00001.parquet"
        path = hf_hub_download(DATASET, filename, repo_type="dataset", revision=REVISION)
        raw[split] = pq.read_table(path).to_pylist()
        print(f"Loaded {len(raw[split])} original {split} rows", flush=True)
    ids, families = set(), set()
    # Use the publisher's official test split for eval; deduplicate globally.
    evaluation = select(raw["test"], "test", 50, tokenizer, data, ids, families, 43)
    train_and_val = select(raw["train"], "train", 300, tokenizer, data, ids, families, 42)
    groups = {"train": train_and_val[:250], "val": train_and_val[250:], "eval_target": evaluation}
    (output / "data").mkdir(parents=True, exist_ok=True)
    (output / "results").mkdir(parents=True, exist_ok=True)
    checksums = {}
    for name, records in groups.items():
        path = output / "data" / f"{name}.jsonl"
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
        checksums[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    regression_source = root / "data/eval_regression.jsonl"
    regression_count = None
    if regression_source.is_file():
        regression = output / "data/eval_regression.jsonl"
        shutil.copy2(regression_source, regression)
        regression_count = len(regression.read_text(encoding="utf-8").splitlines())
        checksums[regression.name] = hashlib.sha256(regression.read_bytes()).hexdigest()
    sample = groups["train"][0]
    proofs = {}
    for mode in ("assistant-only", "response-only"):
        example = data.build_example(tokenizer, messages(sample), max_length=MAX_LENGTH,
                                     mask_mode=mode, enable_thinking=True)
        text = tokenizer.decode([token for token, label in zip(example.input_ids, example.labels)
                                 if label != data.IGNORE_INDEX], skip_special_tokens=False)
        proofs[mode] = {"n_supervised": example.n_supervised, "n_total": len(example.input_ids),
                        "supervised_text": text, "question_is_masked": sample["input"] not in text,
                        "answer_is_supervised": sample["label"]["answer"] in text}
    write_json(output / "results/mask_proof.json", proofs)
    stats = data.token_stats([r["n_tokens"] for records in groups.values() for r in records])
    write_json(output / "results/token_stats.json", stats)
    manifest = {"dataset": DATASET, "revision": REVISION, "license": "MIT", "model": MODEL,
                "sizes": {k: len(v) for k, v in groups.items()}, "max_length": MAX_LENGTH,
                "enable_thinking": True, "system_prompt": SYSTEM, "checksums": checksums,
                "exact_question_duplicates": 0, "numeric_template_duplicates": 0,
                "source_review_exclusions": {k: sorted(v) for k, v in EXCLUSIONS.items()},
                "pretraining_contamination": "Unknown; this public benchmark may have been in Qwen pretraining.",
                "original_CSKH_data_changed": False}
    if regression_count is not None:
        manifest["sizes"]["eval_regression"] = regression_count
    write_json(output / "results/dataset_manifest.json", manifest)
    review = ["# GSM8K bonus — sample review", "", "Review these examples before training.", ""]
    for i, record in enumerate(groups["train"][:10], 1):
        review += [f"## {i}. Original train row {record['source_row']}", "", record["input"], "",
                   record["reasoning"], "", f"Gold answer: `{record['label']['answer']}`", ""]
    (output / "data/REVIEW_SAMPLES.md").write_text("\n".join(review), encoding="utf-8")
    description = f'''# Custom dataset — GSM8K numeric reasoning subset

Source: https://huggingface.co/datasets/{DATASET}
Pinned revision: `{REVISION}`. License: MIT. Original authors: OpenAI GSM8K.
Official source documentation: https://github.com/openai/grade-school-math

Domain: English grade-school math word problems, adapted to reasoning traces followed
by a JSON numeric answer. The original human-written solution is retained; calculator
annotations `<<...>>` are removed as text, never executed. Answers after `####` are
normalized using exact rational arithmetic. Source split and row index are recorded.

Counts: 250 train, 50 validation, 50 eval. Eval comes only from the official test split.
Exact normalized questions and templates with numeric literals replaced are deduplicated
across all three sets before writing. This checks experimental leakage, not all semantic
paraphrases. SHA-256 checksums freeze the selected files before any training.

Before freezing, a manual review of 20 train and 10 eval candidates identified ambiguity,
inconsistent explanatory prose, and unit/wording issues. Excluded source train rows
3341, 7373, 5032 and test rows 389, 1313, replacing them via the same deterministic
sampling procedure. This is a sample review, not a claim that all 350 rows were manually
verified. See SOURCE_QUALITY_REVIEW.md for the actual findings.

Every rendered example is at most {MAX_LENGTH} tokens; none is clipped. Both mask modes
have a nonzero answer loss, exclude the question, and have different supervised spans.
See results/mask_proof.json and results/token_stats.json.

This is a learner-curated subset of published data, not original private data. It is a
different task/domain/response format from the core CSKH corpus. We cannot establish
that it is unseen by Qwen during pretraining; baseline results and this limitation must
be reported honestly. Inspect REVIEW_SAMPLES.md; source annotations are not a guarantee
of perfect labels or faithful model reasoning.
'''
    (output / "data/CUSTOM_DATASET.md").write_text(description, encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    print("DATASET READY: review the examples; no training was run.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.root.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
