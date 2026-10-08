"""B2/B3 isolated math experiment: baseline first, then two matched mask runs.

Run on Colab with --home pointing to the prepared bonus_gsm8k folder on Drive.
Each GPU operation runs in a fresh process; only adapters are exported.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import importlib.util
from importlib import metadata
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

MODES = ("assistant-only", "response-only")
NAIVE = "Solve the math problem and give the final numeric answer."
OPTIMIZED_EXTRA = '''
The final JSON must have exactly one key, answer, whose value is a number written as
a string. Do not put intermediate values in the final JSON. Check arithmetic before
finishing. For example: 3 pens at 4 dollars each gives {"answer": "12"}; half of
18 apples gives {"answer": "9"}. After the thinking block output only the JSON.
'''


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def library(home):
    # The core backup contains the exact labkit used by the student's core run.
    src = home.parent / "src"
    if not (src / "labkit").is_dir():
        raise RuntimeError("The core src/labkit backup is missing next to bonus_gsm8k.")
    sys.path.insert(0, str(src))
    from labkit import data, device, evaluate, generate, modeling, train
    return data, device, evaluate, generate, modeling, train


def prep_module(home):
    path = home / "scripts/prepare_gsm8k_bonus.py"
    spec = importlib.util.spec_from_file_location("gsm8k_preparation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def effective_completion(raw, prompt_open):
    # Qwen's generation prompt can already contain the opening tag. Reconstruct
    # that known prefix for the trace metric; never invent a body or closing tag.
    if prompt_open and not raw.lstrip().startswith("<think>"):
        return "<think>\n" + raw
    return raw


def boxed_values(text):
    values = []
    for match in re.finditer(r"\\boxed\{", text):
        start, depth = match.end(), 1
        for index in range(start, len(text)):
            if text[index] == "{": depth += 1
            elif text[index] == "}": depth -= 1
            if depth == 0:
                value = text[start:index]
                fraction = re.fullmatch(r"\\(?:frac|dfrac|tfrac)\{([+-]?\d+)\}\{([+-]?\d+)\}", value.strip())
                if fraction:
                    value = f"{fraction[1]}/{fraction[2]}"
                values.append(re.sub(r"\\(?:text|mathrm)\{[^{}]*\}", "", value).strip())
                break
    return values


def final_answer(raw, prompt_open, canonical):
    text = effective_completion(raw, prompt_open)
    if "<think>" in text:
        if "</think>" not in text:
            return None, False
        text = text.rsplit("</think>", 1)[-1].strip()
    else:
        text = text.strip()
    def value(obj):
        if not isinstance(obj, dict) or "answer" not in obj or isinstance(obj["answer"], bool):
            raise ValueError("Not a numeric answer object")
        return canonical(obj["answer"])
    try:
        obj = json.loads(text)
        answer = value(obj)
        return answer, set(obj) == {"answer"} and isinstance(obj["answer"], str)
    except (ValueError, TypeError, ZeroDivisionError):
        pass
    # Target can recover a JSON embedded in prose, while format remains failed.
    found = None
    for match in re.finditer(r"\{", text):
        try:
            obj, _ = json.JSONDecoder().raw_decode(text[match.start():])
            found = value(obj)
        except (ValueError, TypeError, ZeroDivisionError):
            continue
    if found is not None:
        return found, False
    # Naive base prompts often produce a boxed numeric answer rather than JSON.
    boxes = boxed_values(text)
    if boxes:
        try:
            return canonical(boxes[-1]), False
        except (ValueError, ZeroDivisionError):
            pass
    try:
        return canonical(text.strip("$* ")), False
    except (ValueError, ZeroDivisionError):
        pass
    marked = re.findall(r"(?:final answer|answer is|answer:|####)\s*[:=]?\s*(?:\*\*)?\s*\$?\s*([+-]?[\d,.]+(?:/\d+)?)", text, re.I)
    if marked:
        try:
            return canonical(marked[-1]), False
        except (ValueError, ZeroDivisionError):
            pass
    return None, False


def read_rows(home, name):
    return [json.loads(line) for line in (home / "data" / name).read_text().splitlines() if line.strip()]


def validate(home):
    manifest = read_json(home / "results/dataset_manifest.json")
    for name, expected in manifest["checksums"].items():
        if sha(home / "data" / name) != expected:
            raise RuntimeError(f"Frozen dataset changed: {name}")
    if manifest["sizes"] != {"train": 250, "val": 50, "eval_target": 50, "eval_regression": 15}:
        raise RuntimeError("Expected the full prepared dataset, not a smoke slice.")
    return manifest


def configuration(home, manifest):
    path = home / "results/math_run_config.json"
    versions = {}
    for name in ("torch", "transformers", "trl", "peft", "accelerate", "datasets", "tokenizers"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            raise RuntimeError(f"Missing {name}; install the core requirements.txt first.")
    source_hashes = {p.name: sha(p) for p in sorted((home.parent / "src/labkit").glob("*.py"))}
    if path.exists():
        cfg = read_json(path)
        if (cfg["data_checksums"] != manifest["checksums"] or cfg["runner_sha256"] != sha(__file__)
                or cfg.get("package_versions") != versions or cfg.get("labkit_hashes") != source_hashes):
            raise RuntimeError("Dataset or experiment code changed after configuration was frozen.")
        return cfg
    if any((home / "adapters" / m).exists() for m in MODES):
        raise RuntimeError("Bonus adapters exist without the frozen baseline configuration.")
    from huggingface_hub import HfApi
    info = HfApi().model_info(manifest["model"])
    cfg = {"model": manifest["model"], "model_revision": info.sha,
           "data_checksums": manifest["checksums"], "runner_sha256": sha(__file__),
           "package_versions": versions, "labkit_hashes": source_hashes,
           "train_system": manifest["system_prompt"], "naive_system": NAIVE,
           "optimized_system": manifest["system_prompt"] + OPTIMIZED_EXTRA,
           "max_length": manifest["max_length"], "max_new_tokens": 512,
           "rank": 16, "alpha": 32, "lr": 1e-4, "epochs": 2, "seed": 42,
           "enable_thinking": True, "regression_enable_thinking": False}
    write_json(path, cfg)
    return cfg


def load_base(home, cfg):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    _, device, _, _, _, _ = library(home)
    if not torch.cuda.is_available():
        raise RuntimeError("Select a T4 GPU runtime before this operation.")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"], revision=cfg["model_revision"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"], revision=cfg["model_revision"], dtype=device.torch_dtype(),
        device_map="auto", trust_remote_code=True)
    model.eval()
    return model, tokenizer


def score(home, cfg, model, tokenizer, system, label):
    _, _, ev, gen, _, _ = library(home)
    canonical = prep_module(home).canonical_number
    rows = read_rows(home, "eval_target.jsonl")
    preds, latency = gen.generate_batch(
        model, tokenizer, [r["input"] for r in rows], system=system,
        max_new_tokens=cfg["max_new_tokens"], enable_thinking=True, batch_size=2, label=label)
    judged = []
    for row, pred in zip(rows, preds):
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": row["input"]}]
        prefix = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True)
        opened = prefix.rfind("<think>") > prefix.rfind("</think>")
        answer, fmt = final_answer(pred, opened, canonical)
        traced = effective_completion(pred, opened)
        judged.append({"id": row["id"], "question": row["input"], "gold": row["label"]["answer"],
                       "raw_completion": pred, "opening_tag_in_prompt": opened,
                       "predicted_answer": answer, "correct": answer == row["label"]["answer"],
                       "format": bool(fmt), "valid_trace": bool(ev.valid_reasoning_trace(traced))})
    regression = read_rows(home, "eval_regression.jsonl")
    rpreds, _ = gen.generate_batch(model, tokenizer, [r["instruction"] for r in regression],
                                 system=None, enable_thinking=False, max_new_tokens=96, batch_size=2,
                                 label=label + "/regression")
    scores = {"target": sum(r["correct"] for r in judged) / len(judged),
              "format": sum(r["format"] for r in judged) / len(judged),
              "valid_trace_rate": sum(r["valid_trace"] for r in judged) / len(judged),
              "regression": sum(ev.keyword_recall(p, r["keywords"]) for p, r in zip(rpreds, regression)) / len(regression),
              "latency_ms": latency, "n_target": len(rows), "n_regression": len(regression)}
    print(label, json.dumps(scores, indent=2), flush=True)
    return {"scores": scores, "predictions": judged, "regression_predictions": rpreds}


def baseline(home, cfg):
    path = home / "results/math_baselines_frozen.json"
    if path.exists():
        print("Baselines already frozen; reusing them.", flush=True)
        return
    model, tokenizer = load_base(home, cfg)
    partial_a = home / "results/math_base_naive.json"
    partial_b = home / "results/math_base_optimized.json"
    if partial_a.exists():
        a = read_json(partial_a)
    else:
        a = score(home, cfg, model, tokenizer, cfg["naive_system"], "base-naive")
        write_json(partial_a, a)
        print("BASE NAIVE SAVED on Drive", flush=True)
    if partial_b.exists():
        b = read_json(partial_b)
    else:
        b = score(home, cfg, model, tokenizer, cfg["optimized_system"], "base-optimized")
        write_json(partial_b, b)
        print("BASE OPTIMIZED SAVED on Drive", flush=True)
    # Use the stronger target baseline even when JSON formatting instructions do
    # not improve mathematical accuracy. Never flatter FT with a weaker reference.
    reference = "baseline_b" if b["scores"]["target"] >= a["scores"]["target"] else "baseline_a"
    write_json(path, {"configuration": cfg, "baseline_a": a, "baseline_b": b,
                      "target_reference": reference, "captured_before_bonus_training": True})
    print("MATH BASELINES FROZEN on Drive:", path, flush=True)


def train_mask(home, cfg, mode):
    if not (home / "results/math_baselines_frozen.json").is_file():
        raise RuntimeError("Run baseline before either mask training run.")
    if (home / "adapters" / mode / "adapter_model.safetensors").is_file():
        if not (home / "results" / (mode + "_train.json")).exists():
            raise RuntimeError("Adapter exists without training metadata; keep it separate and investigate.")
        print("Already trained:", mode, flush=True)
        return
    from transformers import AutoTokenizer, set_seed
    from datasets import Dataset
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer
    data, device, _, gen, modeling, train = library(home)
    from labkit.config import SPECS, get_tier
    set_seed(cfg["seed"])
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"], revision=cfg["model_revision"])
    preparation = prep_module(home)
    rows = []
    for record in read_rows(home, "train.jsonl"):
        rendered_messages = preparation.messages(record)
        if rendered_messages[0]["content"] != cfg["train_system"]:
            raise RuntimeError("Training and evaluation prompts are not aligned.")
        example = data.build_example(tokenizer, rendered_messages, max_length=8192,
                                     mask_mode=mode, enable_thinking=True)
        if len(example.input_ids) > cfg["max_length"] or example.n_supervised == 0:
            raise RuntimeError("A full example would be clipped or has zero answer loss.")
        rows.append({"input_ids": example.input_ids, "labels": example.labels,
                     "attention_mask": [1] * len(example.input_ids)})
    tier = replace(get_tier("T4"), model_id=cfg["model"], max_length=cfg["max_length"])
    spec = replace(SPECS["correct"], key=mode, r=cfg["rank"], alpha=cfg["alpha"], lr=cfg["lr"])
    steps = train.planned_steps(len(rows), tier, cfg["epochs"])
    model, _ = load_base(home, cfg)
    targets = modeling.resolve_target_modules(model, spec.target)
    count = modeling.count_lora_params(model, targets, spec.r)
    local = Path("/kaggle/working/lab21_math_work") / sha(home / "results/math_run_config.json")[:12] / mode
    desired = train.sft_config_kwargs(tier, spec, str(local), max_steps=steps, total_steps=steps,
                                      num_train_epochs=cfg["epochs"], mask_mode=mode)
    kwargs, _ = train.filter_kwargs(SFTConfig, desired, label=mode)
    lora, _ = train.filter_kwargs(LoraConfig, train.lora_config_kwargs(spec, targets), label=mode)
    trainer = SFTTrainer(model=model, args=SFTConfig(**kwargs), train_dataset=Dataset.from_list(rows),
                         processing_class=tokenizer, peft_config=LoraConfig(**lora))
    train.align_trainable_precision(trainer.model)
    start = time.perf_counter()
    result = trainer.train()
    elapsed = time.perf_counter() - start
    trainer.model.save_pretrained(local)
    tokenizer.save_pretrained(local)
    destination = home / "adapters" / mode
    print("Saving completed adapter to Drive:", destination, flush=True)
    shutil.copytree(local, destination, dirs_exist_ok=True, ignore=shutil.ignore_patterns("checkpoint-*"))
    row = train.summarize_run(spec, tier, targets, count, elapsed, gen.peak_vram_gb())
    row.update(mask_mode=mode, max_steps=steps, seed=cfg["seed"], final_loss=result.training_loss)
    write_json(home / "results" / (mode + "_train.json"), row)
    print("TRAIN SAVED:", mode, "steps:", steps, flush=True)


def eval_mask(home, cfg, mode):
    from peft import PeftModel
    model, tokenizer = load_base(home, cfg)
    model = PeftModel.from_pretrained(model, str(home / "adapters" / mode))
    model.eval()
    result = score(home, cfg, model, tokenizer, cfg["train_system"], mode)
    write_json(home / "results" / (mode + "_eval.json"), result)
    print("EVAL SAVED:", mode, flush=True)


def summary(home, cfg):
    frozen = read_json(home / "results/math_baselines_frozen.json")
    baseline_scores = frozen[frozen["target_reference"]]["scores"]
    table = []
    for name in ("baseline_a", "baseline_b"):
        table.append({"run": name, **frozen[name]["scores"]})
    for mode in MODES:
        scores = read_json(home / "results" / (mode + "_eval.json"))["scores"]
        metadata = read_json(home / "results" / (mode + "_train.json"))
        target_delta = scores["target"] - baseline_scores["target"]
        regression_delta = scores["regression"] - baseline_scores["regression"]
        table.append({"run": mode, **scores, "max_steps": metadata["max_steps"],
                      "target_delta": target_delta, "regression_delta": regression_delta,
                      "passed": target_delta > 0 and regression_delta >= -0.02})
    if table[2]["max_steps"] != table[3]["max_steps"]:
        raise RuntimeError("The mask contrast has mismatched training budgets.")
    result = {"configuration": cfg, "dataset": validate(home), "comparison": table,
              "target_reference": frozen["target_reference"],
              "trace_metric": "nonempty closed think block; opening tag from actual generation prompt reconstructed when necessary",
              "limitation": "Public GSM8K may be in pretraining; valid_trace_rate is not proof of faithful mathematical reasoning."}
    write_json(home / "results/math_bonus_summary.json", result)
    print(json.dumps(table, ensure_ascii=False, indent=2), flush=True)
    print("B2/B3 EXPERIMENT FINISHED:", home / "results/math_bonus_summary.json", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", type=Path, required=True)
    parser.add_argument("--stage", choices=("baseline", "train-assistant", "eval-assistant", "train-response", "eval-response", "summary", "all"), required=True)
    args = parser.parse_args()
    home = args.home.resolve()
    manifest = validate(home)
    library(home)
    cfg = configuration(home, manifest)
    if args.stage == "all":
        for stage in ("baseline", "train-assistant", "eval-assistant", "train-response", "eval-response", "summary"):
            subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()), "--home", str(home), "--stage", stage], check=True)
        return
    if args.stage == "baseline": baseline(home, cfg)
    elif args.stage == "train-assistant": train_mask(home, cfg, MODES[0])
    elif args.stage == "train-response": train_mask(home, cfg, MODES[1])
    elif args.stage == "eval-assistant": eval_mask(home, cfg, MODES[0])
    elif args.stage == "eval-response": eval_mask(home, cfg, MODES[1])
    else: summary(home, cfg)


if __name__ == "__main__":
    main()
