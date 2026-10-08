"""B4: train ranks 8/64 and reuse the existing rank-16 core run.

Run from the lab repo on Colab: python /content/bonus_rank_sweep.py
Only bonus adapters/results are written. Each GPU operation uses a fresh process.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import runpy
import subprocess
import sys


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def context(root):
    required = ["results/baselines_frozen.json", "results/runs.csv", "results/verdict.json",
                "adapters/correct/adapter_config.json", "adapters/correct/adapter_model.safetensors",
                "data/split/train.jsonl", "data/split/val.jsonl"]
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise RuntimeError(
            "Saved core files are missing:\n  " + "\n  ".join(missing)
            + "\nRestore the full core ZIP in this repo before running B4. "
            "A downloaded .ipynb contains logs, but not results/ or adapter weights. "
            "No bonus training has started."
        )
    frozen = read_json(root / "results/baselines_frozen.json")
    if frozen.get("smoke_mode"):
        raise RuntimeError("Finish the full core evaluation before B4.")
    with (root / "results/runs.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    core = next(r for r in reversed(rows) if r["run"] == "correct")
    cfg = read_json(root / "adapters/correct/adapter_config.json")
    if (int(core["r"]) != 16 or cfg["r"] != 16 or cfg["lora_alpha"] != 32
            or core["placement"] != "text-linear"
            or core["load_in_4bit"].lower() != "false"):
        raise RuntimeError("B4 requires the existing 16-bit text-linear rank-16 run.")
    if core["model"] != frozen["model"]:
        raise RuntimeError("Core adapter and baselines use different models.")
    paths = ["data/split/train.jsonl", "data/split/val.jsonl",
             "data/eval_target.jsonl", "data/eval_regression.jsonl",
             "results/baselines_frozen.json", "results/verdict.json", "results/runs.csv",
             "adapters/correct/adapter_config.json", "adapters/correct/adapter_model.safetensors",
             "notebooks/03_train_correct.py", "notebooks/05_evaluate_and_verdict.py"]
    paths += [str(p.relative_to(root)) for p in sorted((root / "src/labkit").glob("*.py"))]
    hashes = {name: digest(root / name) for name in paths}
    for group, field in [("target", "n_target"), ("regression", "n_regression")]:
        count = len((root / f"data/eval_{group}.jsonl").read_text().splitlines())
        if count != frozen[field]:
            raise RuntimeError(f"The {group} set differs from the frozen full evaluation.")
    payload = {"core": core, "hashes": hashes}
    payload["fingerprint"] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return payload


def require_same(saved, current):
    if saved.get("fingerprint") != current["fingerprint"]:
        raise RuntimeError("Existing bonus evidence uses a different core/config/dataset. "
                           "Keep it separate; do not mix experiments.")


def configure(root, ctx):
    sys.path.insert(0, str(root / "src"))
    core = ctx["core"]
    os.environ.update(COMPUTE_TIER=core["tier"], BASE_MODEL=core["model"],
                      MASK_MODE=core.get("mask_mode") or "assistant-only", EVAL_LIMIT="0")
    from labkit.config import get_tier
    tier = get_tier()
    n = len((root / "data/split/train.jsonl").read_text().splitlines())
    steps = int(core["max_steps"])
    per_epoch = math.ceil(n / tier.effective_batch)
    # Reuse NB3's epoch recipe only when it reproduces the recorded step budget exactly.
    epochs = steps / per_epoch
    if not epochs.is_integer():
        raise RuntimeError("The existing step budget is not an integer number of epochs.")
    os.environ["EPOCHS"] = str(int(epochs))
    return tier


def train_rank(root, ctx, rank):
    from dataclasses import replace
    configure(root, ctx)
    from labkit import report
    from labkit.config import SPECS
    if float(ctx["core"]["learning_rate"]) != SPECS["correct"].lr:
        raise RuntimeError("The core LR differs from the NB3 recipe.")
    key = f"bonus_rank_{rank}"
    SPECS["correct"] = replace(SPECS["correct"], key=key, r=rank, alpha=2 * rank,
                               label=f"text-linear · r={rank} · controlled rank sweep")
    original_append = report.append_row

    def append_bonus(row, filename="runs.csv", results_dir=None):
        return original_append(row, filename="bonus_rank_runs.csv", results_dir=results_dir)

    report.append_row = append_bonus
    state = runpy.run_path(str(root / "notebooks/03_train_correct.py"), run_name="__main__")
    if state["STEPS"] != int(ctx["core"]["max_steps"]):
        raise RuntimeError("Rank run did not use the core step budget.")
    report.write_json(ctx, "bonus_context.json", results_dir=root / "adapters" / key)


def score_rank(root, ctx, rank):
    configure(root, ctx)
    # Reuse the actual NB5 scorer, stopping before it writes the core verdict/autopsy.
    path = root / "notebooks/05_evaluate_and_verdict.py"
    source = path.read_text(encoding="utf-8").split(
        "scores_ft, preds_ft, rpreds_ft = score_adapter", 1)[0]
    if "def score_adapter(" not in source:
        raise RuntimeError("NB5 scorer was not found.")
    ns = {"__name__": "__bonus_rank_eval__", "__file__": str(path)}
    exec(compile(source, str(path), "exec"), ns)
    scores, preds, regression_preds = ns["score_adapter"](
        root / f"adapters/bonus_rank_{rank}", ns["generate"].NAIVE_PROMPT,
        label=f"rank-{rank}")
    ns["report"].write_json(
        {"rank": rank, "fingerprint": ctx["fingerprint"], "scores": scores.as_dict(),
         "target_predictions": preds, "regression_predictions": regression_preds},
        f"bonus_rank_{rank}.json", results_dir=root / "results")


def checkpoint(root, destination, stage):
    if destination is None:
        return
    sys.path.insert(0, str(root / "scripts"))
    from colab_run_persistent import save_completed
    print(f"Saving B4 checkpoint ({stage}) to Drive...", flush=True)
    destination.mkdir(parents=True, exist_ok=True)
    save_completed(root, destination)
    progress = destination / "bonus_rank_progress.json"
    saved = read_json(progress) if progress.exists() else {"completed": []}
    if stage not in saved["completed"]:
        saved["completed"].append(stage)
    progress.write_text(json.dumps(saved, indent=2), encoding="utf-8")
    print(f"B4 BACKUP SAVED: {stage}", flush=True)


def sweep(root, ctx, backup_dir=None):
    script = Path(__file__).resolve()
    for rank in (8, 64):
        key = f"bonus_rank_{rank}"
        adir = root / "adapters" / key
        if (adir / "adapter_model.safetensors").exists():
            require_same(read_json(adir / "bonus_context.json"), ctx)
            print(f"Resume: {key} already trained", flush=True)
        else:
            subprocess.run([sys.executable, "-u", str(script), "--train", str(rank)],
                           cwd=root, check=True)
        checkpoint(root, backup_dir, f"rank-{rank}-train")
        metrics = root / "results" / f"{key}.json"
        if metrics.exists():
            require_same(read_json(metrics), ctx)
        else:
            subprocess.run([sys.executable, "-u", str(script), "--score", str(rank)],
                           cwd=root, check=True)
        require_same(context(root), ctx)
        checkpoint(root, backup_dir, f"rank-{rank}-eval")

    with (root / "results/bonus_rank_runs.csv").open(encoding="utf-8") as f:
        trained = {r["run"]: r for r in csv.DictReader(f)}
    core_scores = read_json(root / "results/verdict.json")["comparison"][2]
    table = []
    for rank in (8, 16, 64):
        run = ctx["core"] if rank == 16 else trained[f"bonus_rank_{rank}"]
        scores = core_scores if rank == 16 else read_json(
            root / f"results/bonus_rank_{rank}.json")["scores"]
        table.append({"rank": rank, "alpha": 2 * rank, "placement": "text-linear",
                      "lr": float(run["learning_rate"]), "max_steps": int(run["max_steps"]),
                      "trainable_params": int(run["trainable_params"]),
                      "train_loss": float(run["final_loss"]),
                      "peak_vram_gb": float(run["peak_vram_gb"]),
                      **{k: scores[k] for k in ["target", "regression", "format", "latency_ms", "n"]}})
    from labkit import report
    report.write_json({"context": ctx, "rank16_reused_from_core": True, "comparison": table},
                      "bonus_rank_sweep.json", results_dir=root / "results")
    checkpoint(root, backup_dir, "comparison")
    print(report.markdown_table(table), flush=True)
    print("B4 finished: results/bonus_rank_sweep.json", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--train", type=int, choices=(8, 64))
    group.add_argument("--score", type=int, choices=(8, 64))
    parser.add_argument("--backup-dir", type=Path,
                        help="Save completed B4 operations outside the temporary VM.")
    args = parser.parse_args()
    root = Path.cwd()
    if not (root / "src/labkit").is_dir():
        raise RuntimeError("Run from the lab repo directory, after completing NB1–NB5.")
    ctx = context(root)
    configure(root, ctx)
    if args.train:
        train_rank(root, ctx, args.train)
    elif args.score:
        score_rank(root, ctx, args.score)
    else:
        if args.backup_dir:
            destination = args.backup_dir.resolve()
            if destination == root or root in destination.parents:
                raise RuntimeError("Backup must be outside the repo, on mounted Drive.")
        sweep(root, ctx, args.backup_dir)


if __name__ == "__main__":
    main()
