"""Run core stages and save completed work outside the temporary Colab VM.

Usage: python scripts/colab_run_persistent.py --backup-dir /content/drive/MyDrive/... nb1 nb2 nb3 nb4 nb5
The destination must be unique per experiment. Only successfully completed stages
update the backup; a failed stage keeps the preceding completed backup intact.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys


STAGES = {f"nb{i}": path for i, path in enumerate([
    "01_data_and_mask.py", "02_baselines.py", "03_train_correct.py",
    "04_misconfig_autopsy.py", "05_evaluate_and_verdict.py",
    "06_merge_and_serve.py"], start=1)}


def save_completed(root: Path, destination: Path):
    for name in ("results", "adapters", "submission", "notebooks", "src", "scripts", "data"):
        source = root / name
        if source.is_dir():
            shutil.copytree(source, destination / name, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns("merged", "checkpoint-*", "__pycache__"))
    for name in ("requirements.txt", "requirements-cpu.txt", "pyproject.toml", "Makefile", ".env.example"):
        if (root / name).is_file():
            shutil.copy2(root / name, destination / name)


def run(root: Path, destination: Path, stages: list[str]):
    if any(stage not in STAGES for stage in stages):
        raise ValueError("Unknown notebook stage.")
    destination = destination.resolve()
    root = root.resolve()
    if destination == root or root in destination.parents:
        raise ValueError("Backup must be outside the repo, preferably on mounted Drive.")
    destination.mkdir(parents=True, exist_ok=True)
    progress = destination / "progress.json"
    manifest = json.loads(progress.read_text()) if progress.exists() else {"completed": []}
    for stage in stages:
        print(f"\n{stage.upper()} — running; log: {destination / (stage + '.log')}", flush=True)
        with (destination / f"{stage}.log").open("a", encoding="utf-8") as log:
            child = subprocess.Popen(
                [sys.executable, "-u", str(root / "notebooks" / STAGES[stage])],
                cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, errors="replace")
            for line in child.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            code = child.wait()
        if code:
            manifest["failed_stage"] = stage
            progress.write_text(json.dumps(manifest, indent=2))
            print(f"{stage} failed. The last completed backup is preserved at {destination}", flush=True)
            return code
        # Save each completed notebook before starting the next GPU operation.
        save_completed(root, destination)
        if stage not in manifest["completed"]:
            manifest["completed"].append(stage)
        manifest.pop("failed_stage", None)
        progress.write_text(json.dumps(manifest, indent=2))
        print(f"BACKUP SAVED after {stage}: {destination}", flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backup-dir", required=True, type=Path)
    parser.add_argument("stages", nargs="*", default=list(STAGES)[:5])
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    return run(root, args.backup_dir, args.stages or list(STAGES)[:5])


if __name__ == "__main__":
    raise SystemExit(main())
