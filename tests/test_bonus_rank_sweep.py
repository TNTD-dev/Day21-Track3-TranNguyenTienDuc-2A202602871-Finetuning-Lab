"""Run the rank driver through subprocesses with a small fake GPU backend."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/bonus_rank_sweep.py"
spec = importlib.util.spec_from_file_location("bonus_rank_sweep", SCRIPT)
bonus = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bonus)


@pytest.fixture
def lab(tmp_path):
    for folder in ["results", "data/split", "adapters/correct", "notebooks", "src/labkit"]:
        (tmp_path / folder).mkdir(parents=True)
    def write(name, value):
        (tmp_path / name).write_text(value)
    write("data/split/train.jsonl", '{}\n' * 225)
    write("data/split/val.jsonl", '{}\n' * 25)
    write("data/eval_target.jsonl", '{}\n' * 50)
    write("data/eval_regression.jsonl", '{}\n' * 15)
    write("results/baselines_frozen.json", json.dumps({"smoke_mode": False, "model": "model",
                                                     "n_target": 50, "n_regression": 15}))
    scores = {"target": .97, "regression": .52, "format": 1, "latency_ms": 100, "n": 50}
    write("results/verdict.json", json.dumps({"comparison": [{}, {}, scores]}))
    write("results/runs.csv", "run,r,lora_alpha,placement,load_in_4bit,model,tier,mask_mode,max_steps,learning_rate,trainable_params,final_loss,peak_vram_gb\n"
          "correct,16,32,text-linear,False,model,T4,assistant-only,30,0.0001,16000,0.6,8.78\n")
    write("adapters/correct/adapter_config.json", json.dumps({"r": 16, "lora_alpha": 32}))
    write("adapters/correct/adapter_model.safetensors", "original core adapter")
    write("src/labkit/__init__.py", "")
    write("src/labkit/config.py", '''from dataclasses import dataclass
from types import SimpleNamespace
@dataclass(frozen=True)
class Spec:
    key: str = 'correct'
    r: int = 16
    alpha: int = 32
    label: str = 'core'
    lr: float = 1e-4
SPECS = {'correct': Spec()}
def get_tier(): return SimpleNamespace(effective_batch=16)
''')
    write("src/labkit/report.py", '''import csv, json
from pathlib import Path
def append_row(row, filename='runs.csv', results_dir=None):
    path = Path(results_dir) / filename
    exists = path.exists()
    with path.open('a') as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if not exists: writer.writeheader()
        writer.writerow(row)
def write_json(obj, filename, results_dir=None):
    (Path(results_dir) / filename).write_text(json.dumps(obj))
def markdown_table(rows): return json.dumps(rows)
''')
    write("notebooks/03_train_correct.py", '''import json, os
from pathlib import Path
from labkit.config import SPECS
from labkit import report
spec = SPECS['correct']
assert spec.r in (8,64) and spec.alpha == 2*spec.r
assert os.environ['EPOCHS'] == '2'
STEPS = 30
folder = Path('adapters') / spec.key
folder.mkdir()
(folder / 'adapter_model.safetensors').write_text('bonus adapter')
row = {'run':spec.key,'r':spec.r,'learning_rate':spec.lr,'max_steps':STEPS,
       'trainable_params':1000*spec.r,'final_loss':0.3,'peak_vram_gb':9.0}
report.append_row(row, results_dir=Path('results'))
''')
    write("notebooks/05_evaluate_and_verdict.py", '''from types import SimpleNamespace
from labkit import report
generate = SimpleNamespace(NAIVE_PROMPT='classify')
def score_adapter(folder, prompt, label='ft'):
    rank = int(folder.name.split('_')[-1])
    scores = {'target':rank/100,'regression':0.5,'format':1,'latency_ms':100,'n':50}
    return SimpleNamespace(as_dict=lambda: scores), ['{}']*50, ['answer']*15
scores_ft, preds_ft, rpreds_ft = score_adapter(None, None)
raise RuntimeError('the core evaluation must not be executed by B4')
''')
    return tmp_path


def test_sweep_preserves_core_and_resumes_without_retraining(lab):
    before = bonus.context(lab)
    command = [sys.executable, str(SCRIPT)]
    first = subprocess.run(command, cwd=lab, capture_output=True, text=True)
    assert first.returncode == 0, first.stdout + first.stderr
    assert bonus.context(lab) == before
    result = bonus.read_json(lab / "results/bonus_rank_sweep.json")
    rows = result["comparison"]
    assert [row["rank"] for row in rows] == [8, 16, 64]
    assert [row["target"] for row in rows] == [.08, .97, .64]
    assert {row["max_steps"] for row in rows} == {30}
    assert {row["lr"] for row in rows} == {1e-4}
    training_log = (lab / "results/bonus_rank_runs.csv").read_bytes()
    second = subprocess.run(command, cwd=lab, capture_output=True, text=True)
    assert second.returncode == 0, second.stdout + second.stderr
    assert (lab / "results/bonus_rank_runs.csv").read_bytes() == training_log
    assert bonus.context(lab) == before


def test_same_size_dataset_edit_cannot_reuse_bonus_results(lab):
    saved = bonus.context(lab)
    path = lab / "data/eval_target.jsonl"
    path.write_text('{"new":true}\n' + '{}\n' * 49)
    with pytest.raises(RuntimeError, match="different core/config/dataset"):
        bonus.require_same(saved, bonus.context(lab))


def test_smoke_core_is_rejected_before_any_training(lab):
    path = lab / "results/baselines_frozen.json"
    data = bonus.read_json(path)
    data["smoke_mode"] = True
    path.write_text(json.dumps(data))
    with pytest.raises(RuntimeError, match="full core evaluation"):
        bonus.context(lab)


def test_missing_saved_core_is_reported_before_importing_the_gpu_backend(lab):
    (lab / "results/baselines_frozen.json").unlink()
    (lab / "adapters/correct/adapter_model.safetensors").unlink()
    with pytest.raises(RuntimeError, match="Restore the full core ZIP") as error:
        bonus.context(lab)
    message = str(error.value)
    assert "results/baselines_frozen.json" in message
    assert "adapters/correct/adapter_model.safetensors" in message
    assert "No bonus training has started" in message
    assert not (lab / "results/bonus_rank_runs.csv").exists()


def test_completed_rank_runs_are_saved_outside_vm(lab):
    import shutil
    (lab / "scripts").mkdir()
    shutil.copy2(SCRIPT.with_name("colab_run_persistent.py"), lab / "scripts/colab_run_persistent.py")
    destination = lab.parent / "drive_saved"
    command = [sys.executable, str(SCRIPT), "--backup-dir", str(destination)]
    completed = subprocess.run(command, cwd=lab, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "B4 BACKUP SAVED: rank-8-train" in completed.stdout
    progress = bonus.read_json(destination / "bonus_rank_progress.json")
    assert progress["completed"] == ["rank-8-train", "rank-8-eval", "rank-64-train",
                                     "rank-64-eval", "comparison"]
    shutil.rmtree(lab)
    assert (destination / "adapters/bonus_rank_8/adapter_model.safetensors").exists()
    assert (destination / "adapters/bonus_rank_64/adapter_model.safetensors").exists()
    assert (destination / "results/bonus_rank_sweep.json").is_file()
