"""Check that VM loss/failure cannot replace the last completed backup."""
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/colab_run_persistent.py"
spec = importlib.util.spec_from_file_location("colab_run_persistent", SCRIPT)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_failed_stage_preserves_last_completed_artifacts(tmp_path):
    root = tmp_path / "vm"
    destination = tmp_path / "drive"
    (root / "notebooks").mkdir(parents=True)
    (root / "results").mkdir()
    (root / "adapters/merged").mkdir(parents=True)
    (root / "adapters/merged/large-model.bin").write_text("must not copy")
    (root / "adapters/correct/checkpoint-10").mkdir(parents=True)
    (root / "adapters/correct/adapter_model.safetensors").write_text("adapter")
    (root / "adapters/correct/checkpoint-10/optimizer.pt").write_text("must not copy")
    (root / "notebooks/01_data_and_mask.py").write_text(
        "from pathlib import Path\nPath('results/proof.json').write_text('completed')\n")
    (root / "notebooks/02_baselines.py").write_text(
        "from pathlib import Path\nPath('results/proof.json').write_text('partial')\nraise SystemExit(3)\n")
    assert runner.run(root, destination, ["nb1", "nb2"]) == 3
    assert (destination / "results/proof.json").read_text() == "completed"
    assert (destination / "adapters/correct/adapter_model.safetensors").read_text() == "adapter"
    assert not (destination / "adapters/merged").exists()
    assert not (destination / "adapters/correct/checkpoint-10").exists()
    manifest = json.loads((destination / "progress.json").read_text())
    assert manifest == {"completed": ["nb1"], "failed_stage": "nb2"}
    # Simulate the temporary VM filesystem disappearing.
    import shutil
    shutil.rmtree(root)
    assert (destination / "results/proof.json").read_text() == "completed"


def test_backup_inside_repo_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="outside the repo"):
        runner.run(tmp_path, tmp_path / "backup", ["nb1"])
