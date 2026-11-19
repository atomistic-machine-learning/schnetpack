"""spktest re-tests a spktrain run from its folder: the same test_metrics.json as
spktrain wrote after training."""

import json
import os
import shutil
import subprocess
import sys

import pytest

TEST_QM9 = os.path.join(os.path.dirname(__file__), "testdata", "test_qm9.db")


def spk(entry, *args, cwd):
    result = subprocess.run(
        [sys.executable, "-c", f"from schnetpack.cli.{entry} import main; main()"]
        + list(args),
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr[-3000:]


def test_spktest_writes_the_test_metrics_spktrain_wrote(tmp_path):
    os.makedirs(tmp_path / "data")
    shutil.copy(TEST_QM9, tmp_path / "data" / "qm9.db")
    spk(
        "train",
        "experiment=qm9_atomwise",
        f"run.data_dir={tmp_path / 'data'}",
        f"run.path={tmp_path / 'runs'}",
        "run.id=qm9",
        "data.num_train=10",
        "data.num_val=5",
        "data.num_test=5",
        "data.batch_size=5",
        "data.num_workers=0",
        "trainer.max_epochs=2",
        "trainer.accelerator=cpu",
        "seed=0",
        "print_config=False",
        cwd=tmp_path,
    )
    run = tmp_path / "runs" / "qm9"
    trained = json.loads((run / "test_metrics.json").read_text())
    os.remove(run / "test_metrics.json")

    spk("test", str(run), "trainer.accelerator=cpu", cwd=tmp_path)
    tested = json.loads((run / "test_metrics.json").read_text())

    assert trained["checkpoint"].startswith("checkpoints/epoch=")
    assert trained["n_test"] == 5
    units = {k: m["unit"] for k, m in trained["metrics"].items()}
    assert units == {
        "test_loss": None,
        "test_energy_U0_mae": "eV",
        "test_energy_U0_rmse": "eV",
    }
    for key in ("checkpoint", "epoch", "n_test"):
        assert tested[key] == trained[key]
    for name, metric in trained["metrics"].items():
        assert tested["metrics"][name]["unit"] == metric["unit"]
        assert tested["metrics"][name]["value"] == pytest.approx(
            metric["value"], rel=1e-6
        )
