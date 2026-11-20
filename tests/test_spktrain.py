"""spktrain's configs, and what it keeps when a run is resumed."""

import os
import re
import shutil
import subprocess
import sys

import hydra
import pytest
import pytorch_lightning as pl
import yaml
from hydra import compose, initialize_config_dir

import schnetpack

CONFIG_DIR = os.path.join(os.path.dirname(schnetpack.__file__), "configs")
TEST_QM9 = os.path.join(os.path.dirname(__file__), "testdata", "test_qm9.db")


@pytest.mark.parametrize("trainer", ["default_trainer", "debug_trainer"])
def test_trainer_config_builds_a_trainer(trainer):
    with initialize_config_dir(CONFIG_DIR, version_base="1.2"):
        config = compose("train", overrides=[f"trainer={trainer}"])

    assert isinstance(hydra.utils.instantiate(config.trainer, logger=False), pl.Trainer)


def spktrain(root, max_epochs):
    """One spktrain job in its own process, always in the same run folder."""
    result = subprocess.run(
        [sys.executable, "-c", "from schnetpack.cli.train import main; main()"]
        + [
            "experiment=qm9_atomwise",
            f"run.data_dir={root / 'data'}",
            f"run.path={root / 'runs'}",
            "run.id=run",
            "data.num_train=10",
            "data.num_val=5",
            "data.num_test=5",
            "data.batch_size=5",
            "data.num_workers=0",
            f"trainer.max_epochs={max_epochs}",
            "trainer.accelerator=cpu",
            "print_config=False",
        ],
        cwd=root,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr[-3000:]


def test_random_seed_is_stored_and_kept_on_resume(tmp_path):
    os.makedirs(tmp_path / "data")
    shutil.copy(TEST_QM9, tmp_path / "data" / "qm9.db")
    spktrain(tmp_path, max_epochs=1)
    run = tmp_path / "runs" / "run"
    seed = yaml.safe_load((run / "config.yaml").read_text())["seed"]
    assert isinstance(seed, int)

    spktrain(tmp_path, max_epochs=2)
    logs = "".join(p.read_text() for p in run.glob("*.log"))
    logged = re.findall(r"Seed (?:randomly )?with <(\d+)>", logs)
    assert [int(s) for s in logged] == [seed, seed]
