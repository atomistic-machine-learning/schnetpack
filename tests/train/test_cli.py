import os
import re
import subprocess
import sys

import yaml

QM9_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../testdata/test_qm9.db")
)


def spktrain(root, max_epochs):
    """One spktrain job in its own process, always in the same run directory."""
    overrides = [
        "experiment=qm9_atomwise",
        f"data.datapath={QM9_PATH}",
        "data.num_train=12",
        "data.num_val=4",
        "data.num_test=2",
        "data.batch_size=2",
        "data.num_workers=0",
        "trainer.accelerator=cpu",
        f"trainer.max_epochs={max_epochs}",
        f"run.path={root}/runs",
        "run.id=run",
        f"run.data_dir={root}",
    ]
    subprocess.run(
        [sys.executable, "-c", "import schnetpack.cli as c; c.train()", *overrides],
        cwd=root,
        check=True,
        capture_output=True,
    )


def test_random_seed_is_stored_and_kept_on_resume(tmp_path):
    spktrain(tmp_path, max_epochs=1)
    run_dir = tmp_path / "runs" / "run"
    seed = yaml.safe_load((run_dir / "config.yaml").read_text())["seed"]
    assert isinstance(seed, int)

    spktrain(tmp_path, max_epochs=2)
    logged = re.findall(
        r"Seed (?:randomly )?with <(\d+)>", (run_dir / "cli.log").read_text()
    )
    assert [int(s) for s in logged] == [seed, seed]
