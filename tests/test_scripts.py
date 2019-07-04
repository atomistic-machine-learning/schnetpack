import os
from .fixtures import *


class TestQM9Script:
    def test_train(self, script_runner, tmpdir_factory):
        # train model
        modeldir = tmpdir_factory.mktemp("qm9_model_testing").strpath
        ret = script_runner.run(
            "schnetpack_x.py",
            "train",
            "schnet",
            "qm9",
            "tests/data/test_qm9.db",
            modeldir,
            "--max_epochs",
            "4",
            "--split",
            "10",
            "5",
            "--property",
            "energy_U0",
        )
        assert ret.success, ret.stderr
        print(list(os.walk(os.path.join(modeldir, "checkpoints"))))
        assert os.path.exists(os.path.join(modeldir, "best_model"))
        ret = script_runner.run(
            "schnetpack_x.py",
            "train",
            "schnet",
            "qm9",
            "tests/data/test_qm9.db",
            modeldir,
            "--max_epochs",
            "2",
            "--split",
            "10",
            "5",
            "--property",
            "energy_U0",
            "--checkpoint_interval",
            "1",
        )
        assert ret.success, ret.stderr
        print(list(os.walk(os.path.join(modeldir, "checkpoints"))))
        assert os.path.exists(os.path.join(modeldir, "checkpoints",
                                           "checkpoint-4.pth.tar"))
        ret = script_runner.run(
            "schnetpack_x.py",
            "eval",
            "tests/data/test_qm9.db",
            modeldir,
            "--split",
            "test"
        )
        assert ret.success, ret.stderr
        assert os.path.exists(os.path.join(modeldir, "evaluation.txt"))
