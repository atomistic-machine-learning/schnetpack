import pytest
import os

from schnetpack.datasets import QM9
from schnetpack.data import calculate_stats, AtomsLoader


@pytest.fixture(scope="module")
def qm9(tmpdir_factory):
    path = str(tmpdir_factory.mktemp("data").join("qm9.db"))
    qm9 = QM9("/home/kschuett/data/new/qm9.db", num_train=100000, num_val=1000)
    return qm9


def test_qm9(qm9):
    print(qm9)
    qm9.prepare_data()


def test_stats(qm9):
    stats = calculate_stats(
        qm9.train_dataloader(), {qm9.U0: True}, atomref=qm9.dataset.metadata["atomrefs"]
    )
    print(stats)
