import os
import pytest

from schnetpack.datasets import QM9, MD17


@pytest.fixture
def test_qm9_path():
    path = os.path.join(os.path.dirname(__file__), "../data/test_qm9.db")
    return path


@pytest.mark.skip(
    "Run only local, not in CI. Otherwise takes too long and requires downloading the data"
)
def test_qm9(test_qm9_path):
    qm9 = QM9(
        test_qm9_path,
        num_train=10,
        num_val=5,
        batch_size=5,
        remove_uncharacterized=True,
    )
    assert len(qm9.train_dataset) == 10
    assert len(qm9.val_dataset) == 5
    assert len(qm9.test_dataset) == 5

    ds = [b for b in qm9.train_dataloader()]
    assert len(ds) == 2
    ds = [b for b in qm9.val_dataloader()]
    assert len(ds) == 1
    ds = [b for b in qm9.test_dataloader()]
    assert len(ds) == 1


@pytest.fixture
def test_md17_path():
    path = os.path.join(os.path.dirname(__file__), "../data/tmp/test_md17.db")
    return path


@pytest.mark.skip(
    "Run only local, not in CI. Otherwise takes too long and requires downloading the data"
)
def test_md17(test_md17_path):
    md17 = MD17(
        test_md17_path,
        num_train=10,
        num_val=5,
        num_test=5,
        batch_size=5,
        molecule="uracil",
    )
    assert len(md17.train_dataset) == 10
    assert len(md17.val_dataset) == 5
    assert len(md17.test_dataset) == 5

    ds = [b for b in md17.train_dataloader()]
    assert len(ds) == 2
    ds = [b for b in md17.val_dataloader()]
    assert len(ds) == 1
    ds = [b for b in md17.test_dataloader()]
    assert len(ds) == 1
