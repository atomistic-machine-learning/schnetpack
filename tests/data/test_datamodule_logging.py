import logging
from types import SimpleNamespace

import pytest

from schnetpack.data import AtomsDataModule


@pytest.mark.parametrize(
    "trainer, expected",
    [
        (None, ">> msg"),
        (
            SimpleNamespace(global_rank=1, local_rank=2),
            "Global rank: 1, local rank: 2 >> msg",
        ),
    ],
)
def test_log_with_rank(caplog, trainer, expected):
    with caplog.at_level(logging.DEBUG):
        AtomsDataModule._log_with_rank(SimpleNamespace(trainer=trainer), "msg")

    assert expected in caplog.text
