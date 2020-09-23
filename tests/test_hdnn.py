import pytest
import schnetpack as spk

from tests.fixtures import *


@pytest.mark.usefixtures("example_dataset", "batch_size")
def test_triples_exception(example_dataset, batch_size):
    loader = spk.data.AtomsLoader(example_dataset, batch_size)

    reps = spk.representation.BehlerSFBlock(
        n_radial=22, n_angular=5, elements=frozenset(range(100))
    )

    for batch in loader:
        with pytest.raises(spk.representation.HDNNException):
            reps(batch)
        break
