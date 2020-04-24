import pytest
import schnetpack.data
import schnetpack.representation.hdnn as rep

from tests.fixtures import *


@pytest.mark.usefixtures("example_dataset", "batch_size")
def test_triples_exception(example_dataset, batch_size):
    loader = schnetpack.data.AtomsLoader(example_dataset, batch_size)

    reps = rep.BehlerSFBlock(n_radial=22, n_angular=5, elements=frozenset(range(100)))

    for batch in loader:
        with pytest.raises(rep.HDNNException):
            reps(batch)
        break
