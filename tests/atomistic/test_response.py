import numpy as np

import schnetpack as spk
from schnetpack.data.loader import _atoms_collate_fn


def test_strain(environment_periodic):
    cutoff, props, neighbors = environment_periodic
    props.update(neighbors)
    batch = _atoms_collate_fn([props, props])
    strained_batch = spk.atomistic.Strain()(batch)
    assert np.allclose(
        batch[spk.properties.offsets].detach().numpy(),
        strained_batch[spk.properties.offsets].detach().numpy(),
    )
