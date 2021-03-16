import numpy as np
import pytest
from ase import Atoms


@pytest.fixture(scope="session")
def cutoff():
    return 5.0


# example datasets
@pytest.fixture(scope="session")
def max_atoms():
    return 10


@pytest.fixture(scope="session")
def min_atoms():
    return 2


@pytest.fixture(scope="session")
def num_data():
    return 20


@pytest.fixture(scope="session")
def property_shapes():
    return dict(
        property1=[1], derivative1=[-1, 3], contributions1=[-1, 1], property2=[1]
    )


@pytest.fixture(params=[1, 10], ids=["small_batch", "big_batch"])
def batch_size(request):
    return request.param


@pytest.fixture(scope="session")
def example_data(min_atoms, max_atoms, num_data, property_shapes):
    """
    List of (ase.Atoms, data) tuples with different sized atomic systems. Created
    randomly.
    """
    data = []
    for i in range(1, num_data + 1):
        n_atoms = np.random.randint(min_atoms, max_atoms)
        z = np.random.randint(1, 100, size=(n_atoms,))
        r = np.random.randn(n_atoms, 3)
        c = np.random.randn(3, 3)
        pbc = np.random.randint(0, 2, size=(3,)) > 0
        ats = Atoms(numbers=z, positions=r, cell=c, pbc=pbc)

        props = dict()
        for pname, p_shape in property_shapes.items():
            appl_shape = [dim if dim != -1 else n_atoms for dim in p_shape]
            props[pname] = np.random.rand(*appl_shape)

        data.append((ats, props))

    return data
