import os
import pytest
import numpy as np
from ase import Atoms
import schnetpack as spk

__all__ = [
    # general
    "tmp_data_dir",
    "tmp_dbpath",
    "partition_names",
    "min_atoms",
    "max_atoms",
    "num_data",
    "n_train_set",
    "n_validation_set",
    "n_test_set",
    "empty_dataset",
    "property_shapes",
    "example_data",
    "example_dataset",
    "available_properties",
    "main_properties",
    "example_subset",
    "example_concat_dataset",
    "example_concat_dataset2",
    "train_val_test_datasets",
    "dataset_stats",
    "example_loader",
    "train_loader",
    "val_loader",
    "test_loader",
    "batch_size",
    # shared_datadir
    "simulation_hdf5_path",
    "qm9_path",
    "ani1_path",
    "ethanol_path",
    "iso17_path",
    "xyz_path",
    "orca_log_path",
    "hessian_path",
    "target_orca_db_path",
    "molecule_path",
    "md_model_path",
    "hdf5_dataset",
    "qm9_dataset",
    "ani1_dataset",
    "md17_dataset",
    "orca_hessian_targets",
    "orca_main_targets",
]


# temporary paths
@pytest.fixture(scope="session")
def tmp_data_dir(tmpdir_factory):
    return tmpdir_factory.mktemp("data")


@pytest.fixture(scope="session")
def tmp_dbpath(example_dataset):
    return example_dataset.dbpath


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
def n_train_set(num_data):
    return num_data // 2


@pytest.fixture(scope="session")
def n_validation_set(num_data):
    return num_data // 4


@pytest.fixture(scope="session")
def n_test_set(num_data, n_train_set, n_validation_set):
    return num_data - n_train_set - n_validation_set


@pytest.fixture(scope="session")
def empty_dataset(tmp_data_dir, available_properties):
    return spk.data.AtomsData(
        os.path.join(str(tmp_data_dir), "empty_database4tests.db"),
        available_properties=available_properties,
    )


@pytest.fixture(scope="session")
def property_shapes():
    return dict(
        property1=[1], derivative1=[-1, 3], contributions1=[-1, 1], property2=[1]
    )


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


@pytest.fixture(scope="session")
def available_properties(property_shapes):
    return list(property_shapes.keys())


@pytest.fixture(scope="session")
def main_properties(available_properties):
    return [pname for pname in available_properties if pname.startswith("property")]


@pytest.fixture(scope="session")
def example_dataset(tmp_data_dir, example_data, available_properties):
    data = spk.data.AtomsData(
        os.path.join(str(tmp_data_dir), "database4tests.db"),
        available_properties=available_properties,
    )
    # add data
    for ats, props in example_data:
        data.add_system(ats, props)
    return data


@pytest.fixture
def example_subset(example_dataset):
    return spk.data.create_subset(example_dataset, [0, 1])


@pytest.fixture
def example_concat_dataset(example_dataset, example_subset):
    return example_dataset + example_subset


@pytest.fixture
def example_concat_dataset2(example_concat_dataset, example_subset):
    return example_concat_dataset + example_subset


@pytest.fixture(scope="session")
def train_val_test_datasets(example_dataset, n_train_set, n_validation_set):
    return spk.data.train_test_split(example_dataset, n_train_set, n_validation_set)


@pytest.fixture(scope="session")
def dataset_stats(example_dataset, main_properties):
    # empty dicts
    means = {pname: None for pname in main_properties}
    stds = {pname: None for pname in main_properties}

    # read database and calculate statistics
    for pname in main_properties:
        data = [
            example_dataset.get_properties(i)[1][pname]
            for i in range(len(example_dataset))
        ]
        means[pname] = np.mean(data, axis=0)
        stds[pname] = np.std(data, axis=0)

    return means, stds


# example dataloader
@pytest.fixture(params=[1, 10], ids=["small_batch", "big_batch"])
def batch_size(request):
    return request.param


@pytest.fixture
def example_loader(example_dataset, batch_size):
    return spk.data.AtomsLoader(example_dataset, batch_size)


@pytest.fixture
def train_loader(train_val_test_datasets, batch_size):
    return spk.data.AtomsLoader(train_val_test_datasets[0], batch_size)


@pytest.fixture
def val_loader(train_val_test_datasets, batch_size):
    return spk.data.AtomsLoader(train_val_test_datasets[1], batch_size)


@pytest.fixture
def test_loader(train_val_test_datasets, batch_size):
    return spk.data.AtomsLoader(train_val_test_datasets[2], batch_size)


# deprecated
@pytest.fixture(params=[None, ["example1", "example2", "ex3"]])
def partition_names(request):
    return request.param


# data folder
# path declarations
@pytest.fixture
def simulation_hdf5_path(shared_datadir):
    return os.path.join(shared_datadir, "test_simulation.hdf5")


@pytest.fixture
def qm9_path(shared_datadir):
    return os.path.join(shared_datadir, "test_qm9.db")


@pytest.fixture
def ani1_path(shared_datadir):
    return os.path.join(shared_datadir, "test_ani1.db")


@pytest.fixture
def ethanol_path(shared_datadir):
    return os.path.join(shared_datadir, "test_ethanol.db")


@pytest.fixture
def iso17_path(shared_datadir):
    return os.path.join(shared_datadir, "test_iso.db")


@pytest.fixture
def xyz_path(shared_datadir):
    return os.path.join(shared_datadir, "ethanol_snip.xyz")


@pytest.fixture
def molecule_path(shared_datadir):
    return os.path.join(shared_datadir, "test_molecule.xyz")


@pytest.fixture
def orca_log_path(shared_datadir):
    return os.path.join(shared_datadir, "test_orca.log")


@pytest.fixture
def hessian_path(shared_datadir):
    return os.path.join(shared_datadir, "test_orca.hess")


@pytest.fixture
def target_orca_db_path(shared_datadir):
    return os.path.join(shared_datadir, "test_orca_parser.db")


@pytest.fixture(params=[0, 1], ids=["schnet", "acsf"])
def md_model_path(request, shared_datadir):
    models = ["test_md_model.model", "test_md_model_wacsf.model"]
    model = os.path.join(shared_datadir, models[request.param])

    return model


# example datasets
@pytest.fixture
def hdf5_dataset(simulation_hdf5_path):
    return spk.md.utils.hdf5_data.HDF5Loader(simulation_hdf5_path, load_properties=True)


@pytest.fixture
def qm9_dataset(qm9_path):
    return spk.datasets.QM9(qm9_path)


@pytest.fixture
def ani1_dataset(ani1_path):
    return spk.datasets.ANI1(ani1_path)


@pytest.fixture
def md17_dataset(ethanol_path):
    return spk.datasets.MD17(ethanol_path, molecule="ethanol")


@pytest.fixture
def orca_main_targets(shared_datadir):
    return np.load(os.path.join(shared_datadir, "test_orca_main_targets.npz"))


@pytest.fixture
def orca_hessian_targets(shared_datadir):
    return np.load(os.path.join(shared_datadir, "test_orca_hessian_targets.npz"))
