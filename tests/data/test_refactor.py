import pytest
import torch
import numpy as np

import schnetpack.properties as structure
import schnetpack.data.provider as providers_mod
from schnetpack.data.datamodule_v2 import AtomsDataModuleV2
from schnetpack.transform.atomistic import AddOffsets, RemoveOffsets, ScaleProperty


class AtomsDataset:
    """
    Adapter to make the existing `example_data` list fixture behave like a dataset.
    """

    def __init__(self, data):
        self._data = list(data)
        self.transforms = []

        _, props0 = self._data[0]
        self.available_properties = list(props0.keys())

    def __len__(self):
        return len(self._data)

    def subset(self, indices):
        # indices can be list/np array
        idx_list = list(indices)
        sub = AtomsDataset([self._data[i] for i in idx_list])
        # do not carry transforms automatically; DM will attach per split
        return sub

    def __getitem__(self, idx):
        ats, props = self._data[idx]
        out = {}

        # Structure keys expected by SchNetPack transforms/loader
        out[structure.Z] = torch.tensor(ats.numbers, dtype=torch.long)
        out[structure.R] = torch.tensor(np.asarray(ats.positions), dtype=torch.float)
        out[structure.cell] = torch.tensor(
            np.asarray(ats.cell.array), dtype=torch.float
        )
        out[structure.pbc] = torch.tensor(np.asarray(ats.pbc), dtype=torch.bool)
        out[structure.n_atoms] = torch.tensor([len(ats.numbers)], dtype=torch.long)

        # Add properties
        for k, v in props.items():
            # ensure torch tensor
            if isinstance(v, torch.Tensor):
                out[k] = v
            else:
                out[k] = torch.tensor(np.asarray(v), dtype=torch.float)

        # Apply transforms per-system (like BaseAtomsData.__getitem__)
        for t in self.transforms:
            out = t(out)

        return out


def _first_scalar_property_key(dataset: AtomsDataset):
    # choose the first property key, but ensure it's not a structure key
    for p in dataset.available_properties:
        if p not in (
            structure.Z,
            structure.R,
            structure.cell,
            structure.pbc,
            structure.n_atoms,
        ):
            return p
    raise AssertionError(
        "No suitable scalar property found in dataset.available_properties"
    )


def _make_constant_atomrefs(zmax=100, value=1.0):
    atref = torch.zeros((zmax,), dtype=torch.float)
    atref[:] = float(value)
    return atref


@pytest.mark.parametrize("batch_size", [1, 4])
def test_v2_setup_attaches_transforms(example_data, batch_size):
    dataset = AtomsDataset(example_data)

    dm = AtomsDataModuleV2(
        dataset=dataset,
        batch_size=batch_size,
        num_train=0.6,
        num_val=0.2,
        num_test=0.2,
        transforms=[],
        num_workers=0,
        split_file=None,
    )
    dm.setup()

    assert dm.train_dataset is not None
    assert dm.val_dataset is not None
    assert dm.test_dataset is not None

    # transforms attribute exists and is settable
    dm.train_dataset.transforms = []
    dm.val_dataset.transforms = []
    dm.test_dataset.transforms = []


def test_provider_initializes_stats_transforms(example_data):
    dataset = AtomsDataset(example_data)
    prop = _first_scalar_property_key(dataset)

    transforms = [
        RemoveOffsets(
            property=prop, remove_mean=True, remove_atomrefs=False, is_extensive=True
        ),
        ScaleProperty(
            input_key=prop, target_key=prop, output_key=prop, scale_by_mean=False
        ),
    ]

    dm = AtomsDataModuleV2(
        dataset=dataset,
        batch_size=4,
        num_train=0.6,
        num_val=0.2,
        num_test=0.2,
        transforms=transforms,
        num_workers=0,
        split_file=None,
    )
    dm.setup()

    ro = transforms[0]
    sp = transforms[1]

    assert hasattr(ro, "mean")
    assert ro.mean is not None
    assert hasattr(sp, "scale")
    assert sp.scale is not None


@pytest.mark.parametrize("is_extensive", [True, False])
def test_addoffsets_unbatched_and_batched(example_data, is_extensive):
    dataset = AtomsDataset(example_data)
    prop = _first_scalar_property_key(dataset)

    zmax = 100
    atomref_tensor = _make_constant_atomrefs(zmax=zmax, value=1.0)

    t = AddOffsets(
        property=prop,
        add_mean=False,
        add_atomrefs=True,
        is_extensive=is_extensive,
        zmax=zmax,
        atomrefs=atomref_tensor,
    )

    dm = AtomsDataModuleV2(
        dataset=dataset,
        batch_size=4,
        num_train=0.6,
        num_val=0.2,
        num_test=0.2,
        transforms=[t],
        num_workers=0,
        split_file=None,
    )
    dm.setup()

    # Unbatched: dataset[0] (transform runs in __getitem__)
    old = dm.train_dataset.transforms
    dm.train_dataset.transforms = []
    raw = dm.train_dataset[0]
    dm.train_dataset.transforms = old
    one = dm.train_dataset[0]

    y_raw = raw[prop]
    y_one = one[prop]
    delta = (y_one - y_raw).detach().view(-1)[0]

    n_atoms = int(one[structure.n_atoms].view(-1)[0].item())
    expected = float(n_atoms) if is_extensive else 1.0

    assert torch.allclose(delta, torch.tensor(expected, dtype=delta.dtype), atol=1e-6)

    # Batched: loader should include idx_m and not crash
    batch = next(iter(dm.train_dataloader()))
    assert structure.idx_m in batch
    assert prop in batch


def test_provider_caches_stats_calls(example_data, monkeypatch):
    """
    Ensure provider caching prevents recomputing the same stats key multiple times.
    In this test transforms request the same key when:
      - RemoveOffsets is_extensive=True and remove_atomrefs=False -> (prop, True, False)
      - ScaleProperty always requests (prop, True, False)
    """

    dataset = AtomsDataset(example_data)
    prop = _first_scalar_property_key(dataset)

    call_count = {"n": 0}
    real_calculate_stats = providers_mod.calculate_stats

    def wrapped_calculate_stats(*args, **kwargs):
        call_count["n"] += 1
        return real_calculate_stats(*args, **kwargs)

    monkeypatch.setattr(providers_mod, "calculate_stats", wrapped_calculate_stats)

    transforms = [
        RemoveOffsets(
            property=prop, remove_mean=True, remove_atomrefs=False, is_extensive=True
        ),
        ScaleProperty(
            input_key=prop, target_key=prop, output_key=prop, scale_by_mean=False
        ),
    ]

    dm = AtomsDataModuleV2(
        dataset=dataset,
        batch_size=4,
        num_train=0.6,
        num_val=0.2,
        num_test=0.2,
        transforms=transforms,
        num_workers=0,
        split_file=None,
    )
    dm.setup()

    assert call_count["n"] == 1
