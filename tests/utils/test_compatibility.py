"""Loading pickled models with load_model."""

import pickle

import pytest
import torch

import schnetpack as spk
from schnetpack.utils.compatibility import load_model


@pytest.fixture
def model_path(tmp_path):
    model = spk.model.NeuralNetworkPotential(
        representation=spk.model.SchNet(
            n_atom_basis=8,
            n_interactions=1,
            radial_basis=spk.nn.GaussianRBF(n_rbf=5, cutoff=3.0),
            cutoff_fn=spk.nn.CosineCutoff(3.0),
        ),
        input_modules=[spk.model.PairwiseDistances()],
        output_modules=[spk.model.Atomwise(n_in=8, output_key="energy")],
    )
    model.spk_version = spk.__version__
    path = tmp_path / "model.pt"
    torch.save(model, path)
    return path


def test_load_model_accepts_torch_load_kwargs(model_path):
    model = load_model(model_path, weights_only=False, mmap=True)
    assert isinstance(model, spk.model.NeuralNetworkPotential)


def test_load_model_honors_weights_only(model_path):
    with pytest.raises(pickle.UnpicklingError):
        load_model(model_path, weights_only=True)


def test_load_model_hands_a_model_folder_to_the_model_format(model_path, tmp_path):
    model = load_model(model_path)
    spk.save_model(model, str(tmp_path / "best_model"))

    loaded = load_model(str(tmp_path / "best_model"))

    assert isinstance(loaded, spk.model.NeuralNetworkPotential)
    for name, value in model.state_dict().items():
        assert torch.equal(loaded.state_dict()[name], value), name
