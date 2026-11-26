import torch
from ase.build import molecule

import schnetpack as spk
import schnetpack.transform as trn
from schnetpack import properties
from schnetpack.nn import ConditionEmbedding


def layout():
    """Two structures with 3 and 5 atoms."""
    n_atoms = torch.tensor([3, 5])
    return {
        properties.n_atoms: n_atoms,
        properties.idx_m: torch.repeat_interleave(torch.arange(2), n_atoms),
    }


def test_the_standardized_condition_and_its_flag_reach_every_atom():
    module = ConditionEmbedding("gap", 16, mean=6.0, std=2.0)
    out = module(None, {**layout(), "gap": torch.tensor([7.0, 3.0])})
    expected = module.linear(torch.tensor([[0.5, 1.0], [-1.5, 1.0]]))
    assert out.shape == (8, 16)
    torch.testing.assert_close(out[:3], expected[0].expand(3, 16))
    torch.testing.assert_close(out[3:], expected[1].expand(5, 16))


def test_nan_is_no_condition():
    module = ConditionEmbedding("gap", 16, mean=6.0, std=2.0)
    out = module(None, {**layout(), "gap": torch.tensor([7.0, float("nan")])})
    assert torch.isfinite(out).all()
    # value 0 and flag 0: the bias alone, which no given value produces
    torch.testing.assert_close(out[3:], module.linear.bias.expand(5, 16))
    torch.testing.assert_close(
        out[:3], module.linear(torch.tensor([0.5, 1.0])).expand(3, 16)
    )


def conditioned_painn():
    return spk.model.PaiNN(
        n_atom_basis=16,
        n_interactions=2,
        radial_basis=spk.nn.GaussianRBF(n_rbf=8, cutoff=5.0),
        cutoff_fn=spk.nn.CosineCutoff(5.0),
        electronic_embeddings=[ConditionEmbedding("gap", 16, mean=6.0, std=2.0)],
    )


def molecules():
    """Ethanol and water."""
    converter = spk.interfaces.AtomsConverter(
        neighbor_list=trn.MatScipyNeighborList(cutoff=5.0), dtype=torch.float32
    )
    return converter([molecule("CH3CH2OH"), molecule("H2O")])


def test_a_representation_sees_the_condition_of_each_structure():
    torch.manual_seed(0)
    painn = conditioned_painn()
    inputs = spk.model.PairwiseDistances()(molecules())

    def features(gap):
        return painn({**inputs, "gap": torch.tensor(gap)})["scalar_representation"]

    reference = features([7.0, 3.0])
    water_changed = features([7.0, 4.0])
    ethanol_dropped = features([float("nan"), 3.0])
    ethanol = inputs[properties.idx_m] == 0
    torch.testing.assert_close(water_changed[ethanol], reference[ethanol])
    assert not torch.allclose(water_changed[~ethanol], reference[~ethanol])
    torch.testing.assert_close(ethanol_dropped[~ethanol], reference[~ethanol])
    assert not torch.allclose(ethanol_dropped[ethanol], reference[ethanol])
    assert torch.isfinite(ethanol_dropped).all()


def test_a_conditioned_model_is_saved_and_loaded(tmp_path):
    torch.manual_seed(0)
    model = spk.model.NeuralNetworkPotential(
        representation=conditioned_painn(),
        input_modules=[spk.model.PairwiseDistances()],
        output_modules=[spk.model.AtomwiseVector(n_in=16, output_key="eps_pred")],
    ).eval()
    inputs = {**molecules(), "gap": torch.tensor([7.0, float("nan")])}
    expected = model(dict(inputs))["eps_pred"]

    spk.save_model(model, str(tmp_path / "model"))
    loaded = spk.load_model(str(tmp_path / "model"))

    embedding = loaded.representation.electronic_embeddings[0]
    assert (embedding.mean, embedding.std) == (6.0, 2.0)
    torch.testing.assert_close(loaded(dict(inputs))["eps_pred"], expected)
