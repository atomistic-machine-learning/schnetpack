"""The model format: spk.save_model writes a model folder (hyperparameters.json and
model.safetensors) from the constructor arguments every registered class records,
and spk.load_model builds the model from it again, without Hydra.

The saved model in tests/testdata/model_format_1 was written by
``python tests/test_model_format.py``; it guards that files of format 1 keep
loading."""

import copy
import json
import os
import subprocess
import sys

import hydra
import pytest
import pytorch_lightning as pl
import torch
import torchmetrics
from ase.build import molecule
from hydra import compose, initialize_config_dir
from torch import nn

import schnetpack as spk
import schnetpack.transform as trn
from schnetpack import properties
from schnetpack.data import ASEAtomsData
from schnetpack.lightning import AtomisticTask, AtomsDataModule, ModelCheckpoint
from schnetpack.model_format import FORMAT

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(os.path.dirname(spk.__file__), "configs")
BENCHMARK_CONFIGS = os.path.join(os.path.dirname(HERE), "benchmarks", "configs")
TEST_QM9 = os.path.join(HERE, "testdata", "test_qm9.db")
FORMAT_1 = os.path.join(HERE, "testdata", "model_format_1")
FORMAT_1_OUTPUTS = os.path.join(HERE, "testdata", "model_format_1_outputs.json")


def batch(cutoff: float = 5.0) -> dict[str, torch.Tensor]:
    """Ethanol and water, with the time of a generative model."""
    converter = spk.interfaces.AtomsConverter(
        neighbor_list=trn.MatScipyNeighborList(cutoff=min(cutoff, 20.0)),
        dtype=torch.float32,
    )
    inputs = converter([molecule("CH3CH2OH"), molecule("H2O")])
    inputs["t"] = torch.full((len(inputs[properties.Z]),), 0.3)
    return inputs


class Statistics:
    """Stands in for the datamodule, with statistics as it gives them: a 0-d
    float64 mean and float64 atomrefs of 100 elements."""

    def get_stats(self, *args):
        return torch.tensor(-4.2, dtype=torch.float64), torch.tensor(0.37).double()

    def get_atomrefs(self, prop, *args, **kwargs):
        return {prop: torch.linspace(-1.0, -10.0, 100, dtype=torch.float64)}


def trained(model: nn.Module, stats=None) -> nn.Module:
    """What training leaves: statistics from the data and other weights."""
    model.initialize_transforms(stats or Statistics())
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(0.01 * torch.randn(parameter.shape, generator=generator))
    return model.eval()


def potential(output_modules=None, postprocessors=None, **representation):
    return spk.model.NeuralNetworkPotential(
        representation=spk.model.PaiNN(
            n_atom_basis=8,
            n_interactions=1,
            radial_basis=spk.nn.GaussianRBF(n_rbf=6, cutoff=5.0),
            cutoff_fn=spk.nn.CosineCutoff(5.0),
            **representation,
        ),
        input_modules=[spk.model.PairwiseDistances()],
        output_modules=output_modules
        or [spk.model.Atomwise(n_in=8, output_key="energy"), spk.model.Forces()],
        postprocessors=postprocessors
        if postprocessors is not None
        else [
            trn.CastTo64(),
            trn.AddOffsets("energy", add_mean=True, add_atomrefs=True),
        ],
    )


def qm9_datamodule(tmp_path, **kwargs) -> AtomsDataModule:
    """The test QM9 db with its energy: 10 training and 5 validation molecules."""
    return AtomsDataModule(
        ASEAtomsData(
            TEST_QM9,
            load_properties=["energy_U0"],
            transforms=[trn.ASENeighborList(cutoff=5.0)],
        ),
        batch_size=5,
        num_train=10,
        num_val=5,
        split_file=str(tmp_path / "split.npz"),
        num_workers=0,
        **kwargs,
    )


def assert_same_model(model: nn.Module, loaded: nn.Module, cutoff: float = 5.0):
    """Same weights with the same dtypes, and the same outputs."""
    assert model.state_dict().keys() == loaded.state_dict().keys()
    for name, value in model.state_dict().items():
        assert value.dtype == loaded.state_dict()[name].dtype, name
        assert torch.equal(value, loaded.state_dict()[name]), name
    expected, outputs = model(batch(cutoff)), loaded(batch(cutoff))
    assert expected.keys() == outputs.keys()
    for name, value in expected.items():
        assert torch.equal(value, outputs[name]), name


def composed(experiment: str, overrides=()):
    with initialize_config_dir(CONFIG_DIR, version_base="1.2"):
        return compose(
            "train",
            overrides=[f"experiment={experiment}", "run.data_dir=x", *overrides],
        )


EXPERIMENTS = {
    "qm9_atomwise": [],  # PaiNN
    "qm9_atomwise-schnet": ["model/representation=schnet"],
    "qm9_atomwise-so3net": ["model/representation=so3net"],
    "qm9_dipole": [],
    "md17": [],
    "rmd17": [],
    "response": [],  # FieldSchNet
    "distillation": [],
    # the generative benchmarks: a time-agnostic and a time-conditioned model
    "gpff_qm9": [f"hydra.searchpath=[file://{BENCHMARK_CONFIGS}]"],
    "dm_qm9": [f"hydra.searchpath=[file://{BENCHMARK_CONFIGS}]"],
}


@pytest.mark.parametrize("name", EXPERIMENTS)
def test_the_model_of_an_experiment_config_round_trips(name, tmp_path):
    """mace_finetune and mace_scratch are left out: MaceModel stays a pickle."""
    config = composed(name.split("-")[0], EXPERIMENTS[name])
    model = trained(hydra.utils.instantiate(config.model))

    spk.save_model(model, str(tmp_path / "model"))
    loaded = spk.load_model(str(tmp_path / "model"))

    assert sorted(os.listdir(tmp_path / "model")) == [
        "hyperparameters.json",
        "model.safetensors",
    ]
    assert_same_model(model, loaded, float(config.globals.cutoff))


def test_statistics_from_the_data_keep_their_shape_and_dtype(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    datamodule = qm9_datamodule(tmp_path, num_test=5)
    datamodule.setup()
    model = potential(
        output_modules=[
            spk.model.Atomwise(n_in=8, output_key="energy_U0"),
            trn.ScaleProperty("energy_U0"),
        ],
        postprocessors=[
            trn.CastTo64(),
            trn.AddOffsets("energy_U0", add_mean=True, add_atomrefs=True),
        ],
    )
    trained(model, datamodule)
    removal = trn.RemoveOffsets("energy_U0", remove_mean=True)
    removal.initialize(datamodule)

    # as the constructors register them, so that a model built again takes them
    assert model.postprocessors[1].mean.shape == (1,)
    assert model.postprocessors[1].mean.dtype == torch.float64
    assert model.output_modules[1].scale.shape == (1,)
    assert removal.mean.shape == (1,)

    spk.save_model(model, str(tmp_path / "model"))
    assert_same_model(model, spk.load_model(str(tmp_path / "model")))


def test_a_short_atomref_table_from_the_data_is_saved(tmp_path):
    """Atomrefs of 9 entries (MD17's end at oxygen) are padded to the table the
    constructor built, so the model built again has the same buffers."""

    class ShortTable(Statistics):
        def get_atomrefs(self, prop, *args, **kwargs):
            return {prop: torch.linspace(-1.0, -9.0, 9, dtype=torch.float64)}

    model = trained(potential(), ShortTable())
    assert model.postprocessors[1].atomref.shape == (100,)
    spk.save_model(model, str(tmp_path / "model"))
    assert_same_model(model, spk.load_model(str(tmp_path / "model")))


def test_python_built_model_saved_by_model_checkpoint(tmp_path):
    datamodule = qm9_datamodule(tmp_path, batch_transforms=[trn.CastTo32()])
    model = potential(
        output_modules=[spk.model.Atomwise(n_in=8, output_key="energy_U0")],
        postprocessors=[
            trn.CastTo64(),
            trn.AddOffsets("energy_U0", add_mean=True, add_atomrefs=True),
        ],
    )
    task = AtomisticTask(
        model=model,
        outputs=[
            spk.ModelOutput(
                name="energy_U0",
                loss_fn=nn.MSELoss(),
                metrics={"MAE": torchmetrics.MeanAbsoluteError()},
            )
        ],
        optimizer_args={"lr": 1e-3},
    )
    sample = {"n_steps": [32, 64], "grid": {"rho": 5.0}}
    task.model_blocks = {"sample": sample}
    checkpoint = ModelCheckpoint(
        model_path=str(tmp_path / "best_model"),
        dirpath=str(tmp_path / "checkpoints"),
        monitor="val_loss",
    )
    trainer = pl.Trainer(
        max_epochs=1,
        callbacks=[checkpoint],
        accelerator="cpu",
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        default_root_dir=str(tmp_path),
    )
    trainer.fit(task, datamodule=datamodule)

    loaded = spk.load_model(str(tmp_path / "best_model"))
    assert spk.load_hyperparameters(str(tmp_path / "best_model"))["sample"] == sample
    model.eval()
    assert_same_model(model, loaded)


def test_torch_modules_of_the_list_are_arguments(tmp_path):
    """An activation module and a nuclear embedding, as PaiNN takes them."""
    model = trained(
        potential(nuclear_embedding=nn.Embedding(100, 8), activation=nn.SiLU())
    )
    spk.save_model(model, str(tmp_path / "model"))
    hyperparameters = spk.load_hyperparameters(str(tmp_path / "model"))
    embedding = hyperparameters["model"]["representation"]["nuclear_embedding"]
    assert embedding["_target_"] == "torch.nn.Embedding"
    assert_same_model(model, spk.load_model(str(tmp_path / "model")))


@spk.register
class ScaledEnergy(nn.Module):
    """A head of a user's own: a linear energy per atom, scaled and summed."""

    def __init__(self, n_in: int, scale: float, output_key: str = "energy"):
        super().__init__()
        self.output_key = output_key
        self.model_outputs = [output_key]
        self.scale = scale
        self.linear = nn.Linear(n_in, 1)

    def forward(self, inputs):
        energy = self.linear(inputs["scalar_representation"]).squeeze(-1)
        inputs[self.output_key] = self.scale * spk.nn.scatter_add(
            energy, inputs[properties.idx_m], dim_size=len(inputs[properties.n_atoms])
        )
        return inputs


class Unregistered(ScaledEnergy):
    """A subclass of a registered class, not registered itself."""

    def __init__(self, n_in: int):
        super().__init__(n_in, scale=2.0)


def test_a_registered_class_of_your_own_round_trips(tmp_path):
    model = trained(potential(output_modules=[ScaledEnergy(8, scale=3.0)]))

    spk.save_model(model, str(tmp_path / "model"))
    loaded = spk.load_model(str(tmp_path / "model"))

    tree = spk.load_hyperparameters(str(tmp_path / "model"))["model"]
    assert tree["output_modules"][0] == {
        "_target_": f"{__name__}.ScaledEnergy",
        "n_in": 8,
        "scale": 3.0,
    }
    assert_same_model(model, loaded)
    with pytest.raises(ValueError, match="no record of its constructor arguments"):
        spk.save_model(potential(output_modules=[Unregistered(8)]), str(tmp_path / "x"))


def test_a_model_changed_after_construction_is_not_saved(tmp_path):
    model = potential(output_modules=[spk.model.Atomwise(n_in=8, output_key="energy")])
    model.output_modules.append(spk.model.Forces())

    with pytest.raises(ValueError, match="changed after it was built.*Forces"):
        spk.save_model(model, str(tmp_path / "model"))
    assert not (tmp_path / "model").exists()


def test_a_module_replaced_by_one_of_the_same_shapes_is_not_saved(tmp_path):
    """Same class, same weight names and shapes: only the record still names the
    old module, so the file would say cutoff 5.0 for a cutoff of 3.0."""
    model = potential()
    model.representation.cutoff_fn = spk.nn.CosineCutoff(3.0)

    with pytest.raises(ValueError, match=r"model\.representation\.cutoff_fn is no"):
        spk.save_model(model, str(tmp_path / "model"))
    assert not (tmp_path / "model").exists()


def test_an_argument_outside_the_list_is_not_saved(tmp_path):
    model = potential(activation=nn.functional.gelu)

    with pytest.raises(ValueError, match="representation.activation: cannot write"):
        spk.save_model(model, str(tmp_path / "model"))
    assert not (tmp_path / "model").exists()


def test_a_switch_set_after_construction_is_saved_as_it_is(tmp_path):
    model = potential()
    model.do_postprocessing = False

    spk.save_model(model, str(tmp_path / "model"))

    assert not spk.load_model(str(tmp_path / "model")).do_postprocessing


def saved(tmp_path) -> str:
    path = str(tmp_path / "model")
    spk.save_model(potential(), path)
    return path


def edit(path: str, change) -> None:
    file = os.path.join(path, "hyperparameters.json")
    with open(file) as f:
        hyperparameters = json.load(f)
    change(hyperparameters)
    with open(file, "w") as f:
        json.dump(hyperparameters, f)


@pytest.mark.parametrize(
    "node, message",
    [
        ({"_target_": "torch.os.system", "command": "true"}, "not a class"),
        ({"_target_": "schnetpack.utils.load_model"}, "not a class"),
        ({"_target_": "os.system", "_partial_": True}, "not a function"),
        ({"_target_": "torch.nn.functional.silu", "_partial_": False}, "not a func"),
    ],
)
def test_loading_refuses_a_name_outside_the_table(tmp_path, node, message):
    path = saved(tmp_path)
    edit(path, lambda h: h["model"]["representation"].update(cutoff_fn=node))

    with pytest.raises(ValueError, match=message):
        spk.load_model(path)


@pytest.mark.parametrize(
    "name",
    [
        "schnetpack.transform.CachedNeighborList",
        "schnetpack.transform.neighborlist.CachedNeighborList",
    ],
)
def test_loading_refuses_a_class_that_touches_the_file_system(tmp_path, name):
    """CachedNeighborList's constructor creates and copies folders."""
    path = saved(tmp_path)
    cached = {
        "_target_": name,
        "cache_path": str(tmp_path / "created"),
        "neighbor_list": {
            "_target_": "schnetpack.transform.MatScipyNeighborList",
            "cutoff": 5.0,
        },
        "cache_workdir": str(tmp_path / "copied"),
    }
    edit(path, lambda h: h["model"]["postprocessors"].append(cached))

    with pytest.raises(ValueError, match="not a class"):
        spk.load_model(path)
    assert not (tmp_path / "created").exists()
    assert not (tmp_path / "copied").exists()


def test_saving_where_an_earlier_model_file_is_says_so(tmp_path):
    (tmp_path / "best_model").write_bytes(b"a pickled model")

    with pytest.raises(FileExistsError, match="earlier SchNetPack version"):
        spk.save_model(potential(), str(tmp_path / "best_model"))


def test_a_failed_write_leaves_the_saved_model_whole(tmp_path, monkeypatch):
    path = saved(tmp_path)
    first = spk.load_model(path).state_dict()

    def crash(source, target):  # the process dies before the weights are in place
        if target.endswith("model.safetensors"):
            raise OSError("crashed")
        os.rename(source, target)

    monkeypatch.setattr(spk.model_format.os, "replace", crash)
    with pytest.raises(OSError, match="crashed"):
        spk.save_model(trained(potential()), path)

    for name, value in spk.load_model(path).state_dict().items():
        assert torch.equal(value, first[name]), name


def test_loading_refuses_interpolations(tmp_path):
    path = saved(tmp_path)
    edit(path, lambda h: h["model"]["output_modules"][0].update(output_key="${a.b}"))

    with pytest.raises(ValueError, match="interpolations"):
        spk.load_model(path)
    with pytest.raises(ValueError, match="interpolations"):
        spk.save_model(
            potential(output_modules=[spk.model.Atomwise(n_in=8, output_key="${a}")]),
            str(tmp_path / "other"),
        )


def test_loading_refuses_a_newer_format(tmp_path):
    path = saved(tmp_path)
    edit(path, lambda h: h.update(format=FORMAT + 1))

    with pytest.raises(ValueError, match="upgrade SchNetPack"):
        spk.load_model(path)


def test_a_model_scripts_and_deep_copies(tmp_path):
    model = trained(potential(postprocessors=[trn.AddOffsets("energy", add_mean=True)]))

    scripted = torch.jit.script(model)
    for name, value in model(batch()).items():
        # the scripted AddOffsets returns a float64 energy where eager keeps
        # float32, as before the model format: compare the values
        torch.testing.assert_close(scripted(batch())[name], value, check_dtype=False)

    twin = copy.deepcopy(model)
    assert twin._init_args["representation"] is twin.representation
    spk.save_model(twin, str(tmp_path / "model"))
    assert_same_model(model, spk.load_model(str(tmp_path / "model")))


BLOCK_HYDRA = """
import sys

class Block:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] == "hydra":
            raise ImportError("hydra is blocked")

sys.meta_path.insert(0, Block())
import torch
import schnetpack as spk

model = spk.load_model(sys.argv[1])
torch.save(model(torch.load(sys.argv[2])), sys.argv[3])
assert "hydra" not in sys.modules
"""


def test_a_model_loads_without_hydra(tmp_path):
    model = trained(potential())
    spk.save_model(model, str(tmp_path / "model"))
    torch.save(batch(), tmp_path / "batch.pt")

    subprocess.run(
        [sys.executable, "-c", BLOCK_HYDRA, str(tmp_path / "model")]
        + [str(tmp_path / "batch.pt"), str(tmp_path / "outputs.pt")],
        check=True,
    )

    outputs = torch.load(tmp_path / "outputs.pt", weights_only=True)
    for name, value in model(batch()).items():
        assert torch.equal(outputs[name], value), name


def format_1_model() -> nn.Module:
    """The model of tests/testdata/model_format_1: a function, a torch module and a
    tensor as arguments, statistics in float64."""
    torch.manual_seed(0)
    return trained(
        potential(
            output_modules=[
                spk.model.Atomwise(n_in=8, output_key="energy", activation=nn.SiLU()),
                spk.model.Forces(),
            ],
            postprocessors=[
                trn.CastTo64(),
                trn.AddOffsets(
                    "energy",
                    add_mean=True,
                    add_atomrefs=True,
                    atomrefs=torch.linspace(0.0, -2.0, 100, dtype=torch.float64),
                ),
            ],
            activation=nn.functional.silu,
        )
    )


def test_a_saved_format_1_model_loads_with_the_same_outputs():
    with open(FORMAT_1_OUTPUTS) as f:
        expected = json.load(f)

    outputs = spk.load_model(FORMAT_1)(batch())

    assert spk.load_hyperparameters(FORMAT_1)["format"] == 1
    assert sorted(outputs) == sorted(expected)
    # the model computes in float32 and casts its outputs to float64: the stored
    # values are met to float32 accuracy on another platform or Python version
    for name, value in expected.items():
        torch.testing.assert_close(
            outputs[name],
            torch.tensor(value, dtype=outputs[name].dtype),
            rtol=1e-5,
            atol=1e-5,
        )


if __name__ == "__main__":  # writes tests/testdata/model_format_1
    model = format_1_model()
    spk.save_model(model, FORMAT_1)
    with open(FORMAT_1_OUTPUTS, "w") as f:
        json.dump({k: v.tolist() for k, v in model(batch()).items()}, f)
        f.write("\n")
