"""The experiment config, the task and the property keys must agree. Nothing
else composes this yaml short of a real training run."""

import pytest
import torch
from hydra import compose, initialize_config_module
from hydra.utils import instantiate
from omegaconf import open_dict
from omegaconf.errors import MissingMandatoryValue

import schnetpack.cli  # noqa: F401 -- registers the config's resolvers
from schnetpack import properties
from schnetpack.model import FilterShortRange
from schnetpack.train import MaceTeacher, SchNetPackTeacher, teacher_key
from schnetpack.transform import AddOffsets

from .test_mace_teacher import FakeMace


def compose_experiment(overrides):
    with initialize_config_module(
        config_module="schnetpack.configs", version_base="1.2"
    ):
        config = compose(
            config_name="train",
            overrides=["experiment=distillation", *overrides],
            return_hydra_config=True,
        )
    with open_dict(config):
        del config["hydra"]
    return config


@pytest.fixture
def experiment_config(teacher_path):
    return compose_experiment(
        [
            f"task.teacher.model_path={teacher_path}",
            "model.representation.n_atom_basis=16",
            "model.representation.n_interactions=2",
            "run.work_dir=.",
            "run.path=.",
        ]
    )


def test_the_teacher_path_is_mandatory():
    config = compose_experiment([])

    with pytest.raises(MissingMandatoryValue):
        _ = config.task.teacher.model_path


def test_targets_match_the_property_keys(experiment_config):
    targets = [output.target_property for output in experiment_config.task.outputs]

    assert targets == [
        teacher_key(properties.energy),
        teacher_key(properties.forces),
        properties.teacher_hvp,
    ]
    assert experiment_config.task.outputs[2].name == properties.hvp


def test_the_students_energy_offsets_include_atomrefs(experiment_config):
    """Fitted to the teacher's energies, since the data has no labels."""
    model = instantiate(experiment_config.model)

    (offsets,) = [m for m in model.postprocessors if isinstance(m, AddOffsets)]
    assert offsets._property == experiment_config.globals.energy_key
    assert offsets.add_mean and offsets.add_atomrefs and offsets.estimate_atomref


def test_target_keys_follow_the_students_keys(teacher_path, batch):
    config = compose_experiment(
        [
            f"task.teacher.model_path={teacher_path}",
            "globals.energy_key=E",
            "globals.forces_key=F",
            "model.representation.n_atom_basis=16",
            "model.representation.n_interactions=2",
        ]
    )
    model = instantiate(config.model)
    task = instantiate(config.task, model=model, _convert_="partial")

    targets = tuple(output.target_property for output in task.outputs)
    assert (
        targets == task.teacher.target_keys == ("teacher_E", "teacher_F", "teacher_hvp")
    )
    assert torch.isfinite(task.training_step(batch, 0))


def test_the_configured_task_distills_a_batch(experiment_config, batch):
    model = instantiate(experiment_config.model)
    task = instantiate(experiment_config.task, model=model, _convert_="partial")

    assert isinstance(task.teacher, SchNetPackTeacher)
    assert task.teacher.cutoff == experiment_config.globals.teacher_cutoff
    # the student prunes the shared list to its own cutoff
    assert any(isinstance(m, FilterShortRange) for m in task.model.input_modules)
    loss = task.training_step(batch, 0)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in task.model.parameters()
    )


def test_the_teacher_cutoff_is_in_the_teachers_unit():
    """teacher_cutoff is in teacher_distance_unit: the shared list is built at
    the larger cutoff, compared in distance_unit."""
    config = compose_experiment(
        [
            "globals.cutoff=5.",
            "globals.teacher_cutoff=0.6",
            "globals.distance_unit=Ang",
            "globals.teacher_distance_unit=nm",
        ]
    )

    neighbor_list = instantiate(config.data.dataset.transforms[1])
    assert neighbor_list.cutoff == pytest.approx(6.0)
    assert config.task.teacher.cutoff == pytest.approx(0.6)


def test_a_mace_teacher_is_chosen_from_the_teacher_group(tmp_path, batch):
    archive = str(tmp_path / "mace.pt")
    torch.jit.save(torch.jit.script(FakeMace()), archive)
    config = compose_experiment(
        [
            "task/teacher=mace",
            f"task.teacher.model_path={archive}",
            "model.representation.n_atom_basis=16",
            "model.representation.n_interactions=2",
        ]
    )

    task = instantiate(
        config.task, model=instantiate(config.model), _convert_="partial"
    )

    assert isinstance(task.teacher, MaceTeacher)
    assert task.teacher.dtype == torch.float32
    assert task.teacher.cutoff == config.globals.teacher_cutoff == 5.0
    assert torch.isfinite(task.training_step(batch, 0))
