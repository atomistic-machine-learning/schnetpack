"""The boundary between a teacher and training, checked against teachers whose
answers are known in closed form."""

import copy
import pickle

import pytest
import torch
from torch import nn
from torch.autograd import grad

from schnetpack import properties
from schnetpack.dynamics import BatchNeighborList
from schnetpack.train.teacher import TeacherWrapper, teacher_key
from schnetpack.transform import CollectAtomTriples, MatScipyNeighborList
from schnetpack.units import convert_units

from .conftest import CUTOFF, molecules_batch

SPRING = 0.3  # Hartree / Bohr^2
STUDENT_UNITS = dict(student_energy_unit="kcal/mol", student_distance_unit="Ang")


class ForeignHarmonic(nn.Module):
    """A third-party-style model: tensors in, a tuple out, Hartree and Bohr.

    Every atom sits in a spring towards the origin, E = k/2 |r|^2, so its forces
    are -k r and its Hessian is k times the identity.
    """

    def __init__(self, detach_forces=False):
        super().__init__()
        self.detach_forces = detach_forces

    def forward(self, positions, idx_m, n_structures):
        per_atom = 0.5 * SPRING * positions.pow(2).sum(-1)
        energy = torch.zeros(
            n_structures, dtype=positions.dtype, device=positions.device
        ).index_add(0, idx_m, per_atom)
        (dEdR,) = grad(energy.sum(), positions, create_graph=not self.detach_forces)
        return energy, -dEdR


class ForeignTeacher(TeacherWrapper):
    """The hooks a third-party family implements."""

    def __init__(self, detach_forces=False, **kwargs):
        self.detach_forces = detach_forces
        kwargs.setdefault("teacher_energy_unit", "Hartree")
        kwargs.setdefault("teacher_distance_unit", "Bohr")
        super().__init__(**kwargs)

    def load(self):
        return ForeignHarmonic(detach_forces=self.detach_forces)

    def teacher_output(self, inputs, create_graph):
        n_structures = inputs[properties.n_atoms].shape[0]
        energy, forces = self.model(
            inputs[properties.R], inputs[properties.idx_m], n_structures
        )
        return {properties.energy: energy, properties.forces: forces}


class RecordingTeacher(ForeignTeacher):
    """Keeps the inputs it was handed, to inspect units and neighbor lists."""

    def teacher_output(self, inputs, create_graph):
        self.seen = inputs
        self.create_graph = create_graph
        return super().teacher_output(inputs, create_graph)


class CachingTeacher(ForeignTeacher):
    """Builds runtime state in load(), as a family keeping a lookup table does."""

    def load(self):
        self._table = torch.arange(3.0)
        return super().load()


def analytic(batch, probe):
    """Energy, forces and H v of the springs, in kcal/mol and Å."""
    spring = (
        SPRING
        * convert_units("Hartree", "kcal/mol")
        * convert_units("Ang", "Bohr") ** 2
    )
    positions = batch[properties.R].to(torch.float64)
    n_structures = batch[properties.n_atoms].shape[0]
    energy = torch.zeros(n_structures, dtype=torch.float64).index_add(
        0, batch[properties.idx_m], 0.5 * spring * positions.pow(2).sum(-1)
    )
    return energy, -spring * positions, spring * probe.to(torch.float64)


def brute_force_pairs(batch, cutoff):
    """Ordered pairs of distinct atoms closer than ``cutoff``, over all structures."""
    count = 0
    sizes = batch[properties.n_atoms].tolist()
    for positions in torch.split(batch[properties.R].double(), sizes):
        distances = torch.cdist(positions, positions)
        count += int(((distances < cutoff) & (distances > 0)).sum())
    return count


def test_targets_are_converted_to_the_student_units(batch):
    probe = torch.randn_like(batch[properties.R])

    targets = ForeignTeacher(**STUDENT_UNITS)(batch, probe)

    energy, forces, hvp = analytic(batch, probe)
    tolerance = dict(rtol=1e-5, atol=1e-4)
    torch.testing.assert_close(
        targets[teacher_key(properties.energy)], energy, **tolerance
    )
    torch.testing.assert_close(
        targets[teacher_key(properties.forces)].double(), forces, **tolerance
    )
    torch.testing.assert_close(
        targets[properties.teacher_hvp].double(), hvp, **tolerance
    )


def test_energy_and_forces_only_without_a_probe(batch):
    targets = ForeignTeacher(**STUDENT_UNITS)(batch)

    assert set(targets) == {
        teacher_key(properties.energy),
        teacher_key(properties.forces),
    }


@pytest.mark.parametrize("with_probe", [True, False])
def test_teacher_output_is_told_whether_a_graph_is_needed(batch, with_probe):
    """Families whose forces keep their graph through an argument read it here."""
    teacher = RecordingTeacher(**STUDENT_UNITS)
    probe = torch.randn_like(batch[properties.R]) if with_probe else None

    teacher(batch, probe)

    assert teacher.create_graph is with_probe


def test_target_keys_are_named_after_the_students_keys(batch):
    teacher = ForeignTeacher(
        student_energy_key="E", student_force_key="F", **STUDENT_UNITS
    )

    targets = teacher(batch, torch.randn_like(batch[properties.R]))

    assert teacher.target_keys == ("teacher_E", "teacher_F", properties.teacher_hvp)
    assert set(targets) == set(teacher.target_keys)


def test_energy_is_float64_and_the_rest_follows_the_batch(batch):
    targets = ForeignTeacher(**STUDENT_UNITS)(
        batch, torch.randn_like(batch[properties.R])
    )

    assert targets[teacher_key(properties.energy)].dtype == torch.float64
    assert targets[teacher_key(properties.forces)].dtype == torch.float32
    assert targets[properties.teacher_hvp].dtype == torch.float32


def test_targets_follow_a_float64_batch():
    batch = molecules_batch(dtype=torch.float64)

    targets = ForeignTeacher(**STUDENT_UNITS)(
        batch, torch.randn_like(batch[properties.R])
    )

    assert targets[teacher_key(properties.forces)].dtype == torch.float64
    assert targets[properties.teacher_hvp].dtype == torch.float64


def test_targets_are_detached(batch):
    targets = ForeignTeacher(**STUDENT_UNITS)(
        batch, torch.randn_like(batch[properties.R])
    )

    assert not any(target.requires_grad for target in targets.values())


def test_the_batch_is_left_untouched(batch):
    before = {key: value.clone() for key, value in batch.items()}
    positions = batch[properties.R]

    ForeignTeacher(**STUDENT_UNITS)(batch, torch.randn_like(positions))

    assert set(batch) == set(before)
    assert batch[properties.R] is positions
    assert not positions.requires_grad
    for key, value in before.items():
        assert torch.equal(batch[key], value), key


def test_curvature_needs_forces_with_a_graph(batch):
    teacher = ForeignTeacher(detach_forces=True, **STUDENT_UNITS)

    teacher(batch)  # energy and forces alone need no graph
    with pytest.raises(RuntimeError, match="not differentiable"):
        teacher(batch, torch.randn_like(batch[properties.R]))


def test_cutoff_in_teacher_units_prunes_the_batch_list(batch):
    # the teacher and its cutoff work in Bohr, the student in Å
    teacher = RecordingTeacher(cutoff=2.8, **STUDENT_UNITS)

    teacher(batch)

    n_pairs = teacher.seen[properties.idx_i].shape[0]
    assert n_pairs == brute_force_pairs(batch, 2.8 * convert_units("Bohr", "Ang"))
    assert n_pairs != brute_force_pairs(batch, 2.8)
    assert n_pairs < batch[properties.idx_i].shape[0]
    assert teacher.seen[properties.idx_j].shape[0] == n_pairs
    assert teacher.seen[properties.offsets].shape[0] == n_pairs


def with_triples(batch, cutoff):
    """``batch`` with a neighbor list at ``cutoff`` and its atom triples."""
    neighbor_list = BatchNeighborList(
        MatScipyNeighborList(cutoff=cutoff),
        cutoff_skin=0.0,
        transforms=[CollectAtomTriples()],
    )
    return {**batch, **neighbor_list.neighbors(batch)}


def triple_atoms(inputs):
    """Each triple as its center and its unordered pair of neighbors."""
    idx_j = inputs[properties.idx_j]
    return sorted(
        (int(i), *sorted((int(idx_j[j]), int(idx_j[k]))))
        for i, j, k in zip(
            inputs[properties.idx_i_triples],
            inputs[properties.idx_j_triples],
            inputs[properties.idx_k_triples],
        )
    )


def test_a_cutoff_keeps_the_triples_of_the_pairs_within_it(batch):
    """Triples index into the pairs, so pruning renumbers them: the
    teacher sees the triples of a list built at its own cutoff."""
    teacher = RecordingTeacher(cutoff=2.8, **STUDENT_UNITS)

    teacher(with_triples(batch, CUTOFF))

    expected = with_triples(batch, 2.8 * convert_units("Bohr", "Ang"))
    assert triple_atoms(teacher.seen) == triple_atoms(expected)
    assert len(triple_atoms(expected)) > 0


def test_reused_neighbor_list_has_its_offsets_in_teacher_units(batch):
    offsets = torch.randn_like(batch[properties.offsets])
    batch = {**batch, properties.offsets: offsets}
    teacher = RecordingTeacher(**STUDENT_UNITS)

    teacher(batch)

    torch.testing.assert_close(
        teacher.seen[properties.offsets], offsets * convert_units("Ang", "Bohr")
    )
    assert torch.equal(teacher.seen[properties.idx_i], batch[properties.idx_i])


def test_pickles_as_its_configuration(batch):
    teacher = ForeignTeacher(**STUDENT_UNITS)
    assert "model" not in teacher.__getstate__()

    restored = pickle.loads(pickle.dumps(teacher))

    probe = torch.randn_like(batch[properties.R])
    torch.testing.assert_close(restored(batch, probe), teacher(batch, probe))


def test_runtime_state_built_by_load_is_not_pickled():
    """Private attributes are rebuilt by load() on unpickling, never stored."""
    teacher = CachingTeacher(**STUDENT_UNITS)

    state = teacher.__getstate__()
    restored = pickle.loads(pickle.dumps(teacher))

    assert "_table" not in state and "model" not in state
    assert torch.equal(restored._table, torch.arange(3.0))


def test_deepcopy_shares_the_frozen_teacher():
    teacher = ForeignTeacher(**STUDENT_UNITS)

    assert copy.deepcopy(teacher) is teacher


def test_pruning_works_on_the_data_loaders_flat_pbc(batch):
    """The data loader concatenates each sample's (3,) pbc into one flat
    tensor; atoms_to_batch gives (n_structures, 3). Both must work."""
    loader_layout = {**batch, properties.pbc: batch[properties.pbc].reshape(-1)}
    probe = torch.randn_like(batch[properties.R])
    teacher = ForeignTeacher(cutoff=2.0, **STUDENT_UNITS)

    torch.testing.assert_close(teacher(loader_layout, probe), teacher(batch, probe))
