import pytest
import torch

from schnetpack import properties
from schnetpack.dynamics import (
    LBFGS,
    Calculator,
    DirectDenoising,
    ForceCalculator,
    Heun,
    JointCalculator,
    NoiseCalculator,
    PseudoForceCalculator,
    Sampler,
    ScoreCalculator,
    SDEUpdate,
    VelocityCalculator,
    X0Calculator,
)
from schnetpack.generative import (
    VP,
    EpsParametrization,
    FlowMatching,
    GaussianInterpolant,
    ParametrizedModel,
    PseudoForceParametrization,
    ScoreParametrization,
    Variable,
    VelocityParametrization,
)


class InPlaceNeighborList:
    """Writes into its input like spk's neighbor-list transforms; counts calls and resets."""

    def __init__(self):
        self.calls = 0
        self.resets = 0

    def __call__(self, inputs):
        self.calls += 1
        inputs[properties.idx_i] = torch.arange(inputs[properties.R].shape[0])
        return inputs

    def reset(self):
        self.resets += 1


def zero_model(batch):
    assert properties.idx_i in batch
    return {"prediction": torch.zeros_like(batch[properties.R])}


def test_neighbor_list_runs_on_every_model_call_and_never_reaches_the_loop():
    # Heun evaluates twice per step; each call gets a fresh neighbor list,
    # built on a copy, so the loop's batch never holds a derived key.
    nbl = InPlaceNeighborList()
    process = GaussianInterpolant(VP())
    v = Variable(
        "x", properties.R, "atom", process, VelocityParametrization(), "prediction"
    )
    calculator = JointCalculator(
        ParametrizedModel(
            zero_model, VelocityParametrization(), process.schedule, score_key="score"
        ),
        neighbor_list=nbl,
    )
    sampler = Sampler(calculator, [SDEUpdate(v, eta2=0.0, integrator=Heun())], [v])
    batch = {
        properties.R: torch.randn(5, 3),
        properties.idx_m: torch.zeros(5, dtype=torch.long),
        properties.n_atoms: torch.tensor([5]),
    }
    out = sampler.run(batch, 4)
    assert nbl.calls == 8
    assert nbl.resets == 1
    assert properties.idx_i not in out and properties.idx_i not in batch


def test_prepare_moves_floats_to_the_run_dtype_and_leaves_the_rest():
    calculator = Calculator(zero_model, dtype=torch.float64)
    batch = {
        properties.R: torch.zeros(2, 3),
        properties.Z: torch.tensor([1, 6]),
        "flag": 7,
    }
    out = calculator.prepare(batch)
    assert out[properties.R].dtype == torch.float64
    assert out[properties.Z].dtype == torch.int64
    assert out["flag"] == 7
    assert batch[properties.R].dtype == torch.float32


def test_driver_runs_in_the_calculators_dtype():
    seen = []

    def model(batch):
        seen.append(batch[properties.R].dtype)
        return {"prediction": torch.zeros_like(batch[properties.R])}

    sampler = DirectDenoising(
        ForceCalculator(model, kind="pseudo", dtype=torch.float64)
    )
    out = sampler.run({properties.R: torch.randn(3, 3)}, 2)
    assert seen == [torch.float64, torch.float64]
    assert out[properties.R].dtype == torch.float64


def test_grad_policy_is_the_calculators():
    seen = []

    def model(batch):
        seen.append(torch.is_grad_enabled())
        return {"prediction": torch.zeros_like(batch[properties.R])}

    batch = {properties.R: torch.randn(2, 3)}
    Calculator(model)(batch)
    Calculator(model, enable_grad=True)(batch)
    assert seen == [False, True]


# ------------------------------------------------------------------ last-call cache


class CountingModel:
    def __init__(self):
        self.calls = 0

    def __call__(self, batch):
        self.calls += 1
        return {"prediction": batch[properties.R] * 2.0}


def test_no_cache_by_default():
    model = CountingModel()
    calculator = Calculator(model)
    batch = {properties.R: torch.ones(2, 3)}
    calculator(batch)
    calculator(batch)
    assert model.calls == 2


def test_the_cache_answers_an_unchanged_batch():
    model = CountingModel()
    calculator = Calculator(model, cache_last=True)
    batch = {properties.R: torch.ones(2, 3)}
    first = calculator(batch)
    # a new dict holding the same tensors is the same batch
    second = calculator(dict(batch))
    assert model.calls == 1
    assert torch.equal(first["prediction"], second["prediction"])


def test_the_cache_notices_a_replaced_or_edited_tensor():
    model = CountingModel()
    calculator = Calculator(model, cache_last=True)
    batch = {properties.R: torch.ones(2, 3), properties.t: torch.zeros(2)}
    calculator(batch)

    calculator({**batch, properties.R: torch.ones(2, 3)})
    assert model.calls == 2, "a replaced tensor is a new batch, even if equal"

    batch = {properties.R: torch.ones(2, 3), properties.t: torch.zeros(2)}
    calculator(batch)
    batch[properties.t] += 1.0
    calculator(batch)
    assert model.calls == 4, "an in-place edit of any key is noticed"

    calculator({**batch, "extra": torch.zeros(1)})
    assert model.calls == 5, "an added key is a new batch"


def test_reset_clears_the_cache():
    model = CountingModel()
    calculator = Calculator(model, cache_last=True)
    batch = {properties.R: torch.ones(2, 3)}
    calculator(batch)
    calculator.reset()
    calculator(batch)
    assert model.calls == 2


def test_cached_outputs_are_detached():
    def model(batch):
        r = batch[properties.R].requires_grad_()
        return {"prediction": r * 2.0}

    calculator = Calculator(model, enable_grad=True, cache_last=True)
    out = calculator({properties.R: torch.ones(2, 3)})
    assert not out["prediction"].requires_grad


def test_the_drivers_positions_never_require_grad():
    """The model marks its input positions; that must not reach the caller's tensor."""

    def model(batch):
        r = batch[properties.R].requires_grad_()
        return {"prediction": r * 2.0}

    batch = {properties.R: torch.ones(2, 3)}
    Calculator(model, enable_grad=True)(batch)
    assert not batch[properties.R].requires_grad


def test_a_batch_neighbor_list_plugs_in_as_a_transform():
    from schnetpack.dynamics import BatchNeighborList

    assert callable(BatchNeighborList.__call__)
    assert BatchNeighborList.__call__ is not object.__call__


# ------------------------------------------------------------- force-field units

KCAL = 1 / 23.0605480121  # eV per kcal/mol


class RecordingForceField:
    """Unit-sized outputs of every kind, and a record of the inputs it was handed."""

    def __init__(self):
        self.inputs = None
        self.grad_enabled = None

    def __call__(self, batch):
        self.inputs = dict(batch)
        self.grad_enabled = torch.is_grad_enabled()
        n_structures = batch[properties.n_atoms].shape[0]
        return {
            "energy": torch.ones(n_structures),
            "forces": torch.ones_like(batch[properties.R]),
            "stress": torch.ones(n_structures, 3, 3),
            "dipole_moment": torch.ones(n_structures, 3),
        }


def angstrom_batch():
    return {
        properties.n_atoms: torch.tensor([2, 1]),
        properties.R: torch.randn(3, 3),
        properties.cell: torch.randn(2, 3, 3),
    }


def test_the_model_and_the_neighbor_list_see_the_models_units():
    seen = {}

    def neighbor_list(inputs):
        seen["positions"] = inputs[properties.R].clone()
        return inputs

    model = RecordingForceField()
    batch = angstrom_batch()
    before = {key: value.clone() for key, value in batch.items()}
    ForceCalculator(model, neighbor_list=neighbor_list, position_unit="nm")(batch)

    torch.testing.assert_close(seen["positions"], batch[properties.R] / 10.0)
    torch.testing.assert_close(model.inputs[properties.R], batch[properties.R] / 10.0)
    torch.testing.assert_close(
        model.inputs[properties.cell], batch[properties.cell] / 10.0
    )
    for key, value in before.items():
        assert torch.equal(batch[key], value), key


def test_energy_forces_and_stress_come_back_in_ev_and_angstrom():
    calculator = ForceCalculator(
        RecordingForceField(),
        energy_unit="kcal/mol",
        position_unit="nm",
        stress_key="stress",
    )
    out = calculator(angstrom_batch())

    torch.testing.assert_close(out["energy"], torch.full((2,), KCAL))
    torch.testing.assert_close(out["forces"], torch.full((3, 3), KCAL / 10.0))
    torch.testing.assert_close(out["stress"], torch.full((2, 3, 3), KCAL / 1000.0))
    # what is not a force-field quantity passes through as the model reported it
    torch.testing.assert_close(out["dipole_moment"], torch.ones(2, 3))


def test_stress_is_left_alone_without_a_stress_key():
    calculator = ForceCalculator(RecordingForceField(), position_unit="nm")
    torch.testing.assert_close(
        calculator(angstrom_batch())["stress"], torch.ones(2, 3, 3)
    )


def test_a_model_in_ev_and_angstrom_gets_the_positions_as_they_are():
    model = RecordingForceField()
    batch = angstrom_batch()
    ForceCalculator(model, enable_grad=False)(batch)
    assert model.inputs[properties.R] is batch[properties.R]
    assert model.inputs[properties.cell] is batch[properties.cell]


@pytest.mark.parametrize("missing", ["energy", "forces"])
def test_a_force_field_must_return_energy_and_forces(missing):
    def model(batch):
        outputs = RecordingForceField()(batch)
        del outputs[missing]
        return outputs

    with pytest.raises(KeyError, match=missing):
        ForceCalculator(model)(angstrom_batch())


def test_a_force_field_runs_with_grad_by_default():
    model = RecordingForceField()
    ForceCalculator(model)(angstrom_batch())
    assert model.grad_enabled


def test_the_cache_answers_an_unchanged_angstrom_batch():
    calls = []

    def model(batch):
        calls.append(1)
        return {
            "energy": torch.zeros(2),
            "forces": torch.zeros_like(batch[properties.R]),
        }

    calculator = ForceCalculator(model, position_unit="nm", cache_last=True)
    batch = angstrom_batch()
    first = calculator(batch)
    second = calculator(batch)
    assert len(calls) == 1
    torch.testing.assert_close(first["forces"], second["forces"])


@pytest.mark.parametrize(
    "calculator",
    [ForceCalculator(zero_model), VelocityCalculator(zero_model)],
    ids=["force", "velocity"],
)
def test_the_sampler_refuses_a_single_field_calculator(calculator):
    # it reads every field of every variable from one model call
    v = Variable(
        "x", properties.R, "atom", GaussianInterpolant(VP()), EpsParametrization(), "y"
    )
    with pytest.raises(TypeError, match="JointCalculator"):
        Sampler(calculator, [SDEUpdate(v)], [v])


@pytest.mark.parametrize("build", [LBFGS, DirectDenoising], ids=["LBFGS", "DD"])
def test_time_free_drivers_refuse_a_generative_calculator(build):
    # its fields need a time, and it names no units
    calculator = PseudoForceCalculator(zero_model)
    with pytest.raises(TypeError, match="ForceCalculator"):
        build(calculator)


# ------------------------------------------------------------------ pseudo-forces


def test_a_pseudo_force_model_sees_time_zero_and_returns_a_length():
    seen = {}

    def gpff(batch):
        seen["t"] = batch[properties.t]
        return {"prediction": -2.0 * batch[properties.R]}

    batch = angstrom_batch()
    batch[properties.t] = torch.ones(3)
    for unit in ("Ang", "nm"):
        calculator = ForceCalculator(gpff, kind="pseudo", position_unit=unit)
        forces = calculator.forces(batch)
        assert torch.equal(seen["t"], torch.zeros(3))
        # the jump x + F/2 lands on the model's x0 = 0 in any unit
        torch.testing.assert_close(batch[properties.R] + forces / 2, torch.zeros(3, 3))
    assert not calculator.physical
    assert not calculator.enable_grad


def test_a_pseudo_force_has_no_energy_and_no_stress():
    calculator = ForceCalculator(zero_model, kind="pseudo")
    with pytest.raises(TypeError, match="energy"):
        calculator.energy(angstrom_batch())
    with pytest.raises(ValueError, match="stress"):
        ForceCalculator(zero_model, kind="pseudo", stress_key="stress")
    with pytest.raises(ValueError, match="kind"):
        ForceCalculator(zero_model, kind="virtual")


# ------------------------------------------------------------- generative fields


def toy_head(batch):
    x, t = batch[properties.R], batch[properties.t]
    return {"prediction": torch.tanh(x) * (1.0 + t[:, None])}


FIELD_CALCULATORS = [
    (ScoreCalculator, "score"),
    (NoiseCalculator, "noise"),
    (VelocityCalculator, "velocity"),
    (X0Calculator, "x0"),
    (PseudoForceCalculator, "pseudo_force"),
]


@pytest.mark.parametrize(
    "build, field", FIELD_CALCULATORS, ids=[f for _, f in FIELD_CALCULATORS]
)
def test_each_generative_calculator_reads_its_default_key(build, field):
    def model(batch):
        x, t = batch[properties.R], batch[properties.t]
        return {
            name: x * (i + t[:, None]) for i, (_, name) in enumerate(FIELD_CALCULATORS)
        }

    calculator = build(model)
    assert calculator.output_key == field
    x = torch.randn(4, 3, dtype=torch.float64)
    t = torch.full((4,), 0.4, dtype=torch.float64)
    torch.testing.assert_close(
        calculator.field({}, x, t), model({properties.R: x, properties.t: t})[field]
    )
    # a scalar time is spread over the rows
    torch.testing.assert_close(
        calculator.field({}, x, t[0]), calculator.field({}, x, t)
    )


@pytest.mark.parametrize(
    "build, field", FIELD_CALCULATORS, ids=[f for _, f in FIELD_CALCULATORS]
)
def test_a_generative_calculator_needs_its_key_in_the_outputs(build, field):
    calculator = build(toy_head)
    x = torch.randn(4, 3)
    with pytest.raises(KeyError, match=field):
        calculator.field({}, x, torch.tensor(0.4))


def test_the_output_key_can_be_overridden():
    calculator = ScoreCalculator(toy_head, output_key="prediction")
    x = torch.randn(4, 3)
    t = torch.full((4,), 0.4)
    torch.testing.assert_close(
        calculator.field({}, x, t),
        toy_head({properties.R: x, properties.t: t})["prediction"],
    )


@pytest.mark.parametrize(
    "parametrization",
    [EpsParametrization(), VelocityParametrization(), PseudoForceParametrization()],
)
def test_parametrized_model_writes_the_parametrizations_conversions(parametrization):
    schedule = VP()
    model = ParametrizedModel(
        toy_head,
        parametrization,
        schedule,
        score_key="score",
        noise_key="noise",
        velocity_key="velocity",
        x0_key="x0",
        pseudo_force_key="pseudo_force",
    )
    x = torch.randn(4, 3, dtype=torch.float64)
    t = torch.full((4,), 0.4, dtype=torch.float64)
    raw = toy_head({properties.R: x, properties.t: t})["prediction"]
    for build, field in FIELD_CALCULATORS:
        convert = getattr(parametrization, f"to_{field}")
        torch.testing.assert_close(
            build(model).field({}, x, t), convert(schedule, raw, x, t)
        )


def test_parametrized_model_writes_only_the_requested_fields():
    model = ParametrizedModel(
        toy_head, EpsParametrization(), VP(), score_key="s", x0_key="clean"
    )
    x = torch.randn(4, 3)
    t = torch.full((4,), 0.4)
    outputs = model({properties.R: x, properties.t: t})
    assert set(outputs) == {"prediction", "s", "clean"}
    # the model's own outputs are kept as they were
    torch.testing.assert_close(
        outputs["prediction"],
        toy_head({properties.R: x, properties.t: t})["prediction"],
    )


def test_parametrized_model_needs_a_field_key():
    with pytest.raises(ValueError, match="field key"):
        ParametrizedModel(toy_head, EpsParametrization(), VP())


def test_parametrized_model_validates_the_pair():
    with pytest.raises(TypeError, match="Gaussian kernel"):
        ParametrizedModel(
            toy_head, ScoreParametrization(), FlowMatching(), score_key="score"
        )
