"""
The Sampler over a list of updates: model calls per step, stage times,
write order, centered noise and the checks at assembly.
"""

import pytest
import torch

from schnetpack import properties
from schnetpack.dynamics import (
    AncestralUpdate,
    Heun,
    JointCalculator,
    ODEUpdate,
    Sampler,
    SDEUpdate,
    StatisticsStructures,
)
from schnetpack.dynamics.guidance import Guidance
from schnetpack.dynamics.hooks import FreezeScaffold
from schnetpack.dynamics.sample.grids import UniformGrid
from schnetpack.dynamics.sample.integrators import Integrator
from schnetpack.generative import (
    MASK,
    VE,
    VP,
    EpsParametrization,
    FlowMatching,
    GaussianInterpolant,
    GaussianPrior,
    Interpolant,
    LogitsParametrization,
    MaskedTypeWeight,
    MaskNoising,
    ParametrizedModel,
    Variable,
    VelocityParametrization,
)
from schnetpack.generative.noising import ForwardNoising

N_ATOMS = (3, 4)


def network(batch):
    """A smooth toy network with two heads, reading both x and t."""
    r, t = batch[properties.R], batch[properties.t]
    outputs = {
        "eps_R": 0.3 * torch.tanh(r) * (1.0 + t[:, None]),
        "velocity": 0.3 * torch.tanh(r),
    }
    if "y" in batch:
        outputs["eps_y"] = 0.2 * torch.sin(batch["y"]) + batch["t_structure"][:, None]
    return outputs


def structures(seed=1, y=False):
    """Two structures with positions, and a per-structure y if asked."""
    generator = torch.Generator().manual_seed(seed)
    n_atoms = torch.tensor(N_ATOMS)
    batch = {
        properties.n_atoms: n_atoms,
        properties.idx_m: torch.repeat_interleave(torch.arange(len(N_ATOMS)), n_atoms),
        properties.Z: torch.ones(sum(N_ATOMS), dtype=torch.long),
        properties.R: torch.randn(
            sum(N_ATOMS), 3, generator=generator, dtype=torch.float64
        ),
    }
    if y:
        batch["y"] = torch.randn(
            len(N_ATOMS), 2, generator=generator, dtype=torch.float64
        )
    return batch


def positions(schedule=None, centered=False):
    schedule = schedule or VP()
    return Variable(
        name="positions",
        key=properties.R,
        per="atom",
        noising=GaussianInterpolant(
            schedule, prior=GaussianPrior(std=schedule.sigma_max, centered=centered)
        ),
        parametrization=EpsParametrization(),
        output_key="eps_R",
    )


def y_variable():
    """A per-structure variable of width 2 on VP."""
    return Variable(
        name="y",
        key="y",
        per="structure",
        noising=GaussianInterpolant(
            VP(), prior=GaussianPrior(centered=False, key="y", per="structure", dim=2)
        ),
        parametrization=EpsParametrization(),
        output_key="eps_y",
    )


def fields(model, variable, **field_keys):
    """``model`` with ``variable``'s head read as the given fields."""
    return ParametrizedModel(
        model,
        variable.parametrization,
        variable.noising.schedule,
        output_key=variable.output_key,
        key=variable.key,
        time_key=variable.time_key,
        **field_keys,
    )


def counting(calls, model=network):
    def counted(batch):
        calls.append(batch)
        return model(batch)

    return counted


# --- model calls per step --------------------------------------------------- #


@pytest.mark.parametrize(
    "update, n_calls",
    [
        (lambda v: SDEUpdate(v, eta2=1.0), 1),
        (lambda v: SDEUpdate(v, eta2=0.0), 1),
        (lambda v: SDEUpdate(v, integrator=Heun()), 2),
        (lambda v: AncestralUpdate(v), 1),
        (lambda v: ODEUpdate(v), 1),
        (lambda v: ODEUpdate(v, integrator=Heun()), 2),
    ],
    ids=[
        "euler-sde",
        "euler-ode-chart",
        "heun-sde",
        "ancestral",
        "euler-ode",
        "heun-ode",
    ],
)
def test_model_calls_per_step(update, n_calls):
    calls = []
    v = positions()
    model = fields(counting(calls), v, score_key="score", x0_key="x0")
    sampler = Sampler(JointCalculator(model), [update(v)], [v])
    sampler.run(structures(), 3)
    assert len(calls) == 3 * n_calls


# --- several updates --------------------------------------------------------- #


def two_variable_model(seen=None):
    pos, y = positions(), y_variable()

    def recording(batch):
        if seen is not None:
            seen.append(
                {k: batch[k].clone() for k in (properties.R, "y", properties.t)}
            )
        return network(batch)

    model = fields(recording, pos, score_key="score_R")
    model = fields(model, y, x0_key="x0_y")
    return pos, y, model


def test_heun_next_to_an_ancestral_update_calls_twice_and_sees_its_draw():
    seen = []
    pos, y, model = two_variable_model(seen)
    sampler = Sampler(
        JointCalculator(model),
        [SDEUpdate(pos, "score_R", integrator=Heun()), AncestralUpdate(y, "x0_y")],
        [pos, y],
    )
    assert sampler.nodes == (0.0, 1.0)
    out = sampler.run(structures(y=True), 1)
    assert len(seen) == 2
    t, t_next = sampler.grid(1)
    assert torch.allclose(seen[0][properties.t], t.double())
    assert torch.allclose(seen[1][properties.t], t_next.double())
    # the second call sees y at t_next: the value the step ends with
    assert torch.equal(seen[1]["y"], out["y"])
    assert not torch.equal(seen[0]["y"], out["y"])


def test_updates_write_in_list_order_and_decode():
    order = []
    pos, y, model = two_variable_model()
    pos.decode = lambda batch: order.append("positions") or batch
    y.decode = lambda batch: order.append("y") or batch
    sampler = Sampler(
        JointCalculator(model),
        [AncestralUpdate(y, "x0_y"), SDEUpdate(pos, "score_R")],
        [pos, y],
    )
    t, t_next = torch.tensor(0.5), torch.tensor(0.4)
    sampler.step(sampler.calculator.with_time(structures(y=True), t), t, t_next)
    # one write per stage point, then combine writes in list order
    assert order[-2:] == ["y", "positions"]


def test_sample_draws_every_prior_in_order():
    pos, y, model = two_variable_model()
    provider = StatisticsStructures(
        n_atoms=torch.tensor([0, 0, 1, 1]), atom_types=torch.ones(3)
    )
    sampler = Sampler(
        JointCalculator(model, dtype=torch.float64),
        [SDEUpdate(pos, "score_R"), AncestralUpdate(y, "x0_y")],
        [pos, y],
        structures=provider,
    )
    out = sampler.sample(4, 3)
    assert out[properties.R].shape == (int(out[properties.n_atoms].sum()), 3)
    assert out["y"].shape == (4, 2)
    assert out["t_structure"].shape == (4,)


# --- centered noise ----------------------------------------------------------- #


def centers(batch):
    idx_m, r = batch[properties.idx_m], batch[properties.R]
    return torch.stack([r[idx_m == m].mean(0) for m in range(len(N_ATOMS))])


@pytest.mark.parametrize(
    "update",
    [
        lambda v: SDEUpdate(v, eta2=1.0, integrator=Heun()),
        lambda v: AncestralUpdate(v),
        lambda v: AncestralUpdate(v, noise="reverse_diffusion"),
    ],
    ids=["sde", "ancestral", "reverse-diffusion"],
)
def test_centered_noise_keeps_every_center_at_zero(update):
    v = positions(VE(sigma_min=0.01, sigma_max=10.0), centered=True)

    def linear(batch):  # a centered field keeps centered positions centered
        return {"eps_R": 0.3 * batch[properties.R]}

    model = fields(linear, v, score_key="score", x0_key="x0")
    batch = structures()
    batch[properties.R] = GaussianPrior.center(
        batch[properties.R], batch[properties.idx_m]
    )
    out = Sampler(JointCalculator(model), [update(v)], [v]).run(batch, 10)
    assert torch.allclose(
        centers(out), torch.zeros(2, 3, dtype=torch.float64), atol=1e-12
    )


def test_reverse_diffusion_noise_is_wider_than_the_posteriors():
    v = positions(VE(sigma_min=0.01, sigma_max=10.0))
    x0 = torch.zeros(7, 3, dtype=torch.float64)
    batch = structures()
    t, s = (
        torch.tensor(0.5, dtype=torch.float64),
        torch.tensor(0.4, dtype=torch.float64),
    )
    torch.manual_seed(0)
    posterior = AncestralUpdate(v).transition(batch, x0, t, s)
    torch.manual_seed(0)
    reverse = AncestralUpdate(v, noise="reverse_diffusion").transition(batch, x0, t, s)
    sigma = v.noising.schedule.sigma
    mean = (sigma(s) ** 2 / sigma(t) ** 2) * batch[properties.R]
    ratio = (reverse - mean) / (posterior - mean)
    assert torch.allclose(ratio, sigma(t) / sigma(s) * torch.ones_like(ratio))


# --- unmasking types ------------------------------------------------------------- #

N_CLASSES = 10


def types(t_min=1e-2):
    return Variable(
        name="types",
        key=properties.Z,
        per="atom",
        noising=MaskNoising(FlowMatching(t_min=t_min, t_max=1.0)),
        parametrization=LogitsParametrization(N_CLASSES),
        weight=MaskedTypeWeight(),
        output_key="type_logits",
    )


def unmask(z, t, s, excluded=()):
    """One unmasking step from t to s with uniform p(z0 | z_t)."""
    probs = torch.full((z.shape[0], N_CLASSES), 1.0 / (N_CLASSES - 1))
    probs[:, MASK] = 0.0
    update = AncestralUpdate(types(), "x0_Z", excluded=excluded)
    return update.transition({properties.Z: z}, probs, torch.tensor(t), torch.tensor(s))


def test_unmasking_reveals_a_fraction_of_the_step_over_t_and_keeps_revealed():
    torch.manual_seed(0)
    z = torch.full((40000,), MASK)
    z[:1000] = 3
    out = unmask(z, 0.5, 0.45)
    revealed = (out[1000:] != MASK).float().mean().item()
    assert revealed == pytest.approx(0.05 / 0.5, abs=0.006)
    assert (out[:1000] == 3).all()


def test_unmasking_reveals_every_atom_at_the_end_of_the_grid():
    torch.manual_seed(0)
    z = torch.full((1000,), MASK)
    assert (unmask(z, 0.02, 0.01) != MASK).all()  # s = t_min


def test_excluded_classes_are_never_drawn():
    torch.manual_seed(0)
    z = torch.full((5000,), MASK)
    out = unmask(z, 0.02, 0.01, excluded=[2, 3, 4, 5, 6, 7])
    assert set(out.tolist()) == {1, 8, 9}


def typed_network(batch):
    """Positions as in ``network``; type logits that favor 6 and read R."""
    logits = torch.zeros(batch[properties.Z].shape[0], N_CLASSES)
    logits[:, 6] = 2.0 + batch[properties.R][:, 0].float()
    logits[:, 9] = 1.0  # excluded below
    return {**network(batch), "type_logits": logits}


def types_and_positions(seen=None):
    pos, z = positions(VE(sigma_min=0.01, sigma_max=10.0), centered=True), types()

    def recording(batch):
        if seen is not None:
            seen.append(batch[properties.Z].clone())
        return typed_network(batch)

    model = fields(recording, pos, score_key="score_R")
    model = fields(model, z, x0_key="x0_Z")
    return pos, z, model


@pytest.mark.parametrize("integrator", [None, Heun()], ids=["euler", "heun"])
def test_sampling_types_and_positions_unmasks_every_atom(integrator):
    torch.manual_seed(0)
    seen = []
    pos, z, model = types_and_positions(seen)
    provider = StatisticsStructures(
        n_atoms=torch.tensor([0, 0, 0, 1, 1]), atom_types=None
    )
    sampler = Sampler(
        JointCalculator(model, dtype=torch.float64),
        [
            SDEUpdate(pos, "score_R", integrator=integrator),
            AncestralUpdate(z, "x0_Z", excluded=[9]),
        ],
        [z, pos],
        structures=provider,
    )
    out = sampler.sample(64, 20)
    assert (seen[0] == MASK).all()
    masked = [(s == MASK).float().mean().item() for s in seen]
    assert all(a >= b for a, b in zip(masked, masked[1:]))
    assert (out[properties.Z] != MASK).all()
    assert 9 not in out[properties.Z].tolist()
    assert out[properties.R].shape == (int(out[properties.n_atoms].sum()), 3)


def test_the_grid_has_to_reach_the_t_min_of_a_masked_variable():
    pos, z, model = types_and_positions()
    updates = [SDEUpdate(pos, "score_R"), AncestralUpdate(z, "x0_Z")]
    with pytest.raises(ValueError, match="would stay masked"):
        Sampler(JointCalculator(model), updates, [z, pos], grid=UniformGrid(1.0, 0.1))
    # the default grid ends at the largest t_min: the types'
    assert Sampler(JointCalculator(model), updates, [z, pos]).grid.t_end == 1e-2


def test_only_the_ancestral_update_steps_types_and_without_gaussian_options():
    z = types()
    calculator = JointCalculator(typed_network)
    with pytest.raises(TypeError):
        SDEUpdate(z)
    with pytest.raises(TypeError, match="interpolant"):
        Sampler(calculator, [ODEUpdate(z)], [z])
    with pytest.raises(ValueError, match="draws no noise"):
        Sampler(calculator, [AncestralUpdate(z, noise="reverse_diffusion")], [z])
    pos = positions()
    with pytest.raises(ValueError, match="excluded classes"):
        Sampler(calculator, [AncestralUpdate(pos, excluded=[1])], [pos])


# --- hooks and guidance --------------------------------------------------------- #


def test_freeze_scaffold_holds_the_reference_on_a_sampler():
    v = positions()
    batch = structures()
    mask = torch.tensor([True, False, True, False, False, False, False])
    batch = {
        **batch,
        properties.fixed_atoms: mask,
        properties.R_reference: torch.zeros_like(batch[properties.R]),
    }
    sampler = Sampler(
        JointCalculator(fields(network, v, score_key="score")),
        [SDEUpdate(v)],
        [v],
        hooks=[FreezeScaffold()],
    )
    torch.manual_seed(0)
    x = sampler.run(batch, 20)[properties.R]
    assert torch.equal(x[mask], torch.zeros_like(x[mask]))


class Pull(Guidance):
    """A linear pull toward the origin, a guidance term that depends on t."""

    def forward(self, batch):
        return -batch[properties.R] * (1.0 + batch[properties.t][:, None])


def test_guidance_is_added_to_the_guided_fields_at_the_stage_time():
    v = positions()
    model = fields(network, v, score_key="score", x0_key="x0")
    plain = JointCalculator(model)
    guided = JointCalculator(
        model, guidance=[Pull(weight=2.0)], guided_fields=["score"]
    )
    batch, t = structures(), torch.tensor(0.7, dtype=torch.float64)
    shift = guided.fields(batch, t)["score"] - plain.fields(batch, t)["score"]
    assert torch.allclose(shift, -2.0 * 1.7 * batch[properties.R])
    assert torch.equal(guided.fields(batch, t)["x0"], plain.fields(batch, t)["x0"])


def test_guidance_needs_a_field_to_go_to():
    with pytest.raises(ValueError, match="guided_fields"):
        JointCalculator(network, guidance=[Pull()])
    calculator = JointCalculator(network, guidance=[Pull()], guided_fields=["score"])
    with pytest.raises(KeyError, match="'score'"):
        calculator.fields(structures(), torch.tensor(0.5))


def test_calculator_writes_the_time_per_atom_and_per_structure():
    batch = JointCalculator(network).with_time(structures(), torch.tensor(0.3))
    assert batch[properties.t].shape == (7,)
    assert batch["t_structure"].shape == (2,)


# --- checks at assembly ---------------------------------------------------------- #


def test_every_variable_needs_exactly_one_update():
    pos, y, model = two_variable_model()
    calculator = JointCalculator(model)
    with pytest.raises(ValueError, match="without one: \\['y'\\]"):
        Sampler(calculator, [SDEUpdate(pos, "score_R")], [pos, y])
    with pytest.raises(ValueError, match="more than one update"):
        Sampler(calculator, [SDEUpdate(pos), AncestralUpdate(pos)], [pos])
    with pytest.raises(ValueError, match="not among the variables"):
        Sampler(calculator, [SDEUpdate(pos), AncestralUpdate(y)], [pos])


def test_updates_must_fit_the_noising():
    flow = Variable(
        "flow",
        properties.R,
        "atom",
        Interpolant(FlowMatching()),
        VelocityParametrization(),
        "eps_R",
    )
    calculator = JointCalculator(network)
    with pytest.raises(TypeError, match="SDE chart"):
        SDEUpdate(flow)
    with pytest.raises(TypeError, match="Gaussian interpolant"):
        Sampler(calculator, [AncestralUpdate(flow)], [flow])

    class Swap(ForwardNoising):
        schedule = VP()

        def perturb(self, x0, t=None, batch=None, groups=None):
            return x0, x0, None, None

    swap = Variable("swap", properties.R, "atom", Swap(), EpsParametrization(), "y")
    with pytest.raises(TypeError, match="interpolant"):
        Sampler(calculator, [ODEUpdate(swap)], [swap])


class Midpoint(Integrator):
    n_stages = 2
    nodes = (0.0, 0.5)

    def point(self, stage, x, t, t_next, drifts):
        raise NotImplementedError

    def combine(self, x, t, t_next, drifts):
        raise NotImplementedError


def test_updates_must_agree_on_stage_times():
    pos, y, model = two_variable_model()
    with pytest.raises(ValueError, match="disagree on the time of stage 1"):
        Sampler(
            JointCalculator(model),
            [
                SDEUpdate(pos, "score_R", integrator=Midpoint()),
                AncestralUpdate(y, "x0_y"),
            ],
            [pos, y],
        )
    # alone, a midpoint stage is fine
    sampler = Sampler(
        JointCalculator(model),
        [SDEUpdate(pos, "score_R", integrator=Midpoint())],
        [pos],
    )
    assert sampler.nodes == (0.0, 0.5)


def test_an_integrator_needs_one_node_per_stage():
    class Broken(Midpoint):
        nodes = (0.0,)

    with pytest.raises(ValueError, match="2 stages but 1 nodes"):
        SDEUpdate(positions(), integrator=Broken())


def test_a_missing_field_names_what_the_model_returned():
    v = positions()
    sampler = Sampler(
        JointCalculator(fields(network, v, x0_key="x0")), [SDEUpdate(v)], [v]
    )
    with pytest.raises(KeyError, match="reads 'score'"):
        sampler.run(structures(), 1)


def test_sampler_needs_a_joint_calculator():
    from schnetpack.dynamics import ScoreCalculator

    v = positions()
    with pytest.raises(TypeError, match="JointCalculator"):
        Sampler(ScoreCalculator(network), [SDEUpdate(v)], [v])


def test_a_variable_and_its_prior_share_the_key():
    with pytest.raises(ValueError, match="give the prior key='y'"):
        Variable(
            "y", "y", "structure", GaussianInterpolant(VP()), EpsParametrization(), "e"
        )
