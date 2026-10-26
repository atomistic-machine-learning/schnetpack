"""
The Sampler over a list of updates: model calls per step, stage times,
write order, centered noise and the checks at assembly.
"""

import dataclasses

import pytest
import torch

from schnetpack import properties
from schnetpack.dynamics import (
    AncestralUpdate,
    FieldCalculator,
    Heun,
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
    CellParameters,
    EpsParametrization,
    FlowMatching,
    GaussianInterpolant,
    GaussianPrior,
    Interpolant,
    KeyView,
    LogitsParametrization,
    MaskedTypeWeight,
    MaskNoising,
    PeriodicGaussianInterpolant,
    PeriodicPositions,
    Variable,
    VelocityParametrization,
    VPCosine,
    X0Parametrization,
)
from schnetpack.generative.noising import ForwardNoising
from schnetpack.utils.cell import (
    atom_cells,
    minimum_image,
    to_cart,
    to_frac,
)

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
        view=properties.R,
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
        view="y",
        per="structure",
        noising=GaussianInterpolant(
            VP(), prior=GaussianPrior(centered=False, key="y", per="structure", dim=2)
        ),
        parametrization=EpsParametrization(),
        output_key="eps_y",
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
    sampler = Sampler(FieldCalculator(counting(calls)), [update(v)], [v])
    sampler.run(structures(), 3)
    assert len(calls) == 3 * n_calls


# --- fields ------------------------------------------------------------------ #


@pytest.mark.parametrize(
    "update, convert",
    [(SDEUpdate, "to_score"), (ODEUpdate, "to_velocity"), (AncestralUpdate, "to_x0")],
    ids=["score", "velocity", "x0"],
)
def test_an_update_converts_the_head_with_its_variables_parametrization(
    update, convert
):
    v = positions()
    calculator = FieldCalculator(network)
    t = torch.tensor(0.4, dtype=torch.float64)
    batch = calculator.with_time(structures(), t)
    outputs = calculator(batch)
    expected = getattr(v.parametrization, convert)(
        v.noising.schedule, outputs["eps_R"], batch[properties.R], t
    )
    torch.testing.assert_close(update(v).field(batch, outputs, t), expected)


# --- several updates --------------------------------------------------------- #


def two_variable_model(seen=None):
    pos, y = positions(), y_variable()

    def recording(batch):
        if seen is not None:
            seen.append(
                {k: batch[k].clone() for k in (properties.R, "y", properties.t)}
            )
        return network(batch)

    return pos, y, recording


def test_heun_next_to_an_ancestral_update_calls_twice_and_sees_its_draw():
    seen = []
    pos, y, model = two_variable_model(seen)
    sampler = Sampler(
        FieldCalculator(model),
        [SDEUpdate(pos, integrator=Heun()), AncestralUpdate(y)],
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


class Recording(KeyView):
    """A key view that notes every write."""

    def __init__(self, key, order):
        super().__init__(key)
        self.order = order

    def write(self, batch, x):
        self.order.append(self.key)
        return super().write(batch, x)


def test_updates_write_in_list_order():
    order = []
    pos, y, model = two_variable_model()
    pos.view = Recording(properties.R, order)
    y.view = Recording("y", order)
    sampler = Sampler(
        FieldCalculator(model),
        [AncestralUpdate(y), SDEUpdate(pos)],
        [pos, y],
    )
    t, t_next = torch.tensor(0.5), torch.tensor(0.4)
    sampler.step(sampler.calculator.with_time(structures(y=True), t), t, t_next)
    # one write per stage point, then combine writes in list order
    assert order[-2:] == ["y", properties.R]


def test_sample_draws_every_prior_in_order():
    pos, y, model = two_variable_model()
    provider = StatisticsStructures(
        n_atoms=torch.tensor([0, 0, 1, 1]), atom_types=torch.ones(3)
    )
    sampler = Sampler(
        FieldCalculator(model, dtype=torch.float64),
        [SDEUpdate(pos), AncestralUpdate(y)],
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

    batch = structures()
    batch[properties.R] = GaussianPrior.center(
        batch[properties.R], batch[properties.idx_m]
    )
    out = Sampler(FieldCalculator(linear), [update(v)], [v]).run(batch, 10)
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
        view=properties.Z,
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
    update = AncestralUpdate(types(), excluded=excluded)
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

    return pos, z, recording


@pytest.mark.parametrize("integrator", [None, Heun()], ids=["euler", "heun"])
def test_sampling_types_and_positions_unmasks_every_atom(integrator):
    torch.manual_seed(0)
    seen = []
    pos, z, model = types_and_positions(seen)
    provider = StatisticsStructures(
        n_atoms=torch.tensor([0, 0, 0, 1, 1]), atom_types=None
    )
    sampler = Sampler(
        FieldCalculator(model, dtype=torch.float64),
        [
            SDEUpdate(pos, integrator=integrator),
            AncestralUpdate(z, excluded=[9]),
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
    updates = [SDEUpdate(pos), AncestralUpdate(z)]
    with pytest.raises(ValueError, match="would stay masked"):
        Sampler(FieldCalculator(model), updates, [z, pos], grid=UniformGrid(1.0, 0.1))
    # the default grid ends at the largest t_min: the types'
    assert Sampler(FieldCalculator(model), updates, [z, pos]).grid.t_end == 1e-2


def test_only_the_ancestral_update_steps_types_and_without_gaussian_options():
    z = types()
    calculator = FieldCalculator(typed_network)
    with pytest.raises(TypeError):
        SDEUpdate(z)
    with pytest.raises(TypeError, match="interpolant"):
        Sampler(calculator, [ODEUpdate(z)], [z])
    with pytest.raises(ValueError, match="draws no noise"):
        Sampler(calculator, [AncestralUpdate(z, noise="reverse_diffusion")], [z])
    pos = positions()
    with pytest.raises(ValueError, match="excluded classes"):
        Sampler(calculator, [AncestralUpdate(pos, excluded=[1])], [pos])


# --- crystals ------------------------------------------------------------------ #

CELL = CellParameters(length=2.94, angle=99.7, var_length=2.52, var_angle=297.0)


def crystal_variables():
    cell = Variable(
        name="cell",
        view=CELL,
        per="structure",
        noising=GaussianInterpolant(
            VPCosine(t_min=1e-2),
            prior=GaussianPrior(
                std=1.0, centered=False, key="cell_y", per="structure", dim=6
            ),
        ),
        parametrization=EpsParametrization(),
        output_key="eps_cell",
    )
    pos = Variable(
        name="positions",
        view=PeriodicPositions(),
        per="atom",
        noising=PeriodicGaussianInterpolant(VE(sigma_min=0.01, sigma_max=10.0)),
        parametrization=EpsParametrization(),
        output_key="eps_R",
    )
    return cell, types(), pos


Y_TARGET = torch.tensor([1.0, -2.0, 0.5, 3.0, -1.0, 0.0], dtype=torch.float64)


def crystal_network(schedule):
    """
    No noise predicted on the positions, the noise that leads the cell's
    parameters from Y_TARGET to the noisy cell, types 6.
    """

    def network(batch):
        y_t, t = CELL.read(batch), batch["t_structure"][:, None]
        logits = torch.zeros(
            batch[properties.Z].shape[0], N_CLASSES, dtype=torch.float64
        )
        logits[:, 6] = 50.0
        return {
            "eps_R": torch.zeros_like(batch[properties.R]),
            "eps_cell": (y_t - schedule.a(t) * Y_TARGET) / schedule.sigma(t),
            "type_logits": logits,
        }

    return network


def test_a_cell_update_moves_the_atoms_at_fixed_fractional_coordinates():
    torch.manual_seed(0)
    cell, _, _ = crystal_variables()
    cells = torch.tensor([[4.0, 0.5, 0.0], [0.0, 4.5, 0.3], [0.2, 0.0, 5.0]])
    n_atoms = torch.tensor([2, 3])
    batch = {
        properties.n_atoms: n_atoms,
        properties.idx_m: torch.tensor([0, 0, 1, 1, 1]),
        properties.cell: cells.double().expand(2, 3, 3).clone(),
    }
    frac = torch.rand(5, 3, dtype=torch.float64)
    batch[properties.R] = to_cart(frac, atom_cells(batch))
    t, s = (
        torch.tensor(0.5, dtype=torch.float64),
        torch.tensor(0.4, dtype=torch.float64),
    )
    schedule = cell.noising.schedule
    # the noise that leads from Y_TARGET to the cell: x0_hat = Y_TARGET
    outputs = {
        "eps_cell": (CELL.read(batch) - schedule.a(t) * Y_TARGET) / schedule.sigma(t)
    }
    out, y = AncestralUpdate(cell).stage(0, batch, outputs, t, batch, t, s, [])
    torch.testing.assert_close(CELL.read(out), y)
    torch.testing.assert_close(to_frac(out[properties.R], atom_cells(out)), frac)


def test_sampling_a_crystal_keeps_its_atoms_in_the_cell_and_reaches_its_x0():
    torch.manual_seed(0)
    cell, z, pos = crystal_variables()
    sampler = Sampler(
        FieldCalculator(crystal_network(cell.noising.schedule), dtype=torch.float64),
        [
            AncestralUpdate(pos, noise="reverse_diffusion"),
            AncestralUpdate(cell),  # y_t read through the view
            AncestralUpdate(z),
        ],
        [cell, z, pos],
        structures=StatisticsStructures(torch.tensor([0, 0, 1, 1]), None, pbc=True),
    )
    out = sampler.sample(6, 50)
    assert out[properties.pbc].all()
    assert (out[properties.Z] == 6).all()
    frac = to_frac(out[properties.R], atom_cells(out))
    assert frac.min() >= 0 and frac.max() < 1
    # an exact x0 is reached up to the cell's last noise, sigma(t_min) = 0.026
    torch.testing.assert_close(CELL.read(out), Y_TARGET.expand(6, 6), atol=0.15, rtol=0)


PAIR_CELL = torch.tensor(
    [[5.0, 0.0, 0.0], [1.0, 5.5, 0.0], [0.5, -0.7, 6.0]], dtype=torch.float64
)
PAIR = torch.tensor([1.2, 1.9, -1.4], dtype=torch.float64)


def exact_pair_denoiser(schedule):
    """
    x0_hat of two atoms in PAIR_CELL whose separation is +-PAIR: the
    translation and the swap integrate out, and the separation r is wrapped
    N(+-PAIR, 2 sigma^2).
    """
    r = torch.arange(-2, 3, dtype=torch.float64)
    images = torch.cartesian_prod(r, r, r) @ PAIR_CELL

    def model(batch):
        x = batch[properties.R].view(-1, 2, 3)
        sigma = schedule.sigma(batch[properties.t].view(-1, 2)[:, 0])[:, None]
        rel = minimum_image(x[:, 1] - x[:, 0], PAIR_CELL)
        d = torch.stack([rel - PAIR, rel + PAIR], 1)[:, :, None, :] + images
        w = torch.softmax(
            (-(d**2).sum(-1) / (4 * sigma[:, :, None] ** 2)).flatten(1), -1
        ).view(d.shape[:-1])
        grad = -(w[..., None] * d).sum((1, 2)) / (2 * sigma**2)  # d/dr log p
        x0 = x + sigma[:, :, None] ** 2 * torch.stack([-grad, grad], 1)
        return {"x0_R": x0.view(-1, 3)}

    return model


@pytest.mark.parametrize("noise", ["posterior", "reverse_diffusion"])
def test_the_exact_periodic_denoiser_recovers_a_pair_from_the_uniform_start(noise):
    torch.manual_seed(0)
    _, _, pos = crystal_variables()
    # the denoiser is an x0 head
    pos = dataclasses.replace(
        pos, parametrization=X0Parametrization(), output_key="x0_R"
    )
    n = 100

    def pairs(n_samples):
        return {
            properties.n_atoms: torch.full((n_samples,), 2),
            properties.idx_m: torch.repeat_interleave(torch.arange(n_samples), 2),
            properties.Z: torch.full((2 * n_samples,), 14),
            properties.cell: PAIR_CELL.expand(n_samples, 3, 3).clone(),
            properties.pbc: torch.ones(3 * n_samples, dtype=torch.bool),
        }

    sampler = Sampler(
        FieldCalculator(exact_pair_denoiser(pos.noising.schedule), dtype=torch.float64),
        [AncestralUpdate(pos, noise=noise)],
        [pos],
        grid=UniformGrid(1.0, 1e-3),
        structures=pairs,
    )
    out = sampler.sample(n, 200)
    x = out[properties.R].view(n, 2, 3)
    rel = minimum_image(x[:, 1] - x[:, 0], PAIR_CELL)
    error = torch.minimum((rel - PAIR).norm(dim=-1), (rel + PAIR).norm(dim=-1))
    assert error.max() < 0.1  # sigma(t_min) = 0.01 per atom remains


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
        FieldCalculator(network),
        [SDEUpdate(v)],
        [v],
        hooks=[FreezeScaffold()],
    )
    torch.manual_seed(0)
    x = sampler.run(batch, 20)[properties.R]
    assert torch.equal(x[mask], torch.zeros_like(x[mask]))


class Pull(Guidance):
    """A linear pull toward the origin that depends on t; notes the times it sees."""

    def __init__(self, weight=1.0):
        super().__init__(weight)
        self.seen = []

    def forward(self, batch):
        t = batch[properties.t]
        self.seen.append(t)
        return -batch[properties.R] * (1.0 + t[:, None])


def test_guidance_is_added_to_the_updates_field():
    v = positions()
    calculator = FieldCalculator(network)
    t = torch.tensor(0.7, dtype=torch.float64)
    batch = calculator.with_time(structures(), t)
    outputs = calculator(batch)
    guided, plain = SDEUpdate(v, guidance=[Pull(weight=2.0)]), SDEUpdate(v)
    shift = guided.field(batch, outputs, t) - plain.field(batch, outputs, t)
    assert torch.allclose(shift, -2.0 * 1.7 * batch[properties.R])


def test_guidance_is_evaluated_at_every_stage_time():
    v, pull = positions(), Pull()
    sampler = Sampler(
        FieldCalculator(network),
        [SDEUpdate(v, integrator=Heun(), guidance=[pull])],
        [v],
    )
    sampler.run(structures(), 1)
    t, t_next = sampler.grid(1).double()
    assert len(pull.seen) == 2
    assert torch.allclose(pull.seen[0], t.expand(7))
    assert torch.allclose(pull.seen[1], t_next.expand(7))


class Lift(Guidance):
    """A guidance term on the per-structure y: t on every component."""

    def forward(self, batch):
        return batch["t_structure"][:, None].expand_as(batch["y"]).clone()


def test_guidance_is_added_to_the_field_of_any_variable():
    y = y_variable()
    calculator = FieldCalculator(network)
    t = torch.tensor(0.5, dtype=torch.float64)
    batch = calculator.with_time(structures(y=True), t)
    outputs = calculator(batch)
    guided, plain = AncestralUpdate(y, guidance=[Lift(weight=3.0)]), AncestralUpdate(y)
    shift = guided.field(batch, outputs, t) - plain.field(batch, outputs, t)
    assert torch.allclose(shift, torch.full((2, 2), 1.5, dtype=torch.float64))


def test_guidance_has_to_be_shaped_like_the_field():
    y = y_variable()
    calculator = FieldCalculator(network)
    t = torch.tensor(0.5, dtype=torch.float64)
    batch = calculator.with_time(structures(y=True), t)
    update = AncestralUpdate(y, guidance=[Pull()])  # a term on the positions
    with pytest.raises(
        ValueError, match="'y' is shaped \\(7, 3\\), its field \\(2, 2\\)"
    ):
        update.field(batch, calculator(batch), t)


def test_calculator_writes_the_time_per_atom_and_per_structure():
    batch = FieldCalculator(network).with_time(structures(), torch.tensor(0.3))
    assert batch[properties.t].shape == (7,)
    assert batch["t_structure"].shape == (2,)


# --- checks at assembly ---------------------------------------------------------- #


def test_every_variable_needs_exactly_one_update():
    pos, y, model = two_variable_model()
    calculator = FieldCalculator(model)
    with pytest.raises(ValueError, match="without one: \\['y'\\]"):
        Sampler(calculator, [SDEUpdate(pos)], [pos, y])
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
    calculator = FieldCalculator(network)
    with pytest.raises(TypeError, match="SDE chart"):
        SDEUpdate(flow)
    with pytest.raises(TypeError, match="Gaussian interpolant"):
        Sampler(calculator, [AncestralUpdate(flow)], [flow])
    # a head that is no velocity reaches it through the score
    denoised = dataclasses.replace(flow, parametrization=X0Parametrization())
    with pytest.raises(TypeError, match="through the score"):
        Sampler(calculator, [ODEUpdate(denoised)], [denoised])

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
            FieldCalculator(model),
            [SDEUpdate(pos, integrator=Midpoint()), AncestralUpdate(y)],
            [pos, y],
        )
    # alone, a midpoint stage is fine
    sampler = Sampler(
        FieldCalculator(model), [SDEUpdate(pos, integrator=Midpoint())], [pos]
    )
    assert sampler.nodes == (0.0, 0.5)


def test_an_integrator_needs_one_node_per_stage():
    class Broken(Midpoint):
        nodes = (0.0,)

    with pytest.raises(ValueError, match="2 stages but 1 nodes"):
        SDEUpdate(positions(), integrator=Broken())
