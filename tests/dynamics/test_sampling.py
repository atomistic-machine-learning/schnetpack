import pytest
import torch

from schnetpack import properties
from schnetpack.dynamics import (
    AncestralSampler,
    Euler,
    Heun,
    ODESampler,
    ScoreCalculator,
    SDESampler,
    UniformGrid,
    VelocityCalculator,
    X0Calculator,
)
from schnetpack.dynamics.sample import sampler as sample
from schnetpack.generative import (
    SDE,
    VE,
    VP,
    EpsParametrization,
    FlowMatching,
    GaussianInterpolant,
    GaussianPrior,
    Interpolant,
    MatchingLoss,
    ParametrizedModel,
    PermutationCoupling,
    ReverseSDE,
    ScoreParametrization,
    StatisticsStructures,
    VelocityParametrization,
    X0Parametrization,
    expand_t,
)


def batch_model(fn):
    """Wrap a tensor field fn(x, t) into the batch contract batch -> outputs."""
    return lambda batch: {"prediction": fn(batch[properties.R], batch[properties.t])}


#: placeholder calculator for drivers that are only assembled, never run
IDLE = batch_model(lambda x, t: torch.zeros_like(x))


def score_calculator(model, process, parametrization=None, key=properties.R, **kw):
    """The score of a raw-head ``model`` trained on ``process`` under ``parametrization`` (default: score)."""
    parametrization = parametrization or ScoreParametrization()
    return ScoreCalculator(
        ParametrizedModel(
            model, parametrization, process.schedule, key=key, score_key="score"
        ),
        key=key,
        **kw,
    )


def x0_calculator(model, process, parametrization=None, key=properties.R, **kw):
    """The denoised sample of a raw-head ``model`` trained on ``process`` under ``parametrization`` (default: score)."""
    parametrization = parametrization or ScoreParametrization()
    return X0Calculator(
        ParametrizedModel(
            model, parametrization, process.schedule, key=key, x0_key="x0"
        ),
        key=key,
        **kw,
    )


def draw(dynamics, shape, n_steps):
    """Sample from the prior; return the positions."""
    template = {properties.R: torch.empty(shape)}
    return dynamics.run(dynamics.prior.sample_from_batch(template), n_steps)[
        properties.R
    ]


@pytest.fixture
def vp():
    return GaussianInterpolant(VP())


def analytic_score(process, mu0, s0, x1_std=1.0):
    """Exact score of p_t for Gaussian data N(mu0, s0^2) under an affine schedule."""

    def score(x, t, cond=None):
        a = expand_t(process.schedule.a(t), x)
        sigma = expand_t(process.schedule.b(t), x) * x1_std
        var = a**2 * s0**2 + sigma**2
        return -(x - a * mu0) / var

    return score


# --- grids ---------------------------------------------------------------- #


def test_uniform_grid_endpoints_and_monotonicity():
    ts = UniformGrid(1.0, 1e-3)(10)
    assert ts.shape == (11,)
    assert ts[0].item() == pytest.approx(1.0)
    assert ts[-1].item() == pytest.approx(1e-3)
    assert (ts.diff() < 0).all()


def test_uniform_grid_defaults_to_the_unit_interval_reversed():
    ts = UniformGrid()(4)
    assert ts[0].item() == pytest.approx(1.0)
    assert ts[-1].item() == pytest.approx(0.0)


def test_grid_owns_its_range():
    ts = UniformGrid(0.5, 0.1)(4)
    assert ts.shape == (5,)
    assert ts[0].item() == pytest.approx(0.5)
    assert ts[-1].item() == pytest.approx(0.1)


def test_grids_respect_dtype():
    assert UniformGrid(1.0, 0.0)(4, dtype=torch.float64).dtype == torch.float64


# --- priors --------------------------------------------------------------- #


def test_gaussian_prior_has_the_declared_std():
    torch.manual_seed(0)
    samples = GaussianPrior(std=3.0).sample_positions(
        {properties.R: torch.empty(10000, 1)}
    )
    assert samples.std().item() == pytest.approx(3.0, rel=0.05)


def test_sampler_defaults_to_the_forward_process_prior(vp):
    # sigma(t_max) = 1 on VP, so the default start is N(0, I).
    sampler = SDESampler(score_calculator(IDLE, vp), vp)
    assert sampler.prior is vp.prior
    assert sampler.prior.std == pytest.approx(1.0)


def test_sampler_derives_the_scale_from_the_forward_process():
    process = GaussianInterpolant(VE(0.01, 50.0))
    sampler = SDESampler(score_calculator(IDLE, process), process)
    assert sampler.prior is process.prior
    assert sampler.prior.std == pytest.approx(50.0)


def test_sampler_takes_a_given_prior(vp):
    prior = GaussianPrior(std=2.0)
    sampler = SDESampler(score_calculator(IDLE, vp), vp, prior=prior)
    assert sampler.prior is prior


def test_sde_sampler_default_grid_spans_the_schedule():
    process = GaussianInterpolant(VE(0.01, 3.0))
    ts = SDESampler(score_calculator(IDLE, process), process).grid(4)
    assert ts[0].item() == pytest.approx(process.schedule.t_max)
    assert ts[-1].item() == pytest.approx(process.schedule.t_min)


def test_sde_sampler_exposes_its_reverse_sde(vp):
    sampler = SDESampler(score_calculator(IDLE, vp), vp, eta2=0.25)
    assert sampler.forward_process is vp
    assert sampler.eta2 == 0.25
    assert isinstance(sampler.integrator, Euler)
    assert sampler.diffusion(torch.rand(3)) is not None
    assert isinstance(sampler.reverse_sde, ReverseSDE)
    assert sampler.reverse_sde.eta2 == 0.25
    assert sampler.reverse_sde.sde.schedule is vp.schedule


# --- the reverse process -------------------------------------------------- #


def test_reverse_drift_at_eta2_one_matches_the_anderson_form(vp):
    # Anderson: f x - 1/2 (1 + eta^2) g^2 s, at eta = 1 (eta2 = eta^2 = 1).
    rev = ReverseSDE(SDE(vp.schedule), eta2=1.0)

    x = torch.randn(16, 2, dtype=torch.float64)
    t = torch.full((16,), 0.5, dtype=torch.float64)

    s = analytic_score(vp, 0.5, 1.0)(x, t)
    sde = SDE(vp.schedule)
    expected = expand_t(sde.f(t), x) * x - expand_t(sde.g2(t), x) * s
    assert torch.allclose(rev.drift(x, t, s), expected, rtol=1e-8)


def test_reverse_drift_at_eta2_zero_is_the_probability_flow(vp):
    rev = ReverseSDE(SDE(vp.schedule), eta2=0.0)

    x = torch.randn(16, 2, dtype=torch.float64)
    t = torch.full((16,), 0.5, dtype=torch.float64)

    s = analytic_score(vp, 0.5, 1.0)(x, t)
    sde = SDE(vp.schedule)
    expected = expand_t(sde.f(t), x) * x - 0.5 * expand_t(sde.g2(t), x) * s
    assert torch.allclose(rev.drift(x, t, s), expected, rtol=1e-8)


@pytest.mark.parametrize("eta2", [0.0, 0.25, 1.0])
def test_reverse_diffusion(vp, eta2):
    t = torch.tensor([0.3, 0.7], dtype=torch.float64)
    rev = ReverseSDE(SDE(vp.schedule), eta2=eta2)
    expected = torch.sqrt(eta2 * SDE(vp.schedule).g2(t))
    assert torch.allclose(rev.diffusion(t), expected)


def test_reverse_process_exposes_the_reversed_process(vp):
    sde = SDE(vp.schedule)
    rev = ReverseSDE(sde, eta2=0.5)
    assert rev.sde is sde
    assert rev.eta2 == 0.5


def test_sde_reverse_builds_the_reverse_of_the_schedules_sde(vp):
    rev = SDE.reverse(vp.schedule, eta2=0.3)
    assert isinstance(rev, ReverseSDE)
    assert rev.eta2 == 0.3
    assert isinstance(rev.sde, SDE)
    assert rev.sde.schedule is vp.schedule


def test_eta2_zero_has_no_diffusion(vp):
    rev = ReverseSDE(SDE(vp.schedule), eta2=0.0)
    assert torch.equal(rev.diffusion(torch.rand(4)), torch.zeros(4))


# --- assembly: who gets the chart, who never touches it -------------------- #


def test_sde_refuses_a_plain_schedule():
    with pytest.raises(TypeError, match="chart"):
        SDE(FlowMatching())


@pytest.mark.parametrize(
    "process",
    [Interpolant(FlowMatching()), Interpolant(VP(), coupling=PermutationCoupling())],
    ids=["plain-schedule", "coupled"],
)
def test_non_gaussian_interpolant_fails_at_assembly_not_silently(process):
    # An x0/velocity head on a plain schedule validates fine (its target is a
    # plain conditional expectation), and a coupled interpolant has a Gaussian
    # schedule but a non-Gaussian p(x_t | x0); either way the reverse SDE does
    # not apply. The sampler must refuse at construction, naming the
    # obstruction, for every integrator and eta2.
    for parametrization in (X0Parametrization(), VelocityParametrization()):
        calculator = score_calculator(IDLE, process, parametrization)
        for integrator in (Euler(), Heun()):
            for eta2 in (0.0, 1.0):
                with pytest.raises(TypeError, match="GaussianInterpolant"):
                    SDESampler(calculator, process, eta2=eta2, integrator=integrator)
        with pytest.raises(TypeError, match="GaussianInterpolant"):
            AncestralSampler(x0_calculator(IDLE, process, parametrization), process)


def test_sde_sampler_refuses_a_velocity_calculator(vp):
    with pytest.raises(TypeError, match="ScoreCalculator"):
        SDESampler(VelocityCalculator(IDLE), vp)


def test_ancestral_sampler_refuses_a_velocity_calculator(vp):
    with pytest.raises(TypeError, match="X0Calculator"):
        AncestralSampler(VelocityCalculator(IDLE), vp)
    with pytest.raises(TypeError, match="X0Calculator"):
        AncestralSampler(score_calculator(IDLE, vp), vp)


def test_ode_sampler_refuses_a_score_calculator(vp):
    with pytest.raises(TypeError, match="VelocityCalculator"):
        ODESampler(score_calculator(IDLE, vp))


@pytest.mark.parametrize("integrator", ["euler", Euler, object()])
def test_sde_sampler_refuses_a_non_integrator(vp, integrator):
    with pytest.raises(TypeError, match="not an Integrator"):
        SDESampler(score_calculator(IDLE, vp), vp, integrator=integrator)


@pytest.mark.parametrize("integrator", ["euler", Euler, object()])
def test_ode_sampler_refuses_a_non_integrator(integrator):
    with pytest.raises(TypeError, match="not an Integrator"):
        ODESampler(VelocityCalculator(IDLE), integrator=integrator)


def test_ancestral_sampler_defaults_like_the_sde_sampler():
    process = GaussianInterpolant(VE(0.01, 3.0))
    sampler = AncestralSampler(x0_calculator(IDLE, process), process)
    assert sampler.prior is process.prior
    assert sampler.forward_process is process
    assert sampler.schedule is process.schedule
    ts = sampler.grid(4)
    assert ts[0].item() == pytest.approx(process.schedule.t_max)
    assert ts[-1].item() == pytest.approx(process.schedule.t_min)


def test_sde_sampler_is_deterministic_at_eta2_zero(vp):
    sampler = SDESampler(score_calculator(IDLE, vp), vp, eta2=0.0)
    assert sampler.diffusion(torch.rand(3)) is None


# --- end-to-end recovery of the data distribution ------------------------- #


@pytest.mark.parametrize(
    "integrator,n_steps,eta2",
    [
        (Euler(), 400, 1.0),  # reverse SDE
        (Heun(), 100, 0.0),  # probability-flow ODE (EDM-style)
    ],
    ids=["euler-sde", "heun-ode"],
)
def test_reverse_process_recovers_data_stats(vp, integrator, n_steps, eta2):
    torch.manual_seed(0)
    mu0, s0 = 1.5, 0.5
    sampler = SDESampler(
        score_calculator(batch_model(analytic_score(vp, mu0, s0)), vp),
        vp,
        eta2=eta2,
        integrator=integrator,
    )
    samples = draw(sampler, (4096, 1), n_steps)
    assert samples.mean().item() == pytest.approx(mu0, abs=0.1)
    assert samples.std().item() == pytest.approx(s0, abs=0.1)


def test_scaled_ve_recovers_data_stats():
    # The sampler derives its start N(0, sigma_max^2) from the schedule — the
    # full scaled-VE assembly.
    torch.manual_seed(0)
    scale = 10.0
    process = GaussianInterpolant(VE(2e-4 * scale, scale))
    mu0, s0 = 0.5, 1.0
    sampler = SDESampler(
        score_calculator(
            batch_model(analytic_score(process, mu0, s0, x1_std=scale)), process
        ),
        process,
        eta2=1.0,
    )
    samples = draw(sampler, (4096, 1), 500)
    assert samples.mean().item() == pytest.approx(mu0, abs=0.15)
    assert samples.std().item() == pytest.approx(s0, abs=0.15)


def test_run_partial(vp):
    # partial denoising: the grid starts below the prior's end
    torch.manual_seed(0)
    sampler = SDESampler(
        score_calculator(batch_model(analytic_score(vp, 0.0, 1.0)), vp),
        vp,
        grid=UniformGrid(0.5, vp.schedule.t_min),
    )
    x_t = torch.randn(8, 5, 3)
    out = sampler.run({properties.R: x_t}, n_steps=10)
    assert out[properties.R].shape == x_t.shape
    assert torch.allclose(out[properties.t], torch.full((8,), vp.schedule.t_min))


# --- the velocity ODE ------------------------------------------------------ #


def analytic_velocity(mu, s):
    """
    Exact velocity of the linear path x_t = (1 - t) x0 + t x1 for
    x0 ~ N(mu, s^2), x1 ~ N(0, 1): marginal mean (1 - t) mu, variance
    (1 - t)^2 s^2 + t^2.
    """

    def velocity(batch):
        x, t = batch[properties.R], batch[properties.t]
        t = expand_t(t, x)
        mean = (1 - t) * mu
        var = (1 - t) ** 2 * s**2 + t**2
        dvar = -2 * (1 - t) * s**2 + 2 * t
        return {"velocity": -mu + dvar / (2 * var) * (x - mean)}

    return velocity


@pytest.mark.parametrize("integrator", [Euler(), Heun()], ids=["euler", "heun"])
def test_ode_sampler_recovers_data_stats(integrator):
    torch.manual_seed(0)
    mu, s = 1.5, 0.5
    sampler = ODESampler(
        VelocityCalculator(analytic_velocity(mu, s)),
        integrator=integrator,
        prior=GaussianPrior(std=1.0),
    )
    samples = draw(sampler, (20000, 1), 100)
    assert samples.mean().item() == pytest.approx(mu, abs=0.05)
    assert samples.std().item() == pytest.approx(s, abs=0.05)


def test_ode_sampler_default_grid_runs_from_one_to_zero():
    sampler = ODESampler(VelocityCalculator(IDLE))
    ts = sampler.grid(4)
    assert ts[0].item() == pytest.approx(1.0)
    assert ts[-1].item() == pytest.approx(0.0)
    assert isinstance(sampler.integrator, Euler)
    assert sampler.diffusion(torch.rand(3)) is None


# --- generic ancestral sampling ------------------------------------------- #


def test_posterior_matches_the_ve_closed_form():
    # For VE (a = 1) the posterior mean is the convex blend
    # (sigma_s^2/sigma_t^2) x_t + (1 - sigma_s^2/sigma_t^2) x0, and its
    # variance is sigma_s^2 (sigma_t^2 - sigma_s^2)/sigma_t^2.
    process = GaussianInterpolant(VE(0.3, 30.0))
    t = torch.full((16,), 0.6, dtype=torch.float64)
    s = torch.full((16,), 0.4, dtype=torch.float64)
    x_t = torch.randn(16, 1, dtype=torch.float64)
    x0 = torch.randn(16, 1, dtype=torch.float64)

    mean, std = AncestralSampler(x0_calculator(IDLE, process), process).posterior(
        x_t, x0, t, s
    )

    sig_t = process.schedule.sigma(t).reshape(-1, 1)
    sig_s = process.schedule.sigma(s).reshape(-1, 1)
    w = sig_s**2 / sig_t**2
    expected_mean = w * x_t + (1.0 - w) * x0
    expected_var = (sig_s**2 * (sig_t**2 - sig_s**2) / sig_t**2).reshape(-1)

    assert torch.allclose(mean, expected_mean, rtol=1e-8)
    assert torch.allclose(std**2, expected_var, rtol=1e-8)


def test_posterior_recovers_the_ddpm_marginal_variance_on_vp():
    # Sanity on VP: the posterior variance is non-negative and the mean is a
    # genuine interpolation (bounded by the two inputs is not required, but
    # finiteness and the s->0 collapse are).
    process = GaussianInterpolant(VP())
    t = torch.full((8,), 0.5, dtype=torch.float64)
    s = torch.full((8,), 0.5 - 1e-6, dtype=torch.float64)
    x_t = torch.randn(8, 1, dtype=torch.float64)
    x0 = torch.randn(8, 1, dtype=torch.float64)
    mean, std = AncestralSampler(x0_calculator(IDLE, process), process).posterior(
        x_t, x0, t, s
    )
    # as s -> t the posterior concentrates on x_t
    assert torch.allclose(mean, x_t, atol=1e-4)
    assert std.max().item() < 1e-2


def test_ancestral_recovers_data_stats(vp):
    torch.manual_seed(0)
    mu0, s0 = -0.5, 0.8
    sampler = AncestralSampler(
        x0_calculator(batch_model(analytic_score(vp, mu0, s0)), vp), vp
    )
    samples = draw(sampler, (4096, 1), 1000)
    assert samples.mean().item() == pytest.approx(mu0, abs=0.1)
    assert samples.std().item() == pytest.approx(s0, abs=0.15)


def test_ancestral_on_ve_matches_the_score_form_update():
    # On VE the exact-posterior step must reduce to the classic ancestral
    # update x + score (sigma_t^2 - sigma_s^2) + matched noise — the
    # GPFF/NCSN sampler, here recovered rather than reimplemented.
    process = GaussianInterpolant(VE(0.01, 3.0))
    score_fn = analytic_score(process, 0.5, 0.7, x1_std=process.schedule.sigma_max)
    calculator = x0_calculator(batch_model(score_fn), process)

    x = torch.randn(32, 2, dtype=torch.float64)
    t = torch.full((32,), 0.8, dtype=torch.float64)
    dt = torch.tensor(-0.1, dtype=torch.float64)

    torch.manual_seed(1)
    sampler = sample.AncestralSampler(calculator, process)
    stepped = sampler.step({}, x, t, dt)

    sig_t = expand_t(process.schedule.sigma(t), x)
    sig_s = expand_t(process.schedule.sigma(t + dt), x)
    torch.manual_seed(1)
    z = torch.randn_like(x)
    expected = (
        x
        + score_fn(x, t) * (sig_t**2 - sig_s**2)
        + z * (sig_s**2 * (sig_t**2 - sig_s**2) / sig_t**2).sqrt()
    )
    assert torch.allclose(stepped, expected, rtol=1e-10)


def test_ancestral_on_scaled_ve_recovers_data_stats():
    torch.manual_seed(0)
    scale = 10.0
    process = GaussianInterpolant(VE(2e-4 * scale, scale))
    mu0, s0 = 0.5, 1.0
    sampler = AncestralSampler(
        x0_calculator(
            batch_model(analytic_score(process, mu0, s0, x1_std=scale)), process
        ),
        process,
    )
    samples = draw(sampler, (4096, 1), 500)
    assert samples.mean().item() == pytest.approx(mu0, abs=0.15)
    assert samples.std().item() == pytest.approx(s0, abs=0.15)


# --- plumbing ------------------------------------------------------------- #


def test_batch_keys_reach_the_model_and_the_input_batch_is_untouched(vp):
    # Conditioning is just another batch key: the model sees it at every
    # call, next to the moved key and the time. The caller's batch is never
    # modified, and keys the driver does not move are carried along as given
    # (a static neighbor list stays valid; a cutoff list is the calculator's
    # neighbor_list to rebuild).
    seen = []

    def model(batch):
        seen.append(batch)
        return {"prediction": torch.zeros_like(batch[properties.R])}

    x = torch.randn(4, 1)
    batch = {
        properties.R: x,
        "condition": torch.ones(4),
        properties.Rij: torch.zeros(2, 1),
    }
    out = SDESampler(score_calculator(model, vp), vp).run(batch, 3)
    assert len(seen) == 3
    assert all(torch.equal(b["condition"], torch.ones(4)) for b in seen)
    assert all(b[properties.t].shape == (4,) for b in seen)
    assert all(torch.equal(b[properties.Rij], torch.zeros(2, 1)) for b in seen)
    assert batch[properties.R] is x and set(batch) == {
        properties.R,
        "condition",
        properties.Rij,
    }
    assert torch.equal(out["condition"], torch.ones(4))


def test_sampler_moves_any_declared_key(vp):
    # Toy data need not pretend to be positions: the driver moves its keys.
    def model(batch):
        return {"prediction": -batch["x"]}

    sampler = SDESampler(score_calculator(model, vp, key="x"), vp)
    start = {"x": sampler.prior.sample_positions({properties.R: torch.empty(8, 2)})}
    out = sampler.run(start, 5)
    assert out["x"].shape == (8, 2)


def test_sampler_derives_position_shape_from_atom_types(vp):
    sampler = SDESampler(score_calculator(batch_model(lambda x, t: -x), vp), vp)
    batch = {properties.Z: torch.tensor([1, 6, 8])}
    out = sampler.run(sampler.prior.sample_from_batch(batch), 2)
    assert out[properties.R].shape == (3, 3)


def test_sampler_samples_n_structures_from_statistics(vp):
    structures = StatisticsStructures(
        n_atoms=torch.tensor([0, 0, 1, 1]),
        atom_types=torch.tensor([0, 2, 0, 0, 0, 0, 1, 0, 1]),
    )
    sampler = SDESampler(
        score_calculator(batch_model(lambda x, t: -x), vp),
        vp,
        prior=GaussianPrior(structures=structures),
    )
    out = sampler.sample(5, n_steps=2)
    n_atoms = out[properties.n_atoms]
    assert n_atoms.shape == (5,)
    assert set(n_atoms.tolist()) <= {2, 3}
    assert out[properties.Z].shape == (int(n_atoms.sum()),)
    assert set(out[properties.Z].tolist()) <= {1, 6, 8}
    assert out[properties.R].shape == (int(n_atoms.sum()), 3)
    assert torch.equal(
        out[properties.idx_m], torch.repeat_interleave(torch.arange(5), n_atoms)
    )


def test_sample_without_structures_raises(vp):
    sampler = SDESampler(score_calculator(IDLE, vp), vp)
    with pytest.raises(ValueError, match="no structures"):
        sampler.sample(4, n_steps=2)


def test_sampler_accepts_given_starting_states(vp):
    sampler = SDESampler(
        score_calculator(batch_model(analytic_score(vp, 0.0, 1.0)), vp), vp
    )
    x_init = torch.full((8, 1), 3.0)
    out = sampler.run({properties.R: x_init}, 5)
    assert out[properties.R].shape == x_init.shape


# --- trained end to end --------------------------------------------------- #


class ToyNet(torch.nn.Module):
    """A small MLP on (x, t) — enough to actually learn a 1-D Gaussian."""

    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(2, 64),
            torch.nn.SiLU(),
            torch.nn.Linear(64, 64),
            torch.nn.SiLU(),
            torch.nn.Linear(64, 1),
        )

    def forward(self, x, t, cond=None):
        t = t.reshape(-1, 1).expand(x.shape[0], 1)
        return self.layers(torch.cat([x, t], dim=-1))


def train_toy(loss_fn, model, mu, sd, steps=1000):
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(steps):
        loss = loss_fn(model, mu + sd * torch.randn(256, 1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


@pytest.mark.parametrize(
    "process,param_cls,integrator,n_steps,eta2",
    [
        (GaussianInterpolant(VP()), EpsParametrization, Euler(), 200, 1.0),
        (GaussianInterpolant(VP()), EpsParametrization, Heun(), 50, 0.0),
        # the classic (sigma_min=0.01, sigma_max=3) schedule, built in the
        # literature's vocabulary
        (
            GaussianInterpolant(VE(0.01, 3.0)),
            EpsParametrization,
            Euler(),
            500,
            1.0,
        ),
    ],
    ids=["vp-eps-sde", "vp-eps-ode", "ve-eps-sde"],
)
def test_trained_model_recovers_data_stats(
    process, param_cls, integrator, n_steps, eta2
):
    # The analytic-score tests above check the machinery; this checks that each
    # advertised assembly is actually trainable, which an exact score hides.
    # Note the VE endpoint scale must match the data scale (see VE's
    # docstring) — 3.0 exercises the scaled assembly end to end: the prior,
    # the parametrization and the sampler all read the same declared scale.
    torch.manual_seed(0)
    mu, sd = 1.0, 0.5

    parametrization = param_cls()
    model = train_toy(MatchingLoss(process, parametrization), ToyNet(), mu, sd)
    samples = draw(
        SDESampler(
            score_calculator(batch_model(model), process, parametrization),
            process,
            eta2=eta2,
            integrator=integrator,
        ),
        (4096, 1),
        n_steps,
    )

    assert samples.mean().item() == pytest.approx(mu, abs=0.15)
    assert samples.std().item() == pytest.approx(sd, abs=0.15)
