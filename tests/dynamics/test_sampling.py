import pytest
import torch

from schnetpack import properties
from schnetpack.dynamics import (
    Ancestral,
    AncestralDDPM,
    DirectDenoising,
    EulerMaruyama,
    Heun,
    Sampler,
    UniformGrid,
    generate,
)
from schnetpack.generative import (
    VE,
    VP,
    EpsParametrization,
    FlowMatching,
    GaussianPrior,
    MatchingLoss,
    PCVarianceCoupling,
    Prior,
    PseudoForceParametrization,
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


def draw(dynamics, shape, n_steps):
    """Sample from the prior; return the positions."""
    template = {properties.R: torch.empty(shape)}
    return dynamics.denoise(dynamics.prior.sample_from_batch(template), n_steps)[
        properties.R
    ]


@pytest.fixture
def vp():
    return VP()


def analytic_score(process, mu0, s0, x1_std=1.0):
    """Exact score of p_t for Gaussian data N(mu0, s0^2) under an affine schedule."""

    def score(x, t, cond=None):
        a = expand_t(process.a(t), x)
        sigma = expand_t(process.b(t), x) * x1_std
        var = a**2 * s0**2 + sigma**2
        return -(x - a * mu0) / var

    return score


def analytic_velocity(process, mu0, s0):
    """The same process, expressed as a velocity — what an FM net would learn."""

    def velocity(x, t, cond=None):
        score = analytic_score(process, mu0, s0)(x, t)
        return ScoreParametrization().to_velocity(process, score, x, t)

    return velocity


# --- grids ---------------------------------------------------------------- #


def test_uniform_grid_endpoints_and_monotonicity():
    ts = UniformGrid()(1.0, 1e-3, 10)
    assert ts.shape == (11,)
    assert ts[0].item() == pytest.approx(1.0)
    assert ts[-1].item() == pytest.approx(1e-3)
    assert (ts.diff() < 0).all()


def test_grids_respect_dtype():
    assert UniformGrid()(1.0, 0.0, 4, dtype=torch.float64).dtype == torch.float64


# --- priors --------------------------------------------------------------- #


def test_gaussian_prior_has_the_declared_std():
    torch.manual_seed(0)
    samples = GaussianPrior(std=3.0).sample_positions(
        {properties.R: torch.empty(10000, 1)}
    )
    assert samples.std().item() == pytest.approx(3.0, rel=0.05)


def test_sampler_defaults_to_the_processs_sampling_prior(vp):
    # b(t_max) = 1, so the default unit Gaussian diffusion starts at N(0, I).
    sampler = Sampler(IDLE, vp, ScoreParametrization(), EulerMaruyama())
    assert isinstance(sampler.prior, GaussianPrior)
    assert sampler.prior.std == pytest.approx(1.0)
    assert sampler.t_min == vp.t_min
    assert sampler.t_max == vp.t_max


def test_sampler_derives_the_scale_from_the_process():
    # The train/sample tie: the process carries the prior its x1 endpoint was
    # drawn from, and the sampler starts from exactly that — the same object.
    process = VE(scale=50.0)
    sampler = Sampler(IDLE, process, ScoreParametrization(), EulerMaruyama())
    assert isinstance(sampler.prior, GaussianPrior)
    assert sampler.prior.std == pytest.approx(50.0)
    assert sampler.prior is process.prior


def test_sampler_validates_the_pair():
    # The score/noise heads demand a Gaussian kernel; the sampler is where the
    # (process, parametrization) pair meets, so it must refuse at construction.
    reshaped = VP(coupling=PCVarianceCoupling())
    with pytest.raises(TypeError, match="Gaussian kernel"):
        Sampler(
            IDLE,
            reshaped,
            ScoreParametrization(),
            EulerMaruyama(),
            prior=GaussianPrior(),
        )


def test_sampler_refuses_when_the_process_cannot_state_its_start():
    # A coupling that reshapes x1's marginal has no data-free start; the
    # sampler must refuse rather than guess. (churn = 0 with a velocity head
    # so the chart is never demanded — this test is about the start, and
    # churn > 0 on this configuration is refused earlier, for the chart.)
    process = VP(coupling=PCVarianceCoupling())
    with pytest.raises(ValueError, match="marginal"):
        Sampler(IDLE, process, VelocityParametrization(), EulerMaruyama(), churn=0.0)

    # An explicit prior always wins.
    explicit = GaussianPrior()
    sampler = Sampler(
        IDLE,
        process,
        VelocityParametrization(),
        EulerMaruyama(),
        prior=explicit,
        churn=0.0,
    )
    assert sampler.prior is explicit


# --- the reverse process -------------------------------------------------- #


def test_reverse_drift_at_churn_one_matches_the_anderson_form(vp):
    # Anderson: f x - 1/2 (1 + eta^2) g^2 s, at eta = 1 (churn = eta^2 = 1).
    score_fn = analytic_score(vp, 0.5, 1.0)
    rev = ReverseSDE(vp.sde(), score_fn, churn=1.0)

    x = torch.randn(16, 2, dtype=torch.float64)
    t = torch.full((16,), 0.5, dtype=torch.float64)

    s = score_fn(x, t)
    sde = vp.sde()
    expected = expand_t(sde.f(t), x) * x - expand_t(sde.g2(t), x) * s
    assert torch.allclose(rev.drift(x, t), expected, rtol=1e-8)


def test_reverse_drift_at_churn_zero_is_the_probability_flow(vp):
    score_fn = analytic_score(vp, 0.5, 1.0)
    rev = ReverseSDE(vp.sde(), score_fn, churn=0.0)

    x = torch.randn(16, 2, dtype=torch.float64)
    t = torch.full((16,), 0.5, dtype=torch.float64)

    s = score_fn(x, t)
    sde = vp.sde()
    expected = expand_t(sde.f(t), x) * x - 0.5 * expand_t(sde.g2(t), x) * s
    assert torch.allclose(rev.drift(x, t), expected, rtol=1e-8)


@pytest.mark.parametrize("churn", [0.0, 0.25, 1.0])
def test_reverse_diffusion(vp, churn):
    t = torch.tensor([0.3, 0.7], dtype=torch.float64)
    rev = ReverseSDE(vp.sde(), analytic_score(vp, 0.0, 1.0), churn=churn)
    assert torch.allclose(rev.diffusion(t), torch.sqrt(churn * vp.sde().g2(t)))


@pytest.mark.parametrize("churn", [0.0, 1.0])
def test_drift_costs_exactly_one_model_evaluation(vp, churn):
    # The drift is one statement about one score — a single evaluation of
    # the bound field, whatever the churn.
    calls = []

    def counting_score(x, t):
        calls.append(1)
        return analytic_score(vp, 0.0, 1.0)(x, t)

    rev = ReverseSDE(vp.sde(), counting_score, churn=churn)
    rev.drift(torch.randn(4, 2), torch.full((4,), 0.5))
    assert len(calls) == 1


def test_reverse_process_exposes_the_reversed_process(vp):
    rev = ReverseSDE(vp.sde(), analytic_score(vp, 0.0, 1.0), churn=0.5)
    assert rev.process is vp
    assert rev.churn == 0.5


def test_churn_zero_has_no_diffusion(vp):
    rev = ReverseSDE(vp.sde(), analytic_score(vp, 0.0, 1.0), churn=0.0)
    assert torch.equal(rev.diffusion(torch.rand(4)), torch.zeros(4))


# --- assembly: who gets the chart, who never touches it -------------------- #


class ShapedPrior(Prior):
    """A non-Gaussian endpoint *with* a declared scale — the configuration
    that used to sail through f/g2 and Tweedie silently."""

    gaussian = False
    std = 2.0

    def sample_positions(self, batch):
        u = torch.rand_like(batch[properties.R])
        return self.std * (2.0 * u - 1.0)


def test_chart_free_velocity_sampling_runs_without_the_kernel():
    # churn = 0 with a velocity head must assemble the chart-free ReverseODE:
    # on a configuration with no chart, sampling still runs end to end. A
    # wrong dispatch to ReverseSDE would raise at the chart acquisition.
    process = VE(b_min=1e-2, prior=ShapedPrior())
    sampler = Sampler(
        batch_model(lambda x, t, cond=None: torch.zeros_like(x)),
        process,
        VelocityParametrization(),
        Heun(),
        churn=0.0,
    )
    out = draw(sampler, (8, 2), 5)
    assert out.shape == (8, 2)


def test_sde_refuses_without_the_gaussian_kernel():
    with pytest.raises(ValueError, match="chart"):
        VE(b_min=1e-2, prior=ShapedPrior()).sde()


def test_shape_prior_with_declared_scale_fails_at_assembly_not_silently():
    # Defect the split fixes: an x0/velocity head on a non-Gaussian prior with
    # a declared std validates fine (its target is a plain conditional
    # expectation) but its sampling conversions are Gaussian-kernel
    # statements. The Sampler must refuse at construction, naming the
    # obstruction — before this, f/g2 and Tweedie returned wrong numbers
    # without a raise.
    process = VE(b_min=1e-2, prior=ShapedPrior())
    with pytest.raises(ValueError, match="chart"):
        Sampler(IDLE, process, X0Parametrization(), EulerMaruyama(), churn=1.0)
    with pytest.raises(ValueError, match="chart"):
        # the probability-flow ODE converts too
        Sampler(IDLE, process, X0Parametrization(), Heun(), churn=0.0)
    with pytest.raises(ValueError, match="chart"):
        Sampler(IDLE, process, VelocityParametrization(), Ancestral(), churn=0.0)
    with pytest.raises(ValueError, match="chart"):  # churn > 0 crosses it too
        Sampler(IDLE, process, VelocityParametrization(), EulerMaruyama(), churn=1.0)

    # The chart-free assemblies stay open on the same configuration.
    Sampler(IDLE, process, VelocityParametrization(), Heun(), churn=0.0)
    DirectDenoising(IDLE, process, PseudoForceParametrization())


# --- end-to-end recovery of the data distribution ------------------------- #


@pytest.mark.parametrize(
    "integrator,n_steps,churn",
    [
        (EulerMaruyama(), 400, 1.0),  # reverse SDE
        (Heun(), 100, 0.0),  # probability-flow ODE (EDM-style)
    ],
)
def test_reverse_process_recovers_data_stats(vp, integrator, n_steps, churn):
    torch.manual_seed(0)
    mu0, s0 = 1.5, 0.5
    sampler = Sampler(
        batch_model(analytic_score(vp, mu0, s0)),
        vp,
        ScoreParametrization(),
        integrator,
        churn=churn,
    )
    samples = draw(sampler, (4096, 1), n_steps)
    assert samples.mean().item() == pytest.approx(mu0, abs=0.1)
    assert samples.std().item() == pytest.approx(s0, abs=0.1)


def test_ancestral_ddpm_recovers_data_stats(vp):
    torch.manual_seed(0)
    mu0, s0 = -0.5, 0.8
    sampler = Sampler(
        batch_model(analytic_score(vp, mu0, s0)),
        vp,
        ScoreParametrization(),
        AncestralDDPM(),
    )
    samples = draw(sampler, (4096, 1), 1000)
    assert samples.mean().item() == pytest.approx(mu0, abs=0.1)
    assert samples.std().item() == pytest.approx(s0, abs=0.15)


def test_scaled_ve_recovers_data_stats():
    # The endpoint scale (the old sigma_max) rides on the prior; the sampler
    # derives its start from the process — the full scaled-VE assembly.
    torch.manual_seed(0)
    scale = 10.0
    process = VE(scale=scale)
    mu0, s0 = 0.5, 1.0
    sampler = Sampler(
        batch_model(analytic_score(process, mu0, s0, x1_std=scale)),
        process,
        ScoreParametrization(),
        EulerMaruyama(),
        churn=1.0,
    )
    samples = draw(sampler, (4096, 1), 500)
    assert samples.mean().item() == pytest.approx(mu0, abs=0.15)
    assert samples.std().item() == pytest.approx(s0, abs=0.15)


def test_denoise_partial(vp):
    torch.manual_seed(0)
    sampler = Sampler(
        batch_model(analytic_score(vp, 0.0, 1.0)),
        vp,
        ScoreParametrization(),
        EulerMaruyama(),
    )
    x_t = torch.randn(8, 5, 3)
    out = sampler.denoise(
        {properties.R: x_t},
        n_steps=10,
        t_start=0.5,
    )
    assert out[properties.R].shape == x_t.shape


# --- generic ancestral sampling ------------------------------------------- #


def test_ancestral_recovers_data_stats(vp):
    torch.manual_seed(0)
    mu0, s0 = -0.5, 0.8
    sampler = Sampler(
        batch_model(analytic_score(vp, mu0, s0)),
        vp,
        ScoreParametrization(),
        Ancestral(),
    )
    samples = draw(sampler, (4096, 1), 1000)
    assert samples.mean().item() == pytest.approx(mu0, abs=0.1)
    assert samples.std().item() == pytest.approx(s0, abs=0.15)


def test_ancestral_on_ve_matches_the_score_form_update():
    # On VE the exact-posterior step must reduce to the classic ancestral
    # update x + score (sigma_t^2 - sigma_s^2) + matched noise — the
    # GPFF/NCSN sampler, here recovered rather than reimplemented.
    process = VE(0.01, 3.0)
    score_fn = analytic_score(process, 0.5, 0.7, x1_std=process.std)
    rev = ReverseSDE(process.sde(), score_fn)

    x = torch.randn(32, 2, dtype=torch.float64)
    t = torch.full((32,), 0.8, dtype=torch.float64)
    dt = torch.tensor(-0.1, dtype=torch.float64)

    torch.manual_seed(1)
    stepped = Ancestral().step(rev, x, t, dt)

    sig_t = expand_t(process.sigma(t), x)
    sig_s = expand_t(process.sigma(t + dt), x)
    torch.manual_seed(1)
    z = torch.randn_like(x)
    expected = (
        x
        + score_fn(x, t) * (sig_t**2 - sig_s**2)
        + z * (sig_s**2 * (sig_t**2 - sig_s**2) / sig_t**2).sqrt()
    )
    assert torch.allclose(stepped, expected, rtol=1e-10)


def test_ancestral_x0_via_score_round_trips_an_x0_head(vp):
    # Ancestral reads x0 through score -> Tweedie on the chart. For an
    # x0-predicting head the two Tweedie directions cancel algebraically;
    # the round trip must reproduce the head's output.
    par = X0Parametrization()

    def x0_model(x, t):
        return torch.full_like(x, 1.5)

    rev = ReverseSDE(vp.sde(), lambda x, t: par.to_score(vp, x0_model(x, t), x, t))

    x = torch.randn(16, 2, dtype=torch.float64)
    t = torch.full((16,), 0.7, dtype=torch.float64)
    x0_hat = rev.sde.x0_from_score(x, rev.score(x, t), t)
    assert torch.allclose(x0_hat, torch.full_like(x, 1.5), rtol=1e-12)


def test_ancestral_on_scaled_ve_recovers_data_stats():
    torch.manual_seed(0)
    scale = 10.0
    process = VE(scale=scale)
    mu0, s0 = 0.5, 1.0
    sampler = Sampler(
        batch_model(analytic_score(process, mu0, s0, x1_std=scale)),
        process,
        ScoreParametrization(),
        Ancestral(),
    )
    samples = draw(sampler, (4096, 1), 500)
    assert samples.mean().item() == pytest.approx(mu0, abs=0.15)
    assert samples.std().item() == pytest.approx(s0, abs=0.15)


# --- flow matching -------------------------------------------------------- #


def test_fm_velocity_ode_recovers_data_stats():
    torch.manual_seed(0)
    fm = FlowMatching()
    mu0, s0 = 1.0, 0.5
    sampler = Sampler(
        batch_model(analytic_velocity(fm, mu0, s0)),
        fm,
        VelocityParametrization(),
        Heun(),
        churn=0.0,
    )
    samples = draw(sampler, (4096, 1), 100)
    assert samples.mean().item() == pytest.approx(mu0, abs=0.1)
    assert samples.std().item() == pytest.approx(s0, abs=0.1)


def test_fm_ode_never_converts_velocity_to_score(monkeypatch):
    # The reason the reverse family is written around the velocity: at churn=0
    # the singular inverse must not be touched at all.
    fm = FlowMatching()

    def explode(*args, **kwargs):
        raise AssertionError("to_score must not run on the ODE path")

    monkeypatch.setattr(VelocityParametrization, "to_score", explode)

    sampler = Sampler(
        batch_model(analytic_velocity(fm, 0.0, 1.0)),
        fm,
        VelocityParametrization(),
        Heun(),
        churn=0.0,
    )
    samples = draw(sampler, (16, 1), 10)
    assert torch.isfinite(samples).all()


def test_fm_stochastic_sampling_stays_finite():
    # Regression: g^2 = 2 t sigma_max^2 / (1 - t) diverges at t = 1, so a
    # t_max of exactly 1 would produce NaNs the moment churn > 0.
    torch.manual_seed(0)
    fm = FlowMatching()
    sampler = Sampler(
        batch_model(analytic_velocity(fm, 0.0, 1.0)),
        fm,
        VelocityParametrization(),
        EulerMaruyama(),
        churn=1.0,
    )
    samples = draw(sampler, (64, 1), 100)
    assert torch.isfinite(samples).all()


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
    out = Sampler(model, vp, ScoreParametrization(), EulerMaruyama()).denoise(batch, 3)
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

    sampler = Sampler(model, vp, ScoreParametrization(), EulerMaruyama(), key="x")
    start = {"x": sampler.prior.sample_positions({properties.R: torch.empty(8, 2)})}
    out = sampler.denoise(start, 5)
    assert out["x"].shape == (8, 2)


def test_sampler_derives_position_shape_from_atom_types(vp):
    sampler = Sampler(
        batch_model(lambda x, t: -x), vp, ScoreParametrization(), EulerMaruyama()
    )
    batch = {properties.Z: torch.tensor([1, 6, 8])}
    out = sampler.denoise(sampler.prior.sample_from_batch(batch), 2)
    assert out[properties.R].shape == (3, 3)


def test_sampler_samples_n_structures_from_statistics(vp):
    structures = StatisticsStructures(
        n_atoms=torch.tensor([0, 0, 1, 1]),
        atom_types=torch.tensor([0, 2, 0, 0, 0, 0, 1, 0, 1]),
    )
    sampler = Sampler(
        batch_model(lambda x, t: -x),
        vp,
        ScoreParametrization(),
        EulerMaruyama(),
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
    sampler = Sampler(IDLE, vp, ScoreParametrization(), EulerMaruyama())
    with pytest.raises(ValueError, match="no structures"):
        sampler.sample(4, n_steps=2)


def test_sampler_accepts_given_starting_states(vp):
    sampler = Sampler(
        batch_model(analytic_score(vp, 0.0, 1.0)),
        vp,
        ScoreParametrization(),
        EulerMaruyama(),
    )
    x_init = torch.full((8, 1), 3.0)
    out = sampler.denoise({properties.R: x_init}, 5)
    assert out[properties.R].shape == x_init.shape


def test_generate_is_stub():
    with pytest.raises(NotImplementedError):
        generate()


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
    "process,param_cls,integrator,n_steps,churn",
    [
        (VP(), EpsParametrization, EulerMaruyama(), 200, 1.0),
        (VP(), EpsParametrization, Heun(), 50, 0.0),
        (FlowMatching(), VelocityParametrization, Heun(), 50, 0.0),
        (FlowMatching(), VelocityParametrization, EulerMaruyama(), 200, 1.0),
        # the classic (sigma_min=0.01, sigma_max=3) schedule, built in the
        # literature's vocabulary — the split happens inside VE.__init__
        (VE(0.01, 3.0), EpsParametrization, EulerMaruyama(), 500, 1.0),
    ],
    ids=["vp-eps-sde", "vp-eps-ode", "fm-vel-ode", "fm-vel-sde", "ve-eps-sde"],
)
def test_trained_model_recovers_data_stats(
    process, param_cls, integrator, n_steps, churn
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
        Sampler(batch_model(model), process, parametrization, integrator, churn=churn),
        (4096, 1),
        n_steps,
    )

    assert samples.mean().item() == pytest.approx(mu, abs=0.15)
    assert samples.std().item() == pytest.approx(sd, abs=0.15)
