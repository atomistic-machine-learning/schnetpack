import pytest
import torch

from schnetpack.generative import (
    AncestralDDPM,
    EDMPath,
    EulerMaruyama,
    FMPath,
    Heun,
    KarrasGrid,
    KarrasStochasticHeun,
    PathPrior,
    ReverseProcess,
    Sampler,
    UniformGrid,
    VEPath,
    VPPath,
    expand_t,
    generate,
)
from schnetpack.generative.losses import EDMLoss, MatchingLoss
from schnetpack.generative.parametrizations import (
    EpsParametrization,
    ScoreParametrization,
    VelocityParametrization,
    X0Parametrization,
)
from schnetpack.generative.preconditioning import EDMPreconditioner, PrecondDenoiser


@pytest.fixture
def vp():
    return VPPath()


def analytic_score(path, mu0, s0):
    """Exact score of p_t for Gaussian data N(mu0, s0^2) under an affine path."""

    def score(x, t, cond=None):
        alpha = expand_t(path.alpha(t), x)
        sigma = expand_t(path.sigma(t), x)
        var = alpha**2 * s0**2 + sigma**2
        return -(x - alpha * mu0) / var

    return score


def analytic_velocity(path, mu0, s0):
    """The same process, expressed as a velocity — what an FM net would learn."""

    def velocity(x, t, cond=None):
        score = analytic_score(path, mu0, s0)(x, t)
        return ScoreParametrization(path).to_velocity(score, x, t)

    return velocity


def analytic_denoiser(path, mu0, s0):
    """The same process, expressed as a denoiser."""

    def denoise(x, t, cond=None):
        alpha = expand_t(path.alpha(t), x)
        sigma = expand_t(path.sigma(t), x)
        var = alpha**2 * s0**2 + sigma**2
        return (s0**2 * alpha * x + sigma**2 * mu0) / var

    return denoise


# --- grids ---------------------------------------------------------------- #


def test_uniform_grid_endpoints_and_monotonicity():
    ts = UniformGrid()(1.0, 1e-3, 10)
    assert ts.shape == (11,)
    assert ts[0].item() == pytest.approx(1.0)
    assert ts[-1].item() == pytest.approx(1e-3)
    assert (ts.diff() < 0).all()


def test_karras_grid_endpoints_and_monotonicity():
    ts = KarrasGrid(rho=7.0)(80.0, 0.002, 18)
    assert ts.shape == (19,)
    assert ts[0].item() == pytest.approx(80.0, rel=1e-5)
    assert ts[-1].item() == pytest.approx(0.002, rel=1e-5)
    assert (ts.diff() < 0).all()


def test_karras_grid_packs_steps_toward_the_low_noise_end():
    # The point of the warp: steps bunch up where truncation error is worst.
    ts = KarrasGrid(rho=7.0)(80.0, 0.002, 18)
    gaps = (-ts.diff()).tolist()
    assert gaps[0] > gaps[-1]
    assert all(a >= b for a, b in zip(gaps, gaps[1:]))


def test_karras_grid_with_rho_one_is_uniform():
    # In float64: the warp's t_start + i/n (t_end - t_start) cancels badly in
    # float32 near t_end, costing a few digits on the last point. Harmless for
    # sampling, but it would obscure the identity being checked here.
    karras = KarrasGrid(rho=1.0)(80.0, 0.002, 18, dtype=torch.float64)
    uniform = UniformGrid()(80.0, 0.002, 18, dtype=torch.float64)
    assert torch.allclose(karras, uniform, rtol=1e-10)


def test_grids_respect_dtype():
    assert UniformGrid()(1.0, 0.0, 4, dtype=torch.float64).dtype == torch.float64
    assert KarrasGrid()(80.0, 0.002, 4, dtype=torch.float64).dtype == torch.float64


# --- priors --------------------------------------------------------------- #


def test_path_prior_delegates_to_the_path(vp):
    torch.manual_seed(0)
    samples = PathPrior(vp).sample((10000, 1))
    expected = vp.sigma(torch.tensor([vp.t_max])).item()
    assert samples.std().item() == pytest.approx(expected, rel=0.05)


def test_sampler_defaults_to_the_path_prior(vp):
    sampler = Sampler(ScoreParametrization(vp), EulerMaruyama())
    assert isinstance(sampler.prior, PathPrior)
    assert sampler.prior.path is vp
    assert sampler.t_min == vp.t_min
    assert sampler.t_max == vp.t_max


# --- the reverse process -------------------------------------------------- #


def test_reverse_drift_at_churn_one_matches_the_anderson_form(vp):
    # Anderson: f x - 1/2 (1 + eta^2) g^2 s, at eta = 1 (churn = eta^2 = 1).
    score_fn = analytic_score(vp, 0.5, 1.0)
    reverse = ReverseProcess(ScoreParametrization(vp), score_fn, churn=1.0)

    x = torch.randn(16, 2, dtype=torch.float64)
    t = torch.full((16,), 0.5, dtype=torch.float64)

    s = score_fn(x, t)
    expected = expand_t(vp.f(t), x) * x - expand_t(vp.g2(t), x) * s
    assert torch.allclose(reverse.drift(x, t), expected, rtol=1e-8)


def test_reverse_drift_at_churn_zero_is_the_probability_flow(vp):
    score_fn = analytic_score(vp, 0.5, 1.0)
    reverse = ReverseProcess(ScoreParametrization(vp), score_fn, churn=0.0)

    x = torch.randn(16, 2, dtype=torch.float64)
    t = torch.full((16,), 0.5, dtype=torch.float64)

    s = score_fn(x, t)
    expected = expand_t(vp.f(t), x) * x - 0.5 * expand_t(vp.g2(t), x) * s
    assert torch.allclose(reverse.drift(x, t), expected, rtol=1e-8)


@pytest.mark.parametrize("churn", [0.0, 0.25, 1.0])
def test_reverse_diffusion(vp, churn):
    t = torch.tensor([0.3, 0.7], dtype=torch.float64)
    reverse = ReverseProcess(
        ScoreParametrization(vp), analytic_score(vp, 0.0, 1.0), churn=churn
    )
    assert torch.allclose(reverse.diffusion(t), torch.sqrt(churn * vp.g2(t)))


@pytest.mark.parametrize("churn", [0.0, 1.0])
def test_drift_costs_exactly_one_model_evaluation(vp, churn):
    # At churn > 0 the drift needs both a velocity and a score. They must come
    # from one raw output, not two forward passes.
    calls = []

    def counting_model(x, t, cond=None):
        calls.append(1)
        return analytic_score(vp, 0.0, 1.0)(x, t)

    reverse = ReverseProcess(ScoreParametrization(vp), counting_model, churn=churn)
    reverse.drift(torch.randn(4, 2), torch.full((4,), 0.5))
    assert len(calls) == 1


def test_path_reverse_returns_a_reverse_process(vp):
    reverse = ScoreParametrization(vp).reverse(analytic_score(vp, 0.0, 1.0), churn=0.5)
    assert isinstance(reverse, ReverseProcess)
    assert reverse.path is vp
    assert reverse.churn == 0.5


def test_path_probability_flow_is_churn_zero(vp):
    reverse = ScoreParametrization(vp).probability_flow(analytic_score(vp, 0.0, 1.0))
    assert reverse.churn == 0.0
    assert torch.equal(reverse.diffusion(torch.rand(4)), torch.zeros(4))


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
    sampler = Sampler(ScoreParametrization(vp), integrator, churn=churn)
    samples = sampler.sample(analytic_score(vp, mu0, s0), (4096, 1), n_steps)
    assert samples.mean().item() == pytest.approx(mu0, abs=0.1)
    assert samples.std().item() == pytest.approx(s0, abs=0.1)


def test_ancestral_ddpm_recovers_data_stats(vp):
    torch.manual_seed(0)
    mu0, s0 = -0.5, 0.8
    sampler = Sampler(ScoreParametrization(vp), AncestralDDPM())
    samples = sampler.sample(analytic_score(vp, mu0, s0), (4096, 1), 1000)
    assert samples.mean().item() == pytest.approx(mu0, abs=0.1)
    assert samples.std().item() == pytest.approx(s0, abs=0.15)


def test_ve_path_recovers_data_stats():
    torch.manual_seed(0)
    ve = VEPath()
    mu0, s0 = 0.5, 1.0
    sampler = Sampler(ScoreParametrization(ve), EulerMaruyama(), churn=1.0)
    samples = sampler.sample(analytic_score(ve, mu0, s0), (4096, 1), 500)
    assert samples.mean().item() == pytest.approx(mu0, abs=0.15)
    assert samples.std().item() == pytest.approx(s0, abs=0.15)


def test_denoise_partial(vp):
    torch.manual_seed(0)
    sampler = Sampler(ScoreParametrization(vp), EulerMaruyama())
    x_t = torch.randn(8, 5, 3)
    out = sampler.denoise(analytic_score(vp, 0.0, 1.0), x_t, t_start=0.5, n_steps=10)
    assert out.shape == x_t.shape


# --- flow matching -------------------------------------------------------- #


def test_fm_velocity_ode_recovers_data_stats():
    torch.manual_seed(0)
    fm = FMPath()
    mu0, s0 = 1.0, 0.5
    sampler = Sampler(VelocityParametrization(fm), Heun(), churn=0.0)
    samples = sampler.sample(analytic_velocity(fm, mu0, s0), (4096, 1), 100)
    assert samples.mean().item() == pytest.approx(mu0, abs=0.1)
    assert samples.std().item() == pytest.approx(s0, abs=0.1)


def test_fm_ode_never_converts_velocity_to_score(monkeypatch):
    # The reason the reverse family is written around the velocity: at churn=0
    # the singular inverse must not be touched at all.
    fm = FMPath()

    def explode(*args, **kwargs):
        raise AssertionError("to_score must not run on the ODE path")

    monkeypatch.setattr(VelocityParametrization, "to_score", explode)

    sampler = Sampler(VelocityParametrization(fm), Heun(), churn=0.0)
    samples = sampler.sample(analytic_velocity(fm, 0.0, 1.0), (16, 1), 10)
    assert torch.isfinite(samples).all()


def test_fm_stochastic_sampling_stays_finite():
    # Regression: g^2 = 2 t sigma_max^2 / (1 - t) diverges at t = 1, so a
    # t_max of exactly 1 would produce NaNs the moment churn > 0.
    torch.manual_seed(0)
    fm = FMPath()
    sampler = Sampler(VelocityParametrization(fm), EulerMaruyama(), churn=1.0)
    samples = sampler.sample(analytic_velocity(fm, 0.0, 1.0), (64, 1), 100)
    assert torch.isfinite(samples).all()


# --- EDM ------------------------------------------------------------------ #


def test_edm_heun_on_the_karras_grid_recovers_data_stats():
    torch.manual_seed(0)
    edm = EDMPath()
    mu0, s0 = 0.3, 0.5
    sampler = Sampler(
        X0Parametrization(edm), Heun(), grid=KarrasGrid(rho=7.0), churn=0.0
    )
    samples = sampler.sample(analytic_denoiser(edm, mu0, s0), (4096, 1), 32)
    assert samples.mean().item() == pytest.approx(mu0, abs=0.1)
    assert samples.std().item() == pytest.approx(s0, abs=0.1)


def test_edm_velocity_is_the_karras_derivative():
    # On EDMPath (f = 0, g^2 = 2 sigma) the derived velocity must equal the
    # d_i = (x - D) / sigma of Karras Alg. 1 — so the generic Heun step *is*
    # the EDM sampler, with no EDM-specific code.
    edm = EDMPath()
    denoiser = analytic_denoiser(edm, 0.3, 0.5)
    parametrization = X0Parametrization(edm)

    x = torch.randn(16, 1, dtype=torch.float64)
    sigma = torch.linspace(0.01, 20.0, 16, dtype=torch.float64)

    raw = denoiser(x, sigma)
    velocity = parametrization.to_velocity(raw, x, sigma)
    expected = (x - raw) / expand_t(sigma, x)
    assert torch.allclose(velocity, expected, rtol=1e-8)


def test_karras_stochastic_heun_is_a_stub():
    with pytest.raises(NotImplementedError):
        KarrasStochasticHeun().step(None, None, None, None)
    with pytest.raises(NotImplementedError):
        KarrasStochasticHeun().integrate(None, None, None)


# --- plumbing ------------------------------------------------------------- #


def test_cond_is_threaded_through_sampling(vp):
    marker = object()
    seen = []

    def model(x, t, cond=None):
        seen.append(cond)
        return torch.zeros_like(x)

    Sampler(ScoreParametrization(vp), EulerMaruyama()).sample(
        model, (4, 1), 3, cond=marker
    )
    assert seen and all(c is marker for c in seen)


def test_sampler_accepts_given_starting_states(vp):
    sampler = Sampler(ScoreParametrization(vp), EulerMaruyama())
    x_init = torch.full((8, 1), 3.0)
    out = sampler.sample(analytic_score(vp, 0.0, 1.0), (8, 1), 5, x_init=x_init)
    assert out.shape == x_init.shape


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
    "path,param_cls,integrator,n_steps,churn",
    [
        (VPPath(), EpsParametrization, EulerMaruyama(), 200, 1.0),
        (VPPath(), EpsParametrization, Heun(), 50, 0.0),
        (FMPath(), VelocityParametrization, Heun(), 50, 0.0),
        (FMPath(), VelocityParametrization, EulerMaruyama(), 200, 1.0),
        (VEPath(sigma_max=3.0), EpsParametrization, EulerMaruyama(), 500, 1.0),
    ],
    ids=["vp-eps-sde", "vp-eps-ode", "fm-vel-ode", "fm-vel-sde", "ve-eps-sde"],
)
def test_trained_model_recovers_data_stats(path, param_cls, integrator, n_steps, churn):
    # The analytic-score tests above check the machinery; this checks that each
    # advertised assembly is actually trainable, which an exact score hides.
    # Note VEPath needs sigma_max matched to the data scale (see its docstring)
    # — the default 50 against data of scale 0.5 trains to garbage.
    torch.manual_seed(0)
    mu, sd = 1.0, 0.5

    parametrization = param_cls(path)
    model = train_toy(MatchingLoss(parametrization), ToyNet(), mu, sd)
    samples = Sampler(parametrization, integrator, churn=churn).sample(
        model, (4096, 1), n_steps
    )

    assert samples.mean().item() == pytest.approx(mu, abs=0.15)
    assert samples.std().item() == pytest.approx(sd, abs=0.15)


def test_trained_edm_recovers_data_stats():
    torch.manual_seed(0)
    mu, sd = 1.0, 0.5
    path = EDMPath()

    model = PrecondDenoiser(ToyNet(), EDMPreconditioner(sigma_data=sd))
    train_toy(EDMLoss(path, sigma_data=sd), model, mu, sd)

    samples = Sampler(
        X0Parametrization(path), Heun(), grid=KarrasGrid(), churn=0.0
    ).sample(model, (4096, 1), 32)

    assert samples.mean().item() == pytest.approx(mu, abs=0.15)
    assert samples.std().item() == pytest.approx(sd, abs=0.15)
