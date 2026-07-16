import pytest
import torch
from torch import nn

from schnetpack.generative import EDMPath, FMPath, VEPath, VPPath, expand_t
from schnetpack.generative.couplings import IndependentCoupling
from schnetpack.generative.losses import EDMLoss, MatchingLoss
from schnetpack.generative.parametrizations import (
    EpsParametrization,
    ScoreParametrization,
    VelocityParametrization,
    X0Parametrization,
)
from schnetpack.generative.preconditioning import EDMPreconditioner, PrecondDenoiser
from tests.generative.test_preconditioning import InvertingNet


class LinearNet(nn.Module):
    """A trainable stand-in with real parameters."""

    def __init__(self, dim=3):
        super().__init__()
        self.layer = nn.Linear(dim, dim)

    def forward(self, x, t, cond=None):
        return self.layer(x)


class RecordingNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.times = []
        self.conds = []

    def forward(self, x, t, cond=None):
        self.times.append(t)
        self.conds.append(cond)
        return torch.zeros_like(x)


# --- the generic matching loss -------------------------------------------- #


@pytest.mark.parametrize("path_cls", [VPPath, VEPath, FMPath])
@pytest.mark.parametrize(
    "param_cls",
    [
        ScoreParametrization,
        EpsParametrization,
        X0Parametrization,
        VelocityParametrization,
    ],
    ids=lambda c: c.__name__,
)
def test_loss_is_a_scalar_with_gradients(path_cls, param_cls):
    torch.manual_seed(0)
    net = LinearNet()
    loss_fn = MatchingLoss(param_cls(path_cls()))

    loss = loss_fn(net, torch.randn(32, 3))

    assert loss.shape == ()
    assert torch.isfinite(loss)
    loss.backward()
    assert net.layer.weight.grad is not None
    assert torch.isfinite(net.layer.weight.grad).all()


def test_perfect_prediction_gives_zero_loss():
    # A model that emits the target exactly must bottom the loss out, which is
    # the sanity check that target and prediction are compared in the same
    # field.
    path = VPPath()
    parametrization = X0Parametrization(path)

    torch.manual_seed(0)
    x0 = torch.randn(16, 3)

    class Oracle(nn.Module):
        def forward(self, x, t, cond=None):
            return x0

    loss = MatchingLoss(parametrization)(Oracle(), x0)
    assert loss.item() == pytest.approx(0.0, abs=1e-12)


def test_default_times_stay_inside_the_paths_usable_range():
    # Regression: sampling t ~ U[0, 1] would hit the score singularity at 0.
    path = VPPath()
    net = RecordingNet()
    MatchingLoss(ScoreParametrization(path))(net, torch.randn(4096, 3))

    t = net.times[0]
    assert t.shape == (4096,)
    assert t.min().item() >= path.t_min
    assert t.max().item() <= path.t_max
    # and it really covers the range rather than sitting at one end
    assert t.min().item() < path.t_min + 0.05
    assert t.max().item() > path.t_max - 0.05


def test_time_sampler_hook_is_used():
    net = RecordingNet()
    loss_fn = MatchingLoss(
        ScoreParametrization(VPPath()),
        t_sampler=lambda n, device: torch.full((n,), 0.42, device=device),
    )
    loss_fn(net, torch.randn(8, 3))
    assert torch.allclose(net.times[0], torch.full((8,), 0.42))


def test_weight_hook_scales_the_loss():
    torch.manual_seed(0)
    x0 = torch.randn(64, 3)
    net = LinearNet()

    plain = MatchingLoss(
        ScoreParametrization(VPPath()),
        t_sampler=lambda n, device: torch.full((n,), 0.5, device=device),
    )
    weighted = MatchingLoss(
        ScoreParametrization(VPPath()),
        weight=lambda t: torch.full_like(t, 3.0),
        t_sampler=lambda n, device: torch.full((n,), 0.5, device=device),
    )

    torch.manual_seed(1)
    a = plain(net, x0)
    torch.manual_seed(1)
    b = weighted(net, x0)
    assert b.item() == pytest.approx(3.0 * a.item(), rel=1e-6)


def test_coupling_supplies_the_endpoints():
    # The independent coupling must ignore a given x1 and draw its own noise,
    # so passing garbage as x1 changes nothing.
    torch.manual_seed(0)
    x0 = torch.randn(64, 3)
    net = LinearNet()
    loss_fn = MatchingLoss(EpsParametrization(VPPath()), IndependentCoupling())

    torch.manual_seed(1)
    a = loss_fn(net, x0)
    torch.manual_seed(1)
    b = loss_fn(net, x0, x1=torch.full_like(x0, 99.0))
    assert a.item() == pytest.approx(b.item())


def test_cond_reaches_the_model():
    net = RecordingNet()
    marker = object()
    MatchingLoss(ScoreParametrization(VPPath()))(net, torch.randn(4, 3), cond=marker)
    assert net.conds[0] is marker


def test_defaults_are_independent_coupling_and_uniform_weight():
    loss_fn = MatchingLoss(ScoreParametrization(VPPath()))
    assert isinstance(loss_fn.coupling, IndependentCoupling)
    assert torch.equal(loss_fn.weight(torch.rand(5)), torch.ones(5))


# --- EDM ------------------------------------------------------------------ #


def test_edm_loss_is_a_matching_loss_configuration():
    # Not an implementation: the four axes should already cover it.
    loss_fn = EDMLoss()
    assert isinstance(loss_fn, MatchingLoss)
    assert isinstance(loss_fn.path, EDMPath)
    assert isinstance(loss_fn.parametrization, X0Parametrization)
    assert isinstance(loss_fn.coupling, IndependentCoupling)


def test_edm_sigma_sampler_is_lognormal():
    torch.manual_seed(0)
    loss_fn = EDMLoss(P_mean=-1.2, P_std=1.2)
    sigma = loss_fn.t_sampler(100000, None)

    assert (sigma > 0).all()
    log_sigma = torch.log(sigma)
    assert log_sigma.mean().item() == pytest.approx(-1.2, abs=0.02)
    assert log_sigma.std().item() == pytest.approx(1.2, abs=0.02)


def test_edm_weight_is_the_inverse_squared_output_scaling():
    # lambda(sigma) = 1 / c_out(sigma)^2 — the weighting and the preconditioner
    # are two halves of one derivation, and this is the seam between them.
    sigma_data = 0.5
    loss_fn = EDMLoss(sigma_data=sigma_data)
    precond = EDMPreconditioner(sigma_data=sigma_data)

    sigma = torch.logspace(-3, 2, 50, dtype=torch.float64)
    assert torch.allclose(
        loss_fn.weight(sigma), 1.0 / precond.c_out(sigma) ** 2, rtol=1e-8
    )


def test_edm_noising_is_the_ve_form():
    # On EDMPath alpha = 1 and sigma = t, so interpolate must reduce to
    # x0 + sigma * noise.
    path = EDMPath()
    x0, noise = torch.randn(8, 3), torch.randn(8, 3)
    sigma = torch.rand(8) * 10 + 0.1
    assert torch.allclose(
        path.interpolate(x0, noise, sigma), x0 + expand_t(sigma, x0) * noise
    )


def test_edm_loss_equals_one_for_the_optimal_denoiser():
    # With data N(mu0, s0^2) and s0 = sigma_data, the optimal denoiser's error
    # is the posterior variance s0^2 sigma^2 / (s0^2 + sigma^2), and lambda is
    # exactly its reciprocal — so a perfectly trained EDM model sits at loss 1
    # at every noise level. That identity is what makes the weighting correct.
    torch.manual_seed(0)
    mu0 = 0.0
    sigma_data = 0.5

    precond = EDMPreconditioner(sigma_data=sigma_data)
    model = PrecondDenoiser(InvertingNet(precond, mu0, sigma_data), precond)
    loss_fn = EDMLoss(sigma_data=sigma_data)

    x0 = mu0 + sigma_data * torch.randn(200000, 1, dtype=torch.float64)
    loss = loss_fn(model, x0)

    assert loss.item() == pytest.approx(1.0, abs=0.05)


def test_edm_loss_is_flat_across_noise_levels_for_the_optimal_denoiser():
    # The weighting's real job: no noise level should dominate the gradient.
    torch.manual_seed(0)
    mu0, sigma_data = 0.0, 0.5
    precond = EDMPreconditioner(sigma_data=sigma_data)
    model = PrecondDenoiser(InvertingNet(precond, mu0, sigma_data), precond)

    for sigma in [0.01, 0.1, 1.0, 10.0, 80.0]:
        loss_fn = EDMLoss(sigma_data=sigma_data)
        loss_fn.t_sampler = lambda n, device, s=sigma: torch.full(
            (n,), s, device=device, dtype=torch.float64
        )
        x0 = mu0 + sigma_data * torch.randn(200000, 1, dtype=torch.float64)
        assert loss_fn(model, x0).item() == pytest.approx(1.0, abs=0.05), sigma


def test_ve_score_target_spans_orders_of_magnitude():
    # Why VE + ScoreParametrization wants weight=sigma^2: the -x1/sigma target
    # grows like 1/sigma, so an unweighted L2 is dominated by the low-noise end
    # and the model never learns the rest. Documented on VEPath.
    torch.manual_seed(0)
    path = VEPath()
    x1 = torch.randn(4096, 1)

    score_param = ScoreParametrization(path)
    lo = score_param.target(None, x1, torch.zeros(4096)).abs().mean()
    hi = score_param.target(None, x1, torch.ones(4096)).abs().mean()
    assert lo / hi > 1000

    # sigma^2 weighting flattens it: w * (s - (-x1/sigma))^2 == (sigma s + x1)^2,
    # i.e. exactly eps-matching, whose target is x1 at every noise level.
    for t_val in [0.0, 0.5, 1.0]:
        t = torch.full((4096,), t_val)
        weight = path.sigma(t) ** 2
        score_residual = weight * (score_param.target(None, x1, t)) ** 2
        eps_residual = EpsParametrization(path).target(None, x1, t) ** 2
        assert torch.allclose(score_residual, eps_residual, rtol=1e-4)


def test_edm_loss_trains_a_real_net():
    torch.manual_seed(0)
    net = LinearNet(dim=1)
    model = PrecondDenoiser(net, EDMPreconditioner(sigma_data=0.5))
    loss = EDMLoss(sigma_data=0.5)(model, 0.5 * torch.randn(64, 1))

    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(net.layer.weight.grad).all()
