import pytest
import torch
from torch import nn

from schnetpack.generative import (
    EpsParametrization,
    FlowMatching,
    MatchingLoss,
    PCVarianceCoupling,
    PermutationCoupling,
    ScoreParametrization,
    VE,
    VelocityParametrization,
    VP,
    X0Parametrization,
)


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


@pytest.mark.parametrize("process_cls", [VP, VE, FlowMatching])
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
def test_loss_is_a_scalar_with_gradients(process_cls, param_cls):
    torch.manual_seed(0)
    net = LinearNet()
    loss_fn = MatchingLoss(process_cls(), param_cls())

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
    torch.manual_seed(0)
    x0 = torch.randn(16, 3)

    class Oracle(nn.Module):
        def forward(self, x, t, cond=None):
            return x0

    loss = MatchingLoss(VP(), X0Parametrization())(Oracle(), x0)
    assert loss.item() == pytest.approx(0.0, abs=1e-12)


def test_default_times_stay_inside_the_processs_usable_range():
    # Regression: sampling t ~ U[0, 1] would hit the score singularity at 0.
    process = VP()
    net = RecordingNet()
    MatchingLoss(process, ScoreParametrization())(net, torch.randn(4096, 3))

    t = net.times[0]
    assert t.shape == (4096,)
    assert t.min().item() >= process.t_min
    assert t.max().item() <= process.t_max
    # and it really covers the range rather than sitting at one end
    assert t.min().item() < process.t_min + 0.05
    assert t.max().item() > process.t_max - 0.05


def test_time_sampler_hook_is_used():
    net = RecordingNet()
    loss_fn = MatchingLoss(
        VP(),
        ScoreParametrization(),
        t_sampler=lambda n, device: torch.full((n,), 0.42, device=device),
    )
    loss_fn(net, torch.randn(8, 3))
    assert torch.allclose(net.times[0], torch.full((8,), 0.42))


def test_weight_hook_scales_the_loss():
    torch.manual_seed(0)
    x0 = torch.randn(64, 3)
    net = LinearNet()

    plain = MatchingLoss(
        VP(),
        ScoreParametrization(),
        t_sampler=lambda n, device: torch.full((n,), 0.5, device=device),
    )
    weighted = MatchingLoss(
        VP(),
        ScoreParametrization(),
        weight=lambda t: torch.full_like(t, 3.0),
        t_sampler=lambda n, device: torch.full((n,), 0.5, device=device),
    )

    torch.manual_seed(1)
    a = plain(net, x0)
    torch.manual_seed(1)
    b = weighted(net, x0)
    assert b.item() == pytest.approx(3.0 * a.item(), rel=1e-6)


def test_process_supplies_the_endpoints():
    # The identity coupling draws fresh noise per sample; the loss is a real
    # function of (model, x0) once the RNG is fixed.
    torch.manual_seed(0)
    x0 = torch.randn(64, 3)
    net = LinearNet()
    loss_fn = MatchingLoss(VP(), EpsParametrization())

    torch.manual_seed(1)
    a = loss_fn(net, x0)
    torch.manual_seed(1)
    b = loss_fn(net, x0)
    assert a.item() == pytest.approx(b.item())


def test_cond_reaches_the_model():
    net = RecordingNet()
    marker = object()
    MatchingLoss(VP(), ScoreParametrization())(
        net, torch.randn(4, 3), cond=marker
    )
    assert net.conds[0] is marker


def test_defaults_are_uniform_weight_and_the_process_t_sampler():
    loss_fn = MatchingLoss(VP(), ScoreParametrization())
    assert torch.equal(loss_fn.weight(torch.rand(5)), torch.ones(5))
    assert loss_fn.t_sampler is None  # falls through to process.sample_t


def test_loss_validates_the_pair():
    # The loss is where the (process, parametrization) pair meets at training
    # time; a Gaussian-only head on a kernel-less configuration must fail here.
    process = VP(coupling=PCVarianceCoupling())
    with pytest.raises(TypeError, match="Gaussian kernel"):
        MatchingLoss(process, ScoreParametrization())


def test_the_process_carries_the_coupling():
    # The join no longer takes a coupling — it comes with the process, and the
    # pairing is validated at the loss's construction.
    process = VP(coupling=PermutationCoupling())
    loss_fn = MatchingLoss(process, VelocityParametrization())
    assert loss_fn.process is process
    loss = loss_fn(LinearNet(), torch.randn(16, 3))
    assert torch.isfinite(loss)


def test_endpoint_scale_flows_from_the_prior():
    # The prior owns sigma_max; the parametrization reads it through the
    # process. A VE assembly at scale 50 trains without a separate declaration.
    process = VE(scale=50.0)
    loss_fn = MatchingLoss(process, ScoreParametrization())
    loss = loss_fn(LinearNet(), torch.randn(32, 3))
    assert torch.isfinite(loss)
    assert process.prior.std == pytest.approx(50.0)


def test_ve_score_target_spans_orders_of_magnitude():
    # Why VE + ScoreParametrization wants weight=b^2: the -x1/b target
    # grows like 1/b, so an unweighted L2 is dominated by the low-noise end
    # and the model never learns the rest. Documented on VE.
    torch.manual_seed(0)
    process = VE()
    x1 = torch.randn(4096, 1)

    score_param = ScoreParametrization()
    lo = score_param.target(process, None, x1, torch.zeros(4096)).abs().mean()
    hi = score_param.target(process, None, x1, torch.ones(4096)).abs().mean()
    assert lo / hi > 1000

    # b^2 weighting flattens it: w * (s - (-x1/b))^2 == (b s + x1)^2,
    # i.e. exactly eps-matching, whose target is x1 at every noise level.
    eps_param = EpsParametrization()
    for t_val in [0.0, 0.5, 1.0]:
        t = torch.full((4096,), t_val)
        weight = process.b(t) ** 2
        score_residual = weight * (score_param.target(process, None, x1, t)) ** 2
        eps_residual = eps_param.target(process, None, x1, t) ** 2
        assert torch.allclose(score_residual, eps_residual, rtol=1e-4)
