import pytest
import torch

from schnetpack.generative import (
    Coupling,
    DataToDataCoupling,
    EDMPath,
    FMPath,
    IndependentCoupling,
    OTCoupling,
    VEPath,
    VPPath,
    expand_t,
)

# Imported from the module rather than the package root: the new classes share
# their names with the old parametrization module until the C5 cutover.
from schnetpack.generative.parametrizations import (
    EpsParametrization,
    Parametrization,
    PseudoForceParametrization,
    ScoreParametrization,
    VelocityParametrization,
    X0Parametrization,
)

PARAM_CLASSES = [
    ScoreParametrization,
    EpsParametrization,
    X0Parametrization,
    VelocityParametrization,
    PseudoForceParametrization,
]


def all_parametrizations(path):
    return [cls(path) for cls in PARAM_CLASSES]


@pytest.fixture(params=PARAM_CLASSES, ids=lambda c: c.__name__)
def parametrization(request, path):
    return request.param(path)


@pytest.fixture(params=[VPPath, VEPath, FMPath, EDMPath], ids=lambda c: c.__name__)
def path(request):
    return request.param()


def endpoints(path, n=64):
    """A data/noise pair placed at an interior time, in float64."""
    torch.manual_seed(0)
    x0 = torch.randn(n, 3, dtype=torch.float64)
    x1 = torch.randn(n, 3, dtype=torch.float64)
    lo, hi = path.t_min, path.t_max
    t = torch.linspace(
        lo + 0.2 * (hi - lo), hi - 0.2 * (hi - lo), n, dtype=torch.float64
    )
    return x0, x1, t


# --- couplings ------------------------------------------------------------ #


def test_independent_coupling_draws_fresh_noise():
    torch.manual_seed(0)
    x0 = torch.randn(4096, 2)
    out0, x1 = IndependentCoupling().sample(x0)

    assert torch.equal(out0, x0)
    assert x1.shape == x0.shape
    assert x1.mean().item() == pytest.approx(0.0, abs=0.05)
    assert x1.std().item() == pytest.approx(1.0, abs=0.05)

    # independence: the pair should be essentially uncorrelated
    corr = (x0.flatten() * x1.flatten()).mean().item()
    assert abs(corr) < 0.05


def test_independent_coupling_ignores_a_given_x1():
    x0 = torch.randn(8, 3)
    given = torch.full_like(x0, 99.0)
    _, x1 = IndependentCoupling().sample(x0, given)
    assert not torch.allclose(x1, given)


@pytest.mark.parametrize("coupling", [OTCoupling(), DataToDataCoupling()])
def test_unimplemented_couplings_raise(coupling):
    with pytest.raises(NotImplementedError):
        coupling.sample(torch.randn(4, 3), torch.randn(4, 3))


def test_coupling_is_abstract():
    with pytest.raises(TypeError):
        Coupling()


def test_parametrization_is_abstract():
    with pytest.raises(TypeError):
        Parametrization(VPPath())


# --- round trips ---------------------------------------------------------- #


def test_target_round_trips_to_the_score(path, parametrization):
    # Regressing a parametrization's target and converting back must land on
    # the same score, whatever the head predicts.
    x0, x1, t = endpoints(path)
    x_t = path.interpolate(x0, x1, t)

    target = parametrization.target(x0, x1, t)
    score = parametrization.to_score(target, x_t, t)

    expected = -x1 / expand_t(path.sigma(t), x1)
    assert torch.allclose(score, expected, rtol=1e-6, atol=1e-8)


def test_target_round_trips_to_the_velocity(path, parametrization):
    x0, x1, t = endpoints(path)
    x_t = path.interpolate(x0, x1, t)

    target = parametrization.target(x0, x1, t)
    velocity = parametrization.to_velocity(target, x_t, t)

    expected = VelocityParametrization(path).target(x0, x1, t)
    assert torch.allclose(velocity, expected, rtol=1e-6, atol=1e-8)


def test_target_round_trips_to_x0(path, parametrization):
    x0, x1, t = endpoints(path)
    x_t = path.interpolate(x0, x1, t)

    target = parametrization.target(x0, x1, t)
    x0_hat = parametrization.to_x0(target, x_t, t)

    assert torch.allclose(x0_hat, x0, rtol=1e-5, atol=1e-7)


def test_parametrizations_agree_on_every_field(path):
    # All four heads, each fed its own target, must describe one and the same
    # process.
    x0, x1, t = endpoints(path)
    x_t = path.interpolate(x0, x1, t)

    fields = {}
    for p in all_parametrizations(path):
        target = p.target(x0, x1, t)
        fields[type(p).__name__] = (
            p.to_score(target, x_t, t),
            p.to_velocity(target, x_t, t),
            p.to_x0(target, x_t, t),
        )

    ref_score, ref_velocity, ref_x0 = fields["ScoreParametrization"]
    for name, (score, velocity, x0_hat) in fields.items():
        assert torch.allclose(score, ref_score, rtol=1e-5, atol=1e-7), name
        assert torch.allclose(velocity, ref_velocity, rtol=1e-5, atol=1e-7), name
        assert torch.allclose(x0_hat, ref_x0, rtol=1e-5, atol=1e-7), name


# --- direct routes -------------------------------------------------------- #


def test_velocity_parametrization_returns_its_output_untouched():
    # The whole point: no conversion, hence nothing to go singular.
    path = FMPath()
    p = VelocityParametrization(path)
    output = torch.randn(8, 3)
    x_t = torch.randn(8, 3)
    t = torch.full((8,), 0.5)
    assert torch.equal(p.to_velocity(output, x_t, t), output)


def test_x0_parametrization_returns_its_output_untouched():
    path = VPPath()
    p = X0Parametrization(path)
    output = torch.randn(8, 3)
    x_t = torch.randn(8, 3)
    t = torch.full((8,), 0.5)
    assert torch.equal(p.to_x0(output, x_t, t), output)


def test_pseudo_force_magnitude_carries_sigma_on_a_ve_path():
    # The claim the parametrization exists for: on a VE path alpha = 1, so the
    # target collapses to -2 sigma x1 and the spread of F/2 estimates sigma. This
    # is what lets a GPFF head do without a time input.
    path = VEPath()
    p = PseudoForceParametrization(path)
    torch.manual_seed(0)
    x0 = torch.randn(4096, 3, dtype=torch.float64)
    x1 = torch.randn(4096, 3, dtype=torch.float64)

    for t_val in [0.1, 0.5, 0.9]:
        t = torch.full((4096,), t_val, dtype=torch.float64)
        sigma = path.sigma(t)[0].item()

        target = p.target(x0, x1, t)
        assert torch.allclose(target, -2.0 * sigma * x1, rtol=1e-9, atol=1e-9)
        # sigma read back off the prediction alone, told nothing about t
        assert (0.5 * target).std().item() == pytest.approx(sigma, rel=0.05)


def test_pseudo_force_recovers_x0_where_a_sigma_division_would_not():
    # x0 = x_t + F/2 never divides, so unlike the score route it is exact as
    # sigma -> 0 rather than 0/0.
    path = VEPath()
    p = PseudoForceParametrization(path)
    torch.manual_seed(0)
    x_t = torch.randn(8, 3, dtype=torch.float64)
    output = torch.randn(8, 3, dtype=torch.float64)
    t = torch.zeros(8, dtype=torch.float64)  # sigma = sigma_min

    x0_hat = p.to_x0(output, x_t, t)
    assert torch.allclose(x0_hat, x_t + 0.5 * output)
    assert torch.isfinite(x0_hat).all()


def test_direct_routes_survive_where_the_generic_one_would_not():
    # At t -> 0 on an FM path, g^2 -> 0: the velocity's route back through the
    # score is 2 (f x - v) / g^2 and blows up by 1/g^2, while the direct routes
    # stay O(1). Not a NaN — just a drift large enough to wreck a step, which
    # is why churn = 0 must never take that route.
    path = FMPath()
    torch.manual_seed(0)
    x_t = torch.randn(8, 3)
    t = torch.full((8,), 1e-8)
    output = torch.randn(8, 3)

    velocity = VelocityParametrization(path).to_velocity(output, x_t, t)
    x0_hat = X0Parametrization(path).to_x0(output, x_t, t)
    score = VelocityParametrization(path).to_score(output, x_t, t)

    assert velocity.abs().max().item() < 10.0
    assert x0_hat.abs().max().item() < 10.0
    assert score.abs().max().item() > 1e6


# --- shapes --------------------------------------------------------------- #


def test_fields_broadcast_over_per_sample_times(path, parametrization):
    x0 = torch.randn(8, 5, 3)
    x1 = torch.randn(8, 5, 3)
    lo, hi = path.t_min, path.t_max
    t = torch.linspace(lo + 0.2 * (hi - lo), hi - 0.2 * (hi - lo), 8)
    x_t = path.interpolate(x0, x1, t)

    target = parametrization.target(x0, x1, t)
    assert target.shape == x0.shape
    assert parametrization.to_score(target, x_t, t).shape == x0.shape
    assert parametrization.to_velocity(target, x_t, t).shape == x0.shape
    assert parametrization.to_x0(target, x_t, t).shape == x0.shape


def test_target_accepts_but_ignores_bridge_noise(path, parametrization):
    # The eps slot exists for gamma != 0 paths; while gamma is zero it must
    # make no difference.
    x0, x1, t = endpoints(path)
    without = parametrization.target(x0, x1, t)
    with_eps = parametrization.target(x0, x1, t, eps=torch.randn_like(x0))
    assert torch.equal(without, with_eps)
