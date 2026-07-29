import pytest
import torch

from schnetpack.generative import (
    Coupling,
    EpsParametrization,
    FlowMatching,
    GaussianPrior,
    IdentityCoupling,
    OTCoupling,
    Parametrization,
    PCVarianceCoupling,
    PermutationCoupling,
    PseudoForceParametrization,
    ScoreParametrization,
    VE,
    VelocityParametrization,
    VP,
    X0Parametrization,
    expand_t,
)

PARAM_CLASSES = [
    ScoreParametrization,
    EpsParametrization,
    X0Parametrization,
    VelocityParametrization,
    PseudoForceParametrization,
]

# The score and noise parametrizations demand the Gaussian kernel; the rest
# take any process.
GAUSSIAN_ONLY = [ScoreParametrization, EpsParametrization]
ANY_PROCESS = [X0Parametrization, VelocityParametrization, PseudoForceParametrization]


@pytest.fixture(params=PARAM_CLASSES, ids=lambda c: c.__name__)
def parametrization(request):
    return request.param()


@pytest.fixture(params=[VP, VE, FlowMatching], ids=lambda c: c.__name__)
def process_cls(request):
    return request.param


@pytest.fixture
def process(process_cls):
    return process_cls()


def endpoints(process, n=64):
    """A data/noise pair placed at an interior time, in float64."""
    torch.manual_seed(0)
    x0 = torch.randn(n, 3, dtype=torch.float64)
    x1 = torch.randn(n, 3, dtype=torch.float64)
    lo, hi = process.t_min, process.t_max
    t = torch.linspace(
        lo + 0.2 * (hi - lo), hi - 0.2 * (hi - lo), n, dtype=torch.float64
    )
    return x0, x1, t


# --- priors and couplings: who declares what ------------------------------ #


def test_priors_declare_their_endpoint_law():
    # The prior owns "what x1 is": whether it is isotropic Gaussian, and its
    # scale. This is what gates the score/noise targets.
    assert GaussianPrior().gaussian
    assert GaussianPrior(std=50.0).std == pytest.approx(50.0)


def test_gaussian_prior_draws_at_the_declared_scale():
    torch.manual_seed(0)
    x1 = GaussianPrior(std=50.0).sample_like(torch.randn(4096, 3))
    assert x1.shape == (4096, 3)
    assert x1.std().item() == pytest.approx(50.0, rel=0.05)


def test_couplings_declare_whether_they_preserve_the_marginal():
    # Re-ordering leaves x1's marginal untouched; reshaping from the data does
    # not. This is what gates the Gaussian kernel and the derived sampling
    # prior.
    assert IdentityCoupling.preserves_marginal
    assert PermutationCoupling.preserves_marginal
    assert OTCoupling.preserves_marginal
    assert not PCVarianceCoupling.preserves_marginal


def test_identity_coupling_leaves_endpoints_untouched():
    x0 = torch.randn(8, 3)
    x1 = torch.randn(8, 3)
    out0, out1 = IdentityCoupling().pair(x0, x1)
    assert torch.equal(out0, x0)
    assert torch.equal(out1, x1)


def test_permutation_coupling_is_a_pure_reordering():
    # It must return exactly the same multiset of endpoints, only reassigned.
    torch.manual_seed(0)
    x0 = torch.randn(16, 3)
    x1 = torch.randn(16, 3)
    _, paired = PermutationCoupling().pair(x0, x1)
    # every row of the output is a row of the input
    assert paired.shape == x1.shape
    sorted_in = x1[x1[:, 0].argsort()]
    sorted_out = paired[paired[:, 0].argsort()]
    assert torch.allclose(sorted_in, sorted_out)


def test_couplings_never_draw():
    # A custom coupling must opt in to preserving the marginal explicitly.
    class Undeclared(Coupling):
        def pair(self, x0, x1):
            return x0, x1

    assert not Undeclared.preserves_marginal


def test_coupling_is_abstract():
    with pytest.raises(TypeError):
        Coupling()


def test_unimplemented_couplings_raise():
    with pytest.raises(NotImplementedError):
        OTCoupling().pair(torch.randn(4, 3), torch.randn(4, 3))


# --- assembly-time validity ------------------------------------------------ #


def test_score_and_noise_parametrizations_require_the_gaussian_kernel():
    # The score/noise targets are statements about a Gaussian kernel; a
    # configuration without one is refused by validate — which every consumer
    # calls at its own construction.
    reshaped = VP(coupling=PCVarianceCoupling())
    for cls in GAUSSIAN_ONLY:
        with pytest.raises(TypeError, match="Gaussian kernel"):
            cls().validate(reshaped)
        # the same head on a Gaussian configuration is fine
        cls().validate(VP())


def test_the_configuration_is_judged_not_the_class():
    # A permutation coupling re-pairs exchangeable Gaussian draws, which
    # leaves the kernel intact — so the score/noise heads accept it. The old
    # class-based check refused this valid assembly; the property does not.
    repaired = VP(coupling=PermutationCoupling())
    for cls in GAUSSIAN_ONLY:
        cls().validate(repaired)  # must not raise


def test_conditional_expectation_parametrizations_accept_any_process():
    plain = VE(prior=GaussianPrior(30.0), coupling=PCVarianceCoupling())
    for cls in ANY_PROCESS:
        cls().validate(plain)  # must not raise


def test_gpff_pseudo_force_runs_on_any_configuration():
    # The headline: swapping the prior/coupling is one argument, not a new
    # process class. Plain GPFF is a Gaussian VE; shape-prior or aligned-noise
    # GPFF is the same parametrization on the same class, reconfigured.
    PseudoForceParametrization().validate(VE(0.3, 30.0))
    PseudoForceParametrization().validate(
        VE(prior=GaussianPrior(30.0), coupling=PermutationCoupling())
    )


def test_parametrization_is_abstract():
    with pytest.raises(TypeError):
        Parametrization()


# --- round trips ---------------------------------------------------------- #


def test_target_round_trips_to_the_score(process, parametrization):
    # Regressing a parametrization's target and converting back must land on
    # the same score, whatever the head predicts.
    x0, x1, t = endpoints(process)
    x_t = process.interpolate(x0, x1, t)

    target = parametrization.target(process, x0, x1, t)
    score = parametrization.to_score(process, target, x_t, t)

    expected = -x1 / expand_t(process.b(t), x1)
    assert torch.allclose(score, expected, rtol=1e-6, atol=1e-8)


def test_target_round_trips_to_the_velocity(process, parametrization):
    x0, x1, t = endpoints(process)
    x_t = process.interpolate(x0, x1, t)

    target = parametrization.target(process, x0, x1, t)
    velocity = parametrization.to_velocity(process, target, x_t, t)

    expected = VelocityParametrization().target(process, x0, x1, t)
    assert torch.allclose(velocity, expected, rtol=1e-6, atol=1e-8)


def test_target_round_trips_to_x0(process, parametrization):
    x0, x1, t = endpoints(process)
    x_t = process.interpolate(x0, x1, t)

    target = parametrization.target(process, x0, x1, t)
    x0_hat = parametrization.to_x0(process, target, x_t, t)

    assert torch.allclose(x0_hat, x0, rtol=1e-5, atol=1e-7)


def test_parametrizations_agree_on_every_field(process):
    # All five heads, each fed its own target, must describe one and the same
    # process.
    x0, x1, t = endpoints(process)
    x_t = process.interpolate(x0, x1, t)

    fields = {}
    for cls in PARAM_CLASSES:
        p = cls()
        target = p.target(process, x0, x1, t)
        fields[cls.__name__] = (
            p.to_score(process, target, x_t, t),
            p.to_velocity(process, target, x_t, t),
            p.to_x0(process, target, x_t, t),
        )

    ref_score, ref_velocity, ref_x0 = fields["ScoreParametrization"]
    for name, (score, velocity, x0_hat) in fields.items():
        assert torch.allclose(score, ref_score, rtol=1e-5, atol=1e-7), name
        assert torch.allclose(velocity, ref_velocity, rtol=1e-5, atol=1e-7), name
        assert torch.allclose(x0_hat, ref_x0, rtol=1e-5, atol=1e-7), name


def test_scaled_endpoint_keeps_all_fields_consistent(process_cls):
    # The prior owns the endpoint scale: with x1 = s * eps and the process
    # declaring std = s, all parametrizations must still describe one and the
    # same process, and its score must be the true conditional score
    # -(x_t - a x0) / sigma^2 with sigma = b * s.
    s = 7.5
    process = process_cls(scale=s)
    torch.manual_seed(0)
    x0 = torch.randn(64, 3, dtype=torch.float64)
    x1 = s * torch.randn(64, 3, dtype=torch.float64)
    lo, hi = process.t_min, process.t_max
    t = torch.linspace(
        lo + 0.2 * (hi - lo), hi - 0.2 * (hi - lo), 64, dtype=torch.float64
    )
    x_t = process.interpolate(x0, x1, t)

    fields = {}
    for cls in PARAM_CLASSES:
        p = cls()
        target = p.target(process, x0, x1, t)
        fields[cls.__name__] = (
            p.to_score(process, target, x_t, t),
            p.to_velocity(process, target, x_t, t),
            p.to_x0(process, target, x_t, t),
        )

    ref_score, ref_velocity, ref_x0 = fields["ScoreParametrization"]
    for name, (score, velocity, x0_hat) in fields.items():
        assert torch.allclose(score, ref_score, rtol=1e-5, atol=1e-7), name
        assert torch.allclose(velocity, ref_velocity, rtol=1e-5, atol=1e-7), name
        assert torch.allclose(x0_hat, ref_x0, rtol=1e-5, atol=1e-7), name

    a = expand_t(process.a(t), x0)
    sigma = expand_t(process.b(t) * s, x0)
    assert torch.allclose(ref_score, -(x_t - a * x0) / sigma**2, rtol=1e-6)
    assert torch.allclose(ref_x0, x0, rtol=1e-5, atol=1e-6)

    # the eps head regresses unit noise at any endpoint scale
    eps_target = EpsParametrization().target(process, x0, x1, t)
    assert torch.allclose(eps_target, x1 / s)


# --- direct routes -------------------------------------------------------- #


def test_velocity_parametrization_returns_its_output_untouched():
    # The whole point: no conversion, hence nothing to go singular.
    process = FlowMatching()
    p = VelocityParametrization()
    output = torch.randn(8, 3)
    x_t = torch.randn(8, 3)
    t = torch.full((8,), 0.5)
    assert torch.equal(p.to_velocity(process, output, x_t, t), output)


def test_x0_parametrization_returns_its_output_untouched():
    process = VP()
    p = X0Parametrization()
    output = torch.randn(8, 3)
    x_t = torch.randn(8, 3)
    t = torch.full((8,), 0.5)
    assert torch.equal(p.to_x0(process, output, x_t, t), output)


def test_pseudo_force_magnitude_carries_b_on_a_ve_schedule():
    # The claim the parametrization exists for: on VE a = 1, so the target
    # collapses to -2 b x1 and the spread of F/2 estimates b. This is what
    # lets a GPFF head do without a time input.
    process = VE()
    p = PseudoForceParametrization()
    torch.manual_seed(0)
    x0 = torch.randn(4096, 3, dtype=torch.float64)
    x1 = torch.randn(4096, 3, dtype=torch.float64)

    for t_val in [0.1, 0.5, 0.9]:
        t = torch.full((4096,), t_val, dtype=torch.float64)
        b = process.b(t)[0].item()

        target = p.target(process, x0, x1, t)
        assert torch.allclose(target, -2.0 * b * x1, rtol=1e-9, atol=1e-9)
        # b read back off the prediction alone, told nothing about t
        assert (0.5 * target).std().item() == pytest.approx(b, rel=0.05)


def test_pseudo_force_recovers_x0_where_a_b_division_would_not():
    # x0 = x_t + F/2 never divides, so unlike the score route it is exact as
    # b -> 0 rather than 0/0.
    process = VE()
    p = PseudoForceParametrization()
    torch.manual_seed(0)
    x_t = torch.randn(8, 3, dtype=torch.float64)
    output = torch.randn(8, 3, dtype=torch.float64)
    t = torch.zeros(8, dtype=torch.float64)  # b = b_min

    x0_hat = p.to_x0(process, output, x_t, t)
    assert torch.allclose(x0_hat, x_t + 0.5 * output)
    assert torch.isfinite(x0_hat).all()


def test_direct_routes_survive_where_the_generic_one_would_not():
    # At t -> 0 on flow matching, g^2 -> 0: the velocity's route back through
    # the score is 2 (f x - v) / g^2 and blows up by 1/g^2, while the direct
    # routes stay O(1). Not a NaN — just a drift large enough to wreck a step,
    # which is why churn = 0 must never take that route.
    process = FlowMatching()
    torch.manual_seed(0)
    x_t = torch.randn(8, 3)
    t = torch.full((8,), 1e-8)
    output = torch.randn(8, 3)

    velocity = VelocityParametrization().to_velocity(process, output, x_t, t)
    x0_hat = X0Parametrization().to_x0(process, output, x_t, t)
    score = VelocityParametrization().to_score(process, output, x_t, t)

    assert velocity.abs().max().item() < 10.0
    assert x0_hat.abs().max().item() < 10.0
    assert score.abs().max().item() > 1e6


# --- shapes --------------------------------------------------------------- #


def test_fields_broadcast_over_per_sample_times(process, parametrization):
    x0 = torch.randn(8, 5, 3)
    x1 = torch.randn(8, 5, 3)
    lo, hi = process.t_min, process.t_max
    t = torch.linspace(lo + 0.2 * (hi - lo), hi - 0.2 * (hi - lo), 8)
    x_t = process.interpolate(x0, x1, t)

    target = parametrization.target(process, x0, x1, t)
    assert target.shape == x0.shape
    assert parametrization.to_score(process, target, x_t, t).shape == x0.shape
    assert parametrization.to_velocity(process, target, x_t, t).shape == x0.shape
    assert parametrization.to_x0(process, target, x_t, t).shape == x0.shape


def test_target_accepts_but_ignores_bridge_noise(process, parametrization):
    # The eps slot exists for gamma != 0 schedules; while gamma is zero it
    # must make no difference.
    x0, x1, t = endpoints(process)
    without = parametrization.target(process, x0, x1, t)
    with_eps = parametrization.target(process, x0, x1, t, eps=torch.randn_like(x0))
    assert torch.equal(without, with_eps)
