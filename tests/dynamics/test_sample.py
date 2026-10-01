"""
The Sample base: what every step rule inherits from it.
"""

import pytest
import torch

from schnetpack import properties
from schnetpack.dynamics import GenerativeCalculator
from schnetpack.dynamics.constraints.state import Scaffold
from schnetpack.dynamics.guidance import Guidance
from schnetpack.dynamics.sampling import sampler as sample
from schnetpack.generative import (
    VP,
    EpsParametrization,
    FlowMatching,
    VelocityParametrization,
)


def model(batch):
    """A smooth toy head that reads both x and t."""
    x, t = batch[properties.R], batch[properties.t]
    return {"prediction": 0.3 * torch.tanh(x) * (1.0 + t[:, None])}


class Pull(Guidance):
    """A linear pull toward the origin, a guidance term that depends on t."""

    def forward(self, batch):
        return -batch[properties.R] * (1.0 + batch[properties.t][:, None])


def start(n_rows=6, seed=1):
    generator = torch.Generator().manual_seed(seed)
    return {
        properties.R: torch.randn(n_rows, 3, generator=generator, dtype=torch.float64)
    }


def test_sample_draws_from_the_processs_prior():
    driver = sample.Heun(GenerativeCalculator(model, VP(), EpsParametrization()))
    assert driver.prior is not None
    template = {properties.R: torch.empty(5, 3, dtype=torch.float64)}
    out = driver.denoise(driver.prior.sample_from_batch(template), 4)
    assert out[properties.R].shape == (5, 3)
    assert torch.all(out[properties.t] == driver.process.t_min)


def test_sample_refuses_a_calculator_without_a_process():
    with pytest.raises(TypeError, match="GenerativeCalculator"):
        sample.Heun(model)


def test_ancestral_refuses_a_process_without_the_gaussian_kernel():
    from tests.dynamics.test_sampling import ShapedPrior

    calculator = GenerativeCalculator(
        model, FlowMatching(prior=ShapedPrior()), VelocityParametrization()
    )
    sample.Heun(calculator, eta2=0.0)  # chart-free: assembles
    with pytest.raises(ValueError, match="chart"):
        sample.Ancestral(calculator)


def test_scaffold_reads_the_process_and_time_key_from_the_sampler():
    process, parametrization = VP(), EpsParametrization()
    batch = start()
    mask = torch.tensor([True, False, True, False, False, False])
    batch = {
        **batch,
        properties.fixed_atoms: mask,
        properties.R_reference: torch.zeros_like(batch[properties.R]),
    }
    driver = sample.EulerMaruyama(
        GenerativeCalculator(model, process, parametrization), constraints=[Scaffold()]
    )
    torch.manual_seed(0)
    x = driver.denoise(batch, 20)[properties.R]
    # the scaffold rows end on the reference, noised to the final grid time
    torch.testing.assert_close(x[mask], torch.zeros_like(x[mask]), atol=0.1, rtol=0)


@pytest.mark.parametrize(
    "process, parametrization, eta2",
    [
        (VP(), EpsParametrization(), 1.0),
        (VP(), EpsParametrization(), 0.0),
        (FlowMatching(), VelocityParametrization(), 0.0),
    ],
)
def test_drift_costs_exactly_one_model_evaluation(process, parametrization, eta2):
    # The drift is one statement about one score (or one velocity) — a
    # single model call, whatever the eta2 and the route.
    calls = []

    def counting(batch):
        calls.append(1)
        return model(batch)

    driver = sample.EulerMaruyama(
        GenerativeCalculator(counting, process, parametrization), eta2=eta2
    )
    x = start()[properties.R]
    driver.drift({}, x, torch.full((x.shape[0],), 0.5, dtype=x.dtype))
    assert len(calls) == 1


def test_chart_free_route_steps_on_the_reverse_ode():
    from schnetpack.generative import ReverseODE, ReverseSDE

    calculator = GenerativeCalculator(model, FlowMatching(), VelocityParametrization())
    assert isinstance(sample.Heun(calculator, eta2=0.0).reverse, ReverseODE)
    assert isinstance(sample.Heun(calculator, eta2=1.0).reverse, ReverseSDE)
    guided = GenerativeCalculator(
        model, FlowMatching(), VelocityParametrization(), guidance=[Pull()]
    )
    # the calculator holds the chart to fold F into the velocity; the sampler
    # stays on the chart-free route
    assert guided.sde is not None
    assert isinstance(sample.Heun(guided, eta2=0.0).reverse, ReverseODE)
