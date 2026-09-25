import pytest
import torch

from schnetpack import properties
from schnetpack.dynamics import (
    AnnealedNoise,
    DirectDenoising,
    Dynamics,
    EulerMaruyama,
    Sampler,
    Scaffold,
    StateConstraint,
)
from schnetpack.generative import (
    VE,
    VP,
    PseudoForceParametrization,
    ScoreParametrization,
    expand_t,
)
from tests.dynamics.test_sampling import batch_model


class Recorder(StateConstraint):
    """Logs (hook, step, t) and leaves the state alone."""

    def __init__(self):
        self.log = []

    def before_step(self, batch, step, n_steps, dynamics):
        self.log.append(("before", step, batch[dynamics.time_key]))
        return batch

    def after_step(self, batch, step, n_steps, dynamics):
        self.log.append(("after", step, batch[dynamics.time_key]))
        return batch


# --- the loop ------------------------------------------------------------- #


def test_sampler_hooks_see_grid_times_around_each_step():
    vp = VP()
    recorder = Recorder()
    sampler = Sampler(
        batch_model(lambda x, t: -x),
        vp,
        ScoreParametrization(),
        EulerMaruyama(),
        constraints=[recorder],
    )
    n_steps = 4
    sampler.denoise(
        sampler.prior.sample_from_batch({properties.R: torch.empty(3, 1)}), n_steps
    )

    ts = sampler.grid(vp.t_max, vp.t_min, n_steps)
    assert len(recorder.log) == 2 * n_steps
    for i in range(n_steps):
        hook, step, t = recorder.log[2 * i]
        assert (hook, step) == ("before", i)
        assert torch.allclose(t, ts[i].expand(3))
        hook, step, t = recorder.log[2 * i + 1]
        assert (hook, step) == ("after", i + 1)
        assert torch.allclose(t, ts[i + 1].expand(3))


def test_sampler_denoise_starts_at_t_start():
    recorder = Recorder()
    sampler = Sampler(
        batch_model(lambda x, t: -x),
        VP(),
        ScoreParametrization(),
        EulerMaruyama(),
        constraints=[recorder],
    )
    sampler.denoise({properties.R: torch.randn(2, 1)}, 3, t_start=0.5)
    assert torch.allclose(recorder.log[0][2], torch.full((2,), 0.5))


def test_direct_denoising_hooks_see_zero_time_and_reject_t_start():
    recorder = Recorder()
    model = batch_model(lambda x, t: torch.zeros_like(x))
    sampler = DirectDenoising(
        model, VE(0.01, 3.0), PseudoForceParametrization(), constraints=[recorder]
    )
    sampler.denoise(
        sampler.prior.sample_from_batch({properties.R: torch.empty(2, 1)}), 3
    )
    assert [(h, s) for h, s, _ in recorder.log] == [
        ("before", 0),
        ("after", 1),
        ("before", 1),
        ("after", 2),
        ("before", 2),
        ("after", 3),
    ]
    assert all((t == 0.0).all() for _, _, t in recorder.log)

    with pytest.raises(ValueError, match="time-free"):
        sampler.denoise({properties.R: torch.zeros(2, 1)}, 3, t_start=0.5)


def test_stochastic_lambda_is_an_annealed_noise_constraint():
    # The built-in injection and the explicit constraint are the same draw.
    process, param = VE(0.01, 3.0), PseudoForceParametrization()
    model = batch_model(lambda x, t: -x)
    batch = {properties.R: torch.randn(8, 2)}

    torch.manual_seed(1)
    builtin = DirectDenoising(model, process, param, stochastic_lambda=0.7).denoise(
        batch, 5
    )
    torch.manual_seed(1)
    explicit = DirectDenoising(
        model, process, param, stochastic_lambda=0.0, constraints=[AnnealedNoise(0.7)]
    ).denoise(batch, 5)
    assert torch.equal(builtin[properties.R], explicit[properties.R])


# --- scaffold ------------------------------------------------------------- #


def scaffold_batch(mask, reference):
    return {
        properties.R: torch.empty_like(reference),
        properties.fixed_atoms: mask,
        properties.R_reference: reference,
    }


def test_scaffold_direct_denoising_model_sees_clean_scaffold():
    mask = torch.tensor([True, False, True, False])
    reference = torch.tensor([[1.0, 2.0], [0.0, 0.0], [-3.0, 0.5], [0.0, 0.0]])
    seen = []

    def model(x, t):
        seen.append(x[mask].clone())
        return -x  # pseudo force pulling everything to the origin

    sampler = DirectDenoising(
        batch_model(model),
        VE(0.01, 3.0),
        PseudoForceParametrization(),
        stochastic_lambda=1.0,
        constraints=[Scaffold()],
    )
    out = sampler.denoise(
        sampler.prior.sample_from_batch(scaffold_batch(mask, reference)), 6
    )
    x = out[properties.R]

    # scaffold overwrites after the noise injection, so every input holds it
    assert all(torch.equal(s, reference[mask]) for s in seen)
    assert torch.equal(x[mask], reference[mask])
    assert not torch.allclose(x[~mask], torch.zeros(2, 2))  # still moved


def test_scaffold_sampler_renoises_to_the_grid_time():
    torch.manual_seed(0)
    vp = VP()
    n = 4000
    mask = torch.zeros(n, dtype=torch.bool)
    mask[: n - 2] = True
    reference = torch.full((n, 1), 2.0)
    standardized = []

    def model(x, t):
        a = expand_t(vp.a(t), x)[mask]
        b = expand_t(vp.b(t), x)[mask]
        standardized.append((x[mask] - a * reference[mask]) / b)
        return -x

    sampler = Sampler(
        batch_model(model),
        vp,
        ScoreParametrization(),
        EulerMaruyama(),
        constraints=[Scaffold()],
    )
    out = sampler.denoise(
        sampler.prior.sample_from_batch(scaffold_batch(mask, reference)), 5
    )

    # every model input carries the scaffold at the noise level of its t
    for z in standardized:
        assert z.mean().item() == pytest.approx(0.0, abs=0.1)
        assert z.std().item() == pytest.approx(1.0, abs=0.1)
    # and the output holds it exactly
    assert torch.equal(out[properties.R][mask], reference[mask])


class Descent(Dynamics):
    """Minimal non-generative driver: x <- x + 0.5 * force."""

    def denoise(self, batch, n_steps):
        batch = self.calculator.prepare(batch)
        for i in range(n_steps):
            batch = self.before_step(batch, i, n_steps)
            force = self.calculator(batch)["forces"]
            batch = {**batch, self.key: batch[self.key] + 0.5 * force}
            batch = self.after_step(batch, i + 1, n_steps)
        return batch


def test_scaffold_non_generative_dynamics_overwrites():
    mask = torch.tensor([True, False, True])
    reference = torch.tensor([[1.0, 2.0], [0.0, 0.0], [-3.0, 0.5]])
    seen = []

    def model(batch):
        x = batch[properties.R]
        seen.append(x[mask].clone())
        return {"forces": -x}

    batch = scaffold_batch(mask, reference)
    batch[properties.R] = torch.ones(3, 2)
    out = Descent(model, constraints=[Scaffold()]).denoise(batch, 4)
    x = out[properties.R]

    assert all(torch.equal(s, reference[mask]) for s in seen)
    assert torch.equal(x[mask], reference[mask])
    assert torch.allclose(x[~mask], torch.full((1, 2), 0.5**4))


def test_sample_without_prior_raises():
    with pytest.raises(ValueError, match="no prior"):
        Descent(lambda batch: {}).sample(2, 1)


def test_scaffold_validates_its_keys():
    model = batch_model(lambda x, t: x)
    sampler = DirectDenoising(
        model, VE(0.01, 3.0), PseudoForceParametrization(), constraints=[Scaffold()]
    )
    bad_mask = scaffold_batch(torch.tensor([True, False]), torch.zeros(3, 1))
    with pytest.raises(ValueError, match="one flag per row"):
        sampler.denoise(sampler.prior.sample_from_batch(bad_mask), 2)
    bad_reference = scaffold_batch(
        torch.tensor([True, False, False]), torch.zeros(3, 1)
    )
    bad_reference[properties.R_reference] = torch.zeros(2, 1)
    with pytest.raises(ValueError, match="shaped like"):
        sampler.denoise(sampler.prior.sample_from_batch(bad_reference), 2)
