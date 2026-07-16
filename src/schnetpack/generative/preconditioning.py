"""
Preconditioning — rescaling a raw network into a well-conditioned denoiser.

A denoiser has to work across noise levels spanning orders of magnitude. Asked
to predict x0 directly, a network faces inputs whose scale and whose ratio of
signal to noise change completely between sigma_min and sigma_max; at low noise
the answer is nearly the input, at high noise it is nearly the data mean.
Preconditioning (Karras et al. 2022) folds that sigma dependence into four
scalar functions so the network only ever learns the residual:

    D(x, sigma) = c_skip(sigma) x + c_out(sigma) F(c_in(sigma) x, c_noise(sigma))

with c_in normalizing the input variance, c_skip choosing how much of x to
carry over, c_out fixing the target's scale to unit variance, and c_noise
compressing sigma into a sane conditioning input.

This is a *net-to-net* wrapper: :class:`PrecondDenoiser` takes a network and
returns a network. It changes nothing about the model contract — the result is
still ``model(x, t, cond) -> raw output`` — so it composes with
:class:`~schnetpack.generative.parametrizations.X0Parametrization` like any
other denoiser. That is the whole reason EDM needs no special model class.

The same hook covers consistency models, whose boundary condition
c_skip(t_min) = 1, c_out(t_min) = 0 (so D(x, t_min) = x exactly) is just
another :class:`Preconditioner`.
"""

from typing import Optional

import torch
from torch import nn

from schnetpack.generative.paths import expand_t

__all__ = ["Preconditioner", "EDMPreconditioner", "PrecondDenoiser"]


class Preconditioner:
    """
    The four scaling functions of a denoiser, defaulting to no preconditioning.

    c_skip = 0, c_out = 1, c_in = 1 and c_noise = identity leave
    :class:`PrecondDenoiser` equal to the bare network, so the trivial case is
    the base class rather than a special case inside it.
    """

    def c_skip(self, t: torch.Tensor) -> torch.Tensor:
        """Weight on the skip connection from the noisy input."""
        return torch.zeros_like(t)

    def c_out(self, t: torch.Tensor) -> torch.Tensor:
        """Scale applied to the network output."""
        return torch.ones_like(t)

    def c_in(self, t: torch.Tensor) -> torch.Tensor:
        """Scale applied to the network input."""
        return torch.ones_like(t)

    def c_noise(self, t: torch.Tensor) -> torch.Tensor:
        """
        Transform of the noise level fed to the network as its time input.

        The identity, deliberately: a log would diverge as t -> 0, and the base
        class must stay usable on every path, including those whose t_min is 0.
        """
        return t


class EDMPreconditioner(Preconditioner):
    """
    The EDM preconditioning of Karras et al. (2022), for sigma-space paths.

    Derived by requiring unit variance of the network input and of the
    effective training target, given data of scale ``sigma_data``:

        c_skip = sd^2 / (sigma^2 + sd^2)
        c_out  = sigma sd / sqrt(sigma^2 + sd^2)
        c_in   = 1 / sqrt(sigma^2 + sd^2)
        c_noise = 1/4 log(sigma)

    ``sigma_data`` must match the one used by
    :class:`~schnetpack.generative.losses.EDMLoss`, whose weighting is exactly
    1 / c_out^2 — the two are halves of one derivation, and mismatching them
    mistrains silently rather than raising.

    Intended for :class:`~schnetpack.generative.paths.EDMPath`, where t is
    literally sigma.
    """

    def __init__(self, sigma_data: float = 0.5):
        """
        Args:
            sigma_data: standard deviation of the (normalized) data
        """
        self.sigma_data = sigma_data

    def c_skip(self, t):
        return self.sigma_data**2 / (t**2 + self.sigma_data**2)

    def c_out(self, t):
        return t * self.sigma_data / torch.sqrt(t**2 + self.sigma_data**2)

    def c_in(self, t):
        return 1.0 / torch.sqrt(t**2 + self.sigma_data**2)

    def c_noise(self, t):
        return 0.25 * torch.log(t)


class PrecondDenoiser(nn.Module):
    """
    Wraps a network into a preconditioned denoiser predicting the clean sample.

    Satisfies the model contract ``(x, t, cond) -> raw output``, so it drops
    straight into a sampler or loss paired with
    :class:`~schnetpack.generative.parametrizations.X0Parametrization`.

    The EDM assembly is::

        model = PrecondDenoiser(net, EDMPreconditioner(sigma_data=0.5))
        # + X0Parametrization() + EDMPath()

    The network sees ``c_noise(t)`` rather than t as its time input, still
    per-sample and unexpanded; only the input and output scalings are
    broadcast against x.
    """

    def __init__(self, net: nn.Module, precond: Optional[Preconditioner] = None):
        """
        Args:
            net: network predicting the preconditioned residual F
            precond: scaling functions (default: no preconditioning)
        """
        super().__init__()
        self.net = net
        self.precond = precond if precond is not None else Preconditioner()

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond=None) -> torch.Tensor:
        """
        Denoise x at noise level t.

        Args:
            x: noisy sample, shape (n_samples, ...)
            t: noise level, per-sample or scalar
            cond: conditioning passed through to the network
        """
        p = self.precond
        f = self.net(x * expand_t(p.c_in(t), x), p.c_noise(t), cond)
        return expand_t(p.c_skip(t), x) * x + expand_t(p.c_out(t), x) * f
