from math import pi

import torch
import torch.nn as nn

__all__ = [
    "gaussian_rbf",
    "GaussianRBF",
    "GaussianRBFCentered",
    "BesselRBF",
    "BernsteinRBF",
    "PhysNetBasisRBF",
]

from torch import nn as nn


def gaussian_rbf(inputs: torch.Tensor, offsets: torch.Tensor, widths: torch.Tensor):
    coeff = -0.5 / torch.pow(widths, 2)
    diff = inputs[..., None] - offsets
    y = torch.exp(coeff * torch.pow(diff, 2))
    return y


class GaussianRBF(nn.Module):
    r"""Gaussian radial basis functions."""

    def __init__(
        self, n_rbf: int, cutoff: float, start: float = 0.0, trainable: bool = False
    ):
        """
        Args:
            n_rbf: total number of Gaussian functions, :math:`N_g`.
            cutoff: center of last Gaussian function, :math:`\mu_{N_g}`
            start: center of first Gaussian function, :math:`\mu_0`.
            trainable: If True, widths and offset of Gaussian functions
                are adjusted during training process.
        """
        super(GaussianRBF, self).__init__()
        self.n_rbf = n_rbf

        # compute offset and width of Gaussian functions
        offset = torch.linspace(start, cutoff, n_rbf)
        widths = torch.FloatTensor(
            torch.abs(offset[1] - offset[0]) * torch.ones_like(offset)
        )
        if trainable:
            self.widths = nn.Parameter(widths)
            self.offsets = nn.Parameter(offset)
        else:
            self.register_buffer("widths", widths)
            self.register_buffer("offsets", offset)

    def forward(self, inputs: torch.Tensor):
        return gaussian_rbf(inputs, self.offsets, self.widths)


class GaussianRBFCentered(nn.Module):
    r"""Gaussian radial basis functions centered at the origin."""

    def __init__(
        self, n_rbf: int, cutoff: float, start: float = 1.0, trainable: bool = False
    ):
        """
        Args:
            n_rbf: total number of Gaussian functions, :math:`N_g`.
            cutoff: width of last Gaussian function, :math:`\mu_{N_g}`
            start: width of first Gaussian function, :math:`\mu_0`.
            trainable: If True, widths of Gaussian functions
                are adjusted during training process.
        """
        super(GaussianRBFCentered, self).__init__()
        self.n_rbf = n_rbf

        # compute offset and width of Gaussian functions
        widths = torch.linspace(start, cutoff, n_rbf)
        offset = torch.zeros_like(widths)
        if trainable:
            self.widths = nn.Parameter(widths)
            self.offsets = nn.Parameter(offset)
        else:
            self.register_buffer("widths", widths)
            self.register_buffer("offsets", offset)

    def forward(self, inputs: torch.Tensor):
        return gaussian_rbf(inputs, self.offsets, self.widths)


class BesselRBF(nn.Module):
    """
    Sine for radial basis functions with coulomb decay (0th order bessel).

    References:

    .. [#dimenet] Klicpera, Groß, Günnemann:
       Directional message passing for molecular graphs.
       ICLR 2020
    """

    def __init__(self, n_rbf: int, cutoff: float):
        """
        Args:
            cutoff: radial cutoff
            n_rbf: number of basis functions.
        """
        super(BesselRBF, self).__init__()
        self.n_rbf = n_rbf

        freqs = torch.arange(1, n_rbf + 1) * pi / cutoff
        self.register_buffer("freqs", freqs)

    def forward(self, inputs):
        ax = inputs[..., None] * self.freqs
        sinax = torch.sin(ax)
        norm = torch.where(inputs == 0, torch.tensor(1.0, device=inputs.device), inputs)
        y = sinax / norm[..., None]
        return y


class BernsteinRBF(torch.nn.Module):
    r"""Bernstein radial basis functions.

    According to
    B_{v,n}(x) = \binom{n}{v} x^v (1 - x)^{n - v}
    with
        B as the Bernstein polynomial of degree v
        binom{k}{n} as the binomial coefficient n! / (k! * (n - k)!)
        they become in logaritmic form log(n!) - log(k!) - log((n - k)!)
        n as index running from 0 to degree k

    The logarithmic form of the k-th Bernstein polynominal of degree n is

        log(B_{k}_{n}) = logBinomCoeff + k * log(x) - (n-k) * log(1-x)
        k_term is here k*log(x)
        n_k_term is here (n-k)*log(1-x)
        x is here the radial basis expansion : exp[-alpha*d]

        logBinomCoeff is a scalar
        k_term is a vector
        n_k_term is also a vector

    log to avoid numerical overflow errors, and ensure stability
    """

    def __init__(self, n_rbf: int, cutoff: float, init_alpha: float = 0.95):
        """
        Args:
            n_rbf: total number of Bernstein functions, :math:`N_g`.
            cutoff: center of last Bernstein function, :math:`\mu_{N_g}`
        """

        super(BernsteinRBF, self).__init__()
        self.n_rbf = n_rbf

        # log binomal coefficient vector
        b = self.calculate_log_binomial_coefficients(n_rbf)
        n_idx = torch.arange(0, n_rbf)
        n_k_idx = n_rbf - 1 - n_idx

        # register buffers and parameters
        self.register_buffer("cutoff", torch.tensor(cutoff))
        self.register_buffer("b", b)
        self.register_buffer("n", n_idx)
        self.register_buffer("n_k", n_k_idx)
        self.register_buffer("init_alpha", torch.tensor(init_alpha))

    # log of factorial (n! or k! or n-k!)
    def log_factorial(self, n):
        # log of factorial degree n
        return torch.sum(torch.log(torch.arange(1, n + 1)))

    # calculate log binominal coefficient
    def log_binomial_coefficient(self, n, k):
        # n_factorial - k_factorial - n_k_factorial
        return self.log_factorial(n) - (
            self.log_factorial(k) + self.log_factorial(n - k)
        )

    # vector of log binominal coefficients
    def calculate_log_binomial_coefficients(self, n_rbf):
        # store the log binomial coefficients
        # Loop through each value from 0 to n_rbf-1
        log_binomial_coeffs = [
            self.log_binomial_coefficient(n_rbf - 1, x) for x in range(n_rbf)
        ]
        return torch.tensor(log_binomial_coeffs)

    def forward(self, inputs):
        exp_x = -self.init_alpha * inputs[..., None]
        x = torch.exp(exp_x)
        k_term = self.n * torch.where(self.n != 0, torch.log(x), torch.zeros_like(x))
        n_k_term = self.n_k * torch.where(
            self.n_k != 0, torch.log(1 - x), torch.zeros_like(x)
        )
        y = torch.exp(self.b + k_term + n_k_term)
        return y


class PhysNetBasisRBF(torch.nn.Module):
    """
    Expand distances in the basis used in PhysNet (see https://arxiv.org/abs/1902.08408)

    width (beta_k) = (2K^⁻1 * (1 - exp(-cutoff)))^-2)
    center (mu_k) = equally spaced between exp(-cutoff) and 1

    """

    def __init__(self, n_rbf: int, cutoff: float, trainable: bool):
        """
        Args:
            n_rbf: total number of basis functions.
            cutoff: cutoff basis functions
        """

        super(PhysNetBasisRBF, self).__init__()
        self.n_rbf = n_rbf

        # compute offset and width of Gaussian functions
        widths = ((2 / self.n_rbf) * (1 - torch.exp(torch.Tensor([-cutoff])))) ** (-2)
        r_0 = torch.exp(torch.Tensor([-cutoff])).item()
        centers = torch.linspace(r_0, 1, self.n_rbf)

        if trainable:
            self.widths = torch.nn.Parameter(widths)
            self.centers = torch.nn.Parameter(centers)
        else:
            self.register_buffer("widths", widths)
            self.register_buffer("centers", centers)

    def forward(self, inputs: torch.Tensor):
        return torch.exp(
            -abs(self.widths) * (torch.exp(-inputs[..., None]) - self.centers) ** 2
        )
