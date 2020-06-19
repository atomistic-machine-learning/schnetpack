import torch
import torch.nn as nn
import torch.nn.functional as F
import schnetpack as spk
import numpy as np


class ExponentialGaussianFunctions(nn.Module):
    def __init__(
        self, num_basis_functions, ini_alpha=0.9448630629184640, exp_weighting=False
    ):
        # ini_alpha is 0.5/Bohr converted to 1/Angstrom
        super(ExponentialGaussianFunctions, self).__init__()
        self.num_basis_functions = num_basis_functions
        self.ini_alpha = ini_alpha
        self.exp_weighting = exp_weighting
        self.register_buffer(
            "center",
            torch.linspace(1, 0, self.num_basis_functions, dtype=torch.float64),
        )
        self.register_buffer(
            "width", torch.tensor(1.0 * self.num_basis_functions, dtype=torch.float64)
        )
        self.register_parameter(
            "_alpha", nn.Parameter(torch.tensor(1.0, dtype=torch.float64))
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self._alpha, spk.nn.softplus_inverse(self.ini_alpha))

    def forward(self, r, cutoff_values):
        expalphar = torch.exp(-F.softplus(self._alpha) * r.view(-1, 1))
        rbf = cutoff_values.view(-1, 1) * torch.exp(
            -self.width * (expalphar - self.center) ** 2
        )
        if self.exp_weighting:
            return rbf * expalphar
        else:
            return rbf


class ExponentialBernsteinPolynomials(nn.Module):
    def __init__(
        self,
        num_basis_functions,
        no_basis_function_at_infinity=False,
        ini_alpha=0.9448630629184640,
        exp_weighting=False,
    ):  # ini_alpha is 0.5/Bohr converted to 1/Angstrom
        super(ExponentialBernsteinPolynomials, self).__init__()
        self.num_basis_functions = num_basis_functions
        self.ini_alpha = ini_alpha
        self.no_basis_function_at_infinity = no_basis_function_at_infinity
        self.exp_weighting = exp_weighting
        if (
            self.no_basis_function_at_infinity
        ):  # increase number of basis functions by one
            num_basis_functions += 1
        # compute values to initialize buffers
        logfactorial = np.zeros((num_basis_functions))
        for i in range(2, num_basis_functions):
            logfactorial[i] = logfactorial[i - 1] + np.log(i)
        v = np.arange(0, num_basis_functions)
        n = (num_basis_functions - 1) - v
        logbinomial = logfactorial[-1] - logfactorial[v] - logfactorial[n]
        if self.no_basis_function_at_infinity:  # remove last basis function at infinity
            v = v[:-1]
            n = n[:-1]
            logbinomial = logbinomial[:-1]
        # register buffers and parameters
        self.register_buffer("logc", torch.tensor(logbinomial, dtype=torch.float64))
        self.register_buffer("n", torch.tensor(n, dtype=torch.float64))
        self.register_buffer("v", torch.tensor(v, dtype=torch.float64))
        self.register_parameter(
            "_alpha", nn.Parameter(torch.tensor(1.0, dtype=torch.float64))
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self._alpha, spk.nn.softplus_inverse(self.ini_alpha))

    def forward(self, r, cutoff_values):
        alphar = -F.softplus(self._alpha) * r.view(-1, 1)
        x = self.logc + self.n * alphar + self.v * torch.log(-torch.expm1(alphar))
        # x[torch.isnan(x)] = 0.0 #removes nan for r == 0, should not be necessary
        rbf = cutoff_values.view(-1, 1) * torch.exp(x)
        if self.exp_weighting:
            return rbf * torch.exp(alphar)
        else:
            return rbf


class BernsteinPolynomials(nn.Module):
    def __init__(self, num_basis_functions, cutoff):
        super(BernsteinPolynomials, self).__init__()
        self.num_basis_functions = num_basis_functions
        # compute values to initialize buffers
        logfactorial = np.zeros((num_basis_functions))
        for i in range(2, num_basis_functions):
            logfactorial[i] = logfactorial[i - 1] + np.log(i)
        v = np.arange(0, num_basis_functions)
        n = (num_basis_functions - 1) - v
        logbinomial = logfactorial[-1] - logfactorial[v] - logfactorial[n]
        # register buffers and parameters
        self.register_buffer("cutoff", torch.tensor(cutoff, dtype=torch.float64))
        self.register_buffer("logc", torch.tensor(logbinomial, dtype=torch.float64))
        self.register_buffer("n", torch.tensor(n, dtype=torch.float64))
        self.register_buffer("v", torch.tensor(v, dtype=torch.float64))
        self.reset_parameters()

    def reset_parameters(self):
        pass

    def forward(self, r, cutoff_values):
        x = r.view(-1, 1) / self.cutoff
        x = torch.where(x < 1.0, x, 0.5 * torch.ones_like(x))  # prevent NaNs
        x = torch.log(x)
        x = self.logc + self.n * x + self.v * torch.log(-torch.expm1(x))
        # x[torch.isnan(x)] = 0.0 #removes nan for r == 0, should not be necessary
        rbf = cutoff_values.view(-1, 1) * torch.exp(x)
        return rbf


class GaussianFunctions(nn.Module):
    def __init__(self, num_basis_functions, cutoff):
        super(GaussianFunctions, self).__init__()
        self.num_basis_functions = num_basis_functions
        self.register_buffer("cutoff", torch.tensor(cutoff, dtype=torch.float64))
        self.register_buffer(
            "center",
            torch.linspace(0, cutoff, self.num_basis_functions, dtype=torch.float64),
        )
        self.register_buffer(
            "width",
            torch.tensor(self.num_basis_functions / cutoff, dtype=torch.float64),
        )
        self.reset_parameters()

    def reset_parameters(self):
        pass

    def forward(self, r, cutoff_values):
        rbf = cutoff_values.view(-1, 1) * torch.exp(
            -self.width * (r.view(-1, 1) - self.center) ** 2
        )
        return rbf
