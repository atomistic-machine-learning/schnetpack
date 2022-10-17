import torch


def binom(n: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """
    Compute binomial coefficients (n k)
    """
    return torch.exp(
        torch.lgamma(n + 1) - torch.lgamma((n - k) + 1) - torch.lgamma(k + 1)
    )
