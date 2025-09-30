import torch
import torch.nn as nn
from schnetpack.units import convert_units

from typing import Dict, List

__all__ = ["Aggregation", "NSAggregation", "NSwPrecondAggregation", "ForceAggregation"]


class Aggregation(nn.Module):
    """
    Aggregate predictions into a single output variable.

    Args:
        keys (list(str)): List of properties to be added.
        output_key (str): Name of new property in output.
    """

    def __init__(self, keys: List[str], output_key: str = "y"):
        super(Aggregation, self).__init__()

        self.keys: List[str] = list(keys)
        self.output_key = output_key
        self.model_outputs = [output_key]

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        energy = torch.stack([inputs[key] for key in self.keys]).sum(0)
        inputs[self.output_key] = energy
        return inputs


class NSAggregation(nn.Module):
    """
    Calculate newton step from forces and inverse hessian.

    Args:
        output_key (str): Name of new property in output.
    """

    def __init__(
        self, force_key: str, inv_hess_key: str, output_key: str = "newton_step_pd"
    ):
        super(NSAggregation, self).__init__()

        self.force_key = force_key
        self.inv_hess_key = inv_hess_key
        self.output_key = output_key
        self.model_outputs = [output_key]

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        newton_step_pd = []
        for idx_m in range(inputs["_n_atoms"].shape[0]):
            f = inputs[self.force_key][inputs["_idx_m"] == idx_m].flatten()
            ns = inputs[self.inv_hess_key][idx_m] @ f
            ns = ns.reshape(-1, 3)
            newton_step_pd.append(ns)
        inputs[self.output_key] = torch.cat(newton_step_pd, dim=0)
        return inputs


class ForceAggregation(nn.Module):
    """
    Calculate newton step from forces and inverse hessian.

    Args:
        output_key (str): Name of new property in output.
    """

    def __init__(
        self,
        ns_key: str,
        hess_key: str,
        output_key: str = "forces",
        energy_unit: str = "kcal/mol",
        position_unit: str = "Ang",
        damping: float = 0.1,
        damping_strategy: str = "simple",
    ):
        super(ForceAggregation, self).__init__()

        self.energy_conversion = convert_units("Hartree", energy_unit)
        self.position_conversion = convert_units("Bohr", position_unit)

        self.ns_key = ns_key
        self.hess_key = hess_key
        self.output_key = output_key
        self.model_outputs = [output_key]

        self.damping = damping
        if damping_strategy == "simple":
            self._damping_strategy = self._simple_damping
        elif damping_strategy == "lm":
            self._damping_strategy = self._levenberg_marquardt_regularization
        else:
            raise ValueError(f"Unknown damping strategy: {damping_strategy}")

    def _levenberg_marquardt_regularization(
        self, hessian: torch.tensor
    ) -> torch.tensor:
        n_atoms = torch.tensor(hessian.shape[0] // 3, device=hessian.device)
        eigvals = torch.linalg.eigvalsh(hessian)
        min_eigval = torch.min(eigvals)
        if min_eigval > 1e-8 * self.energy_conversion / self.position_conversion**2:
            return torch.zeros_like(hessian, device=hessian.device)
        factor = (
            -min_eigval
            + self.damping * self.energy_conversion / self.position_conversion**2
        )
        return torch.eye(n_atoms.item() * 3, device=hessian.device) * factor

    def _simple_damping(self, hessian: torch.tensor) -> torch.tensor:
        n_atoms = torch.tensor(hessian.shape[0] // 3, device=hessian.device)
        factor = self.damping * self.energy_conversion / self.position_conversion**2
        return torch.eye(n_atoms.item() * 3, device=hessian.device) * factor

    def _modified_cholesky(self, hessian, beta=1e-8):
        """
        Perform a modified Cholesky decomposition on matrix H to ensure positive definiteness.
        H: Input Hessian matrix (must be symmetric)
        beta: Small positive constant to ensure positive definiteness
        Returns: L such that H ≈ L @ L.T and L is lower triangular
        """
        n = hessian.shape[0]
        L = torch.zeros_like(hessian, device=hessian.device)
        D = torch.zeros(n, device=hessian.device)

        for j in range(n):
            dj = hessian[j, j] - torch.sum(L[j, :j] ** 2 * D[:j])
            D[j] = max(abs(dj), beta)
            L[j, j] = 1.0

            for i in range(j + 1, n):
                L[i, j] = (hessian[i, j] - torch.sum(L[i, :j] * L[j, :j] * D[:j])) / D[
                    j
                ]

        H_new = L @ torch.diag(D) @ L.T
        return H_new

    def eigenvalue_modification(self, hessian):
        eigvals, eigvectors = torch.linalg.eigh(hessian)

        min_eigval = torch.min(eigvals)

        cutoff = 1e-3
        # modified_eigvals = torch.nn.functional.softplus(eigvals - cutoff) + cutoff
        modified_eigvals = torch.abs(eigvals)
        # modified_eigvals = torch.maximum(eigvals, torch.tensor([1.0], device=hessian.device))

        return eigvectors.T @ torch.diag(modified_eigvals) @ eigvectors

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        forces = []
        _idx_m_hess = inputs["_idx_m"].repeat_interleave(3)
        for idx_m, n_atoms in enumerate(inputs["_n_atoms"]):
            ns = inputs[self.ns_key][inputs["_idx_m"] == idx_m].flatten()
            hess = inputs[self.hess_key][_idx_m_hess == idx_m]

            damping_matrix = self._damping_strategy(hess)
            f = hess @ ns + damping_matrix @ ns

            # hess_pd = self.eigenvalue_modification(hess)
            # f = hess_pd @ ns

            forces.append(f.reshape(-1, 3))
        inputs[self.output_key] = torch.cat(forces, dim=0)
        return inputs


class NSwPrecondAggregation(nn.Module):
    """
    Calculate newton step from forces and inverse hessian with preconditioning.

    Args:
        output_key (str): Name of new property in output.
    """

    def __init__(
        self, force_key: str, precond_key: str, output_key: str = "newton_step_pd"
    ):
        super(NSwPrecondAggregation, self).__init__()

        self.force_key = force_key
        self.precond_key = precond_key
        self.output_key = output_key
        self.model_outputs = [output_key]

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        ns = inputs[self.precond_key] * inputs[self.force_key]
        inputs[self.output_key] = ns
        return inputs
