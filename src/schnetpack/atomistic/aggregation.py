import torch
import torch.nn as nn

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

    def __init__(self, ns_key: str, hess_key: str, output_key: str = "forces"):
        super(ForceAggregation, self).__init__()

        self.ns_key = ns_key
        self.hess_key = hess_key
        self.output_key = output_key
        self.model_outputs = [output_key]

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        forces = []
        _idx_m_hess = inputs["_idx_m"].repeat_interleave(3)
        for idx_m in range(inputs["_n_atoms"].shape[0]):
            ns = inputs[self.ns_key][inputs["_idx_m"] == idx_m].flatten()
            hess = inputs[self.hess_key][_idx_m_hess == idx_m]
            f = hess @ ns
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
