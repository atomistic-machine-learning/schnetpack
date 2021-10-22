from __future__ import annotations

from typing import Dict, Optional, List

from schnetpack.transform import Transform

import torch
import torch.nn as nn

__all__ = ["AtomisticModel", "NeuralNetworkPotential"]


class AtomisticModel(nn.Module):
    """
    Base class for all SchNetPack models.

    Models should override the `forward` method and call `BaseAtomisticModel.forward`` to enable
    postprocessing and assembling the results dictionary.
    """

    def __init__(
        self,
        postprocessors: Optional[List[Transform]] = None,
        input_dtype: torch.dtype = torch.float32,
        enable_postprocess: bool = False,
    ):
        super().__init__()
        self.input_dtype = input_dtype
        self._do_postprocess = enable_postprocess
        self.postprocessors = nn.ModuleList(postprocessors)
        self.required_derivatives: Optional[List[str]] = None

    def collect_derivatives(self) -> List[str]:
        self.required_derivatives = None
        required_derivatives = set()
        for m in self.modules():
            if (
                hasattr(m, "required_derivatives")
                and m.required_derivatives is not None
            ):
                required_derivatives.update(m.required_derivatives)
        required_derivatives: List[str] = list(required_derivatives)
        return required_derivatives

    def initialize_derivatives(
        self, inputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        for p in self.required_derivatives:
            if p in inputs.keys():
                inputs[p].requires_grad_()
        return inputs

    def enable_postproces(self):
        self._do_postprocess = True

    def disable_postproces(self):
        self._do_postprocess = False

    def initialize_postprocessors(self, datamodule):
        for pp in self.postprocessors:
            pp.datamodule(datamodule)

    def postprocess(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if self._do_postprocess:
            # apply postprocessing
            for pp in self.postprocessors:
                inputs = pp(inputs)
        return inputs


class NeuralNetworkPotential(AtomisticModel):
    """
    A generic neural network potential class that sequentially applies a list of input modules, a representation module and
    a list of output modules.

    This can be flexibly configured for various, e.g. property prediction or potential energy sufaces with response
    properties.
    """

    def __init__(
        self,
        representation: nn.Module,
        input_modules: List[nn.Module] = None,
        output_modules: List[nn.Module] = None,
        postprocessors: Optional[List[Transform]] = None,
        input_dtype: torch.dtype = torch.float32,
        enable_postprocess: bool = False,
    ):
        super().__init__(
            input_dtype=input_dtype,
            postprocessors=postprocessors,
            enable_postprocess=enable_postprocess,
        )
        self.representation = representation
        self.input_modules = nn.Sequential(*input_modules)
        self.output_modules = nn.Sequential(*output_modules)

        self.required_derivatives = self.collect_derivatives()

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # inititalize derivatives for response properties
        inputs = self.initialize_derivatives(inputs)

        inputs = self.input_modules(inputs)
        inputs = self.representation(inputs)
        inputs = self.output_modules(inputs)

        # apply (optional) postprocessing
        inputs = self.postprocess(inputs)
        return inputs
