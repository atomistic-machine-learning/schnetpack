from __future__ import annotations

from typing import Dict, Optional, List

from schnetpack.transform import Transform
import schnetpack.properties as properties
from schnetpack.utils import as_dtype

import torch
import torch.nn as nn

__all__ = ["AtomisticModel", "NeuralNetworkPotential"]


class AtomisticModel(nn.Module):
    """
    Base class for all SchNetPack models.

    SchNetPack models should subclass `AtomisticModel` implement the forward method.
    To use the automatic collection of required derivatives, each submodule that
    requires gradients w.r.t to the input, should list them as strings in
    `submodule.required_derivatives = ["input_key"]`. The model needs to call
    `self.collect_derivatives()` at the end of its `__init__`.

    To make use of post-processing transform, the model should call
    `input = self.postprocess(input)` at the end of its `forward`. The post processors
    will only be applied if `do_postprocessing=True`.

    Example:
         class SimpleModel(AtomisticModel):
            def __init__(
                self,
                representation: nn.Module,
                output_module: nn.Module,
                postprocessors: Optional[List[Transform]] = None,
                input_dtype_str: str = "float32",
                do_postprocessing: bool = True,
            ):
                super().__init__(
                    input_dtype_str=input_dtype_str,
                    postprocessors=postprocessors,
                    do_postprocessing=do_postprocessing,
                )
                self.representation = representation
                self.output_modules = output_modules

                self.collect_derivatives()

            def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
                inputs = self.initialize_derivatives(inputs)

                inputs = self.representation(inputs)
                inputs = self.output_module(inputs)

                # apply postprocessing (if enabled)
                inputs = self.postprocess(inputs)
                return inputs

    """

    def __init__(
        self,
        postprocessors: Optional[List[Transform]] = None,
        input_dtype_str: str = "float32",
        do_postprocessing: bool = True,
    ):
        """
        Args:
            postprocessors: Post-processing transforms tha may be initialized using the
                `datamodule`, but are not applied during training.
            input_dtype_str: The dtype of real inputs as string.
            do_postprocessing: If true, post-processing is activated.
        """
        super().__init__()
        self.input_dtype_str = input_dtype_str
        self.do_postprocessing = do_postprocessing
        self.postprocessors = nn.ModuleList(postprocessors)
        self.required_derivatives: Optional[List[str]] = None
        self.model_outputs: Optional[List[str]] = None

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
        self.required_derivatives = required_derivatives

    def collect_outputs(self) -> List[str]:
        self.model_outputs = None
        model_outputs = set()
        for m in self.modules():
            if hasattr(m, "model_outputs") and m.model_outputs is not None:
                model_outputs.update(m.model_outputs)
        model_outputs: List[str] = list(model_outputs)
        self.model_outputs = model_outputs

    def initialize_derivatives(
        self, inputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        for p in self.required_derivatives:
            if p in inputs.keys():
                inputs[p].requires_grad_()
        return inputs

    def initialize_transforms(self, datamodule):
        for module in self.modules():
            if isinstance(module, Transform):
                module.datamodule(datamodule)

    def postprocess(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if self.do_postprocessing:
            # apply postprocessing
            for pp in self.postprocessors:
                inputs = pp(inputs)
        return inputs

    def extract_outputs(
        self, inputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        results = {k: inputs[k] for k in self.model_outputs}
        return results


class NeuralNetworkPotential(AtomisticModel):
    """
    A generic neural network potential class that sequentially applies a list of input
    modules, a representation module and a list of output modules.

    This can be flexibly configured for various, e.g. property prediction or potential
    energy sufaces with response properties.
    """

    def __init__(
        self,
        representation: nn.Module,
        input_modules: List[nn.Module] = None,
        output_modules: List[nn.Module] = None,
        postprocessors: Optional[List[Transform]] = None,
        input_dtype_str: str = "float32",
        do_postprocessing: Optional[bool] = None,
    ):
        """
        Args:
            representation: The module that builds representation from inputs.
            input_modules: Modules that are applied before representation, e.g. to
                modify input or add additional tensors for response properties.
            output_modules: Modules that predict output properties from the
                representation.
            postprocessors: Post-processing transforms that may be initialized using the
                `datamodule`, but are not applied during training.
            input_dtype_str: The dtype of real inputs.
            do_postprocessing: If true, post-processing is activated.
        """
        super().__init__(
            input_dtype_str=input_dtype_str,
            postprocessors=postprocessors,
            do_postprocessing=do_postprocessing,
        )
        self.representation = representation
        self.input_modules = nn.ModuleList(input_modules)
        self.output_modules = nn.ModuleList(output_modules)

        self.collect_derivatives()
        self.collect_outputs()

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # initialize derivatives for response properties
        inputs = self.initialize_derivatives(inputs)

        for m in self.input_modules:
            inputs = m(inputs)

        inputs = self.representation(inputs)

        for m in self.output_modules:
            inputs = m(inputs)

        # apply postprocessing (if enabled)
        inputs = self.postprocess(inputs)
        results = self.extract_outputs(inputs)

        return results
