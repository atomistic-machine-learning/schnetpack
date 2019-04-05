import torch
from torch import nn as nn
from schnetpack.data import Structure


class ModelError(Exception):
    pass


class AtomisticModel(nn.Module):
    """
    Join a representation model with output modules.

    Args:
        representation (torch.nn.Module): Representation block of the model.
        output_model (schnetpack.atomwise.OutputBlock): Output block of the model.
            Needed for predicting properties.

    Returns:
         dict: property predictions
    """
    def __init__(
            self,
            representation,
            output_model
    ):
        super(AtomisticModel, self).__init__()
        self.representation = representation
        self.output_layer = output_model
        self.requires_dr = output_model.requires_dr

    def forward(self, inputs):
        if self.requires_dr:
            inputs[Structure.R].requires_grad_()
        inputs["representation"] = self.representation(inputs)
        return self.output_layer(inputs)


class OutputBlock(nn.Module):
    """
    Forward representation trough multiple output models.

    Args:
        output_modules (list): list of output modules
    """

    def __init__(
            self,
            output_modules
    ):
        super(OutputBlock, self).__init__()
        if type(output_modules) == list:
            output_modules = nn.ModuleList(output_modules)
        self.output_modules = output_modules
        self.requires_dr = any([om.dr_property for om in output_modules])

    def forward(self, inputs):
        """
        Forward inputs through output modules.

        Returns:
            dict: properties and predictions
        """
        outs = {}
        for output_model in self.output_modules:
            outs.update(output_model(inputs))
        return outs
