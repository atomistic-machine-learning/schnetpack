import schnetpack
import torch
import torch.nn as nn

from .md_config import *
from .normal_model_transformation import *
from .thermostat_utils import *

from typing import Optional


class CalculatorError(Exception):
    pass


def activate_model_stress(
    model: schnetpack.model.AtomisticModel, stress_key: str
) -> schnetpack.model.AtomisticModel:
    """
    Utility function for activating computation of stress in models not explicitly trained on the stress tensor.
    Used for e.g. simulations under constant pressure and in cells.

    Args:
        model (AtomisticTask): loaded schnetpack model for which stress computation should be activated.
        stress_key (str): name of stress tensor in model.

    Returns:
        model (AtomisticTask): schnetpack model with activated stress tensor.
    """
    stress = False

    # Check if a module suitable for stress computation is present
    for module in model.output_modules:
        if isinstance(module, schnetpack.atomistic.response.Forces) or isinstance(
            module, schnetpack.atomistic.Response
        ):
            # for `Forces` module
            if hasattr(module, "calc_stress"):
                # activate internal stress computation flag
                module.calc_stress = True

                # append stress label to output list and update required derivatives in the module
                module.model_outputs.append(stress_key)
                module.required_derivatives.append(schnetpack.properties.strain)

                # if not set in the model, also update output list and required derivatives so that:
                #   a) required derivatives are computed and
                #   b) property is added to the model outputs
                if stress_key not in model.model_outputs:
                    model.model_outputs.append(stress_key)
                    model.required_derivatives.append(schnetpack.properties.strain)

                stress = True

            # for `Response` module
            if hasattr(module, "basic_derivatives"):
                # activate internal stress computation flag
                module.calc_stress = True
                module.basic_derivatives["dEds"] = schnetpack.properties.strain
                module.derivative_instructions["dEds"] = True
                module.basic_derivatives["dEds"] = schnetpack.properties.strain

                module.map_properties[schnetpack.properties.stress] = (
                    schnetpack.properties.stress
                )

                # append stress label to output list and update required derivatives in the module
                module.model_outputs.append(stress_key)
                module.required_derivatives.append(schnetpack.properties.strain)

                # if not set in the model, also update output list and required derivatives so that:
                #   a) required derivatives are computed and
                #   b) property is added to the model outputs
                if stress_key not in model.model_outputs:
                    model.model_outputs.append(stress_key)
                    model.required_derivatives.append(schnetpack.properties.strain)

                stress = True

    # If stress computation has been enables, insert preprocessing for strain computation
    if stress:
        model.input_modules.insert(0, schnetpack.atomistic.Strain())

    if not stress:
        raise CalculatorError("Failed to activate stress computation")

    return model


class UninitializedMixin(nn.modules.lazy.LazyModuleMixin):
    """
    Custom mixin for lazy initialization of buffers used in the MD system and simulation hooks.
    This can be used to add buffers with a certain dtype in an uninitialized state.
    """

    def register_uninitialized_buffer(
        self, name: str, dtype: Optional[torch.dtype] = None
    ):
        """
        Register an uninitialized buffer with the requested dtype. This can be used to reserve variable which are not
        known at the initialization of `schnetpack.md.System` and simulation hooks.

        Args:
            name (str): Name of the uninitialized buffer to register.
            dtype (torch.dtype): If specified, buffer will be set to requested dtype. If None is given, this will
                                 default to float64 type.
        """
        if dtype is None:
            dtype = torch.float64

        self.register_buffer(name, nn.parameter.UninitializedBuffer(dtype=dtype))
