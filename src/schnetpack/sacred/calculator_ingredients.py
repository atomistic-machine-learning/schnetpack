from sacred import Ingredient
import os
import torch

from schnetpack.md.calculators import SchnetPackCalculator
from schnetpack.md.utils import MDUnits

calculator_ingradient = Ingredient('calculator')


@calculator_ingradient.config
def config():
    """configuration for the calculator ingredient"""
    calculator = 'schnet_calculator'
    required_properties = ['y', 'dydx']
    force_handle = 'dydx'
    position_conversion = 1.0 / MDUnits.angs2bohr
    force_conversion = 1.0 / MDUnits.auforces2aseforces
    property_conversion = {}

    model_path = 'eth_ens_01.model'
    # If model is a directory, search for best_model file
    if os.path.isdir(model_path):
        model_path = os.path.join(model_path, 'best_model')


@calculator_ingradient.capture
def load_model(_log, model_path, device):
    _log.info('Loaded model from {:s}'.format(model_path))
    model = torch.load(model_path).to(device)
    return model


@calculator_ingradient.capture
def build_calculator(_log, required_properties, force_handle,
                     position_conversion, force_conversion,
                     property_conversion, calculator, device):
    """
    Build the calculator object from the provided settings.

    Args:
        model (torch.nn.module): the model which is used for property calculation
        required_properties (list): list of properties that are calculated by the model
        force_handle (str): name of the forces property in the model output
        position_conversion (float): conversion factor for positions
        force_conversion (float): conversion factor for forces
        property_conversion (dict): dictionary with conversion factors for other properties
        calculator (src.schnetpack.md.calculator.Calculator): calculator object

    Returns:
        the calculator object
    """
    _log.info(f'Using {calculator}')

    if calculator == 'schnet_calculator':

        model = load_model(device=device)
        return SchnetPackCalculator(model,
                                    required_properties=required_properties,
                                    force_handle=force_handle,
                                    position_conversion=position_conversion,
                                    force_conversion=force_conversion,
                                    property_conversion=property_conversion)
    else:
        raise NotImplementedError
