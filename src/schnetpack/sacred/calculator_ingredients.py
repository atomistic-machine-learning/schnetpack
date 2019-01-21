from sacred import Ingredient

from schnetpack.md.calculators import SchnetPackCalculator
from schnetpack.md.utils import MDUnits


calculator_ingradient = Ingredient('calculator')


@calculator_ingradient.config
def config():
    """configuration for the calculator ingredient"""
    calculator = 'schnet_calculator'
    required_properties = ['energy', 'forces']
    force_handle = 'forces'
    position_conversion = 1.0 / MDUnits.angs2bohr
    force_conversion = 1.0 / MDUnits.auforces2aseforces
    property_conversion = {}


@calculator_ingradient.capture
def build_calculator(model, required_properties, force_handle,
                     position_conversion, force_conversion,
                     property_conversion, calculator):
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
    if calculator == 'schnet_calculator':
        return SchnetPackCalculator(model=model,
                                    required_properties=required_properties,
                                    force_handle=force_handle,
                                    position_conversion=position_conversion,
                                    force_conversion=force_conversion,
                                    property_conversion=property_conversion)
    else:
        raise NotImplementedError

