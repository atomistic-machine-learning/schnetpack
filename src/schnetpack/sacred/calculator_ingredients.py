from sacred import Ingredient

from schnetpack.md.calculators import SchnetPackCalculator
from schnetpack.md.utils import MDUnits


calculator_ingradient = Ingredient('calculator')


@calculator_ingradient.config
def config():
    calculator = 'schnet_calculator'
    required_properties = []
    force_handle = 0
    position_conversion = 1.0 / MDUnits.angs2bohr
    force_conversion = 1.0 / MDUnits.auforces2aseforces
    property_conversion = {}


@calculator_ingradient.capture
def build_calculator(model, required_properties, force_handle,
                     position_conversion, force_conversion,
                     property_conversion, calculator):
    if calculator == 'schnet_calculator':
        return SchnetPackCalculator(model=model,
                                    required_properties=required_properties,
                                    force_handle=force_handle,
                                    position_conversion=position_conversion,
                                    force_conversion=force_conversion,
                                    property_conversion=property_conversion)
    else:
        raise NotImplementedError

