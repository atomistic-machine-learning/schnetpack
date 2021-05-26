import re
from typing import Union

from ase import units as aseunits

__all__ = ["convert_units"]


def _conversion_factor(unit: str):
    """Get units by string"""
    return getattr(aseunits, unit)


def _parse_unit(unit):
    if type(unit) == str:
        # If a string is given, split into parts.
        parts = re.split("(\W)", unit)

        conversion = 1.0
        divide = False
        for part in parts:
            if part == "/":
                divide = True
            elif part == "" or part == " ":
                pass
            else:
                p = _conversion_factor(part)
                if divide:
                    conversion /= p
                    divide = False
                else:
                    conversion *= p
        return conversion
    else:
        # If unit is given as number, return number
        return unit


def convert_units(src_unit: Union[str, float], tgt_unit: Union[str, float]):
    """Return conversion factor for given units"""
    return _parse_unit(src_unit) / _parse_unit(tgt_unit)
