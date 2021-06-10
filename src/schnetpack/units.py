import re
from typing import Union, Dict

from ase import units as aseunits
from ase.units import Units
import numpy as np

__all__ = ["convert_units"]

# Internal units (MD internal -> ASE internal)
__md_base_units__ = {
    "energy": "kJ / mol",
    "length": "Angstrom",
    "mass": 1.0,  # 1 Dalton in ASE reference frame
}


def setup_md_units(md_base_units: Dict[str, Union[str, float]]):
    """
    Define the units used in molecular dynamics. This is done based on the base units for energy, length and mass
    from which all other quantities are derived.

    Args:
        md_base_units (dict): Dictionary defining the basic units for molecular dynamics simulations

    Returns:
        dict(str, float):
    """
    # Initialize basic unit system
    md_base_units = {u: _parse_unit(md_base_units[u]) for u in md_base_units}

    # Set up unit dictionary
    units = Units(md_base_units)

    # Derived units (MD internal -> ASE internal)
    units["time"] = units["length"] * np.sqrt(units["mass"] / units["energy"])

    # Constants (internal frame)
    units["kB"] = aseunits.kB / units["energy"]  # Always uses Kelvin
    units["hbar"] = (
        aseunits._hbar * (aseunits.J * aseunits.s) / (units["energy"] * units["time"])
    )  # hbar is given in J/s by ASE

    # For spectra
    units["hbar2icm"] = units["hbar"] * 100.0 * aseunits._c * aseunits._aut

    # TODO: pressure !

    return units


# Placeholders for expected unit entries
energy = 0.0
length = 0.0
mass = 0.0
time = 0.0
kB = 0.0
hbar = 0.0
hbar2icm = 0.0


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


globals().update(setup_md_units(__md_base_units__))
