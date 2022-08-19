import re
from typing import Union, Dict

from ase import units as aseunits
from ase.units import Units
import numpy as np

__all__ = ["convert_units"]

# Internal units (MD internal -> ASE internal)
__md_base_units__ = {
    "energy": "kJ / mol",
    "length": "nm",
    "mass": 1.0,  # 1 Dalton in ASE reference frame
    "charge": 1.0,  # Electron charge
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
    units["force"] = units["energy"] / units["length"]
    units["stress"] = units["energy"] / units["length"] ** 3
    units["pressure"] = units["stress"]

    # Conversion of length units
    units["A"] = aseunits.Angstrom / units["length"]
    units["Ang"] = units["A"]
    units["Angs"] = units["A"]
    units["Angstrom"] = units["A"]
    units["nm"] = aseunits.nm / units["length"]
    units["a0"] = aseunits.Bohr / units["length"]
    units["Bohr"] = units["a0"]

    # Conversion of energy units
    units["kcal"] = aseunits.kcal / units["energy"]
    units["kJ"] = aseunits.kJ / units["energy"]
    units["eV"] = aseunits.eV / units["energy"]
    units["Hartree"] = aseunits.Hartree / units["energy"]
    units["Ha"] = units["Hartree"]

    # Time units
    units["fs"] = aseunits.fs / units["time"]
    units["s"] = aseunits.s / units["time"]
    units["aut"] = aseunits._aut * aseunits.s / units["time"]

    # Pressure units
    units["Pascal"] = aseunits.Pascal / units["pressure"]
    units["bar"] = 1e5 * units["Pascal"]

    # Mol
    units["mol"] = aseunits.mol

    # Mass
    units["Dalton"] = 1.0 / units["mass"]
    units["amu"] = aseunits._amu / units["mass"]

    # Charge distributions
    units["Debye"] = aseunits.Debye / (units["charge"] * units["length"])
    units["C"] = aseunits.C / units["charge"]

    # Constants (internal frame)
    units["kB"] = aseunits.kB / units["energy"]  # Always uses Kelvin
    units["hbar"] = (
        aseunits._hbar * (aseunits.J * aseunits.s) / (units["energy"] * units["time"])
    )  # hbar is given in J*s by ASE
    units["ke"] = (
        units["a0"] * units["Ha"] / units["charge"] ** 2
    )  # Coulomb constant is 1 in atomic units

    # For spectra
    units["hbar2icm"] = units["hbar"] * 100.0 * aseunits._c * aseunits._aut

    return units


# Placeholders for expected unit entries
(
    energy,
    length,
    mass,
    charge,
    time,
    force,
    stress,
    pressure,
    kB,
    hbar,
    hbar2icm,
    A,
    Ang,
    Angs,
    Angstrom,
    nm,
    a0,
    Bohr,
    kcal,
    kJ,
    eV,
    Hartree,
    Ha,
    fs,
    s,
    aut,
    mol,
    Dalton,
    amu,
    Debye,
    C,
    ke,
    bar,
    Pascal,
) = [0.0] * 34


def _conversion_factor_ase(unit: str):
    """Get units by string and convert to ase unit system."""
    if unit == "A":
        raise Warning(
            "The unit string 'A' specifies Ampere. For Angstrom, please use 'Ang' or 'Angstrom'."
        )
    return getattr(aseunits, unit)


def _conversion_factor_internal(unit: str):
    """Get units by string and convert to internal unit system."""
    return globals()[unit]


def _parse_unit(unit, conversion_factor=_conversion_factor_ase):
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
                p = conversion_factor(part)
                if divide:
                    conversion /= p
                    divide = False
                else:
                    conversion *= p
        return conversion
    else:
        # If unit is given as number, return number
        return unit


def unit2internal(src_unit: Union[str, float]):
    """
    Convert unit to internal unit system defined above.

    Args:
        src_unit (str, float): Name of unit

    Returns:
        float: conversion factor from external to internal unit system.
    """
    return _parse_unit(src_unit, conversion_factor=_conversion_factor_internal)


def convert_units(src_unit: Union[str, float], tgt_unit: Union[str, float]):
    """Return conversion factor for given units"""
    return _parse_unit(src_unit) / _parse_unit(tgt_unit)


globals().update(setup_md_units(__md_base_units__))
