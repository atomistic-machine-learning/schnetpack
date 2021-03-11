from ase import units
import numpy as np
import re
import warnings

__all__ = ["MDUnits"]


class MDUnits:
    """
    Basic conversion factors to atomic units used internally:
        fs2atu (time): femtoseconds to atomic time units
        eV2Ha (energy): electron Volt to Hartree
        d2amu (mass): Dalton to atomic mass units
        angs2bohr (length): Angstrom to Bohr
        auforces2aseforces (forces): Hartee per Bohr to electron Volt per Angstrom

    Definitions for constants:
        kB: Boltzmann constant in units of Hartree per Kelvin.
        hbar: Reduced Planck constant in atomic units.
    """

    # Internal units (MD internal -> ASE internal)
    energy_unit = units.kJ / units.mol
    length_unit = units.nm
    mass_unit = 1.0  # 1 Dalton in ASE reference frame

    # Derived units (MD internal -> ASE internal)
    time_unit = length_unit * np.sqrt(mass_unit / energy_unit)

    # General utility units for conversion
    fs2internal = units.fs / time_unit
    da2internal = 1.0 / mass_unit
    angs2internal = units.Angstrom / length_unit
    bar2internal = 1e5 * units.Pascal / (energy_unit / length_unit ** 3)

    # Constants (internal frame)
    kB = units.kB / energy_unit  # Always uses Kelvin
    hbar = (
        units._hbar * (units.J * units.s) / (energy_unit * time_unit)
    )  # hbar is given in J/s by ASE

    # For spectra
    h_bar2icm = hbar * 100 * units._c * units._aut

    # Conversion units use when reading in MD inputs
    # These always convert to ASE frame of reference
    conversions = {
        "mol": units.mol,
        "kcal": units.kcal / energy_unit,
        "kj": units.kJ / energy_unit,
        "ev": units.eV / energy_unit,
        "hartree": units.Ha / energy_unit,
        "ha": units.Ha / energy_unit,
        "bohr": units.Bohr / length_unit,
        "angstrom": units.Angstrom / length_unit,
        "angs": units.Angstrom / length_unit,
        "a": units.Angstrom / length_unit,
        "nm": units.nm / length_unit,
        "fs": units.fs / time_unit,
        "s": units.s / time_unit,
        "aut": units._aut * units.s / time_unit,  # _aut is given in s
    }

    @staticmethod
    def parse_mdunit(unit):
        """
        Auxiliary functions, used to determine the conversion factor of position and force units between MD propagation
        and the provided ML Calculator. Allowed units are:
            mol, kcal, eV, Bohr, Angstrom, Hartree and all combinations using '/' thereof (e.g. kcal/mol/Angstrom).

        Args:
            unit (str/float): Unit to be used to convert forces from Calculator units to atomic units used in the MD.
                              Can be a str, which is converted to the corresponding numerical value or a float, which
                              is returned.

        Returns:
            float: Factor used for conversion in the Calculator.

        """
        warnings.warn(
            "Routine is deprecated, please use unit2internal, internal2unit or unit2unit instead.",
            DeprecationWarning,
        )
        if type(unit) == str:
            parts = unit.lower().replace(" ", "").split("/")
            scaling = 1.0
            for part in parts:
                if part not in MDUnits.conversions:
                    raise KeyError("Unrecognized unit {:s}".format(part))
                scaling *= MDUnits.conversions[part]
            return scaling
        else:
            return unit

    @staticmethod
    def _parse_unit(unit):
        if type(unit) == str:
            # If a string is given, split into parts.
            parts = re.split("(\W)", unit.lower())

            conversion = 1.0
            divide = False
            for part in parts:
                if part == "/":
                    divide = True
                elif part == "" or part == " ":
                    pass
                else:
                    if divide:
                        conversion /= MDUnits.conversions[part]
                        divide = False
                    else:
                        conversion *= MDUnits.conversions[part]
            return conversion
        else:
            # If unit is given as number, return number
            return unit

    @staticmethod
    def unit2internal(unit):
        conversion = MDUnits._parse_unit(unit)
        return conversion

    @staticmethod
    def internal2unit(unit):
        conversion = MDUnits._parse_unit(unit)
        return 1.0 / conversion

    @staticmethod
    def unit2unit(unit1, unit2):
        conversion1 = MDUnits.unit2internal(unit1)
        conversion2 = MDUnits.internal2unit(unit2)
        return conversion1 * conversion2
