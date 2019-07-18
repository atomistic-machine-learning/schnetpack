from ase import units

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

    # Unit conversions
    fs2atu = 1e-15 / units._aut
    eV2Ha = units.eV / units.Ha
    d2amu = units._amu / units._me
    angs2bohr = units.Angstrom / units.Bohr
    auforces2aseforces = angs2bohr / eV2Ha
    Ha2kcalpmol = units.Ha * units.mol / units.kcal

    # Constants
    kB = units.kB / units.Ha
    hbar = 1.0

    # For spectra
    h_bar2icm = hbar * 100 * units._c * units._aut

    # Conversion units use when reading in MD inputs
    conversions = {
        "kcal": units.kcal / units.Hartree,
        "mol": 1 / units.mol,
        "ev": units.eV / units.Hartree,
        "bohr": 1.0,
        "angstrom": units.Bohr,
        "a": units.Bohr,
        "angs": units.Bohr,
        "hartree": 1.0,
        "ha": 1.0,
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
