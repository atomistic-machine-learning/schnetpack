import schnetpack as spk
import pytest
from ase import units

from tests.fixtures import *


def test_properties(hdf5_dataset):
    # Test general properties
    assert "energy" in hdf5_dataset.properties
    assert "forces" in hdf5_dataset.properties
    assert "dipole_moment" in hdf5_dataset.properties

    # energy
    # 1) check for general shape
    energy = hdf5_dataset.properties["energy"]
    assert energy.shape == (2, 1, 1, 1)
    # 2) Check for extraction
    energy = hdf5_dataset.get_property("energy")
    assert energy.shape == (2, 1)
    # 2) mol index
    energy = hdf5_dataset.get_property("energy", mol_idx=0)
    assert energy.shape == (2, 1)
    # 2) centroid index
    energy = hdf5_dataset.get_property("energy", replica_idx=0)
    assert energy.shape == (2, 1)

    # Other properties
    dipole_moment = hdf5_dataset.get_property("dipole_moment")
    assert dipole_moment.shape == (2, 3)

    forces = hdf5_dataset.get_property("forces")
    assert forces.shape == (2, 16, 3)


def test_molecule(hdf5_dataset):
    # Test molecule properties
    assert spk.Properties.R in hdf5_dataset.properties
    assert spk.Properties.Z in hdf5_dataset.properties
    assert "velocities" in hdf5_dataset.properties

    # Check positions
    positions = hdf5_dataset.get_positions()
    assert positions.shape == (2, 16, 3)

    # Check atom_types
    atom_types = hdf5_dataset.get_property(spk.Properties.Z)
    assert atom_types.shape == (16,)

    # Check velocities
    velocities = hdf5_dataset.get_velocities()
    assert velocities.shape == (2, 16, 3)


# TODO: Think about good scheme.


@pytest.fixture
def unit_conversion():
    conversions = {
        "kcal / mol": units.kcal / units.mol / spk.md.utils.MDUnits.energy_unit,
        "kcal/mol": units.kcal / units.mol / spk.md.utils.MDUnits.energy_unit,
        "kJ /mol": 1.0,
        "A": 1.0 / spk.md.utils.MDUnits.length_unit,
        "kcal / mol / Bohr": units.kcal
        / units.mol
        / units.Bohr
        / (spk.md.utils.MDUnits.energy_unit / spk.md.utils.MDUnits.length_unit),
        "kcal / mol / A": units.kcal
        / units.mol
        / units.Angstrom
        / (spk.md.utils.MDUnits.energy_unit / spk.md.utils.MDUnits.length_unit),
        "kcal / mol / Angs": units.kcal
        / units.mol
        / units.Angstrom
        / (spk.md.utils.MDUnits.energy_unit / spk.md.utils.MDUnits.length_unit),
        "kcal / mol / Angstrom": units.kcal
        / units.mol
        / units.Angstrom
        / (spk.md.utils.MDUnits.energy_unit / spk.md.utils.MDUnits.length_unit),
        0.57667: 0.57667,
    }
    return conversions


@pytest.fixture
def unit_conversion_dual():
    conversions = {
        ("kcal / mol", "kJ / mol"): units.kcal / units.kJ,
        ("kcal/mol", "kcal/mol"): 1.0,
        (2.0, 2.0): 1.0,
    }
    return conversions


def test_unit_conversion(unit_conversion, unit_conversion_dual):
    for unit, factor in unit_conversion.items():
        assert abs(spk.md.utils.md_units.MDUnits.unit2internal(unit) - factor) < 1e-6
        assert (
            abs(spk.md.utils.md_units.MDUnits.internal2unit(unit) - 1.0 / factor) < 1e-6
        )

    for unit_duple, factor in unit_conversion_dual.items():
        unit1, unit2 = unit_duple
        assert (
            abs(spk.md.utils.md_units.MDUnits.unit2unit(unit1, unit2) - factor) < 1e-6
        )
