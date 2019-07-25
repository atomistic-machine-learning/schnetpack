import os
import pytest

from schnetpack import Properties

import schnetpack.md.utils.hdf5_data
import schnetpack.md.utils.md_units

from ase import units


@pytest.fixture
def hdf5_dataset():
    hdf5_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "data/test_simulation.hdf5"
    )
    return schnetpack.md.utils.hdf5_data.HDF5Loader(hdf5_path, load_properties=True)


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
    assert Properties.R in hdf5_dataset.properties
    assert Properties.Z in hdf5_dataset.properties
    assert "velocities" in hdf5_dataset.properties

    # Check positions
    positions = hdf5_dataset.get_positions()
    assert positions.shape == (2, 16, 3)

    # Check atom_types
    atom_types = hdf5_dataset.get_property(Properties.Z)
    assert atom_types.shape == (16,)

    # Check velocities
    velocities = hdf5_dataset.get_velocities()
    assert velocities.shape == (2, 16, 3)


@pytest.fixture
def unit_conversion():
    conversions = {
        "kcal / mol": units.kcal / units.Hartree / units.mol,
        "kcal/mol": units.kcal / units.Hartree / units.mol,
        "kcal / mol / Angstrom": units.kcal / units.Hartree / units.mol * units.Bohr,
        "kcal / mol / Angs": units.kcal / units.Hartree / units.mol * units.Bohr,
        "kcal / mol / A": units.kcal / units.Hartree / units.mol * units.Bohr,
        "kcal / mol / Bohr": units.kcal / units.Hartree / units.mol * units.Angstrom,
        "eV": units.eV / units.Ha,
        "Ha": 1.0,
        "Hartree": 1.0,
        0.57667: 0.57667,
    }
    return conversions


def test_unit_conversion(unit_conversion):
    for unit, factor in unit_conversion.items():
        assert (
            abs(schnetpack.md.utils.md_units.MDUnits.parse_mdunit(unit) - factor) < 1e-6
        )
