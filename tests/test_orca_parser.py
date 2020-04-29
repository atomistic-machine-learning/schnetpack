import os
import numpy as np
import pytest
from schnetpack.md.parsers.orca_parser import *
from schnetpack import Properties
from schnetpack.data import AtomsData

from tests.fixtures import *


@pytest.fixture
def main_path(shared_datadir):
    return os.path.join(shared_datadir, "test_orca.log")


@pytest.fixture
def hessian_path(shared_datadir):
    return os.path.join(shared_datadir, "test_orca.hess")


@pytest.fixture
def targets_main(shared_datadir):
    return np.load(os.path.join(shared_datadir, "test_orca_main_targets.npz"))


@pytest.fixture
def targets_hessian(shared_datadir):
    return np.load(os.path.join(shared_datadir, "test_orca_hessian_targets.npz"))


@pytest.fixture
def target_orca_db(shared_datadir):
    return os.path.join(shared_datadir, "test_orca_parser.db")


def test_main_file_parser(main_path, targets_main):
    main_parser = OrcaMainFileParser(properties=OrcaMainFileParser.properties)
    main_parser.parse_file(main_path)

    results = main_parser.get_parsed()
    results[Properties.Z] = results["atoms"][0]
    results[Properties.R] = results["atoms"][1]
    results.pop("atoms", None)

    for p in targets_main:
        assert p in results

        if p == Properties.Z:
            assert np.array_equal(results[p], targets_main[p])
        else:
            assert np.allclose(results[p], targets_main[p])


def test_hessian_file_parser(hessian_path, targets_hessian):
    hessian_parser = OrcaHessianFileParser(properties=OrcaHessianFileParser.properties)
    hessian_parser.parse_file(hessian_path)

    results = hessian_parser.get_parsed()

    for p in targets_hessian:
        assert p in results
        assert np.allclose(results[p], targets_hessian[p])


def test_orca_parser(tmpdir, main_path, target_orca_db):
    db_path = os.path.join(tmpdir, "test_orca_parser.db")

    all_properties = OrcaMainFileParser.properties + OrcaHessianFileParser.properties

    orca_parser = OrcaParser(db_path, properties=all_properties)
    orca_parser.file_extensions[Properties.hessian] = ".hess"
    orca_parser.parse_data([main_path])

    db_target = AtomsData(target_orca_db)
    db_test = AtomsData(db_path)

    target_atoms, target_properties = db_target.get_properties(0)
    test_atoms, test_properties = db_test.get_properties(0)

    assert np.allclose(
        target_atoms.get_atomic_numbers(), test_atoms.get_atomic_numbers()
    )
    assert np.allclose(target_atoms.positions, test_atoms.positions)

    for p in target_properties:
        assert p in test_properties
        assert np.allclose(test_properties[p], target_properties[p])
