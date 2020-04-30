import os
import numpy as np
from schnetpack.md.parsers.orca_parser import *
from schnetpack import Properties
from schnetpack.data import AtomsData

from tests.fixtures import *


def test_main_file_parser(orca_log_path, orca_main_targets):
    main_parser = OrcaMainFileParser(properties=OrcaMainFileParser.properties)
    main_parser.parse_file(orca_log_path)

    results = main_parser.get_parsed()
    results[Properties.Z] = results["atoms"][0]
    results[Properties.R] = results["atoms"][1]
    results.pop("atoms", None)

    for p in orca_main_targets:
        assert p in results

        if p == Properties.Z:
            assert np.array_equal(results[p], orca_main_targets[p])
        else:
            assert np.allclose(results[p], orca_main_targets[p])


def test_hessian_file_parser(hessian_path, orca_hessian_targets):
    hessian_parser = OrcaHessianFileParser(properties=OrcaHessianFileParser.properties)
    hessian_parser.parse_file(hessian_path)

    results = hessian_parser.get_parsed()

    for p in orca_hessian_targets:
        assert p in results
        assert np.allclose(results[p], orca_hessian_targets[p])


def test_orca_parser(testdir, orca_log_path, target_orca_db_path):
    db_path = os.path.join(testdir, "test_orca_parser.db")

    all_properties = OrcaMainFileParser.properties + OrcaHessianFileParser.properties

    orca_parser = OrcaParser(db_path, properties=all_properties)
    orca_parser.file_extensions[Properties.hessian] = ".hess"
    orca_parser.parse_data([orca_log_path])

    db_target = AtomsData(target_orca_db_path)
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
