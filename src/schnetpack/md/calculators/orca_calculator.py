from __future__ import annotations
import torch
import os
import subprocess
from ase import Atoms
from ase.data import chemical_symbols
import numpy as np

from typing import List, Union, Dict, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from schnetpack.md import System

from schnetpack.md.calculators.base_calculator import QMCalculatorError, QMCalculator
from schnetpack.md.parsers.orca_parser import OrcaMainFileParser
from schnetpack import units as spk_units

__all__ = ["OrcaCalculator"]


class OrcaCalculator(QMCalculator):
    """
    Calculator for interfacing the ORCA code package [#orca1]_ with SchNetPack molecular dynamics.
    Requires ORCA to be installed and an input file template.
    This template is a standard ORCA input file, with everything past the specification of coordinate
    format, charge and multiplicity removed (coordinates and final `*`).
    If desired, a Queuer can be give, which will attempt to send all jobs to a grid engine queue.

    In general, the calculator will take the current System to generate inputs, perform the calculation
    with ORCA, extract data from the ouput file (using the OrcaParser class) and update the System.

    Args:
        required_properties (list): List of the property names which will be passed to the simulator
        force_key (str): Name of the property corresponding to the forces.
        compdir (str): Directory in which computations are performed.
        qm_executable (str): Path to the ORCA executable.
        orca_template (str): Path to an ORCA template which will be used to generate input files. This should be a full
                             ORCA input, where the geometry section between *xyz and * is replaced by the string
                             `{geometry}`.
        energy_unit (str, float): Energy units returned by the internal computation model.
        position_unit (str, float): Unit conversion for the length used in the model computing all properties. E.g. if
                             the model needs Angstrom, one has to provide the conversion factor converting from the
                             atomic units used internally (Bohr) to Angstrom: 0.529177.
                             Is used together with `energy_unit` to determine units of force and stress.
        energy_key (str, optional): Name of the property corresponding to the energy.
        stress_key (str, optional): Name of the property corresponding to the stress.
        property_conversion (dict, optional): Convert properties to requested units. If left empty, no conversion
                                              is performed.
        overwrite (bool): Overwrite previous computation results. Default is true.
        adaptive (bool, optional): Specify, whether the calculator should be used for adaptive sampling.
        basename (str, optional): Basename of the generated input files.
        orca_parser (schnetpack.md.parsers.OrcaParser, optional): Parser used to extract data from output files.

    References
    ----------
    .. [#orca1] Neese:
       The ORCA program system.
       WIREs Comput Mol Sci, 2 (1), 73-78. 2012.
    """

    def __init__(
        self,
        required_properties: List,
        force_key: str,
        compdir: str,
        qm_executable: str,
        orca_template: str,
        energy_unit: Union[str, float],
        position_unit: Union[str, float],
        energy_key: Optional[str] = None,
        stress_key: Optional[str] = None,
        property_conversion: Dict[str, Union[str, float]] = {},
        overwrite: bool = True,
        adaptive: bool = False,
        basename: str = "input",
        orca_parser=OrcaMainFileParser,
    ):
        super(OrcaCalculator, self).__init__(
            required_properties=required_properties,
            force_key=force_key,
            compdir=compdir,
            qm_executable=qm_executable,
            energy_unit=energy_unit,
            position_unit=position_unit,
            energy_key=energy_key,
            stress_key=stress_key,
            property_conversion=property_conversion,
            overwrite=overwrite,
            adaptive=adaptive,
        )

        self.orca_template = open(orca_template, "r").read()
        self.basename = basename
        self.orca_parser = orca_parser(target_properties=required_properties)

    def _generate_orca_inputs(
        self, molecules: List[Tuple[np.array, np.array]], current_compdir: str
    ):
        """
        Generate input files for all molecules in the current System.

        Args:
            molecules (list((np.array, np.array)): List of tuples of numpy arrays containing the atom types and structures
                                           of all molecules to be computed.
            current_compdir (str): Path to the computation directory in which the input files should be generated.

        Returns:
            list: List of all generated input files.
        """
        input_files = []
        for idx, molecule in enumerate(molecules):
            # Convert data and generate input files
            atom_types, positions = molecule
            # Convert inputs to Angstrom for input file, since this is the default length unit there
            positions *= spk_units.convert_units(
                self.position_conversion * spk_units.length, "Angstrom"
            )
            input_file_name = os.path.join(
                current_compdir, "{:s}_{:06d}.oinp".format(self.basename, idx + 1)
            )
            self._write_orca_input(input_file_name, atom_types, positions)
            input_files.append(input_file_name)

        return input_files

    def _write_orca_input(
        self, input_file_name: str, atom_types: np.array, positions: np.array
    ):
        """
        Write the ORCA input file using the provided template.

        Args:
            input_file_name (str): Name of the input file.
            atom_types (numpy.array): Integer array holding the atomic numbers of each species.
            positions (numpy.array): Array of all atom positions in Angstrom.
        """
        input_file = open(input_file_name, "w")
        geometry_block = []
        for idx in range(len(atom_types)):
            geometry_block.append(
                "{:2s} {:15.8f} {:15.8f} {:15.8f}".format(
                    chemical_symbols[atom_types[idx]], *positions[idx]
                )
            )
        input_file.write(self.orca_template.format(geometry="\n".join(geometry_block)))
        input_file.close()

    def _run_computation(
        self, molecules: List[Tuple[np.array, np.array]], current_compdir: str
    ):
        """
        Perform the actual computation.
        First, inputs are generated, then computations are performed, finally the data is extracted
        from the output files and returned.

        Args:
            molecules (list): List of atom_types, position tuples
            current_compdir (str): Path to computation directory.

        Returns:
            dict: Dictionary holding the requested properties which were computed.
        """
        # Generate input files
        input_files = self._generate_orca_inputs(molecules, current_compdir)

        # Perform computations
        for input_file in input_files:
            command = "{:s} {:s}".format(self.qm_executable, input_file, input_file)
            with open("{:s}.log".format(input_file), "wb") as out:
                computation = subprocess.Popen(command.split(), stdout=out)
                computation.wait()

        # Extract the results
        outputs = []
        for input_file in input_files:
            self.orca_parser.parse_file("{:s}.log".format(input_file))

            orca_outputs = self.orca_parser.get_parsed()

            for p in self.required_properties:
                if orca_outputs[p] is None:
                    raise QMCalculatorError(
                        "Requested property {:s} was not computed in {:s}".format(
                            p, input_file
                        )
                    )

            outputs.append(orca_outputs)

        return outputs

    def _format_calc(self, outputs: Dict[str, np.array], system: System):
        """
        Format the extracted properties into the form used by the schnetpack.md.System
        class (zero padding, reshaping, etc.).

        Args:
            outputs (dict(str, np.array): Dictionary of ouput arrays.
            system (schnetpack.md.System): System from the molecular dynamics simulation.

        Returns:
            dict: Dictionary of all extracted computation results formatted and padded to the
                  right dimensions used in the system class.
        """
        results = {p: [] for p in self.required_properties}

        for output in outputs:
            for p in self.required_properties:
                # Check for convergence
                if output[p] is None:
                    raise QMCalculatorError("Errors encountered during computation.")

                results[p].append(torch.from_numpy(output[p]))

        for p in self.required_properties:
            results[p] = torch.stack(results[p]).to(system.device, system.dtype)

        return results

    def _format_ase(self, molecules, outputs):
        """
        Format the ouput for storage in an ASE database (for adaptive sampling).

        Args:
            molecules (list): List of tuples containing the atom types (integer numpy.array)
                      and positions (float numpy.array).
            outputs (list): Paths to output files.
        """
        atom_buffer = []
        property_buffer = []
        for idx, molecule in enumerate(molecules):
            atom_types, positions = molecule
            atoms = Atoms(atom_types.cpu(), positions.cpu())

            props = outputs[idx]

            atom_buffer.append(atoms)
            property_buffer.append(props)
        return atom_buffer, property_buffer
