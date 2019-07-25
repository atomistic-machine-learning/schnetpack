import torch
import os
import subprocess
from ase import Atoms
from ase.data import chemical_symbols

from schnetpack.md.calculators.basic_calculators import (
    QMCalculatorError,
    QMCalculator,
    Queuer,
)
from schnetpack.md.utils import MDUnits
from schnetpack.md.parsers.orca_parser import OrcaMainFileParser
from schnetpack import Properties


class OrcaCalculator(QMCalculator):
    """
    Calculator for interfacing the ORCA code package [#orca1]_ with SchNetPack molecular dynamics.
    Requires ORCA to be installed and an input file template.
    This template is a standard ORCA input file, with everything past the specification of coordinate
    format, charge and multiplicity removed (coordinates and final `*`).
    If desired, a Queuer can be give, which will attempt to send all jobs to a grid engine queue.

    In general, the calculator will take the current System to generate inputs, perform the calculation
    with ORCA, extract data from the ouput file (useing the OrcaParser class) and update the System.

    Args:
        required_properties (list): List of properties which should be extracted from output.
        force_handle (str): Indicator for molecular forces.
        compdir (str): Directory in which computations are performed.
        qm_executable (str): Path to the ORCA executable.
        orca_template (str): Path to an ORCA template which will be used to generate input files.
        orca_parser (schnetpack.md.parsers.OrcaParser, optional): Parser used to extract data from output files.
        position_conversion (str/float, optional): Conversion of positions from atomic units.
        force_conversion (str/float, optional): Conversion of forces to atomic units.
        property_conversion (dict, optional): Convert properties to requested units. If left empty, no conversion
                                              is performed.
        queuer (schnetpack.md.calculator.Queuer, optional): If given, jobs will be submitted to a grid engine queue.
        adaptive (bool, optional): Specify, whether the calculator should be used for adaptive sampling.
        basename (str, optional): Basename of the generated input files.

    References
    ----------
    .. [#orca1] Neese:
       The ORCA program system.
       WIREs Comput Mol Sci, 2 (1), 73-78. 2012.
    """

    is_atomistic = [Properties.forces, Properties.shielding]

    def __init__(
        self,
        required_properties,
        force_handle,
        compdir,
        qm_executable,
        orca_template,
        orca_parser=OrcaMainFileParser,
        position_conversion=1.0 / MDUnits.angs2bohr,
        force_conversion=1.0,
        property_conversion={},
        queuer=None,
        adaptive=False,
        basename="input",
    ):
        super(OrcaCalculator, self).__init__(
            required_properties,
            force_handle,
            compdir,
            qm_executable,
            position_conversion=position_conversion,
            force_conversion=force_conversion,
            property_conversion=property_conversion,
            adaptive=adaptive,
        )

        self.orca_template = open(orca_template, "r").read()
        self.basename = basename
        self.orca_parser = orca_parser(properties=required_properties)

        self.queuer = queuer

    def _generate_orca_inputs(self, molecules, current_compdir):
        """
        Generate input files for all molecules in the current System.

        Args:
            molecules (list of ase.Atoms): List of ASE Atoms objects containing the structures and atom types
                                           of all molecules to be computed.
            current_compdir (str): Path to the computation directory in which the input files should be generated.

        Returns:
            list: List of all generated input files.
        """
        input_files = []
        for idx, molecule in enumerate(molecules):
            # Convert data and generate input files
            atom_types, positions = molecule
            input_file_name = os.path.join(
                current_compdir, "{:s}_{:06d}.oinp".format(self.basename, idx + 1)
            )
            self._write_orca_input(input_file_name, atom_types, positions)
            input_files.append(input_file_name)

        return input_files

    def _write_orca_input(self, input_file_name, atom_types, positions):
        """
        Write the ORCA input file using the provided template.

        Args:
            input_file_name (str): Name of the input file.
            atom_types (numpy.array): Integer array holding the atomic numbers of each species.
            positions (numpy.array): Array of all atom positions in Angstrom.
        """
        input_file = open(input_file_name, "w")
        input_file.write(self.orca_template)
        for idx in range(len(atom_types)):
            input_file.write(
                "{:2s} {:15.8f} {:15.8f} {:15.8f}\n".format(
                    chemical_symbols[atom_types[idx]], *positions[idx]
                )
            )
        input_file.write("*")
        input_file.close()

    def _run_computation(self, molecules, current_compdir):
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
        if self.queuer is None:
            for input_file in input_files:
                command = "{:s} {:s}".format(self.qm_executable, input_file, input_file)
                with open("{:s}.log".format(input_file), "wb") as out:
                    computation = subprocess.Popen(command.split(), stdout=out)
                    computation.wait()
        else:
            self.queuer.submit(input_files, current_compdir)

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

    def _format_calc(self, outputs, system):
        """
        Format the extracted properties into the form used by the schnetpack.md.System
        class (zero padding, reshaping, etc.).

        Args:
            outputs (list): Paths to output files.
            system (schnetpack.md.System): System from the molecular dynamics simulation.

        Returns:
            dict: Dictionary of all extracted computation results formatted and padded to the
                  right dimensions used in the system class.
        """
        max_natoms = system.max_n_atoms
        results = {p: [] for p in self.required_properties}

        for output in outputs:
            for p in self.required_properties:
                # Check for convergence
                if output[p] is None:
                    raise QMCalculatorError("Errors encountered during computation.")
                if p in self.is_atomistic:
                    padded_output = torch.zeros(max_natoms, *output[p].shape[1:])
                    padded_output[: output[p].shape[0], ...] = torch.from_numpy(
                        output[p]
                    )
                    results[p].append(padded_output)
                else:
                    results[p].append(torch.from_numpy(output[p]))

        for p in self.required_properties:
            results[p] = torch.stack(results[p]).to(system.device)

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

            properties = outputs[idx]

            atom_buffer.append(atoms)
            property_buffer.append(properties)
        return atom_buffer, property_buffer


class OrcaQueuer(Queuer):
    """
    Class for interfacing the ORCA calculator with a grid engine queue.
    Will create the directory structure and a submission command, which it will pass to the
    given queue. Jobs will be submitted as array jobs, where the number of concurrently running
    jobs can be changed. Should be adapted to work with the individual grid setups.

    Args:
        queue (str): Indentifier for the queue the job should be sent to.
        orca_executable (str): Path to the executable to be called.
        concurrent (str): How many concurrent jobs should be run in each array. (default=100)
        basename (str): Basic identifier used for job (default='input')
        cleanup (bool): Whether directories should be deleted (default=True)
    """

    QUEUE_FILE = """
#!/usr/bin/env bash
##############################
#$ -cwd
#$ -V
#$ -q {queue}
#$ -N {jobname}
#$ -t 1-{array_range}
#$ -tc {concurrent}
#$ -S /bin/bash
#$ -e /dev/null
#$ -o /dev/null
#$ -r n
#$ -sync y
##############################

task_name={basename}_$(printf "%06d" $SGE_TASK_ID)

export QSUB_WORKDIR=/tmp/$USER/$task_name.$$
mkdir -p $QSUB_WORKDIR

cp {compdir}/$task_name.oinp $QSUB_WORKDIR
cd $QSUB_WORKDIR

{orca_path} $task_name.oinp > $task_name.oinp.log

cp -f $QSUB_WORKDIR/$task_name* {compdir}/
rm -rf $QSUB_WORKDIR
"""

    def __init__(
        self, queue, orca_executable, concurrent=100, basename="input", cleanup=True
    ):
        super(OrcaQueuer, self).__init__(
            queue,
            orca_executable,
            concurrent=concurrent,
            basename=basename,
            cleanup=cleanup,
        )

    def _create_submission_command(self, n_inputs, compdir, jobname):
        """
        Use the QUEUE_FILE template to creates the submission command.

        Args:
            n_inputs (int): Number of input files to be submitted
            compdir (str): Current working directory
            jobname (str): Name of each job

        Returns:
            str: Submission command.
        """
        submission_command = self.QUEUE_FILE.format(
            queue=self.queue,
            basename=self.basename,
            array_range=n_inputs,
            concurrent=self.concurrent,
            compdir=compdir,
            orca_path=self.executable,
            jobname=jobname,
        )
        return submission_command
