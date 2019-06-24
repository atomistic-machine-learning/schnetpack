from schnetpack.md.calculators.calculators import QMCalculatorError, QMCalculator, Queuer
from schnetpack.md.utils import MDUnits
from schnetpack.md.parsers.orca_parser import OrcaMainFileParser

from schnetpack.data.definitions import Properties

import torch

import os
import subprocess

from ase.data import chemical_symbols
from ase import Atoms


class OrcaCalculator(QMCalculator):
    is_atomistic = [Properties.forces, Properties.shielding]

    def __init__(self, required_properties,
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
                 basename='input'):
        super(OrcaCalculator, self).__init__(required_properties,
                                             force_handle,
                                             compdir,
                                             qm_executable,
                                             position_conversion=position_conversion,
                                             force_conversion=force_conversion,
                                             property_conversion=property_conversion,
                                             adaptive=adaptive)

        self.orca_template = open(orca_template, 'r').read()
        self.basename = basename
        self.orca_parser = orca_parser(properties=required_properties)

        self.queuer = queuer

    def _generate_orca_inputs(self, molecules, current_compdir):
        input_files = []
        for idx, molecule in enumerate(molecules):
            # Convert data and generate input files
            atom_types, positions = molecule
            input_file_name = os.path.join(current_compdir, '{:s}_{:06d}.oinp'.format(self.basename, idx + 1))
            self._write_orca_input(input_file_name, atom_types, positions)
            input_files.append(input_file_name)

        return input_files

    def _write_orca_input(self, input_file_name, atom_types, positions):
        """Write orca input file"""
        input_file = open(input_file_name, 'w')
        input_file.write(self.orca_template)
        for idx in range(len(atom_types)):
            input_file.write('{:2s} {:15.8f} {:15.8f} {:15.8f}\n'.format(
                chemical_symbols[atom_types[idx]],
                *positions[idx]
            ))
        input_file.write('*')
        input_file.close()

    def _run_computation(self, molecules, current_compdir):
        # Generate input files
        input_files = self._generate_orca_inputs(molecules, current_compdir)

        # Perform computations
        if self.queuer is None:
            for input_file in input_files:
                command = '{:s} {:s}'.format(self.qm_executable, input_file, input_file)
                with open('{:s}.log'.format(input_file), 'wb') as out:
                    computation = subprocess.Popen(command.split(), stdout=out)
                    computation.wait()
        else:
            self.queuer.submit(input_files, current_compdir)

        # Extract the results
        outputs = []
        for input_file in input_files:
            self.orca_parser.parse_file('{:s}.log'.format(input_file))

            orca_outputs = self.orca_parser.get_parsed()

            for p in self.required_properties:
                if orca_outputs[p] is None:
                    raise QMCalculatorError('Requested property {:s} was not computed in {:s}'.format(p, input_file))

            outputs.append(orca_outputs)

        return outputs

    def _format_calc(self, outputs, system):
        max_natoms = system.max_n_atoms
        results = {p: [] for p in self.required_properties}

        for output in outputs:
            for p in self.required_properties:
                # Check for convergence
                if output[p] is None:
                    raise QMCalculatorError('Errors encountered during computation.')
                if p in self.is_atomistic:
                    padded_output = torch.zeros(max_natoms, *output[p].shape[1:])
                    padded_output[:output[p].shape[0], ...] = torch.from_numpy(output[p])
                    results[p].append(padded_output)
                else:
                    results[p].append(torch.from_numpy(output[p]))

        for p in self.required_properties:
            results[p] = torch.stack(results[p]).to(system.device)

        return results

    def _format_ase(self, molecules, outputs):
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

    def __init__(self, queue, orca_executable, concurrent=100, basename='input', cleanup=True):
        super(OrcaQueuer, self).__init__(queue, orca_executable, concurrent=concurrent, basename=basename,
                                         cleanup=cleanup)

    def _create_submission_command(self, n_inputs, compdir, jobname):
        submission_command = self.QUEUE_FILE.format(queue=self.queue,
                                                    basename=self.basename,
                                                    array_range=n_inputs,
                                                    concurrent=self.concurrent,
                                                    compdir=compdir,
                                                    orca_path=self.executable,
                                                    jobname=jobname)
        return submission_command
