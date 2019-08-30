import os
import subprocess

from schnetpack.md.utils import MDUnits


class MDCalculatorError(Exception):
    """
    Exception for MDCalculator base class.
    """

    pass


class MDCalculator:
    """
    Base MDCalculator class for computing and updating the forces of the simulated system, as well as other
    properties predicted by the model.

    Args:
        required_properties (list): List of the property names which will be passed to the simulator
        force_handle (str): Name of the property corresponding to the forces.
        position_conversion (float): Unit conversion for the length used in the model computing all properties. E.g. if
                             the model needs Angstrom, one has to provide the conversion factor converting from the
                             atomic units used internally (Bohr) to Angstrom: 0.529177...
        force_conversion (float): Conversion factor converting the forces returned by the used model back to atomic
                                  units (Hartree/Bohr).
        property_conversion (dict(float)): Optional dictionary of conversion factors for other properties predicted by
                                           the model. Only changes the units used for logging the various outputs.
        detach (bool): Detach property computation graph after every calculator call. Enabled by default. Should only
                       be disabled if one wants to e.g. compute derivatives over short trajectory snippets.
    """

    def __init__(
        self,
        required_properties,
        force_handle,
        position_conversion=1.0,
        force_conversion=1.0,
        property_conversion={},
        detach=True,
    ):
        self.results = {}
        self.force_handle = force_handle
        self.required_properties = required_properties

        # Perform automatic conversion of units
        self.position_conversion = MDUnits.parse_mdunit(position_conversion)
        self.force_conversion = MDUnits.parse_mdunit(force_conversion)
        self.property_conversion = {
            p: MDUnits.parse_mdunit(property_conversion[p]) for p in property_conversion
        }
        self._init_default_conversion()

        self.detach = detach

    def calculate(self, system):
        """
        Main calculator routine, which needs to be implemented individually.
        This routine should take the current system state, perform the appropriate computations to get the forces
        and use them to update the system forces stored in system.forces

        To this end, results should be stored in the dictionary self.results using the keys contained in
        self.required_properties
        Afterwards, the routine self._update_system(system) can be used to update the system state.

        Args:
            system (schnetpack.md.System): System object containing current state of the simulation.
        """
        raise NotImplementedError

    def _init_default_conversion(self):
        """
        Auxiliary routine to initialize default conversion factors (1.0) if no alternatives are given in
        property_conversion upon initializing the calculator.
        """
        for p in self.required_properties:
            if p not in self.property_conversion:
                self.property_conversion[p] = 1.0

    def _update_system(self, system):
        """
        Routine, which looks in self.results for the properties defined in self.required_properties and uses them to
        update the forces and properties of the provided system. If required, reformatting is carried out here.

        Args:
            system (schnetpack.md.System): System object containing current state of the simulation.
        """

        # Collect all requested properties (including forces)
        for p in self.required_properties:
            if p not in self.results:
                raise MDCalculatorError(
                    "Requested property {:s} not in " "results".format(p)
                )
            else:
                # Detach properties if requested
                if self.detach:
                    self.results[p] = self.results[p].detach()

                dim = self.results[p].shape
                system.properties[p] = (
                    self.results[p].view(
                        system.n_replicas, system.n_molecules, *dim[1:]
                    )
                    * self.property_conversion[p]
                )

            # Set the forces for the system (at this point, already detached)
            self._set_system_forces(system)

    def _get_system_neighbors(self, system):
        """
        Auxiliary function, which extracts neighbor lists formatted for schnetpack models from the system class.
        This is done by collapsing the replica and molecule dimension into one batch dimension.

        Args:
            system (schnetpack.md.System): System object containing current state of the simulation.

        Returns:
            torch.LongTensor: (n_replicas*n_molecules) x n_atoms x (n_atoms-1) tensor holding the indices of all
                              neighbor atoms.
            torch.LongTensor: (n_replicas*n_molecules) x n_atoms x (n_atoms-1) binary tensor indicating padded
                              dimensions.
        """
        if system.neighbor_list is None:
            raise ValueError("System does not have neighbor list.")
        neighbor_list, neighbor_mask = system.neighbor_list.get_neighbors()

        neighbor_list = neighbor_list.view(
            -1, system.max_n_atoms, system.max_n_atoms - 1
        )
        neighbor_mask = neighbor_mask.view(
            -1, system.max_n_atoms, system.max_n_atoms - 1
        )
        return neighbor_list, neighbor_mask

    def _get_system_molecules(self, system):
        """
        Routine to extract positions, atom_types and atom_masks formatted in a manner suitable for schnetpack models
        from the system class. This is done by collapsing the replica and molecule dimension into one batch dimension.

        Args:
            system (schnetpack.md.System): System object containing current state of the simulation.

        Returns:
            torch.FloatTensor: (n_replicas*n_molecules) x n_atoms x 3 tensor holding nuclear positions
            torch.FloatTensor: (n_replicas*n_molecules) x n_atoms tensor holding nuclear charges
            torch.FloatTensor: (n_replicas*n_molecules) x n_atoms binary tensor indicating padded atom dimensions
        """
        positions = (
            system.positions.view(-1, system.max_n_atoms, 3) * self.position_conversion
        )

        atom_types = system.atom_types.view(-1, system.max_n_atoms)
        atom_masks = system.atom_masks.view(-1, system.max_n_atoms)
        return positions, atom_types, atom_masks

    def _set_system_forces(self, system):
        """
        Function to reformat and update the forces of the system from the computed forces stored in self.results.
        The string contained in self.force_handle is used as an indicator. The single batch dimension is recast to the
        original replica x molecule dimensions used by the system.

        Args:
            system (schnetpack.md.System): System object containing current state of the simulation.
        """
        forces = self.results[self.force_handle]
        system.forces = (
            forces.view(system.n_replicas, system.n_molecules, system.max_n_atoms, 3)
            * self.force_conversion
        )

    def _get_ase_molecules(self, system):
        """
        Dummy function to get molecules in ASE format.
        """
        pass


class QMCalculatorError(Exception):
    """
    Exception for the QM calculator base class
    """

    pass


class QMCalculator(MDCalculator):
    """
    Basic calculator for interfacing quantum chemistry codes with SchNetPack molecular dynamics.



    Calculator for interfacing the ORCA code package with SchNetPack molecular dynamics.
    Requires ORCA to be installed and an input file template.
    This template is a standard ORCA input file, with everything past the specification of coordinate
    format, charge and multiplicity removed (coordinates and final *).
    If desired, a Queuer can be give, which will attempt to send all jobs to a grid engine queue.

    In general, the calculator will take the current System to generate inputs, perform the calculation
    with ORCA, extract data from the ouput file (useing the OrcaParser class) and update the System.

    Args:
        required_properties (list): List of properties which should be extracted from output.
        force_handle (str): Indicator for molecular forces.
        compdir (str): Directory in which computations are performed.
        qm_executable (str): Path to the ORCA executable.
        position_conversion (str/float, optional): Conversion of positions from atomic units.
        force_conversion (str/float, optional): Conversion of forces to atomic units.
        property_conversion (dict, optional): Convert properties to requested units. If left empty, no conversion
                                              is performed.
        adaptive (bool, optional): Specify, whether the calculator should be used for adaptive sampling.
    """

    is_atomistic = []

    def __init__(
        self,
        required_properties,
        force_handle,
        compdir,
        qm_executable,
        position_conversion=1.0 / MDUnits.angs2bohr,
        force_conversion=1.0,
        property_conversion={},
        adaptive=False,
    ):

        super(QMCalculator, self).__init__(
            required_properties,
            force_handle,
            position_conversion=position_conversion,
            force_conversion=force_conversion,
            property_conversion=property_conversion,
        )

        self.qm_executable = qm_executable

        self.compdir = compdir
        if not os.path.exists(self.compdir):
            os.makedirs(compdir)

        # Set the force handle to be an atomistic property
        self.is_atomistic = force_handle

        self.adaptive = adaptive
        self.step = 0

    def calculate(self, system, samples=None):
        """
        Perform the calculation with a quantum chemistry code.
        If samples is given, only a subset of molecules is selected.

        Args:
            system (schnetpack.md.System): System from the molecular dynamics simulation.
            samples (np.array, optional): Integer array specifying whether only particular
                                          replicas and molecules in the system should be used for
                                          computations. Only works with adaptive sampling.

        Returns:
            (list,list):
                atom_buffer:
                    List of ASE atoms objects of every computed molecule.
                    Only returned if adaptive sampling is activated.

                property_buffer:
                    List of property dictionaries for every computation.
                    Only returned if adaptive sampling is activated.
        """
        # Use of samples only makes sense in conjunction with adaptive sampling
        if not self.adaptive and samples is not None:
            raise QMCalculatorError(
                "Usage of subsamples only allowed during adaptive sampling."
            )

        # Generate director for current step
        # current_compdir = os.path.join(self.compdir, 'step_{:06d}'.format(self.step))
        current_compdir = os.path.join(self.compdir, "step_X")
        if not os.path.exists(current_compdir):
            os.makedirs(current_compdir)

        # Get molecules (select samples if requested)
        molecules = self._extract_molecules(system, samples=samples)

        # Run computation
        outputs = self._run_computation(molecules, current_compdir)

        # Increment internal step
        self.step += 1

        # Prepare output
        # a) either parse to update system properties
        if not self.adaptive:
            self.results = self._format_calc(outputs, system)
            self._update_system(system)
        # b) or append to the database (just return everything as molecules/atoms objects)
        else:
            atom_buffer, property_buffer = self._format_ase(molecules, outputs)
            return atom_buffer, property_buffer

    def _extract_molecules(self, system, samples=None):
        """
        Extract atom types and molecular structures from the system. and convert to
        appropriate units.

        Args:
            system (schnetpack.md.System): System from the molecular dynamics simulation.
            samples (np.array, optional): Integer array specifying whether only particular
                                          replicas and molecules in the system should be used for
                                          computations. Only works with adaptive sampling.

        Returns:
            list: List of tuples containing the atom types (integer numpy.array) and positions
                  (float numpy.array).
        """
        molecules = []
        for rep_idx in range(system.n_replicas):
            for mol_idx in range(system.n_molecules):
                # Check which geometries need samples in adaptive setup
                if samples is not None:
                    if not samples[rep_idx, mol_idx]:
                        continue
                atom_types = system.atom_types[
                    rep_idx, mol_idx, : system.n_atoms[mol_idx]
                ]
                # Convert Bohr to Angstrom
                positions = (
                    system.positions[rep_idx, mol_idx, : system.n_atoms[mol_idx], ...]
                    * self.position_conversion
                )
                # Store atom types and positions for ase db during sampling
                molecules.append((atom_types, positions))

        return molecules

    def _run_computation(self, molecules, current_compdir):
        """
        Placeholder performing the computation.

        Args:
            molecules (list): List of tuples containing the atom types (integer numpy.array)
                      and positions (float numpy.array).
            current_compdir (str): Path to the current computation directory.
        """
        raise NotImplementedError

    def _format_calc(self, outputs, system):
        """
        Placeholder to format the computation output if no adaptive sampling is used.

        Args:
            outputs (list): Paths to output files.
            system (schnetpack.md.System): System from the molecular dynamics simulation.
        """
        raise NotImplementedError

    def _format_ase(self, molecules, outputs):
        """
        Placeholder to format the ouput for storage in an ASE database (for adaptive sampling).

        Args:
            molecules (list): List of tuples containing the atom types (integer numpy.array)
                      and positions (float numpy.array).
            outputs (list): Paths to output files.
        """
        raise NotImplementedError


class Queuer:
    """
    Base class for interfacing a calculator with a grid engine queue.
    Will create the directory structure and a submission command, which it will pass to the
    given queue. Jobs will be submitted as array jobs, where the number of concurrently running
    jobs can be changed. In order to the script to work, the QUEUE_FILE string needs to be updated
    and the _create_submission_command function has to be implemented.

    Args:
        queue (str): Indentifier for the queue the job should be sent to.
        executable (str): Path to the executable to be called.
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

# Adapt here
"""

    def __init__(
        self, queue, executable, concurrent=100, basename="input", cleanup=True
    ):
        self.queue = queue
        self.executable = executable
        self.concurrent = concurrent
        self.basename = basename
        self.cleanup = cleanup

    def submit(self, input_files, current_compdir):
        """
        Submit an array of jobs to a grid engine and wait for their completion.

        Args:
            input_files (list): List of all inputs.
            current_compdir (str): Path to the directory used for computations.
        """
        jobname = os.path.basename(current_compdir)
        compdir = os.path.abspath(current_compdir)
        n_inputs = len(input_files)

        submission_command = self._create_submission_command(n_inputs, compdir, jobname)

        script_name = os.path.join(current_compdir, "submit.sh")
        with open(script_name, "w") as submission_script:
            submission_script.write(submission_command)

        computation = subprocess.Popen(["qsub", script_name], stdout=subprocess.PIPE)
        computation.wait()

        if self.cleanup:
            os.remove(script_name)

    def _create_submission_command(self, n_inputs, compdir, jobname):
        """
        Creates the submission command and needs to be implemeneted for specific tasks.

        Args:
            n_inputs (int): Number of input files to be submitted
            compdir (str): Current working directory
            jobname (str): Name of each job

        Returns:
            str: Submission command.
        """
        raise NotImplementedError
