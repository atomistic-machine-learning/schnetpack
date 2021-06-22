from typing import List, Union, Dict

import torch
import torch.nn as nn
from schnetpack import units as spk_units
from schnetpack import properties
from schnetpack.md import System

__all__ = ["MDCalculator", "MDCalculatorError"]


class MDCalculatorError(Exception):
    """
    Exception for MDCalculator base class.
    """

    pass


class MDCalculator(nn.Module):
    """
    Base MDCalculator class for computing and updating the forces of the simulated system, as well as other
    properties predicted by the model.

    Args:
        required_properties (list): List of the property names which will be passed to the simulator
        force_label (str): Name of the property corresponding to the forces.
        energy_units (str, float): Energy units returned by the internal computation model.
        position_units (str, float): Unit conversion for the length used in the model computing all properties. E.g. if
                             the model needs Angstrom, one has to provide the conversion factor converting from the
                             atomic units used internally (Bohr) to Angstrom: 0.529177.
                             Is used together with `energy_units` to determine units of force and stress.
        energy_label (str, optional): Name of the property corresponding to the energy.
        stress_label (str, optional): Name of the property corresponding to the stress.
        property_conversion (dict(float, str)): Optional dictionary of conversion factors for other properties predicted
                             by the model. Only changes the units used for logging the various outputs.
    """

    def __init__(
        self,
        required_properties: List,
        force_label: str,
        energy_units: Union[str, float],
        position_units: Union[str, float],
        energy_label: str = None,
        stress_label: str = None,
        property_conversion: Dict[str, Union[str, float]] = {},
    ):
        super(MDCalculator, self).__init__()
        # Get required properties and filer Nones for easier init in derived classes
        self.required_properties = [rp for rp in required_properties if rp is not None]
        print(self.required_properties, required_properties, "PPP")

        self.results = {}

        self.energy_label = energy_label
        self.force_label = force_label
        self.stress_label = stress_label

        # Default conversion (1.0) for all units
        self.property_conversion = {p: 1.0 for p in self.required_properties}

        # Set requested conversion factors to internal unit system
        for p in property_conversion:
            self.property_conversion[p] = spk_units.unit2internal(
                property_conversion[p]
            )

        # Special unit conversions
        self.energy_conversion = spk_units.convert_units(energy_units, spk_units.energy)
        self.position_conversion = spk_units.convert_units(
            position_units, spk_units.length
        )

        # Derived conversions
        self.force_conversion = self.energy_conversion / self.position_conversion
        self.stress_conversion = self.energy_conversion / self.position_conversion ** 3

    def calculate(self, system: System):
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

    def _update_system(self, system: System):
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
                dim = self.results[p].shape
                print(p, self.property_conversion)
                # Bring to general structure of MD code. Second dimension can be n_mol or n_mol x n_atoms.
                system.properties[p] = (
                    self.results[p].view(system.n_replicas, -1, *dim[1:])
                    * self.property_conversion[p]
                )

        # Set the forces for the system (at this point, already detached)
        self._set_system_forces(system)

        # Set stress of the system if requested:
        if self.stress_label is not None:
            self._set_system_stress(system)

    def _get_system_molecules(self, system: System):
        """
        Routine to extract positions, atom_types and atom_masks formatted in a manner suitable for schnetpack models
        from the system class. This is done by collapsing the replica and molecule dimension into one batch dimension.

        Args:
            system (schnetpack.md.System): System object containing current state of the simulation.

        Returns:
            dict(str, torch.Tensor): Input batch for schnetpack models without neighbor information.
        """
        # Get atom types
        atom_types = system.atom_types.repeat(system.n_molecules)

        # Get n_atoms
        n_atoms = system.n_atoms.repeat(system.n_molecules)

        # Get positions
        positions = system.positions.view(-1, 3) / self.position_conversion

        # Construct index vector for all replicas and molecules
        index_m = (
            system.index_m.repeat(system.n_molecules, 1)
            + system.n_molecules
            * torch.arange(system.n_molecules, device=system.device)
            .long()
            .unsqueeze(-1)
        ).view(-1)

        # Get cells and PBC
        cells = system.cells
        pbc = system.pbc

        if system.cells is not None:
            cells = cells.view(-1, 3, 3) / self.position_conversion
            pbc = pbc.repeat(system.n_replicas, 1, 1).view(-1, 3)

        inputs = {
            properties.Z: atom_types,
            properties.n_atoms: n_atoms,
            properties.R: positions,
            properties.idx_m: index_m,
            properties.cell: cells,
            properties.pbc: pbc,
        }

        return inputs

    def _set_system_forces(self, system: System):
        """
        Function to reformat and update the forces of the system from the computed forces stored in self.results.
        The string contained in self.force_handle is used as an indicator. The single batch dimension is recast to the
        original replica x molecule dimensions used by the system.

        Args:
            system (schnetpack.md.System): System object containing current state of the simulation.
        """
        forces = self.results[self.force_label]
        system.forces = (
            forces.view(system.n_replicas, system.total_n_atoms, 3)
            * self.force_conversion
        )

    def _set_system_stress(self, system: System):
        stress = self.results[self.stress_label]
        system.stress = (
            stress.view(system.n_replicas, system.n_molecules, 3, 3)
            * self.stress_conversion
        )


# class QMCalculator(MDCalculator):
#     """
#     Basic calculator for interfacing quantum chemistry codes with SchNetPack molecular dynamics.
#
#
#
#     Calculator for interfacing the ORCA code package with SchNetPack molecular dynamics.
#     Requires ORCA to be installed and an input file template.
#     This template is a standard ORCA input file, with everything past the specification of coordinate
#     format, charge and multiplicity removed (coordinates and final *).
#     If desired, a Queuer can be give, which will attempt to send all jobs to a grid engine queue.
#
#     In general, the calculator will take the current System to generate inputs, perform the calculation
#     with ORCA, extract data from the ouput file (useing the OrcaParser class) and update the System.
#
#     Args:
#         required_properties (list): List of properties which should be extracted from output.
#         force_handle (str): Indicator for molecular forces.
#         compdir (str): Directory in which computations are performed.
#         qm_executable (str): Path to the ORCA executable.
#         position_conversion (str/float, optional): Conversion of positions from atomic units.
#         force_conversion (str/float, optional): Conversion of forces to atomic units.
#         property_conversion (dict, optional): Convert properties to requested units. If left empty, no conversion
#                                               is performed.
#         adaptive (bool, optional): Specify, whether the calculator should be used for adaptive sampling.
#     """
#
#     is_atomistic = []
#
#     def __init__(
#             self,
#             required_properties,
#             force_handle,
#             compdir,
#             qm_executable,
#             position_conversion="Angstrom",
#             force_conversion=1.0,
#             property_conversion={},
#             adaptive=False,
#     ):
#
#         super(QMCalculator, self).__init__(
#             required_properties,
#             force_handle,
#             position_conversion=position_conversion,
#             force_conversion=force_conversion,
#             property_conversion=property_conversion,
#         )
#
#         self.qm_executable = qm_executable
#
#         self.compdir = compdir
#         if not os.path.exists(self.compdir):
#             os.makedirs(compdir)
#
#         # Set the force handle to be an atomistic property
#         self.is_atomistic = force_handle
#
#         self.adaptive = adaptive
#         self.step = 0
#
#     def calculate(self, system, samples=None):
#         """
#         Perform the calculation with a quantum chemistry code.
#         If samples is given, only a subset of molecules is selected.
#
#         Args:
#             system (schnetpack.md.System): System from the molecular dynamics simulation.
#             samples (np.array, optional): Integer array specifying whether only particular
#                                           replicas and molecules in the system should be used for
#                                           computations. Only works with adaptive sampling.
#
#         Returns:
#             (list,list):
#                 atom_buffer:
#                     List of ASE atoms objects of every computed molecule.
#                     Only returned if adaptive sampling is activated.
#
#                 property_buffer:
#                     List of property dictionaries for every computation.
#                     Only returned if adaptive sampling is activated.
#         """
#         # Use of samples only makes sense in conjunction with adaptive sampling
#         if not self.adaptive and samples is not None:
#             raise QMCalculatorError(
#                 "Usage of subsamples only allowed during adaptive sampling."
#             )
#
#         # Generate director for current step
#         # current_compdir = os.path.join(self.compdir, 'step_{:06d}'.format(self.step))
#         current_compdir = os.path.join(self.compdir, "step_X")
#         if not os.path.exists(current_compdir):
#             os.makedirs(current_compdir)
#
#         # Get molecules (select samples if requested)
#         molecules = self._extract_molecules(system, samples=samples)
#
#         # Run computation
#         outputs = self._run_computation(molecules, current_compdir)
#
#         # Increment internal step
#         self.step += 1
#
#         # Prepare output
#         # a) either parse to update system properties
#         if not self.adaptive:
#             self.results = self._format_calc(outputs, system)
#             self._update_system(system)
#         # b) or append to the database (just return everything as molecules/atoms objects)
#         else:
#             atom_buffer, property_buffer = self._format_ase(molecules, outputs)
#             return atom_buffer, property_buffer
#
#     def _extract_molecules(self, system, samples=None):
#         """
#         Extract atom types and molecular structures from the system. and convert to
#         appropriate units.
#
#         Args:
#             system (schnetpack.md.System): System from the molecular dynamics simulation.
#             samples (np.array, optional): Integer array specifying whether only particular
#                                           replicas and molecules in the system should be used for
#                                           computations. Only works with adaptive sampling.
#
#         Returns:
#             list: List of tuples containing the atom types (integer numpy.array) and positions
#                   (float numpy.array).
#         """
#         molecules = []
#         for rep_idx in range(system.n_replicas):
#             for mol_idx in range(system.n_molecules):
#                 # Check which geometries need samples in adaptive setup
#                 if samples is not None:
#                     if not samples[rep_idx, mol_idx]:
#                         continue
#                 atom_types = system.atom_types[
#                              rep_idx, mol_idx, : system.n_atoms[mol_idx]
#                              ]
#                 # Convert Bohr to Angstrom
#                 positions = (
#                         system.positions[rep_idx, mol_idx, : system.n_atoms[mol_idx], ...]
#                         * self.position_conversion
#                 )
#                 # Store atom types and positions for ase db during sampling
#                 molecules.append((atom_types, positions))
#
#         return molecules
#
#     def _run_computation(self, molecules, current_compdir):
#         """
#         Placeholder performing the computation.
#
#         Args:
#             molecules (list): List of tuples containing the atom types (integer numpy.array)
#                       and positions (float numpy.array).
#             current_compdir (str): Path to the current computation directory.
#         """
#         raise NotImplementedError
#
#     def _format_calc(self, outputs, system):
#         """
#         Placeholder to format the computation output if no adaptive sampling is used.
#
#         Args:
#             outputs (list): Paths to output files.
#             system (schnetpack.md.System): System from the molecular dynamics simulation.
#         """
#         raise NotImplementedError
#
#     def _format_ase(self, molecules, outputs):
#         """
#         Placeholder to format the ouput for storage in an ASE database (for adaptive sampling).
#
#         Args:
#             molecules (list): List of tuples containing the atom types (integer numpy.array)
#                       and positions (float numpy.array).
#             outputs (list): Paths to output files.
#         """
#         raise NotImplementedError
