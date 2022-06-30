from __future__ import annotations
from typing import List, Union, Dict, Optional, Tuple

from typing import TYPE_CHECKING
from contextlib import nullcontext

import numpy as np

if TYPE_CHECKING:
    from schnetpack.md import System

import torch
import torch.nn as nn
from schnetpack import units as spk_units
from schnetpack import properties

import os

__all__ = ["MDCalculator", "MDCalculatorError", "QMCalculator", "QMCalculatorError"]


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
        force_key (str): Name of the property corresponding to the forces.
        energy_unit (str, float): Energy units returned by the internal computation model.
        position_unit (str, float): Unit conversion for the length used in the model computing all properties. E.g. if
                             the model needs Angstrom, one has to provide the conversion factor converting from the
                             atomic units used internally (Bohr) to Angstrom: 0.529177.
                             Is used together with `energy_unit` to determine units of force and stress.
        energy_key (str, optional): Name of the property corresponding to the energy.
        stress_key (str, optional): Name of the property corresponding to the stress.
        property_conversion (dict(float, str)): Optional dictionary of conversion factors for other properties predicted
                             by the model. Only changes the units used for logging the various outputs.
        gradients_required (bool): If set to true, enable accumulation of computational graph in calculator.
    """

    def __init__(
        self,
        required_properties: List,
        force_key: str,
        energy_unit: Union[str, float],
        position_unit: Union[str, float],
        energy_key: Optional[str] = None,
        stress_key: Optional[str] = None,
        property_conversion: Dict[str, Union[str, float]] = {},
        gradients_required: bool = False,
    ):
        super(MDCalculator, self).__init__()
        # Get required properties and filter non-unique entries
        self.required_properties = list(set(required_properties))

        if force_key not in self.required_properties:
            self.required_properties.append(force_key)

        if energy_key is not None and energy_key not in self.required_properties:
            self.required_properties.append(energy_key)

        if stress_key is not None and stress_key not in self.required_properties:
            self.required_properties.append(stress_key)

        self.results = {}

        self.energy_key = energy_key
        self.force_key = force_key
        self.stress_key = stress_key

        # Default conversion (1.0) for all units
        self.property_conversion = {p: 1.0 for p in self.required_properties}

        # Set requested conversion factors to internal unit system
        for p in property_conversion:
            self.property_conversion[p] = spk_units.unit2internal(
                property_conversion[p]
            )

        # Special unit conversions
        self.energy_conversion = spk_units.convert_units(energy_unit, spk_units.energy)
        self.position_conversion = spk_units.convert_units(
            position_unit, spk_units.length
        )

        # Derived conversions
        self.force_conversion = self.energy_conversion / self.position_conversion
        self.stress_conversion = self.energy_conversion / self.position_conversion**3

        # set up gradient context for passing results
        if gradients_required:
            self.grad_context = nullcontext()
        else:
            self.grad_context = torch.no_grad()

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
        with self.grad_context:
            # Collect all requested properties (including forces)
            for p in self.required_properties:
                if p not in self.results:
                    raise MDCalculatorError(
                        "Requested property {:s} not in " "results".format(p)
                    )
                else:
                    dim = self.results[p].shape
                    # Bring to general structure of MD code. Second dimension can be n_mol or n_mol x n_atoms.
                    system.properties[p] = (
                        self.results[p].view(system.n_replicas, -1, *dim[1:])
                        * self.property_conversion[p]
                    )

            # Set the forces for the system (at this point, already detached)
            self._set_system_forces(system)

            # Store potential energy to system if requested:
            if self.energy_key is not None:
                self._set_system_energy(system)

            # Set stress of the system if requested:
            if self.stress_key is not None:
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
        atom_types = system.atom_types.repeat(system.n_replicas)

        # Get n_atoms
        n_atoms = system.n_atoms.repeat(system.n_replicas)

        # Get positions
        positions = system.positions.view(-1, 3) / self.position_conversion

        # Construct index vector for all replicas and molecules
        index_m = (
            system.index_m.repeat(system.n_replicas, 1)
            + system.n_molecules
            * torch.arange(system.n_replicas, device=system.device).long().unsqueeze(-1)
        ).view(-1)

        # Get cells and PBC
        cells = system.cells.view(-1, 3, 3) / self.position_conversion
        pbc = system.pbc.repeat(system.n_replicas, 1, 1).view(-1, 3)

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
        forces = self.results[self.force_key]
        system.forces = (
            forces.view(system.n_replicas, system.total_n_atoms, 3)
            * self.force_conversion
        )

    def _set_system_energy(self, system: System):
        energy = self.results[self.energy_key]
        system.energy = (
            energy.view(system.n_replicas, system.n_molecules, 1)
            * self.energy_conversion
        )

    def _set_system_stress(self, system: System):
        stress = self.results[self.stress_key]
        system.stress = (
            stress.view(system.n_replicas, system.n_molecules, 3, 3)
            * self.stress_conversion
        )


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
        required_properties (list): List of the property names which will be passed to the simulator
        force_key (str): Name of the property corresponding to the forces.
        compdir (str): Directory in which computations are performed.
        qm_executable (str): Path to the ORCA executable.
        energy_unit (str, float): Energy units returned by the internal computation model.
        position_unit (str, float): Unit conversion for the length used in the model computing all properties. E.g. if
                             the model needs Angstrom, one has to provide the conversion factor converting from the
                             atomic units used internally (Bohr) to Angstrom: 0.529177.
                             Is used together with `energy_unit` to determine units of force and stress.
        energy_key (str, optional): Name of the property corresponding to the energy.
        stress_key (str, optional): Name of the property corresponding to the stress.
        property_conversion (dict(float, str)): Optional dictionary of conversion factors for other properties predicted
                             by the model. Only changes the units used for logging the various outputs.
        overwrite (bool): Overwrite previous computation results. Default is true.
        adaptive (bool): Flag for adaptive sampling.
    """

    is_atomistic = []

    def __init__(
        self,
        required_properties: List,
        force_key: str,
        compdir: str,
        qm_executable: str,
        energy_unit: Union[str, float],
        position_unit: Union[str, float],
        energy_key: Optional[str] = None,
        stress_key: Optional[str] = None,
        property_conversion: Dict[str, Union[str, float]] = {},
        overwrite: bool = True,
        adaptive: bool = False,
    ):
        super(QMCalculator, self).__init__(
            required_properties=required_properties,
            force_key=force_key,
            energy_unit=energy_unit,
            position_unit=position_unit,
            energy_key=energy_key,
            stress_key=stress_key,
            property_conversion=property_conversion,
        )

        from os import path

        self.qm_executable = os.path.abspath(qm_executable)

        self.compdir = os.path.abspath(compdir)
        if not os.path.exists(self.compdir):
            os.makedirs(compdir)

        # Initialize computation counter
        self.step = 0

        self.overwrite = overwrite
        self.adaptive = adaptive

    def calculate(self, system: System, samples: Optional[np.array] = None):
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
        if self.overwrite:
            current_compdir = os.path.join(self.compdir)
        else:
            current_compdir = os.path.join(
                self.compdir, "step_{:06d}".format(self.step)
            )

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

    def _extract_molecules(self, system: System, samples: Optional[np.array] = None):
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
        all_molecules = system.get_ase_atoms(position_unit_output=spk_units.length)

        molecules = []

        # flatten samples
        if samples is not None:
            samples = samples.flatten()

        for idx, mol in enumerate(all_molecules):
            if samples is not None:
                if not samples[idx]:
                    continue

            atom_types = mol.get_atomic_numbers()
            positions = mol.get_positions() / self.position_conversion

            # Store atom types and positions for ase db during sampling
            molecules.append((atom_types, positions))

        return molecules

    def _run_computation(
        self, molecules: List[Tuple[np.array, np.array]], current_compdir: str
    ):
        """
        Placeholder performing the computation.

        Args:
            molecules (list): List of tuples containing the atom types (integer numpy.array)
                      and positions (float numpy.array).
            current_compdir (str): Path to the current computation directory.
        """
        raise NotImplementedError

    def _format_calc(self, outputs: List[str], system: System):
        """
        Placeholder to format the computation output if no adaptive sampling is used.

        Args:
            outputs (list): Paths to output files.
            system (schnetpack.md.System): System from the molecular dynamics simulation.
        """
        raise NotImplementedError

    def _format_ase(
        self, molecules: List[Tuple[np.array, np.array]], outputs: List[str]
    ):
        """
        Placeholder to format the ouput for storage in an ASE database (for adaptive sampling).

        Args:
            molecules (list): List of tuples containing the atom types (integer numpy.array)
                      and positions (float numpy.array).
            outputs (list): Paths to output files.
        """
        raise NotImplementedError
