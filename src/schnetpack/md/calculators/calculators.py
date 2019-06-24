from schnetpack.data import Structure
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
        self.position_conversion = position_conversion
        self.force_conversion = force_conversion

        self.detach = detach

        self.property_conversion = property_conversion
        self._init_default_conversion()

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


class SchnetPackCalculator(MDCalculator):
    """
    MD calculator for schnetpack models.

    Args:
        model (object): Loaded schnetpack model.
        required_properties (list): List of properties to be computed by the calculator
        force_handle (str): String indicating the entry corresponding to the molecular forces
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
            model,
            required_properties,
            force_handle,
            position_conversion=1.0 / MDUnits.angs2bohr,
            force_conversion=1.0 / MDUnits.auforces2aseforces,
            property_conversion={},
            detach=True,
    ):
        super(SchnetPackCalculator, self).__init__(
            required_properties,
            force_handle,
            position_conversion,
            force_conversion,
            property_conversion,
            detach,
        )

        self.model = model

    def calculate(self, system):
        """
        Main routine, generates a properly formatted input for the schnetpack model from the system, performs the
        computation and uses the results to update the system state.

        Args:
            system (schnetpack.md.System): System object containing current state of the simulation.
        """
        inputs = self._generate_input(system)
        self.results = self.model(inputs)
        self._update_system(system)

    def _generate_input(self, system):
        """
        Function to extracts neighbor lists, atom_types, positions e.t.c. from the system and generate a properly
        formatted input for the schnetpack model.

        Args:
            system (schnetpack.md.System): System object containing current state of the simulation.

        Returns:
            dict(torch.Tensor): Schnetpack inputs in dictionary format.
        """
        positions, atom_types, atom_masks = self._get_system_molecules(system)
        neighbors, neighbor_mask = self._get_system_neighbors(system)

        inputs = {
            Structure.R: positions,
            Structure.Z: atom_types,
            Structure.atom_mask: atom_masks,
            Structure.cell: None,
            Structure.cell_offset: None,
            Structure.neighbors: neighbors,
            Structure.neighbor_mask: neighbor_mask,
        }

        return inputs
