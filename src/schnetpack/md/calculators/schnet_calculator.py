import schnetpack
from schnetpack import Properties

from schnetpack.md.calculators import MDCalculator
from schnetpack.md.calculators.ensemble_calculator import EnsembleCalculator
from schnetpack.md.neighbor_lists import SimpleNeighborList
import logging

from schnetpack.md.utils import MDUnits, check_triples_required
from schnetpack.environment import collect_atom_triples_batch


class SchnetPackCalculator(MDCalculator):
    """
    MD calculator for schnetpack models.

    Args:
        model (schnetpack.atomistic.AtomisticModel): Loaded schnetpack model.
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
        neighbor_list (schnetpack.md.neighbor_list.MDNeighborList): Neighbor list object for determining which
                                                                    interatomic distances should be computed.
        cutoff (float): Cutoff radius for computing the neighbor interactions. If this is set to a negative number,
                        the cutoff is determined automatically based on the model (default=-1.0). Units are the distance
                        units used in the model.
        cutoff_shell (float): Second shell around the cutoff region. The neighbor lists only are recomputed when atoms move
                              a distance further than this shell (default=1 Angstrom).
    """

    def __init__(
        self,
        model,
        required_properties,
        force_handle,
        position_conversion="Angstrom",
        force_conversion="eV / Angstrom",
        property_conversion={},
        stress_handle=None,
        stress_conversion="eV / Angstrom / Angstrom / Angstrom",
        detach=True,
        neighbor_list=SimpleNeighborList,
        cutoff=-1.0,
        cutoff_shell=1.0,
        cutoff_lr=None,
    ):
        super(SchnetPackCalculator, self).__init__(
            required_properties,
            force_handle,
            position_conversion=position_conversion,
            force_conversion=force_conversion,
            stress_handle=stress_handle,
            stress_conversion=stress_conversion,
            property_conversion=property_conversion,
            detach=detach,
        )
        self.model = model

        # Check if atom triples need to be computed (e.g. for Behler functions)
        self.triples_required = check_triples_required(model)

        # If stress is required, activate stress computation in model
        if stress_handle is not None:
            self._activate_stress(stress_handle)

        self.neighbor_list = self._init_neighbor_list(
            neighbor_list, cutoff, cutoff_shell, cutoff_lr
        )

    def _activate_stress(self, stress_handle):
        schnetpack.utils.activate_stress_computation(self.model, stress=stress_handle)

    def calculate(self, system):
        """
        Main routine, generates a properly formatted input for the schnetpack model from the system, performs the
        computation and uses the results to update the system state.

        Args:
            system (schnetpack.md.System): System object containing current state of the simulation.
        """
        # set model to evaluation mode to disable graph creation
        self.model.eval()

        inputs = self._generate_input(system)
        self.results = self.model(inputs)
        self._update_system(system)

    def _init_neighbor_list(self, neighbor_list, cutoff, cutoff_shell, cutoff_lr):
        """
        Function for properly setting up the neighbor list used for the SchNetPack calculator.
        This automatically checks, whether a proper cutoff has been provided and moves neighbor lists which support
        CUDA to the appropriate device.

        Args:
            neighbor_list (schnetpack.md.neighbor_lists.MDNeighborList.__init__): Uninitialized neighbor list class.
            cutoff (float): Cutoff radius for computing the neighbor interactions. If this is set to a negative number,
                            the cutoff is determined automatically based on the model (default=-1.0). Units are the
                            distance units used in the model.
            cutoff_shell (float): Second shell around the cutoff region. The neighbor lists only are recomputed when
                                  atoms move a distance further than this shell (default=1 Angstrom).

        Returns:
            schnetpack.md.neighbor_lists.MDNeighborList: Initialized neighbor list.
        """
        # Determine the cutoff for the neighbor list
        if cutoff is not None:
            if cutoff < 0.0:
                # Get cutoff automatically if given one is negative
                cutoff = self._get_model_cutoff()
                # cutoff *= self.position_conversion
            else:
                # Check whether set cutoff is reasonable
                try:
                    model_cutoff = self._get_model_cutoff()
                    # cutoff *= self.position_conversion
                    if cutoff < model_cutoff:
                        logging.warning(
                            f"Specified cutoff for neighbor list {cutoff} smaller than cutoff in model {model_cutoff}."
                        )
                except:
                    print("No cutoff found in model, using {:f}".format(cutoff))

            # Convert from model units to internal units
            cutoff /= self.position_conversion
            cutoff_shell /= self.position_conversion

        if isinstance(neighbor_list, schnetpack.md.neighbor_lists.TorchNeighborList):
            device = next(self.model.parameters()).device
        else:
            device = None

        if cutoff_lr is not None:
            cutoff_lr /= self.position_conversion
            neighbor_list = schnetpack.md.neighbor_lists.DualNeighborList(
                cutoff, cutoff_lr, neighbor_list, shell=cutoff_shell, device=device
            )
        else:
            neighbor_list = neighbor_list(cutoff, shell=cutoff_shell, device=device)

        return neighbor_list

    def _generate_input(self, system):
        """
        Function to extracts neighbor lists, atom_types, positions e.t.c. from the system and generate a properly
        formatted input for the schnetpack model.

        Args:
            system (schnetpack.md.System): System object containing current state of the simulation.

        Returns:
            dict(torch.Tensor): Schnetpack inputs in dictionary format.
        """
        positions, atom_types, atom_masks, cells, pbc = self._get_system_molecules(
            system
        )

        inputs = {
            Properties.R: positions,
            Properties.Z: atom_types,
            Properties.atom_mask: atom_masks,
            Properties.cell: cells,
            Properties.pbc: pbc,
        }

        neighbors = self._get_system_neighbors(system)
        inputs.update(neighbors)

        return inputs

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
            torch.LongTensor: (n_replicas*n_molecules) x n_atoms x (n_atoms-1) x 3 tensor of the cell offset vectors
                              for periodic boundary conditions or cell simulations.
        """
        if self.neighbor_list is None:
            raise ValueError("System does not have neighbor list.")

        neighbor_list, neighbor_mask, offsets = self.neighbor_list.get_neighbors(system)

        neighbor_list, neighbor_mask, offsets = self._format_neighbors(
            neighbor_list,
            neighbor_mask,
            offsets,
            system.max_n_atoms,
            self.neighbor_list.max_neighbors,
        )
        neighbors = {
            Properties.neighbors: neighbor_list,
            Properties.neighbor_mask: neighbor_mask,
            Properties.cell_offset: offsets,
        }

        # Check if triples are required
        if self.triples_required:
            (
                nbh_j,
                nbh_k,
                offset_idx_j,
                offset_idx_k,
                pair_mask,
            ) = collect_atom_triples_batch(neighbor_list, neighbor_mask)
            neighbors[Properties.neighbor_pairs_j] = nbh_j
            neighbors[Properties.neighbor_pairs_k] = nbh_k
            neighbors[Properties.neighbor_offsets_j] = offset_idx_j
            neighbors[Properties.neighbor_offsets_k] = offset_idx_k
            neighbors[Properties.neighbor_pairs_mask] = pair_mask

        # Check if two cutoffs are present
        if isinstance(
            self.neighbor_list, schnetpack.md.neighbor_lists.DualNeighborList
        ):
            (
                neighbor_list_lr,
                neighbor_mask_lr,
                offsets_lr,
            ) = self.neighbor_list.get_neighbors_lr(system)

            neighbor_list_lr, neighbor_mask_lr, offsets_lr = self._format_neighbors(
                neighbor_list_lr,
                neighbor_mask_lr,
                offsets_lr,
                system.max_n_atoms,
                self.neighbor_list.max_neighbors_lr,
            )
            neighbors.update(
                {
                    Properties.neighbors_lr: neighbor_list_lr,
                    Properties.neighbor_mask_lr: neighbor_mask_lr,
                    Properties.cell_offset_lr: offsets_lr,
                }
            )

        return neighbors

    @staticmethod
    def _format_neighbors(
        neighbor_list, neighbor_mask, offsets, max_n_atoms, max_neighbors
    ):
        neighbor_list = neighbor_list.view(-1, max_n_atoms, max_neighbors)
        neighbor_mask = neighbor_mask.view(-1, max_n_atoms, max_neighbors)

        # Offsets need not be transformed to units, only cell
        if offsets is not None:
            offsets = offsets.view(-1, max_n_atoms, max_neighbors, 3)
        return neighbor_list, neighbor_mask, offsets

    def _get_model_cutoff(self):
        """
        Function to check the model passed to the calculator for already set cutoffs. This depends on the representation
        used, currently SchNet and wACSF models are supported.

        Returns:
            float: Model cutoff in model position units.
        """

        # Get representation
        if hasattr(self.model, "module"):
            representation = self.model.module.representation
        else:
            representation = self.model.representation

        # Check for different models, cutoff is set in different functions
        if isinstance(representation, schnetpack.representation.SchNet):
            model_cutoff = representation.interactions[0].cutoff_network.cutoff
        elif isinstance(representation, schnetpack.representation.BehlerSFBlock):
            model_cutoff = representation.cutoff.cutoff
        elif isinstance(representation, schnetpack.representation.StandardizeSF):
            model_cutoff = representation.SFBlock.cutoff.cutoff
        else:
            raise ValueError(
                "Unrecognized model representation {:s }for cutoff detection.".format(
                    representation
                )
            )

        # Convert from torch tensor and print out the detected cutoff
        model_cutoff = float(model_cutoff[0].cpu().numpy())
        logging.info("Detected cutoff radius of {:5.3f}...".format(model_cutoff))

        return model_cutoff


class EnsembleSchnetPackCalculator(EnsembleCalculator, SchnetPackCalculator):
    def __init__(
        self,
        models,
        required_properties,
        force_handle,
        position_conversion="Angstrom",
        force_conversion="eV / Angstrom",
        property_conversion={},
        stress_handle=None,
        stress_conversion="eV / Angstrom / Angstrom / Angstrom",
        detach=True,
        neighbor_list=SimpleNeighborList,
        cutoff=-1.0,
        cutoff_shell=1.0,
        cutoff_lr=None,
    ):
        self.models = models
        required_properties = self._update_required_properties(required_properties)

        super(EnsembleSchnetPackCalculator, self).__init__(
            models[0],
            required_properties=required_properties,
            force_handle=force_handle,
            position_conversion=position_conversion,
            force_conversion=force_conversion,
            property_conversion=property_conversion,
            stress_handle=stress_handle,
            stress_conversion=stress_conversion,
            detach=detach,
            neighbor_list=neighbor_list,
            cutoff=cutoff,
            cutoff_shell=cutoff_shell,
            cutoff_lr=cutoff_lr,
        )
