from __future__ import annotations
from typing import Union, List, Dict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from schnetpack.md import System
    from schnetpack.model import PESModel
    from schnetpack.md.neighborlist_md import NeighborListMD

import schnetpack as spk
import torch
from schnetpack import properties
from schnetpack.md.calculators import MDCalculator, MDCalculatorError
from schnetpack.md.neighborlist_md import ASENeighborListMD
import logging

import time

log = logging.getLogger(__name__)

__all__ = ["SchnetPackCalculator"]


# from schnetpack.md.calculators.ensemble_calculator import EnsembleCalculator


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
        model_file: str,
        force_label: str,
        energy_units: Union[str, float],
        position_units: Union[str, float],
        energy_label: str = None,
        stress_label: str = None,
        property_conversion: Dict[str, Union[str, float]] = {},
        neighbor_list: NeighborListMD = ASENeighborListMD,
        cutoff: float = -1.0,
        cutoff_shell: float = 1.0,
    ):
        super(SchnetPackCalculator, self).__init__(
            required_properties=[energy_label, force_label, stress_label],
            force_label=force_label,
            energy_units=energy_units,
            position_units=position_units,
            energy_label=energy_label,
            stress_label=stress_label,
            property_conversion=property_conversion,
        )
        self.model = self._load_model(model_file)
        # TODO: ?????????
        # Activate properties if required
        # self.model.targets[properties.energy] = energy_label
        # self.model.targets[properties.forces] = force_label
        # self.model.targets[properties.stress] = stress_label
        self.model.eval()

        # Set up the neighbor list:
        self.neighbor_list = self._init_neighbor_list(
            neighbor_list, cutoff, cutoff_shell
        )

    def _load_model(self, model_file: str) -> PESModel:
        log.info("Loading model from {:s}".format(model_file))
        model = torch.jit.load(model_file)
        model.inference_mode = False
        # TODO !
        #    -> CASTING TO PRECISION IS PROBLEMATIC
        return model

    def _init_neighbor_list(
        self, neighbor_list: NeighborListMD, cutoff: float, cutoff_shell: float
    ):
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
        # Check if model has cutoff stored
        if hasattr(self.model, "cutoff"):
            model_cutoff = self.model.cutoff.item()
        else:
            model_cutoff = False

        # If a negative cutoff is given, use model cutoff
        if cutoff < 0.0:
            if not model_cutoff:
                raise MDCalculatorError(
                    "No cutoff found in model, please specify calculator cutoff via calculator.cutoff= ..."
                )
            else:
                log.info(
                    "Using model cutoff of {:.2f} (model units)...".format(model_cutoff)
                )
                cutoff = model_cutoff
        else:
            # Check if cutoff is sensible and raise a warning
            if cutoff < model_cutoff:
                log.warning(
                    "Cutoff {:.2f} for neighbor list smaller than cutoff in model {:.2f} (model units)...".format(
                        cutoff, model_cutoff
                    )
                )
                cutoff = model_cutoff
            else:
                log.info("Using cutoff of {:.2f} (model units)...".format(cutoff))

        # Check if atom triples need to be computed (e.g. for Behler functions)
        triples_required = self._check_triples_required()
        if triples_required:
            log.info("Enabling computation of atom triples")

        # Initialize the neighbor list
        neighbor_list = neighbor_list(
            cutoff=cutoff, cutoff_shell=cutoff_shell, requires_triples=triples_required
        )

        return neighbor_list

    def _check_triples_required(self):
        # Turn on collection of atom triples if representation requires angles
        if isinstance(self.model.representation, spk.representation.SymmetryFunctions):
            if self.model.representation.n_basis_angular > 0:
                log.info("Enabling collection of atom triples for angular functions...")
                return True
            else:
                return False
        else:
            return False

    def calculate(self, system: System):
        """
        Main routine, generates a properly formatted input for the schnetpack model from the system, performs the
        computation and uses the results to update the system state.

        Args:
            system (schnetpack.md.System): System object containing current state of the simulation.
        """
        inputs = self._generate_input(system)
        self.results = self.model(inputs)
        self._update_system(system)

    def _generate_input(self, system: System) -> Dict[str, torch.Tensor]:
        """
        Function to extracts neighbor lists, atom_types, positions e.t.c. from the system and generate a properly
        formatted input for the schnetpack model.

        Args:
            system (schnetpack.md.System): System object containing current state of the simulation.

        Returns:
            dict(torch.Tensor): Schnetpack inputs in dictionary format.
        """
        inputs = self._get_system_molecules(system)
        a = time.time()
        neighbors = self.neighbor_list.get_neighbors(inputs)
        b = time.time()
        print(b - a, "NBL")
        inputs.update(neighbors)
        return inputs


# class EnsembleSchnetPackCalculator(EnsembleCalculator, SchnetPackCalculator):
#     def __init__(
#         self,
#         models,
#         required_properties,
#         force_handle,
#         position_conversion="Angstrom",
#         force_conversion="eV / Angstrom",
#         property_conversion={},
#         stress_handle=None,
#         stress_conversion="eV / Angstrom / Angstrom / Angstrom",
#         detach=True,
#         neighbor_list=SimpleNeighborList,
#         cutoff=-1.0,
#         cutoff_shell=1.0,
#         cutoff_lr=None,
#     ):
#         self.models = models
#         required_properties = self._update_required_properties(required_properties)
#
#         super(EnsembleSchnetPackCalculator, self).__init__(
#             models[0],
#             required_properties=required_properties,
#             force_handle=force_handle,
#             position_conversion=position_conversion,
#             force_conversion=force_conversion,
#             property_conversion=property_conversion,
#             stress_handle=stress_handle,
#             stress_conversion=stress_conversion,
#             detach=detach,
#             neighbor_list=neighbor_list,
#             cutoff=cutoff,
#             cutoff_shell=cutoff_shell,
#             cutoff_lr=cutoff_lr,
#         )
