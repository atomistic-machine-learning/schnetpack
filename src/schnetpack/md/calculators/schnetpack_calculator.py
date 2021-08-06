from __future__ import annotations
from typing import Union, List, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from schnetpack.md import System
    from schnetpack.atomistic.model import AtomisticModel
    from schnetpack.md.neighborlist_md import NeighborListMD

import torch
import logging

from schnetpack.md.calculators.base_calculator import MDCalculator, MDCalculatorError
from schnetpack.md.calculators.ensemble_calculator import EnsembleCalculator
from schnetpack.md.neighborlist_md import ASENeighborListMD

import time

log = logging.getLogger(__name__)

__all__ = ["SchnetPackCalculator", "SchnetPackEnsembleCalculator"]


class SchnetPackCalculator(MDCalculator):
    """
    MD calculator for schnetpack models.

    Args:
        model_file (str): Path to stored schnetpack model.
        force_label (str): String indicating the entry corresponding to the molecular forces
        energy_units (float, float): Conversion factor converting the energies returned by the used model back to
                                     internal MD units.
        position_units (float, float): Conversion factor for converting the system positions to the units required by
                                       the model.
        energy_label (str, optional): If provided, label is used to store the energies returned by the model to the
                                      system.
        stress_label (str, optional): If provided, label is used to store the stress returned by the model to the
                                      system (required for constant pressure simulations).
        required_properties (list): List of properties to be computed by the calculator
        property_conversion (dict(float)): Optional dictionary of conversion factors for other properties predicted by
                                           the model. Only changes the units used for logging the various outputs.
        neighbor_list (schnetpack.md.neighbor_list.MDNeighborList): Neighbor list object for determining which
                                                                    interatomic distances should be computed.
        cutoff (float): Cutoff radius for computing the neighbor interactions. If this is set to a negative number,
                        the cutoff is determined automatically based on the model (default=-1.0). Units are the distance
                        units used in the model.
        cutoff_shell (float): Second shell around the cutoff region. The neighbor lists only are recomputed when atoms
                              move a distance further than this shell (default=1 model unit).
    """

    def __init__(
        self,
        model_file: str,
        force_label: str,
        energy_units: Union[str, float],
        position_units: Union[str, float],
        energy_label: str = None,
        stress_label: str = None,
        required_properties: List = [],
        property_conversion: Dict[str, Union[str, float]] = {},
        neighbor_list: NeighborListMD = ASENeighborListMD,
        cutoff: float = -1.0,
        cutoff_shell: float = 1.0,
    ):
        super(SchnetPackCalculator, self).__init__(
            required_properties=required_properties
            + [energy_label, force_label, stress_label],
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

    @staticmethod
    def _load_model(model_file: str) -> AtomisticModel:
        log.info("Loading model from {:s}".format(model_file))
        model = torch.jit.load(model_file)
        model.inference_mode = False
        # TODO:
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

        # Initialize the neighbor list
        neighbor_list = neighbor_list(
            cutoff=cutoff, cutoff_shell=cutoff_shell, requires_triples=False
        )

        return neighbor_list

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


class SchnetPackEnsembleCalculator(EnsembleCalculator, SchnetPackCalculator):
    """
    Ensemble calculator using schnetpack models. Uncertainties are computed as the variance of all model predictions.
    """

    def __init__(
        self,
        model_files: List[str],
        force_label: str,
        energy_units: Union[str, float],
        position_units: Union[str, float],
        energy_label: str = None,
        stress_label: str = None,
        required_properties: List = [],
        property_conversion: Dict[str, Union[str, float]] = {},
        neighbor_list: NeighborListMD = ASENeighborListMD,
        cutoff: float = -1.0,
        cutoff_shell: float = 1.0,
    ):
        """
        Args:
            model_files (list(str)): List of paths to stored schnetpack model to be used in ensemble.
            force_label (str): String indicating the entry corresponding to the molecular forces
            energy_units (float, float): Conversion factor converting the energies returned by the used model back to
                                         internal MD units.
            position_units (float, float): Conversion factor for converting the system positions to the units required by
                                           the model.
            energy_label (str, optional): If provided, label is used to store the energies returned by the model to the
                                          system.
            stress_label (str, optional): If provided, label is used to store the stress returned by the model to the
                                          system (required for constant pressure simulations).
            required_properties (list): List of properties to be computed by the calculator
            property_conversion (dict(float)): Optional dictionary of conversion factors for other properties predicted by
                                               the model. Only changes the units used for logging the various outputs.
            neighbor_list (schnetpack.md.neighbor_list.MDNeighborList): Neighbor list object for determining which
                                                                        interatomic distances should be computed.
            cutoff (float): Cutoff radius for computing the neighbor interactions. If this is set to a negative number,
                            the cutoff is determined automatically based on the model (default=-1.0). Units are the distance
                            units used in the model.
            cutoff_shell (float): Second shell around the cutoff region. The neighbor lists only are recomputed when atoms
                                  move a distance further than this shell (default=1 model unit).
        """
        required_properties = required_properties + [
            energy_label,
            force_label,
            stress_label,
        ]
        self._update_required_properties(required_properties)
        self.models = self._load_models(model_files)
        super(SchnetPackEnsembleCalculator, self).__init__(
            model_file=self.models[0],
            required_properties=required_properties,
            force_label=force_label,
            energy_units=energy_units,
            position_units=position_units,
            energy_label=energy_label,
            stress_label=stress_label,
            property_conversion=property_conversion,
            neighbor_list=neighbor_list,
            cutoff=cutoff,
            cutoff_shell=cutoff_shell,
        )
        self.models = torch.nn.ModuleList(self.models)

    @staticmethod
    def _load_models(model_files: List[str]):
        """
        Load models for ensemble.

        Args:
            model_files (list(str)): List of paths to models used in ensemble.

        Returns:
             list(schnetpack.atomistic.model.AtomisticModel): Loaded models.
        """
        models = []
        for model_file in model_files:
            log.info("Loading model from {:s}".format(model_file))
            model = torch.jit.load(model_file)
            model.inference_mode = False
            models.append(model)
        return models

    @staticmethod
    def _load_model(model: AtomisticModel) -> AtomisticModel:
        """
        Dummy model loading routine.

        Args:
            model (schnetpack.atomistic.model.AtomisticModel): loaded schnetpack model.

        Returns:
            schnetpack.atomistic.model.AtomisticModel: Loaded model
        """
        return model
