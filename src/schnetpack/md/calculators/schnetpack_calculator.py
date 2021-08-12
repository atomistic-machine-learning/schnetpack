from __future__ import annotations
from typing import Union, List, Dict, TYPE_CHECKING

import schnetpack.atomistic.response

if TYPE_CHECKING:
    from schnetpack.md import System
    from schnetpack.atomistic.model import AtomisticModel
    from schnetpack.md.neighborlist_md import NeighborListMD

import torch
import logging

from schnetpack.md.calculators.base_calculator import MDCalculator, MDCalculatorError
from schnetpack.md.calculators.ensemble_calculator import EnsembleCalculator
from schnetpack.md.neighborlist_md import ASENeighborListMD

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
        script_model (bool): convert loaded model to torchscript.
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
        script_model: bool = False,
    ):
        super(SchnetPackCalculator, self).__init__(
            required_properties=required_properties,
            force_label=force_label,
            energy_units=energy_units,
            position_units=position_units,
            energy_label=energy_label,
            stress_label=stress_label,
            property_conversion=property_conversion,
        )
        self.script_model = script_model
        self.model = self._prepare_model(model_file)

        # Set up the neighbor list:
        self.neighbor_list = self._init_neighbor_list(
            neighbor_list, cutoff, cutoff_shell
        )

    def _prepare_model(self, model_file: str) -> AtomisticModel:
        """
        Load an individual model.

        Args:
            model_file (str): path to model.

        Returns:
           AtomisticModel: loaded schnetpack model
        """
        return self._load_model(model_file)

    def _load_model(self, model_file: str) -> AtomisticModel:
        """
        Load an individual model, activate stress computation and convert to torch script if requested.

        Args:
            model_file (str): path to model.

        Returns:
           AtomisticModel: loaded schnetpack model
        """

        log.info("Loading model from {:s}".format(model_file))
        model = torch.load(model_file).to(torch.float32)
        model = model.eval()

        if self.stress_label is not None:
            log.info("Activating stress computation...")
            model = self._activate_stress(model)

        if self.script_model:
            log.info("Converting model to torch script...")
            model = model.to_torchscript(None, "script")

        log.info("Deactivating inference mode for simulation...")
        self._deactivate_inference_mode(model)

        return model

    @staticmethod
    def _deactivate_inference_mode(model: AtomisticModel) -> AtomisticModel:
        if hasattr(model, "postprocessors"):
            for pp in model.postprocessors:
                if isinstance(pp, schnetpack.transform.AddOffsets):
                    log.info("Found `AddOffsets` postprocessing module...")
                    log.info(
                        "Constant offset of {:20.11f} will be removed...".format(
                            pp.mean.detach().cpu().numpy()[0]
                        )
                    )
        model.inference_mode = False
        return model

    @staticmethod
    def _activate_stress(model: AtomisticModel) -> AtomisticModel:
        """
        Activate stress computations for simulations in cells.

        Args:
            model (AtomisticModel): loaded schnetpack model for which stress computation should be activated.

        Returns:

        """
        print(model.output_modules[1].calc_stress)
        # model.output_modules[1].calc_stress = True
        stress = False
        for module in model.output_modules:
            if isinstance(module, schnetpack.atomistic.response.Forces):
                if hasattr(module, "calc_stress"):
                    module.calc_stress = True
                    stress = True
        if not stress:
            raise MDCalculatorError("Failed to activate stress computation")

        print(model.output_modules[1].calc_stress)
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
        neighbors = self.neighbor_list.get_neighbors(inputs)
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
        script_model: bool = True,
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
            script_model (bool): convert loaded model to torchscript.
        """
        super(SchnetPackEnsembleCalculator, self).__init__(
            model_file=model_files,
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
            script_model=script_model,
        )
        # Update the required properties
        self._update_required_properties()
        # Convert list of models to module list
        self.models = torch.nn.ModuleList(self.model)

    def _prepare_model(self, model_files: List[str]) -> List[AtomisticModel]:
        """
        Load multiple models.

        Args:
            model_files (list(str)): List of stored models.

        Returns:
            list(AtomisticModel): list of loaded models.
        """
        return [self._load_model(model_file) for model_file in model_files]
