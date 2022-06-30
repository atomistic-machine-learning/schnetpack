from __future__ import annotations
from typing import Union, List, Dict, TYPE_CHECKING

import schnetpack.atomistic.response

if TYPE_CHECKING:
    from schnetpack.md import System
    from schnetpack.model import AtomisticModel
    from schnetpack.md.neighborlist_md import NeighborListMD

import torch
import logging

from schnetpack.md.calculators.base_calculator import MDCalculator
from schnetpack.md.calculators.ensemble_calculator import EnsembleCalculator
from schnetpack.md.utils import activate_model_stress

log = logging.getLogger(__name__)

__all__ = ["SchNetPackCalculator", "SchNetPackEnsembleCalculator"]


class SchNetPackCalculator(MDCalculator):
    """
    MD calculator for schnetpack models.

    Args:
        model_file (str): Path to stored schnetpack model.
        force_key (str): String indicating the entry corresponding to the molecular forces
        energy_unit (float, float): Conversion factor converting the energies returned by the used model back to
                                     internal MD units.
        position_unit (float, float): Conversion factor for converting the system positions to the units required by
                                       the model.
        neighbor_list (schnetpack.md.neighbor_list.MDNeighborList): Neighbor list object for determining which
                                                                    interatomic distances should be computed.
        energy_key (str, optional): If provided, label is used to store the energies returned by the model to the
                                      system.
        stress_key (str, optional): If provided, label is used to store the stress returned by the model to the
                                      system (required for constant pressure simulations).
        required_properties (list): List of properties to be computed by the calculator
        property_conversion (dict(float)): Optional dictionary of conversion factors for other properties predicted by
                                           the model. Only changes the units used for logging the various outputs.
        script_model (bool): convert loaded model to torchscript.
    """

    def __init__(
        self,
        model_file: str,
        force_key: str,
        energy_unit: Union[str, float],
        position_unit: Union[str, float],
        neighbor_list: NeighborListMD,
        energy_key: str = None,
        stress_key: str = None,
        required_properties: List = [],
        property_conversion: Dict[str, Union[str, float]] = {},
        script_model: bool = False,
    ):
        super(SchNetPackCalculator, self).__init__(
            required_properties=required_properties,
            force_key=force_key,
            energy_unit=energy_unit,
            position_unit=position_unit,
            energy_key=energy_key,
            stress_key=stress_key,
            property_conversion=property_conversion,
        )
        self.script_model = script_model
        self.model = self._prepare_model(model_file)
        self.neighbor_list = neighbor_list

    def _prepare_model(self, model_file: str) -> AtomisticModel:
        """
        Load an individual model.

        Args:
            model_file (str): path to model.

        Returns:
           AtomisticTask: loaded schnetpack model
        """
        return self._load_model(model_file)

    def _load_model(self, model_file: str) -> AtomisticModel:
        """
        Load an individual model, activate stress computation and convert to torch script if requested.

        Args:
            model_file (str): path to model.

        Returns:
           AtomisticTask: loaded schnetpack model
        """

        log.info("Loading model from {:s}".format(model_file))
        # load model and keep it on CPU, device can be changed afterwards
        model = torch.load(model_file, map_location="cpu").to(torch.float64)
        model = model.eval()

        if self.stress_key is not None:
            log.info("Activating stress computation...")
            model = activate_model_stress(model, self.stress_key)

        if self.script_model:
            log.info("Converting model to torch script...")
            model = torch.jit.script(model)

        log.info("Deactivating inference mode for simulation...")
        self._deactivate_postprocessing(model)

        return model

    @staticmethod
    def _deactivate_postprocessing(model: AtomisticModel) -> AtomisticModel:
        if hasattr(model, "postprocessors"):
            for pp in model.postprocessors:
                if isinstance(pp, schnetpack.transform.AddOffsets):
                    log.info("Found `AddOffsets` postprocessing module...")
                    log.info(
                        "Constant offset of {:20.11f} per atom  will be removed...".format(
                            pp.mean.detach().cpu().numpy()
                        )
                    )
        model.do_postprocessing = False
        return model

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


class SchNetPackEnsembleCalculator(EnsembleCalculator, SchNetPackCalculator):
    """
    Ensemble calculator using schnetpack models. Uncertainties are computed as the variance of all model predictions.
    """

    def __init__(
        self,
        model_files: List[str],
        force_key: str,
        energy_unit: Union[str, float],
        position_unit: Union[str, float],
        neighbor_list: NeighborListMD,
        energy_key: str = None,
        stress_key: str = None,
        required_properties: List = [],
        property_conversion: Dict[str, Union[str, float]] = {},
        script_model: bool = True,
    ):
        """
        Args:
            model_files (list(str)): List of paths to stored schnetpack model to be used in ensemble.
            force_key (str): String indicating the entry corresponding to the molecular forces
            energy_unit (float, float): Conversion factor converting the energies returned by the used model back to
                                         internal MD units.
            position_unit (float, float): Conversion factor for converting the system positions to the units required by
                                           the model.
            neighbor_list (schnetpack.md.neighbor_list.MDNeighborList): Neighbor list object for determining which
                                                                        interatomic distances should be computed.
            energy_key (str, optional): If provided, label is used to store the energies returned by the model to the
                                          system.
            stress_key (str, optional): If provided, label is used to store the stress returned by the model to the
                                          system (required for constant pressure simulations).
            required_properties (list): List of properties to be computed by the calculator
            property_conversion (dict(float)): Optional dictionary of conversion factors for other properties predicted by
                                               the model. Only changes the units used for logging the various outputs.
            script_model (bool): convert loaded model to torchscript.
        """
        super(SchNetPackEnsembleCalculator, self).__init__(
            model_file=model_files,
            required_properties=required_properties,
            force_key=force_key,
            energy_unit=energy_unit,
            position_unit=position_unit,
            neighbor_list=neighbor_list,
            energy_key=energy_key,
            stress_key=stress_key,
            property_conversion=property_conversion,
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
