import logging
from copy import deepcopy
import numpy as np
from math import sqrt
from os.path import isfile

from ase.optimize.optimize import Dynamics
from ase.parallel import world, barrier
from ase.io import write
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
import ase

import torch
from torch import nn
from schnetpack.units import convert_units
from schnetpack.model import NeuralNetworkPotential
import schnetpack
from schnetpack.interfaces.ase_interface import AtomsConverter, SpkCalculator

from typing import List, Optional, Dict, Callable, Union


class AtomsConverterError(Exception):
    pass


class EnsembleAverageStrategy:
    """
    base class for Ensemble Average Strategies
    """
    def __init__(self):
        pass

    def correct_dimension(self, num_atoms, inputs:torch.Tensor):

        """
        Args:
            inputs:
                stacked output tensors of predicted property (e.g Energy or Forces)
            num_atoms:
                number of atoms in mol. Needed for correct dimension reshaping

        Returns:
            correct dimension for reshaping the inputs accordingly
            [num_models, num_mols in batch, num_atoms in mol, property dimension]
            this way no distinction between single point and batchwise optimization has to be done in Ensemble
        """

        n_models = inputs.shape[0]
        batch_size = num_atoms.size()[0]
        n_atoms = num_atoms.unique()[0].item()
        property_dim = inputs.shape[-1]

        if len(inputs.shape) == 2:
            n_atoms = 1
            property_dim = 1

        return (n_models,batch_size,n_atoms,property_dim)

    def uncertainty_estimation(self, inputs: torch.Tensor, num_atoms):
        """
        Args:
            inputs:
                stacked output tensors of predicted property (e.g Energy or Forces)
            num_atoms:
                number of atoms in mol. Needed for correct dimension reshaping

        Returns:
            custom uncertainty estimation
        """
        raise NotImplementedError
    
    def fallback(self, conditions):
        if conditions.sum() == 0:
            logging.info(
                f"All models fail to predict properties with the given filter criteria: {self.filter_criteria.item()}\n"
                f"Please consider to change the filter criteria or" 
                f"to lower the applied model drop threshold of currently {self.model_drop_threshold * 100} % "
                f"Per default now only the first model is considered for the current step"
            )
            conditions[0] = True
        else:
            pass

        return conditions


class SimpleEnsembleAverage(EnsembleAverageStrategy):
    """

    Simply ensemble average class
    Model output is dropped if output exceeds mean +/- factor*standarddeviation
    Models are dropped if number of dropped model outputs exceeds threshold

    """
    def __init__(
            self,
            filter_criteria : Optional[float] = 1.,
            model_drop_threshold : Optional[float] = 0.5
            ):
        """
        Args:
            filter_criteria:
                numerical criteria applied to inputs
            model_drop_threshold:
                threshold when to drop specific model (default = 0.5)
        """
        self.filter_criteria = filter_criteria
        self.model_drop_threshold = model_drop_threshold

        super().__init__()

    def uncertainty_estimation(
            self,
            num_atoms,
            inputs: torch.Tensor
            ):
        """
        Args:
            inputs:
                stacked output tensors of predicted property (e.g Energy or Forces)
        Returns:
            ...
        """

        n_models, batch_size, n_atoms, property_dim = self.correct_dimension(num_atoms,inputs)

        # consistent with _default_average_strategy, detach avoids num precision error in mean
        inputs = torch.reshape(inputs, (n_models, batch_size, n_atoms, property_dim)).detach().cpu().numpy()
        conditions = np.zeros(shape=(n_models, batch_size), dtype=bool)

        for batch in range(batch_size):
            mean = np.mean(inputs[:, batch, :, :], axis=0)
            std = np.std(inputs[:, batch, :, :], axis=0) * self.filter_criteria

            for model in range(n_models):

                check = np.logical_and(
                    inputs[model, batch, :, :] >= (mean - std),
                    inputs[model, batch, :, :] <= (mean + std)).sum() > round(n_atoms * property_dim * self.model_drop_threshold)
                
                conditions[model, batch] = check

        # needed for batch optimization mode
        if batch_size > 1:
            conditions = conditions.sum(axis=1) >= round(batch_size * self.model_drop_threshold)

        # check if all models fail
        conditions = self.fallback(conditions)

        processed_input = inputs[conditions].reshape(
                            conditions.sum().item(),
                            batch_size*n_atoms,
                            property_dim)
        mean = np.squeeze(np.mean(processed_input, axis=0))
        return mean


class EnsembleCalculator(SpkCalculator):
    """
    Calculator for neural network models for ensemble calculations.
    Requires multiple models
    """
    # TODO: maybe calculate function should calculate all properties always
    #       doc string
    def __init__(
            self,
            model: Union[List[str], List[nn.Module]],
            neighbor_list: schnetpack.transform.Transform,
            energy_key: str = "energy",
            force_key: str = "forces",
            stress_key: Optional[str] = None,
            energy_unit: Union[str, float] = "kcal/mol",
            position_unit: Union[str, float] = "Angstrom",
            device: Union[str, torch.device] = "cpu",
            dtype: torch.dtype = torch.float32,
            converter: callable = AtomsConverter,
            transforms: Union[
                schnetpack.transform.Transform, List[schnetpack.transform.Transform]
            ] = None,
            additional_inputs: Dict[str, torch.Tensor] = None,
            auxiliary_output_modules: Optional[List] = None,
            ensemble_average_strategy: Optional[EnsembleAverageStrategy] = None,
            **kwargs,
    ):
        """
        Args:
            model_file: path to trained models OR list of preloaded models
            neighbor_list (schnetpack.transform.Transform): SchNetPack neighbor list
            energy_key (str): name of energies in model (default="energy")
            force_key (str): name of forces in model (default="forces")
            stress_key (str): name of stress tensor in model. Will not be computed if set to None (default=None)
            energy_unit (str, float): energy units used by model (default="kcal/mol")
            position_unit (str, float): position units used by model (default="Angstrom")
            device (torch.device): device used for calculations (default="cpu")
            dtype (torch.dtype): select model precision (default=float32)
            converter (callable): converter used to set up input batches
            transforms (schnetpack.transform.Transform, list): transforms for the converter. More information
                can be found in the AtomsConverter docstring.
            additional_inputs (dict): additional inputs required for some transforms in the converter.
            ensemble_average_strategy : User defined class to calculate ensemble average
            **kwargs: Additional arguments for basic ase calculator class
        """

        SpkCalculator.__init__(
            self,
            model=model,
            neighbor_list=neighbor_list,
            energy_key=energy_key,
            force_key=force_key,
            stress_key=stress_key,
            energy_unit=energy_unit,
            position_unit=position_unit,
            device=device,
            dtype=dtype,
            converter=converter,
            transforms=transforms,
            additional_inputs=additional_inputs,
            auxiliary_output_modules=auxiliary_output_modules,
            **kwargs
        )

        self.ensemble_average_strategy = ensemble_average_strategy

    def _load_model(self, model):
        ensemble_model = nn.ModuleDict()
        for model_idx, m in enumerate(model):
            if type(m) is str:
                m = torch.load(m, map_location="cpu")
            for auxiliary_output_module in self.auxiliary_output_modules:
                m.output_modules.insert(1, auxiliary_output_module)
            ensemble_model.update({"model{}".format(model_idx): m})
        ensemble_model = ensemble_model.eval()
        return ensemble_model

    def _default_average_strategy(self, prop, stacked_model_results):

        mean = torch.mean(stacked_model_results, dim=0) * self.property_units[prop]
        std = torch.std(stacked_model_results, dim=0) * self.property_units[prop]

        results = {}
        if prop == self.energy or prop == self.stress:
            # ase calculator should return scalar energy
            results[prop] = mean.detach().cpu().numpy().item()
            results[prop + "_std"] = std.detach().cpu().numpy().item()
        else:
            results[prop] = mean.detach().cpu().numpy()
            results[prop + "_std"] = std.detach().cpu().numpy()
        return results

    def _calculate(self, atoms: Union[ase.Atoms, List[ase.Atoms]], properties: List[str]) -> None:
        inputs = self.converter(atoms)

        # empty dict for model results
        model_results = {}
        for prop in properties:
            model_prop = self.property_map[prop]
            model_results[model_prop] = []

        # get results of all models
        for model_key in self.model.keys():
            model = self.model[model_key]
            x = deepcopy(inputs)
            predictions = model(x)
            for prop in properties:
                model_prop = self.property_map[prop]
                if model_prop in predictions:
                    model_results[model_prop].append(predictions[model_prop])
                else:
                    raise AtomsConverterError(
                        "'{:s}' is not a property of your model. Please "
                        "check the model "
                        "properties!".format(prop)
                    )

        results = {}
        for prop in properties:
            model_prop = self.property_map[prop]
            stacked_model_results = torch.stack(model_results[model_prop])

            if self.ensemble_average_strategy:
                results[prop] = self.ensemble_average_strategy.uncertainty_estimation(
                    inputs=stacked_model_results,
                    num_atoms=x["_n_atoms"]
                ) * self.property_units[prop]
            else:
                results.update(self._default_average_strategy(prop, stacked_model_results))

        self.results = results
