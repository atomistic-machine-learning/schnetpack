import pickle
import time
import random
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

    def uncertainty_estimation(self, inputs: torch.Tensor, device: torch.device = torch.device("cpu")):#, **kwargs):
        """
        Args:
            inputs:
                stacked output tensors of predicted property (e.g Energy or Forces)

            device:
                device used for calculations (default="cpu")

        Returns:
            custom uncertainty estimation
        """
        raise NotImplementedError


class SimpleEnsembleAverage(EnsembleAverageStrategy):
    """
    Simply ensemble average function, mostly for testing only
    Model outputs are dropped if exceeding mean +/- factor*standarddeviation
    """
    def __init__(self,
                 mode: str = "single",
                 criteria: Optional[torch.Tensor] = torch.tensor([1.])):
        """
        Args:
            criteria: torch.Tensor
                factor to be multiplied with standard deviation default value 1
        """

        self.criteria = torch.Tensor([criteria])
        self.mode = mode
        super().__init__()

    def uncertainty_estimation(
            self,
            inputs: torch.Tensor,
            device: torch.device = torch.device("cpu")
            ):
        """
        Args:
            inputs:
                stacked output tensors of predicted property (e.g Energy or Forces)

            device:
                device used for calculations (default="cpu")

        Returns:
            ...
        """

        num_dim = inputs.size()[-1]
        criteria = self.criteria.to(device)
        mean = torch.mean(inputs, dim=0)
        std = torch.std(inputs, dim=0) * criteria

        condition = torch.logical_and(inputs >= (mean - std), inputs <= (mean + std))

        if self.mode == "single":
            if num_dim > 1:
                num_models, num_atoms, num_dim = inputs.size()
                N = torch.tensor(condition.size()[1] * condition.size()[2], device=device)

                new_dim = 0
                for n in range(condition.size()[0]):

                    if torch.round(condition[n].sum()) > torch.round(N/2):
                        condition[n].fill_(True)
                        new_dim +=1
                    else:
                        condition[n].fill_(False)

                processed_input = torch.reshape(inputs[condition],(new_dim,num_atoms,num_dim))

            else:
                processed_input = inputs[condition]

        if self.mode == "batchwise":
            if inputs.dim() == 2:
                num_models, batchsize = inputs.size()
                N = torch.tensor(num_models,device=device)
                idx = tuple(range(num_models))
                new_dim = 0
                for n in range(batchsize):

                    if torch.round(condition[idx,n].sum()) >= torch.round(N/2):
                        condition[idx,n].fill_(True)
                        new_dim +=1
                    else:
                        condition[idx,n].fill_(False)

                processed_input = torch.reshape(inputs[condition],
                                                (int(inputs[condition].size()[0] / (batchsize)),
                                                batchsize))

            if inputs.dim() == 3:
                num_models, (num_atoms), num_dim = inputs.size()
                mean = torch.mean(inputs, dim=0)
                std = torch.std(inputs, dim=0) * criteria

                condition = torch.logical_and(inputs >= (mean-std), inputs <= (mean + std))
                
                N = torch.tensor(condition.size()[1] * condition.size()[2], device=device)

                new_dim = 0
                for n in range(condition.size()[0]):

                    if torch.round(condition[n].sum()) > torch.round(N/2):
                        condition[n].fill_(True)
                        new_dim +=1
                    else:
                        condition[n].fill_(False)

                processed_input = torch.reshape(inputs[condition], (new_dim, num_atoms, num_dim))

        return torch.mean(processed_input, dim=0)


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
                m = torch.load(m)
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
                    #device=self.device
                ).detach().cpu().numpy()
            else:
                results.update(self._default_average_strategy(prop, stacked_model_results))

        self.results = results
