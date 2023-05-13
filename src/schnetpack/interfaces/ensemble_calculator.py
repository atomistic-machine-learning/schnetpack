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

import torch
from torch import nn
from schnetpack.units import convert_units
from schnetpack.model import NeuralNetworkPotential

from typing import List, Optional, Dict, Callable, Union

class EnsembleAverageStrategy:

    '''
    base class for Ensemble Average Stragies
    '''

    def __init__(self):

        pass


    def uncertainty_estimation(self,inputs,device,**kwargs):
        
        '''
        Args:
            inputs: kwargs["inputs"] torch.Tensor
            stacked output tensors of predicted property (e.g Energy or Forces)

            device: kwargs["device"] torch.device
            device used for calculations (default="cpu") 

        Returns:
            ensemble averages based on user defined uncertainty estimation 
        
        '''
        raise NotImplementedError



class SimpleEnsembleAverage(EnsembleAverageStrategy):

    '''
    simply ensemble average function, mostly for testing only
    Model outputs are dropped if exceeding mean +/- factor*standarddeviation
    
    '''
    def __init__(self,
                 mode="single",
                 batchsize = torch.tensor([1],dtype=torch.int8),
                 criteria: Optional[torch.Tensor] = torch.tensor([1.])):

        self.criteria = torch.Tensor([criteria])
        self.mode = mode
        self.batchsize = batchsize 

    def uncertainty_estimation(
            self,
            inputs,
            device
            ):

        '''
        inputs: torch.Tensor
            stacked output tensors of predicted property (e.g Energy or Forces)

        device: torch.device
            device used for calculations (default="cpu")  

        criteria: torch.Tensor
            factor to be multiplied with standard deviation default value 1
        '''

        num_dim = inputs.size()[-1]
        batchsize = self.batchsize.to(device)
        criteria = self.criteria.to(device)
        mean = torch.mean(inputs,dim=0)
        std = torch.std(inputs,dim=0) * criteria

        condition = torch.logical_and(inputs >= (mean-std), inputs <= (mean + std))

        if self.mode == "single":
            if num_dim > 1:
                num_models, num_atoms, num_dim = inputs.size()
                N = torch.tensor(condition.size()[1] * condition.size()[2],device=device)

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


                num_models, (num_atoms) , num_dim = inputs.size()
                mean = torch.mean(inputs,dim=0)
                std = torch.std(inputs,dim=0) * criteria

                condition = torch.logical_and(inputs >= (mean-std), inputs <= (mean + std))
                
                N = torch.tensor(condition.size()[1] * condition.size()[2],device=device)

                new_dim = 0
                for n in range(condition.size()[0]):

                    if torch.round(condition[n].sum()) > torch.round(N/2):
                        condition[n].fill_(True)
                        new_dim +=1
                    else:
                        condition[n].fill_(False)

                processed_input = torch.reshape(inputs[condition],(new_dim,num_atoms,num_dim))



        return torch.mean(processed_input,dim=0)



class EnsembleCalculator(Calculator):
    """
    Calculator for neural network models for ensemble calculations.
    Requires multiple models
    """

    energy = "energy"
    forces = "forces"
    stress = "stress"

    def __init__(
        self,
        model_file: Union[List[str],List[nn.Module]],
        atoms_converter,
        device="cpu",
        auxiliary_output_modules=None,
        energy_key="energy",
        force_key="forces",
        stress_key="stress",
        energy_unit="kcal/mol",
        position_unit="Ang",
        dtype = torch.float32,
        ensemble_average_strategy : Optional[EnsembleAverageStrategy] = None):
        """
        model_file: str
            path to trained models
            has to be a list of paths
            OR
            list of preloaded models

        atoms_converter: schnetpack.interfaces.AtomsConverter
            Class used to convert ase Atoms objects to schnetpack input

        device: torch.device
            device used for calculations (default="cpu")

        energy_key: str
            name of energies in model (default="energy")

        force_key: str
            name of forces in model (default="forces")

        energy_unit: str, float
            energy units used by model (default="eV")

        position_unit: str, float
            position units used by model (default="Angstrom")

        ensemble_average_strategy : User defined class to
            to calculate ensemble average
        """

        super(EnsembleCalculator, self).__init__()

        self.device = device
        self.atoms_converter = atoms_converter
        self.model_results = None
        self.model_file = model_file 
        self.dtype = dtype
        self.ensemble_average_strategy = ensemble_average_strategy

        # TODO if auxiliary_output_modules must have same len as model list
        self.auxiliary_output_modules = auxiliary_output_modules or []

        self.energy_key = energy_key
        self.force_key = force_key
        self.stress_key = stress_key

        # set up basic conversion factors
        self.energy_conversion = convert_units(energy_unit, "eV")
        self.position_conversion = convert_units(position_unit, "Angstrom")

        # Unit conversion to default ASE units
        self.property_units = {
            self.energy: self.energy_conversion,
            self.forces: self.energy_conversion / self.position_conversion,
            self.stress: self.energy_conversion / self.position_conversion**3,
        }

        self._load_model(model_file)

    def _load_model(self, model_file):
        
        if isinstance(model_file[0],NeuralNetworkPotential):

            # TODO
            model = nn.ModuleDict(
                {"model"+str(n): model_file[n].to(self.device) for n in range(len(model_file))}
            )

        # TODO test if better with nn.ModuleDict or nn.ModuleList, idea behind dict was that in the 
        # specific user uncertainity calculation function an uncertainity for every model could be provided
        else:
            model = nn.ModuleDict(
                {"model"+str(n): torch.load(model_file[n],map_location=self.device) for n in range(len(model_file))}
            )
            #for n in range(len(model_file)):
            #    model["model"+str(n)].output_modules[1].calc_stress = True
        # for now outcommented because will be checked later
        #for auxiliary_output_module in self.auxiliary_output_modules:
            #self.model.output_modules.insert(1, auxiliary_output_module)
        self.model = model.eval()
        self.model.to(device=self.device,dtype=self.dtype)

    def _update_model_inputs(self, atoms):
        self.model_inputs = self.atoms_converter(atoms)

    def _requires_calculation(self, property_keys, atoms):
        if self.model_results is None:
            return True
        for name in property_keys:
            if name not in self.model_results:
                return True
        if len(self.atoms) != len(atoms):
            return True
        for atom, atom_ref in zip(atoms, self.atoms):
            if atom != atom_ref:
                return True
            
    def get_forces(self, atoms, fixed_atoms_mask=None):
        if self._requires_calculation(
            property_keys=[self.energy_key, self.force_key], atoms=atoms
        ):
            self.calculate(atoms)
        f = (
            self.model_results[self.force_key]
            * self.property_units[self.forces]
        )
        if fixed_atoms_mask is not None:
            f[fixed_atoms_mask] *= 0.0
        return f

    def get_potential_energy(self, atoms):
        if self._requires_calculation(property_keys=[self.energy_key], atoms=atoms):
            self.calculate(atoms)
        return (
            self.model_results[self.energy_key]
            * self.property_units[self.energy]
        )

    def calculate(
            self,
            atoms):

        inputs = deepcopy(self.atoms_converter(atoms))

        self.accumulated_results = {key: self.model[key](deepcopy(inputs)) for key in list(self.model.keys())  }
        self.model_results = {}

        # property names (keys)
        props = list(self.accumulated_results[list(self.accumulated_results.keys())[0]].keys())
        for p in props:

            
            # applying user defined ensemble average strategy based on uncertainty estimation
            stacked_prop = torch.stack([ (self.accumulated_results[result][p] *torch.tensor(random.uniform(1.0,1.10))) for result in self.accumulated_results])
            
            if self.ensemble_average_strategy:

                stacked_prop = self.ensemble_average_strategy.uncertainty_estimation(
                    inputs=stacked_prop,
                    device=self.device
                    ) 
            
                self.model_results[p] = stacked_prop.detach().cpu().numpy()

            else:

                # tmp fix for ase-gui error: setting an rray element with a sequence. The requested array would exceed
                # the maximum number of dimension of 1
                # only effects energy

                if stacked_prop.dim() == 2:

                    self.model_results[p] = torch.mean(stacked_prop,dim=0)[0].detach().cpu().numpy()
                    self.model_results[p+"_std"] = torch.std(stacked_prop,dim=0)[0].detach().cpu().numpy()
                else:
                   self.model_results[p] = torch.mean(stacked_prop,dim=0).detach().cpu().numpy()
                   self.model_results[p+"_std"] = torch.std(stacked_prop,dim=0).detach().cpu().numpy() 


        self.atoms = atoms.copy()
    





# class Calculator:

#     """
#     Base class for ase calculators.
#     """

#     def __init__(self):
#         self.results = None
#         self.atoms = None

#     def calculation_required(self, atoms, properties=None):
#         if self.atoms is None or not self.atoms == atoms:
#             return True
#         return False

#     def get_forces(
#         self,
#         atoms,
#     ):
#         if self.calculation_required(atoms):
#             self.calculate(atoms)
#         return self.results["forces"]

#     def get_potential_energy(
#         self,
#         atoms,
#     ):
#         if self.calculation_required(atoms):
#             self.calculate(atoms)
#         return self.results["energy"]

#     # TODO not implemented and I am not sure how to implement it
#     #if view(mol) or Trajcetory writer of ASE called, this causes an error, if not uncommented
#     def get_stress(
#         self,
#         atoms,
#     ):
#         if self.calculation_required(atoms):
#             self.calculate(atoms)
#         return self.results["stress"]

#     def calculate(self, atoms):
#         pass