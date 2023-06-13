import logging
import numpy as np
import torch
from typing import List, Optional, Dict, Callable, Union


class EnsembleAverageStrategy:
    """
    Base class for Ensemble Average Strategies
    """
    def __init__(self):
        pass

    def uncertainty_estimation(self, inputs: torch.Tensor, num_atoms: torch.Tensor):
        """
        Calculates the mean and standard deviation of the outputs regarding all models in the ensemble.

        Args:
            inputs: stacked output tensors of predicted property, e.g., energy or forces.
            num_atoms: umber of atoms in the molecules. Used for dimension reshaping.
        """
        # consistent with _default_average_strategy, detach avoids num precision error in mean
        processed_inputs = inputs.detach().cpu().numpy()

        mean = np.squeeze(np.mean(processed_inputs, axis=0))
        std = np.squeeze(np.std(processed_inputs, axis=0))

        return mean, std


class SimpleEnsembleAverage(EnsembleAverageStrategy):
    """
    Simply ensemble average class
    Model output is dropped if output exceeds mean +/- factor*standarddeviation
    Models are dropped if number of dropped model outputs exceeds threshold
    """

    def __init__(
            self,
            filter_criteria: Optional[float] = 1.,
            model_drop_threshold: Optional[float] = 0.5
            ):
        """
        Args:
            filter_criteria: numerical criteria applied to inputs
            model_drop_threshold: threshold when to drop specific model (default = 0.5)
        """
        self.filter_criteria = filter_criteria
        self.model_drop_threshold = model_drop_threshold

        super().__init__()

    def correct_dimension(self, num_atoms, inputs: torch.Tensor):
        """
        Reshaping of the inputs as (num_models, num_mols in batch, num_atoms in mol, property dimension)
        this way no distinction between single molecule optimization and batchwise optimization
        has to be done in the Ensemble Average Strategy

        Args:
            inputs: stacked output tensors of predicted property (e.g Energy or Forces)
            num_atoms: number of atoms in mol. Needed for correct dimension reshaping
        """

        n_models = inputs.shape[0]
        batch_size = num_atoms.size()[0]
        n_atoms = num_atoms.unique()[0].item()
        property_dim = inputs.shape[-1]

        if len(inputs.shape) == 2:
            n_atoms = 1
            property_dim = 1

        return n_models, batch_size, n_atoms, property_dim

    def uncertainty_estimation(self, inputs: torch.Tensor, num_atoms: torch.Tensor = None):
        """
        Args:
            inputs: stacked output tensors of predicted property, e.g., energy or forces.
            num_atoms: umber of atoms in the molecules. Used for dimension reshaping.
        """

        n_models, batch_size, n_atoms, property_dim = self.correct_dimension(num_atoms, inputs)

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
        std = np.squeeze(np.std(processed_input, axis=0))
        return mean, std

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
