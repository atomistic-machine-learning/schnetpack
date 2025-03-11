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