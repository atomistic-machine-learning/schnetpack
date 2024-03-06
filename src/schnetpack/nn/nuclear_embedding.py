import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from schnetpack.electron_configurations import electron_config


class NuclearEmbedding(nn.Module):
    """
    Embedding which maps scalar nuclear charges Z to vectors in a
    (num_features)-dimensional feature space. The embedding consists of a freely
    learnable parameter matrix [Zmax, num_features] and a learned linear mapping
    from the electron configuration to a (num_features)-dimensional vector. The
    latter part encourages alchemically meaningful representations without
    restricting the expressivity of learned embeddings.
    Using complexe nuclear embedding can have negative impact on the model performance, when spin charge embedding is activated
    Negative performance in regard of the duration until the model converges. The model will converge to a lower value, but the duration is longer.

    Arguments:
        num_features (int):
            Dimensions of feature space.
        Zmax (int):
            Maximum nuclear charge of atoms. The default is 100, so all
            elements up to Fermium (Fe) (Z=100) are supported. 
            Can be kept at the default value (has minimal memory impact).
        zero_init (bool):
            If True, initialize the embedding with zeros. Otherwise, use
            uniform initialization.
    """

    def __init__(
        self,
        num_features: int,
        Zmax: int = 100, 
        zero_init: bool = True
        ) -> None:
        """ Initializes the NuclearEmbedding class. """
        super(NuclearEmbedding, self).__init__()
        self.num_features = num_features
        self.register_buffer("electron_config", torch.tensor(electron_config))
        self.register_parameter(
            "element_embedding", nn.Parameter(torch.Tensor(Zmax, self.num_features))
        )
        self.register_buffer(
            "embedding", torch.Tensor(Zmax, self.num_features), persistent=False
        )
        self.config_linear = nn.Linear(
            self.electron_config.size(1), self.num_features, bias=False
        )
        self.reset_parameters(zero_init)

    def reset_parameters(self, zero_init: bool = True) -> None:
        """ Initialize parameters. """
        if zero_init:
            nn.init.zeros_(self.element_embedding)
            nn.init.zeros_(self.config_linear.weight)
        else:
            nn.init.uniform_(self.element_embedding, -math.sqrt(3), math.sqrt(3))
            nn.init.orthogonal_(self.config_linear.weight)

    def train(self, mode: bool = True) -> None:
        """ Switch between training and evaluation mode. """
        super(NuclearEmbedding, self).train(mode=mode)
        if not self.training:
            with torch.no_grad():
                self.embedding = self.element_embedding + self.config_linear(
                    self.electron_config
                )

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        """
        Assign corresponding embeddings to nuclear charges.
        N: Number of atoms.
        num_features: Dimensions of feature space.

        Arguments:
            Z (LongTensor [N]):
                Nuclear charges (atomic numbers) of atoms.

        Returns:
            x (FloatTensor [N, num_features]):
                Embeddings of all atoms.
        """
        if self.training:  # during training, the embedding needs to be recomputed
            self.embedding = self.element_embedding + self.config_linear(
                self.electron_config
            )
        if self.embedding.device.type == "cpu":  # indexing is faster on CPUs
            return self.embedding[Z]
        else:  # gathering is faster on GPUs
            return torch.gather(
                self.embedding, 0, Z.view(-1, 1).expand(-1, self.num_features)
            )
