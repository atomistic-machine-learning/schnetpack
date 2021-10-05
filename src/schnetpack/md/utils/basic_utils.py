import torch
from schnetpack import representation

__all__ = ["compute_centroid", "batch_inverse", "check_triples_required"]


def compute_centroid(ensemble):
    """
    Compute centroids of the system properties (e.g. positions, momenta) given in ensemble with respect to the replica
    dimension (0). The original dimensionality of the tensor is kept for the purpose of broadcasting and logging. This
    routine is primarily intended to be used for ring polymer simulations.

    Args:
        ensemble (torch.Tensor): System property tensor (e.g. positions, momenta) of the general dimension
                                 n_replicas x n_molecules x ...

    Returns:
        torch.Tensor: Centroid averaged over the replica dimension with the general shape 1 x n_molecules x ...
    """
    centroid = torch.mean(ensemble, 0, keepdim=True)
    return centroid


def batch_inverse(tensor):
    """
    Compute the matrix inverse of a batch of square matrices. This routine is used for removing rotational motion
    during the molecular dynamics simulation.

    Args:
        tensor (torch.Tensor):  Tensor of square matrices with the shape n_batch x dim1 x dim1

    Returns:
        torch.Tensor: Tensor of the inverted square matrices with the same shape as the input tensor.
    """
    return torch.linalg.inv(tensor)


class RunningAverage:
    """
    Running average class for logging purposes. Accumulates the average of a given tensor over the course of the
    simulation.
    """

    def __init__(self):
        # Initialize running average and item count
        self.average = 0
        self.counts = 0

    def update(self, value):
        """
        Update the running average.

        Args:
            value (torch.Tensor): Tensor containing the property whose average should be accumulated.
        """
        self.average = (self.counts * self.average + value) / (self.counts + 1)
        self.counts += 1


def check_triples_required(model):
    """
    Check, if the model has a representation which requires the computation of triples between atoms.
    E.g. angular terms in Behler symmetry functions.
    Args:
        model (schnetpack.model.AtomisticModel):
    Returns:
        bool: Indicator whether model needs triples or not.
    """
    # check if DataParallel or conventional
    if hasattr(model, "module"):
        representation_layer = model.module.representation
    else:
        representation_layer = model.representation

    # If representation is standardized, extract it from standardization layer
    if isinstance(representation_layer, representation.StandardizeSF):
        representation_layer = representation_layer.SFBlock

    # Check if representation uses SFs
    if isinstance(representation_layer, representation.SymmetryFunctions):
        # Check if angular terms are present:
        if representation_layer.n_angular > 0:
            return True
        else:
            return False
    else:
        return False
