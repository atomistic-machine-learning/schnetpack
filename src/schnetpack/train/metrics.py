import torch
from torchmetrics import Metric
from torchmetrics.functional.regression.mae import (
    _mean_absolute_error_compute,
    _mean_absolute_error_update,
)

from typing import Optional, Tuple

__all__ = ["TensorDiagonalMeanAbsoluteError"]


class TensorDiagonalMeanAbsoluteError(Metric):
    """
    Custom torch metric for monitoring the mean absolute error on the diagonals and offdiagonals of tensors, e.g.
    polarizability.
    """

    is_differentiable = True
    higher_is_better = False
    sum_abs_error: torch.Tensor
    total: torch.Tensor

    def __init__(
        self,
        diagonal: Optional[bool] = True,
        diagonal_dims: Optional[Tuple[int, int]] = (-2, -1),
        dist_sync_on_step=False,
    ) -> None:
        """

        Args:
            diagonal (bool): If true, diagonal values are used, if False off-diagonal.
            diagonal_dims (tuple(int,int)): axes of the square matrix for which the diagonals should be considered.
            dist_sync_on_step (bool): synchronize.
        """
        # call `self.add_state`for every internal state that is needed for the metrics computations
        # dist_reduce_fx indicates the function that should be used to reduce
        # state from multiple processes
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("sum_abs_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

        self.diagonal = diagonal
        self.diagonal_dims = diagonal_dims
        self._diagonal_mask = None

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """
        Update the metric.

        Args:
            preds (torch.Tensor): network predictions.
            target (torch.Tensor): reference values.
        """
        # update metric states
        preds = self._input_format(preds)
        target = self._input_format(target)

        sum_abs_error, n_obs = _mean_absolute_error_update(preds, target)

        self.sum_abs_error += sum_abs_error
        self.total += n_obs

    def compute(self) -> torch.Tensor:
        """
        Compute the final metric.

        Returns:
            torch.Tensor: mean absolute error of diagonal or offdiagonal elements.
        """
        # compute final result
        return _mean_absolute_error_compute(self.sum_abs_error, self.total)

    def _input_format(self, x) -> torch.Tensor:
        """
        Extract diagonal / offdiagonal elements from input tensor.

        Args:
            x (torch.Tensor): input tensor.

        Returns:
            torch.Tensor: extracted and flattened elements (diagonal / offdiagonal)
        """
        if self._diagonal_mask is None:
            self._diagonal_mask = self._init_diag_mask(x)
        return x.masked_select(self._diagonal_mask)

    def _init_diag_mask(self, x: torch.Tensor) -> torch.Tensor:
        """
        Initialize the mask for extracting the diagonal elements based on the given axes and the shape of the
        input tensor.

        Args:
            x (torch.Tensor): input tensor.

        Returns:
            torch.Tensor: Boolean diagonal mask.
        """
        tensor_shape = x.shape
        dim_0 = tensor_shape[self.diagonal_dims[0]]
        dim_1 = tensor_shape[self.diagonal_dims[1]]

        if not dim_0 == dim_1:
            raise AssertionError(
                "Found different size for diagonal dimensions, expected square sub matrix."
            )

        view = [1 for _ in tensor_shape]
        view[self.diagonal_dims[0]] = dim_0
        view[self.diagonal_dims[1]] = dim_1

        diag_mask = torch.eye(dim_0, device=x.device, dtype=torch.long).view(view)

        if self.diagonal:
            return diag_mask == 1
        else:
            return diag_mask != 1
