from torch import Tensor
from torch import nn
import torch.nn.functional as F


class ScaledMSELoss(nn.MSELoss):

    def __init__(self, size_average=None, reduce=None, reduction: str = "mean") -> None:
        super().__init__(size_average=size_average, reduce=reduce, reduction=reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:

        target_norm = target.norm(p=2, dim=1, keepdim=True) + 1e-6
        target = target / target_norm
        input = input / target_norm

        return F.mse_loss(input, target, reduction=self.reduction)


class WeightedMSELoss(nn.MSELoss):

    def __init__(self, size_average=None, reduce=None, reduction: str = "mean") -> None:
        super().__init__(size_average=size_average, reduce=reduce, reduction=reduction)

    def forward(
        self, input: Tensor, target: Tensor, weight_target: Tensor, _idx_m: Tensor
    ) -> Tensor:
        _idx_m_hess = _idx_m.repeat_interleave(3)
        residual = target - input
        flat_residual = residual.flatten()
        loss = 0.0
        for idx_m in range(max(_idx_m_hess).item() + 1):
            spl_residual = flat_residual[_idx_m_hess == idx_m]
            spl_weight = weight_target[_idx_m_hess == idx_m]

            loss += spl_residual.T @ spl_weight @ spl_residual

        return loss
