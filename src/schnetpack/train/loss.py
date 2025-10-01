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
