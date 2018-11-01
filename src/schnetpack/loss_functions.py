import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from schnetpack.config_model import Hyperparameters


class MSELoss(_Loss, Hyperparameters):
    r"""
    Custom MSELoss
    """
    def __init__(self, size_average=None, reduce=None, reduction='elementwise_mean',
                 input_key=None, target_key=None):

        Hyperparameters.__init__(self, locals())
        _Loss.__init__(self, size_average, reduce, reduction)
        self.input_key = input_key
        self.target_key = target_key

    def forward(self, input, target):
        i = input if not self.input_key else input[self.input_key]
        t = target if not self.target_key else target[self.target_key]
        return F.mse_loss(i, t, reduction=self.reduction)