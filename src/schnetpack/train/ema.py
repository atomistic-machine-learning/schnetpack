import torch
from torch_ema import ExponentialMovingAverage as TorchEMA

__all__ = ["EMA"]


class EMA(TorchEMA):
    """torch_ema's average with the update as one foreach call: s - (1 - d)(s - p) is
    lerp(s, p, 1 - d). torch_ema loops over the parameters in Python (3 kernels each),
    which cost 12% of a fast QM9 training epoch."""

    def update(self, parameters=None, gate: torch.Tensor | None = None):
        """
        Args:
            parameters: as torch_ema's
            gate: a 0-d tensor, 1.0 to update and 0.0 to leave the average as it
                is (a skipped optimizer step), decided on the GPU without a host
                sync; None updates
        """
        parameters = list(self._get_parameters(parameters))
        decay = self.decay
        if self.num_updates is not None:
            self.num_updates += 1
            decay = min(decay, (1 + self.num_updates) / (10 + self.num_updates))
        with torch.no_grad():
            if gate is None:
                torch._foreach_lerp_(self.shadow_params, parameters, 1.0 - decay)
            else:
                diff = torch._foreach_sub(parameters, self.shadow_params)
                torch._foreach_mul_(diff, gate * (1.0 - decay))
                torch._foreach_add_(self.shadow_params, diff)
