"""
Inference on spk batches: the one place a driver's model call goes through.

A :class:`Calculator` owns everything between "here is a batch" and "here are
the model outputs": moving the batch to the model's device and dtype, the
neighbor list, the gradient policy and the model call itself. Drivers own
none of it; they hand the calculator a batch and read the outputs.

The calculator never writes into the batch it is given. The neighbor list
and the model both write into their inputs, so they get a shallow copy, and
the keys they add (neighbor lists, distance vectors, outputs) never reach the
driver's batch. That is what keeps a moving structure free of stale derived
keys: they only ever exist on the copy built for one call.
"""

from typing import Any, Callable, Dict, Mapping, Optional

import torch
from torch import nn

__all__ = ["Calculator", "as_calculator"]


class Calculator:
    """
    Run a model on spk batches: ``calculator(batch) -> outputs``.

    Stateless between calls apart from whatever the neighbor list caches
    (a skin-based list keeps the lists it built); :meth:`reset` clears that
    between runs. No output caching: an integrator such as Heun evaluates
    different structures within one step, so a cache would never hit.
    """

    def __init__(
        self,
        model: Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]],
        neighbor_list: Optional[Callable[[Dict], Dict]] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        enable_grad: bool = False,
    ):
        """
        Args:
            model: batch -> outputs, e.g. a
                :class:`~schnetpack.model.NeuralNetworkPotential`
            neighbor_list: batch -> batch, run on every call to rebuild the
                neighbor list of the current structures. Without one, the
                batch's own neighbor keys are used as they are — right for a
                static (e.g. fully connected) list, stale for a cutoff list
                once the structures move.
            device: device to run on; the model and every batch are moved
                there (default: leave both where they are)
            dtype: floating dtype to run in; the model and the floating
                tensors of every batch are cast to it (default: unchanged)
            enable_grad: run the model with autograd enabled. Needed for
                models that differentiate their outputs (forces from an
                energy); generative heads run faster without it.
        """
        if isinstance(model, nn.Module) and (device is not None or dtype is not None):
            model = model.to(device=device, dtype=dtype)
        self.model = model
        self.neighbor_list = neighbor_list
        self.device = device
        self.dtype = dtype
        self.enable_grad = enable_grad

    def prepare(self, batch: Mapping[str, Any]) -> Dict[str, Any]:
        """
        Move a batch to the calculator's device and dtype, once per run.

        Drivers call this at loop entry, so the loop's batch already lives
        where the model runs and no call pays a transfer. Returns a new
        dict; non-tensor values pass through.
        """
        out = {}
        for key, value in batch.items():
            if torch.is_tensor(value):
                dtype = (
                    self.dtype
                    if self.dtype is not None and value.is_floating_point()
                    else None
                )
                value = value.to(device=self.device, dtype=dtype)
            out[key] = value
        return out

    def reset(self) -> None:
        """Forget per-run caches (the neighbor list's), before a new run."""
        reset = getattr(self.neighbor_list, "reset", None)
        if reset is not None:
            reset()

    def __call__(self, batch: Mapping[str, Any]) -> Dict[str, torch.Tensor]:
        inputs = dict(batch)
        if self.neighbor_list is not None:
            inputs = self.neighbor_list(inputs)
        with torch.set_grad_enabled(self.enable_grad):
            return self.model(inputs)


def as_calculator(calculator) -> Calculator:
    """Pass a :class:`Calculator` through; wrap a bare ``batch -> outputs`` callable."""
    if isinstance(calculator, Calculator):
        return calculator
    return Calculator(calculator)
