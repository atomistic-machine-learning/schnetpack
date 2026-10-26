"""Run a model as one traced and compiled graph, derivatives included.

``torch.compile`` alone cannot train on forces: its backward is not differentiable
again, and the force loss needs the gradient of the forces. Here the model's forward,
including the ``autograd.grad`` call of its response modules (forces, stress), is first
recorded into one graph with ``make_fx``, so the derivatives become ordinary ops. That
graph is compiled with dynamic shapes; training then needs only a single backward
through it, and inference runs it without autograd.
"""

import warnings

import torch
from torch._dynamo.source import ConstantSource
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.func import functional_call
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.experimental.symbolic_shapes import ShapeEnv

from schnetpack.model.base import AtomisticModel

__all__ = ["CompiledPotential", "compile_potential", "compile_supported"]

# make_fx over autograd.grad drops the gradient through saved outputs (exp, tanh,
# sigmoid, sqrt) on torch 2.7.1; 2.14.1 traces them correctly
_MIN_TORCH = (2, 14)


def compile_supported() -> bool:
    """Whether this torch version traces the gradients of a force loss correctly."""
    return torch.torch_version.TorchVersion(torch.__version__) >= _MIN_TORCH


class CompiledPotential:
    """
    Calls ``model`` through traced and compiled graphs with the same dict interface.

    Not a module: it keeps no parameters of its own and passes the model's current
    parameters and buffers on every call, so optimizer steps, ``.to()``, checkpoints and
    ``torch.save`` keep working on ``model`` itself. One graph is traced and compiled per
    train/eval mode, postprocessing flag, set of input keys, dtypes and devices, and per
    input dimension of size 0 or 1 (e.g. a batch of a single molecule); batches of other
    sizes reuse it (dynamic shapes).

    In eval mode the graph runs without autograd: forces and stress are already part of
    it, and the outputs carry no gradient history.

    Pickling and copying keep the model but drop the compiled graphs, which are traced
    again on the next call.
    """

    def __init__(self, model: AtomisticModel, **compile_kwargs):
        """
        Args:
            model: the model to run, e.g. a ``NeuralNetworkPotential`` with ``Forces``
            compile_kwargs: passed on to ``torch.compile`` (``dynamic=True`` by default)
        """
        if not compile_supported():
            warnings.warn(
                f"compile_potential needs torch >= {'.'.join(map(str, _MIN_TORCH))} to "
                f"train correctly; torch {torch.__version__} can trace wrong parameter "
                "gradients of a force loss.",
                stacklevel=2,
            )
        self.model = model
        self.compile_kwargs = {"dynamic": True, **compile_kwargs}
        self._graphs = {}

    @property
    def do_postprocessing(self) -> bool:
        return self.model.do_postprocessing

    @do_postprocessing.setter
    def do_postprocessing(self, value: bool):
        self.model.do_postprocessing = value

    def __getstate__(self):
        return {**self.__dict__, "_graphs": {}}

    def __call__(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        inputs = {k: v for k, v in inputs.items() if isinstance(v, torch.Tensor)}
        params = dict(self.model.named_parameters())
        buffers = dict(self.model.named_buffers())
        key = (
            self.model.training,
            self.model.do_postprocessing,
            tuple(
                (k, v.dtype, v.device, tuple(min(n, 2) for n in v.shape))
                for k, v in sorted(inputs.items())
            ),
        )
        if key not in self._graphs:
            graph = self._trace(params, buffers, inputs)
            self._graphs[key] = torch.compile(graph, **self.compile_kwargs)
        if self.model.training:
            return self._graphs[key](params, buffers, inputs)
        with torch.no_grad():
            return self._graphs[key](params, buffers, inputs)

    def _trace(self, params, buffers, inputs):
        def forward(params, buffers, inputs):
            return functional_call(self.model, (params, buffers), (dict(inputs),))

        # make_fx does not guard the graph it returns, so a size it specializes on holds
        # for every later batch. Sizes 0 and 1 are part of the graph key; sizes that
        # happen to be equal (duck shapes) must not share a symbol. Parameter and
        # buffer shapes stay fixed.
        fake_mode = FakeTensorMode(
            allow_non_fake_inputs=True, shape_env=ShapeEnv(duck_shape=False)
        )

        def fixed(tensors):
            return {
                k: fake_mode.from_tensor(v, static_shapes=True)
                for k, v in tensors.items()
            }

        # the response modules call autograd.grad, also when the caller is in no_grad
        derivatives = self.model.required_derivatives or []
        example = {
            k: fake_mode.from_tensor(
                v.detach().requires_grad_() if k in derivatives else v,
                source=ConstantSource(f"inputs[{k!r}]"),
            )
            for k, v in inputs.items()
        }
        with torch.enable_grad():
            return make_fx(
                forward, tracing_mode="symbolic", _allow_non_fake_inputs=True
            )(fixed(params), fixed(buffers), example)


def compile_potential(model: AtomisticModel, **compile_kwargs) -> CompiledPotential:
    """
    Wrap ``model`` so that it runs as one compiled graph including its derivatives.

    Args:
        model: the model to run, e.g. a ``NeuralNetworkPotential`` with ``Forces``
        compile_kwargs: passed on to ``torch.compile`` (``dynamic=True`` by default)

    Returns:
        a callable with the model's dict-in, dict-out interface
    """
    return CompiledPotential(model, **compile_kwargs)
