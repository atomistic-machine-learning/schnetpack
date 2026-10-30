import torch

from schnetpack.objectives import (
    ModelOutput,
    calculate_padded_loss,
    extract_targets,
    predict_without_postprocessing,
)

__all__ = ["GraphStep"]


class GraphStep:
    """
    The training step as one CUDA graph. Building the batch, the model, the loss and
    its gradients are recorded at the first call and replayed at every later one:
    one launch instead of hundreds of kernels, and no host synchronization. After
    the replay the gradients are clipped and the optimizer steps.

    The batch must have the same shapes in every step and be built on the GPU from
    the structure indices alone, as :meth:`~schnetpack.data.device.DeviceAtomsLoader.build_batch`
    does. With a fused optimizer (Adam or AdamW with ``fused=True``), a step whose
    loss exceeds ``skip_loss_above`` is skipped on the GPU, without a host sync.

    In a training loop of your own::

        train = DeviceAtomsLoader(data, train_idx, 100, transforms, batching="budget")
        step = GraphStep(
            CompiledPotential(model, dynamic=False), outputs, optimizer,
            train.build_batch,
        )
        for sel in train.index_batches:
            loss = step(sel)

    :class:`~schnetpack.lightning.AtomisticTask` with ``cuda_graph`` runs this step.
    """

    def __init__(
        self,
        potential: torch.nn.Module,
        outputs: list[ModelOutput],
        optimizer,
        build_batch,
        grad_clip: float | None = None,
        skip_loss_above: float | None = None,
    ):
        """
        Args:
            potential: the model, usually ``CompiledPotential(model,
                dynamic=False)``
            outputs: the outputs of the loss (``MSELoss`` or ``L1Loss`` without
                loss masks, or ``VariableOutput``)
            optimizer: steps the parameters; a wrapper holding the torch optimizer
                as ``.optimizer`` (Lightning's) works too
            build_batch: the batch of a tensor of structure indices, on the GPU and
                with the same shapes for every step
            grad_clip: clip the norm of the gradients to this value
            skip_loss_above: skip the optimizer step of a batch whose loss exceeds
                this value or is not finite
        """
        self.potential, self.outputs = potential, outputs
        self.optimizer, self.build_batch = optimizer, build_batch
        self.grad_clip, self.skip_loss_above = grad_clip, skip_loss_above
        self.graph: torch.cuda.CUDAGraph | None = None
        #: 1.0 / 0.0 whether the last step updated the parameters (the gate of
        #: :meth:`~schnetpack.train.EMA.update`); None without skip_loss_above
        self.keep: torch.Tensor | None = None
        #: the steps skipped so far, on the GPU
        self.skipped: torch.Tensor | None = None
        #: the last step's batch, predictions and targets: tensors of the graph,
        #: overwritten by the next step
        self.batch = self.pred = self.targets = None

    def _parameters(self) -> list[torch.Tensor]:
        return [p for group in self.optimizer.param_groups for p in group["params"]]

    def _forward_backward(self):
        batch = self.build_batch(self._input)
        targets = extract_targets(self.outputs, batch)
        pred = predict_without_postprocessing(self.potential, batch)
        loss = calculate_padded_loss(self.outputs, pred, targets, batch)
        loss.backward()
        return loss, pred, targets, batch

    def _record(self, sel: torch.Tensor):
        device = self._parameters()[0].device
        self._input = sel.to(device).clone()
        side = torch.cuda.Stream(device)
        side.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(side):
            for _ in range(3):  # compiles here, outside the capture
                self._forward_backward()
        torch.cuda.current_stream(device).wait_stream(side)
        for p in self._parameters():
            p.grad = None
        self.graph = torch.cuda.CUDAGraph()
        # the AccumulateGrad nodes of the parameters remember the stream of an
        # earlier backward; let the capturing stream take over
        if hasattr(torch.autograd.graph, "set_override_stale_capture_stream"):
            torch.autograd.graph.set_override_stale_capture_stream(True)
        with torch.cuda.graph(self.graph, stream=side):
            self._loss, self.pred, self.targets, self.batch = self._forward_backward()

    def __call__(self, sel: torch.Tensor) -> torch.Tensor:
        """
        One training step on the structures sel.

        Returns:
            the loss of the batch (before the optimizer step)
        """
        if self.graph is None:
            self._record(sel)
        self._input.copy_(sel, non_blocking=True)
        self.graph.replay()
        loss = self._loss.detach()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self._parameters(), self.grad_clip)
        if self.skip_loss_above is None:
            self.optimizer.step()
            return loss.clone()
        skip = (~(loss <= self.skip_loss_above)).to(torch.float32)
        self.keep = 1.0 - skip
        self.skipped = skip if self.skipped is None else self.skipped + skip
        torch_optimizer = getattr(self.optimizer, "optimizer", self.optimizer)
        if torch_optimizer.defaults.get("fused", False):
            # the fused step skips itself where found_inf is 1
            torch_optimizer.found_inf = skip
            self.optimizer.step()
        elif not bool(skip):  # a host sync
            self.optimizer.step()
        return loss.clone()
