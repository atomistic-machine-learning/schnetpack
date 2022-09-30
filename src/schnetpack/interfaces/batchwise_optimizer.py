import torch
import logging
from tqdm import tqdm
from schnetpack import properties

__all__ = ["TorchStructureLBFGS"]


class TorchStructureLBFGS(torch.optim.LBFGS):
    """
    LBFGS optimizer that allows for relaxation of multiple structures in parallel. The approximation of the inverse
    hessian is shared across the entire batch (all structures). Hence, it is recommended to use this optimizer
    preferably for batches of similar structures/compositions. In other cases, please utilize the ASELBFGS optimizer,
    which is particularly constructed for batches of different structures/compositions.

    This optimizer is an extension/adaptation of the torch.optim.LBFGS optimizer particularly designed for relaxation
    of atomic structures. In addition to the inherited features, this optimizer allows for fixing the positions of a
    set of atoms during the relaxation and a method to run the optimizer. Latter allows for setting a convergence
    criterium. Furthermore, we implemented a logging method that prints out the largest force in the system after each
    optimization iteration.
    """

    def __init__(
        self,
        model,
        model_inputs,
        fixed_atoms_mask,
        maxstep=None,
        logging_function=None,
        lr: float = 1.0,
        energy_key: str = "energy",
        position_key: str = properties.R,
    ):
        """
        Args:
            model (schnetpack.model.AtomisticModel): ml force field model
            model_inputs: input batch containing all structures
            fixed_atoms_mask (list(bool)): list of booleans indicating to atoms with positions fixed in space.
            maxstep (float): how far is a single atom allowed to move. (default: None)
            logging_function: function that logs the structure of the systems during the relaxation
            lr (float): learning rate (default: 1)
            energy_key (str): name of energies in model (default="energy")
            position_key (str): name of atomic positions in model (default="_positions")
        """

        self.model = model
        self.energy_key = energy_key
        self.position_key = position_key
        self.fixed_atoms_mask = fixed_atoms_mask
        self.model_inputs = model_inputs
        self.logging_function = logging_function
        self.fmax = None
        self.maxstep = maxstep

        R = self.model_inputs[self.position_key]
        R.requires_grad = True
        super().__init__(params=[R], lr=lr)

    def _gather_flat_grad(self):
        """override this function to allow for keeping atoms fixed during the relaxation"""
        views = []
        for p in self._params:
            if p.grad is None:
                view = p.new(p.numel()).zero_()
            elif p.grad.is_sparse:
                view = p.grad.to_dense().view(-1)
            else:
                view = p.grad.view(-1)
            views.append(view)
        flat_grad = torch.cat(views, 0)
        if self.fixed_atoms_mask is not None:
            flat_grad[self.fixed_atoms_mask] = 0.0
        self.flat_grad = flat_grad
        return flat_grad

    def _add_grad(self, step_size, update):
        offset = 0

        if self.maxstep is not None:
            step_size = self.determine_step_size(step_size, update)

        for p in self._params:
            numel = p.numel()
            # view as to avoid deprecated pointwise semantics
            p.add_(update[offset : offset + numel].view_as(p), alpha=step_size)
            offset += numel
        assert offset == self._numel()

    def determine_step_size(self, step_size, update):
        """Determine step to take according to maxstep

        Normalize all steps as the largest step. This way
        we still move along the eigendirection.
        """
        reshaped_update = update.view(-1, 3)
        steplengths = ((step_size * reshaped_update) ** 2).sum(1) ** 0.5
        longest_step = torch.max(steplengths)
        # check if any step in entire batch is greater than maxstep
        if longest_step >= self.maxstep:
            # rescale all steps
            logging.info("normalized integration step")
            step_size *= self.maxstep / longest_step
        return step_size

    def closure(self):
        results = self.model(self.model_inputs)
        self.zero_grad()
        loss = results[self.energy_key].sum()
        loss.backward()
        return loss

    def log(self, forces=None):
        """log relaxation results such as max force in the system"""
        if forces is None:
            forces = self.flat_grad.view(-1, 3)
        if not self.converged():
            logging.info("NOT CONVERGED")
        logging.info(
            "max. atomic force: {}".format(torch.sqrt((forces**2).sum(axis=1).max()))
        )

    def converged(self, forces=None):
        """Did the optimization converge?"""
        if forces is None:
            forces = self.flat_grad.view(-1, 3)
        return (forces**2).sum(axis=1).max() < self.fmax**2

    def run(self, fmax, max_opt_steps):
        """run relaxation"""
        self.fmax = fmax

        # optimization
        for opt_step in tqdm(range(max_opt_steps)):
            self.step(self.closure)

            # log structure
            if self.logging_function is not None:
                self.logging_function(opt_step)

            # stop optimization if max force is smaller than threshold
            if self.converged():
                break
        self.log()

    def get_relaxed_structure(self):
        return self.model_inputs[self.position_key]
