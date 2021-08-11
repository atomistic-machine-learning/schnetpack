import torch.nn as nn
import schnetpack.md as spkmd

__all__ = ["RemoveCOMMotion", "SimulationHook"]


class SimulationHook(nn.Module):
    """
    Basic class for simulator hooks
    """

    def on_step_begin(self, simulator):
        pass

    def on_step_middle(self, simulator):
        pass

    def on_step_end(self, simulator):
        pass

    def on_step_failed(self, simulator):
        pass

    def on_simulation_start(self, simulator):
        pass

    def on_simulation_end(self, simulator):
        pass


class RemoveCOMMotion(SimulationHook):
    """
    Periodically remove motions of the center of mass from the system.

    Args:
        every_n_steps (int): Frequency with which motions are removed.
        remove_rotation (bool): Also remove rotations.
        wrap_positions: Wrap atom positions back to box in periodic simulations.
    """

    def __init__(self, every_n_steps: int, remove_rotation: bool, wrap_positions: bool):
        super(RemoveCOMMotion, self).__init__()
        self.every_n_steps = every_n_steps
        self.remove_rotation = remove_rotation
        self.wrap_positions = wrap_positions

    def on_step_end(self, simulator: spkmd.Simulator):
        if simulator.step % self.every_n_steps == 0:
            simulator.system.remove_center_of_mass()
            simulator.system.remove_translation()

            if self.remove_rotation:
                simulator.system.remove_com_rotation()

            if self.wrap_positions:
                simulator.system.wrap_positions()
