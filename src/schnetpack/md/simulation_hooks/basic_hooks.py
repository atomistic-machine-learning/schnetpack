__all__ = ["RemoveCOMMotion", "SimulationHook"]


class SimulationHook:
    """
    Basic class for simulator hooks
    """

    @property
    def state_dict(self):
        return {}

    @state_dict.setter
    def state_dict(self, state_dict):
        pass

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
        remove_rotation (bool): Also remove rotations (default=False).
    """

    def __init__(self, every_n_steps=10, remove_rotation=True):
        self.every_n_steps = every_n_steps
        self.remove_rotation = remove_rotation

    def on_step_end(self, simulator):
        if simulator.step % self.every_n_steps == 0:
            simulator.system.remove_com()
            simulator.system.remove_com_translation()
            if self.remove_rotation:
                simulator.system.remove_com_rotation()
