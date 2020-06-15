from schnetpack.md.simulation_hooks import SimulationHook


class AdaptiveSamplingHook(SimulationHook):
    def __init__(self):
        raise NotImplementedError

    def on_simulation_start(self, simulator):
        # Set all things
        raise NotImplementedError

    def on_step_end(self, simulator):
        # Do all the stuff
        raise NotImplementedError
