"""
All molecular dynamics in SchNetPack is performed using the :obj:`schnetpack.md.Simulator` class.
This class collects the atomistic system (:obj:`schnetpack.md.System`), calculators (:obj:`schnetpack.md.calculators`),
integrators (:obj:`schnetpack.md.integrators`) and various simulation hooks (:obj:`schnetpack.md.simulation_hooks`)
and performs the time integration.
"""
from tqdm import tqdm


class Simulator:
    """
    Main driver of the molecular dynamics simulation. Uses an integrator to
    propagate the molecular system defined in the system class according to
    the forces yielded by a provided calculator.

    In addition, hooks can be applied at five different stages of each
    simulation step:
     - Start of the simulation (e.g. for initializing thermostats)
     - Before first integrator half step (e.g. thermostats)
     - After computation of the forces and before main integrator step (e.g.
      for accelerated MD)
     - After second integrator half step (e.g. thermostats, output routines)
     - At the end of the simulation (e.g. general wrap up of file writes, etc.)

    This routine has a state dict which can be used to restart a previous
    simulation.

    Args:
        system (object): Instance of the system class defined in
                         molecular_dynamics.system holding the structures,
                         masses, atom type, momenta, forces and properties of
                         all molecules and their replicas
        integrator (object): Integrator for propagating the molecular
                             dynamics simulation, defined in
                             schnetpack.md.integrators
        calculator (object): Calculator class used to compute molecular
                             forces for propagation and (if requested)
                             various other properties.
        simulator_hooks (list(object)): List of different hooks to be applied
                                        during simulations. Examples would be
                                        file loggers and thermostats.
        step (int): Index of the initial simulation step.
        restart (bool): Indicates, whether the simulation is restarted. E.g. if set to True, the simulator tries to
                        continue logging in the previously created dataset. (default=False)
                        This is set automatically by the restart_simulation function. Enabling it without the function
                        currently only makes sense if independent simulations should be written to the same file.
    """

    def __init__(
        self, system, integrator, calculator, simulator_hooks=[], step=0, restart=False
    ):

        self.system = system
        self.integrator = integrator
        self.calculator = calculator
        self.simulator_hooks = simulator_hooks
        self.step = step
        self.n_steps = None
        self.restart = restart

        # Keep track of the actual simulation steps performed with simulate calls
        self.effective_steps = 0

    def simulate(self, n_steps=10000):
        """
        Main simulation function. Propagates the system for a certain number
        of steps.

        Args:
            n_steps (int): Number of simulation steps to be performed (
                           default=10000)
        """

        self.n_steps = n_steps

        # Perform initial computation of forces
        if self.system.forces is None:
            self.calculator.calculate(self.system)

        # Call hooks at the simulation start
        for hook in self.simulator_hooks:
            hook.on_simulation_start(self)

        for _ in tqdm(range(n_steps), ncols=120):

            # Call hook berfore first half step
            for hook in self.simulator_hooks:
                hook.on_step_begin(self)

            # Do half step momenta
            self.integrator.half_step(self.system)

            # Do propagation MD/PIMD
            self.integrator.main_step(self.system)

            # Compute new forces
            self.calculator.calculate(self.system)

            # Call hook after forces
            for hook in self.simulator_hooks:
                hook.on_step_middle(self)

            # Do half step momenta
            self.integrator.half_step(self.system)

            # Call hooks after second half step
            for hook in self.simulator_hooks:
                hook.on_step_end(self)

            self.step += 1
            self.effective_steps += 1

        # Call hooks at the simulation end
        for hook in self.simulator_hooks:
            hook.on_simulation_end(self)

    @property
    def state_dict(self):
        """
        State dict used to restart the simulation. Generates a dictionary with
        the following entries:
            - step: current simulation step
            - systems: state dict of the system holding current positions,
                       momenta, forces, etc...
            - simulator_hooks: dict of state dicts of the various hooks used
                               during simulation using their basic class
                               name as keys.

        Returns:
            dict: State dict containing the current step, the system
                  parameters (positions, momenta, etc.) and all
                  simulator_hook state dicts

        """
        state_dict = {
            "step": self.step,
            "system": self.system.state_dict,
            "simulator_hooks": {
                hook.__class__: hook.state_dict for hook in self.simulator_hooks
            },
        }
        return state_dict

    @state_dict.setter
    def state_dict(self, state_dict):
        """
        Set the current state dict of the simulator using a state dict
        defined in state_dict. This routine assumes, that the identity of all
        hooks has not changed and the order is preserved. A more general
        method to restart simulations is provided below.

        Args:
            state_dict (dict): state dict containing the entries 'step',
            'simulator_hooks' and 'system'.

        """
        self.step = state_dict["step"]
        self.system.state_dict = state_dict["system"]

        # Set state dicts of all hooks
        for hook in self.simulator_hooks:
            if hook.__class__ in state_dict["simulator_hooks"]:
                hook.state_dict = state_dict["simulator_hooks"][hook.__class__]

    def restart_simulation(self, state_dict, soft=False):
        """
        Routine for restarting a simulation. Reads the current step, as well
        as system state from the provided state dict. In case of the
        simulation hooks, only the states of the thermostat hooks are
        restored, as all other hooks do not depend on previous simulations.

        If the soft option is chosen, only restores states of thermostats if
        they are present in the current simulation and the state dict.
        Otherwise, all thermostats found in the state dict are required to be
        present in the current simulation.

        Args:
            state_dict (dict): State dict of the current simulation
            soft (bool): Flag to toggle hard/soft thermostat restarts (
                         default=False)

        """
        self.step = state_dict["step"]
        self.system.state_dict = state_dict["system"]

        if soft:
            # Do the same as in a basic state dict setting
            for hook in self.simulator_hooks:
                if hook.__class__ in state_dict["simulator_hooks"]:
                    hook.state_dict = state_dict["simulator_hooks"][hook.__class__]
        else:
            # Hard restart, require all thermostats to be there
            for hook in self.simulator_hooks:
                # Check if hook is thermostat
                if hasattr(hook, "temperature_bath"):
                    if hook.__class__ not in state_dict["simulator_hooks"]:
                        raise ValueError(
                            f"Could not find restart information for {hook.__class__} in state dict."
                        )
                    else:
                        hook.state_dict = state_dict["simulator_hooks"][hook.__class__]

        # In this case, set restart flag automatically
        self.restart = True

    def load_system_state(self, state_dict):
        """
        Routine for only loading the system state of previous simulations.
        This can e.g. be used for production runs, where an equilibrated system
        is loaded, but the thermostat is changed.

        Args:
            state_dict (dict): State dict of the current simulation
        """
        self.system.state_dict = state_dict["system"]
