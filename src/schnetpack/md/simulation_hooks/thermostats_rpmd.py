import torch

from typing import Optional

from schnetpack.md.simulation_hooks import LangevinThermostat

__all__ = ["PILELocalThermostat"]


class PILELocalThermostat(LangevinThermostat):
    """
    Langevin thermostat for ring polymer molecular dynamics as introduced in [#stochastic_thermostats2]_.
    Applies specially initialized Langevin thermostats to the beads of the ring polymer in normal mode representation.

    Args:
        temperature_bath (float): Temperature of the external heat bath in Kelvin.
        time_constant (float): Thermostat time constant in fs
        nm_transformation (schnetpack.md.utils.NormalModeTransformer): Module use dto transform between beads and normal
                                                                       model representation in ring polymer dynamics.
        thermostat_centroid (bool): Whether a thermostat should be applied to the centroid of the ring polymer in
                                    normal mode representation (relevant e.g. for TRPMD, default is True)
        damping_factor (float): If specified, damping factor is applied to the current momenta of the system (used in TRPMD,
                         default is no damping).

    References
    ----------
    .. [#stochastic_thermostats2] Ceriotti, Parrinello, Markland, Manolopoulos:
       Efficient stochastic thermostatting of path integral molecular dynamics.
       The Journal of Chemical Physics, 133 (12), 124104. 2010.
    """

    def __init__(
        self,
        temperature_bath: float,
        time_constant: float,
        thermostat_centroid: Optional[bool] = True,
        damping_factor: Optional[float] = 1.0,
    ):
        super(PILELocalThermostat, self).__init__(
            temperature_bath=temperature_bath, time_constant=time_constant
        )
        self.register_buffer("thermostat_centroid", torch.tensor(thermostat_centroid))
        self.register_buffer("dampinf_factor", torch.tensor(damping_factor))

    def _init_thermostat(self, simulator):
        """
        Initialize the Langevin matrices based on the normal mode frequencies of the ring polymer. If the centroid is to
        be thermostatted, the suggested value of 1/time_constant is used.

        Args:
            simulator (schnetpack.simulation_hooks.simulator.Simulator): Main simulator class containing information on the
                                                                 time step, system, etc.
        """
        if not isinstance(simulator.integrator, RingPolymer):
            raise ThermostatError("PILE thermostats can only be used in RPMD")

        # Initialize friction coefficients
        gamma_normal = 2 * simulator.integrator.omega_normal

        # Use seperate coefficient for centroid mode (default, unless using thermostatted RPMD)
        if self.thermostat_centroid:
            gamma_normal[0] = 1.0 / self.time_constant

        # Apply TRPMD damping factor if provided
        if self.damping is not None:
            gamma_normal *= self.damping

        if self.nm_transformation is None:
            raise ThermostatError(
                "Normal mode transformation required for " "PILE thermostat"
            )

        # Initialize coefficient matrices
        c1 = torch.exp(-0.5 * simulator.integrator.time_step * gamma_normal)
        c2 = torch.sqrt(1 - c1 ** 2)

        self.c1 = c1.to(self.device)[:, None, None, None]
        self.c2 = c2.to(self.device)[:, None, None, None]

        # Get mass and temperature factors
        self.thermostat_factor = torch.sqrt(
            simulator.system.masses
            * MDUnits.kB
            * self.n_replicas
            * self.temperature_bath
        )
        exit()

    @property
    def state_dict(self):
        state_dict = {
            "c1": self.c1,
            "c2": self.c2,
            "thermostat_factor": self.thermostat_factor,
            "temperature_bath": self.temperature_bath,
            "n_replicas": self.n_replicas,
            "damping": self.damping,
            "thermostat_centroid": self.thermostat_centroid,
        }
        return state_dict

    @state_dict.setter
    def state_dict(self, state_dict):
        self.c1 = state_dict["c1"]
        self.c2 = state_dict["c2"]
        self.thermostat_factor = state_dict["thermostat_factor"]
        self.temperature_bath = state_dict["temperature_bath"]
        self.n_replicas = state_dict["n_replicas"]
        self.damping = state_dict["damping"]
        self.thermostat_centroid = state_dict["thermostat_centroid"]

        # Set initialized flag
        self.initialized = True
