:mod:`schnetpack.md`
====================


.. automodule:: schnetpack.md


Calculators
-----------

.. automodule:: schnetpack.md.calculators

.. autoclass:: schnetpack.md.calculators.MDCalculator
   :members:

.. autoclass:: schnetpack.md.calculators.QMCalculator
   :members:

.. autoclass:: schnetpack.md.calculators.SchnetPackCalculator
   :members:

.. autoclass:: schnetpack.md.calculators.OrcaCalculator
   :members:

.. autoclass:: schnetpack.md.calculators.SGDMLCalculator
   :members:


Integrators
-----------

.. automodule:: schnetpack.md.integrators


.. autoclass:: schnetpack.md.VelocityVerlet
   :members:

.. autoclass:: schnetpack.md.RingPolymer
   :members:


System
------

.. automodule:: schnetpack.md.system


.. autoclass:: schnetpack.md.System
   :members:


Initial Conditions
------------------

.. automodule:: schnetpack.md.initial_conditions


.. autoclass:: schnetpack.md.Initializer
   :members:

.. autoclass:: schnetpack.md.MaxwellBoltzmannInit
   :members:


Neighbor Lists
--------------

.. automodule:: schnetpack.md.neighbor_lists


.. autoclass:: schnetpack.md.MDNeighborList
   :members:

.. autoclass:: schnetpack.md.SimpleNeighborList
   :members:


Simulator
---------

.. automodule:: schnetpack.md.simulator

.. autoclass:: schnetpack.md.simulator.Simulator
   :members:


Simulation Hooks
----------------

.. automodule:: schnetpack.md.simulation_hooks

.. autoclass:: schnetpack.md.simulation_hooks.SimulationHook
   :members:

.. autoclass:: schnetpack.md.simulation_hooks.RemoveCOMMotion
   :members:


Logging
^^^^^^^

.. automodule:: schnetpack.md.simulation_hooks.logging_hooks

.. autoclass:: schnetpack.md.simulation_hooks.logging_hooks.Checkpoint
   :members:

.. autoclass:: schnetpack.md.simulation_hooks.logging_hooks.TemperatureLogger
   :members:

.. autoclass:: schnetpack.md.simulation_hooks.logging_hooks.FileLogger
   :members:

.. autoclass:: schnetpack.md.simulation_hooks.logging_hooks.MoleculeStream
   :members:

.. autoclass:: schnetpack.md.simulation_hooks.logging_hooks.DataStream
   :members:

.. autoclass:: schnetpack.md.simulation_hooks.logging_hooks.PropertyStream
   :members:

.. autoclass:: schnetpack.md.simulation_hooks.logging_hooks.SimulationStream
   :members:


Sampling
^^^^^^^^

.. automodule:: schnetpack.md.simulation_hooks.sampling

.. autoclass:: schnetpack.md.simulation_hooks.sampling.BiasPotential
   :members:

.. autoclass:: schnetpack.md.simulation_hooks.sampling.AcceleratedMD
   :members:

.. autoclass:: schnetpack.md.simulation_hooks.sampling.MetaDyn
   :members:

.. autoclass:: schnetpack.md.simulation_hooks.sampling.CollectiveVariable
   :members:

.. autoclass:: schnetpack.md.simulation_hooks.sampling.BondColvar
   :members:


Thermostats
^^^^^^^^^^^

.. automodule:: schnetpack.md.simulation_hooks.thermostats

.. autoclass:: schnetpack.md.simulation_hooks.thermostats.ThermostatHook
   :members:

.. autoclass:: schnetpack.md.simulation_hooks.thermostats.BerendsenThermostat
   :members:

.. autoclass:: schnetpack.md.simulation_hooks.thermostats.GLEThermostat
   :members:

.. autoclass:: schnetpack.md.simulation_hooks.thermostats.PIGLETThermostat
   :members:

.. autoclass:: schnetpack.md.simulation_hooks.thermostats.LangevinThermostat
   :members:

.. autoclass:: schnetpack.md.simulation_hooks.thermostats.PILELocalThermostat
   :members:

.. autoclass:: schnetpack.md.simulation_hooks.thermostats.PILEGlobalThermostat
   :members:

.. autoclass:: schnetpack.md.simulation_hooks.thermostats.NHCThermostat
   :members:

.. autoclass:: schnetpack.md.simulation_hooks.thermostats.NHCRingPolymerThermostat
   :members:

.. autoclass:: schnetpack.md.simulation_hooks.thermostats.TRPMDThermostat
   :members:


Utils
-----

.. automodule:: schnetpack.md.utils

.. autoclass:: schnetpack.md.utils.MDUnits
   :members:

.. autoclass:: schnetpack.md.utils.NormalModeTransformer
   :members:


HDF5-Loader
^^^^^^^^^^^

.. automodule:: schnetpack.md.utils.hdf5_data

.. autoclass:: schnetpack.md.utils.HDF5Loader
   :members:


Spectra
^^^^^^^

.. automodule:: schnetpack.md.utils.spectra

.. autoclass:: schnetpack.md.utils.VibrationalSpectrum
   :members:

.. autoclass:: schnetpack.md.utils.PowerSpectrum
   :members:

.. autoclass:: schnetpack.md.utils.IRSpectrum
   :members:


Parsers
-------

.. automodule:: schnetpack.md.parsers


ORCA Parsers
^^^^^^^^^^^^

.. automodule:: schnetpack.md.parsers.orca_parser

.. autoclass:: schnetpack.md.parsers.orca_parser.OrcaParserException
   :members:

.. autoclass:: schnetpack.md.parsers.orca_parser.OrcaParser
   :members:

.. autoclass:: schnetpack.md.parsers.orca_parser.OrcaParser
   :members:

.. autoclass:: schnetpack.md.parsers.orca_parser.OrcaOutputParser
   :members:

.. autoclass:: schnetpack.md.parsers.orca_parser.OrcaPropertyParser
   :members:

.. autoclass:: schnetpack.md.parsers.orca_parser.OrcaFormatter
   :members:

.. autoclass:: schnetpack.md.parsers.orca_parser.OrcaMainFileParser
   :members:

.. autoclass:: schnetpack.md.parsers.orca_parser.OrcaHessianFileParser
   :members:
