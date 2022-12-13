schnetpack.md
=============
.. currentmodule:: md

This module contains all functionality for performing various molecular dynamics simulations using SchNetPack.

System
------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    System


Initial Conditions
------------------

.. currentmodule:: md.initial_conditions

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    Initializer
    MaxwellBoltzmannInit
    UniformInit


Integrators
-----------

.. currentmodule:: md.integrators

Integrators for NVE and NVT simulations:

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    Integrator
    VelocityVerlet
    RingPolymer

Integrators for NPT simulations:

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    NPTVelocityVerlet
    NPTRingPolymer


Calculators
-----------

.. currentmodule:: md.calculators

Basic calculators:

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    MDCalculator
    QMCalculator
    EnsembleCalculator
    LJCalculator

Neural network potentials and ORCA calculators:

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    SchNetPackCalculator
    SchNetPackEnsembleCalculator
    OrcaCalculator


Neighbor List
-------------

.. currentmodule:: md.neighborlist_md

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    NeighborListMD


Simulator
---------

.. currentmodule:: md

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    Simulator


Simulation hooks
----------------

.. currentmodule:: md.simulation_hooks

Basic hooks:

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    SimulationHook
    RemoveCOMMotion

Thermostats:

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    ThermostatHook
    BerendsenThermostat
    LangevinThermostat
    NHCThermostat
    GLEThermostat

Thermostats for ring-polymer MD:

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    PILELocalThermostat
    PILEGlobalThermostat
    TRPMDThermostat
    RPMDGLEThermostat
    PIGLETThermostat
    NHCRingPolymerThermostat

Barostats:

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    BarostatHook
    NHCBarostatIsotropic
    NHCBarostatAnisotropic

Barostats for ring-polymer MD:

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    PILEBarostat

Logging and callback

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    Checkpoint
    DataStream
    MoleculeStream
    PropertyStream
    FileLogger
    BasicTensorboardLogger
    TensorBoardLogger


Simulation data and postprocessing
----------------------------------

.. currentmodule:: md.data

Data loading:

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    HDF5Loader

Vibrational spectra:

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    VibrationalSpectrum
    PowerSpectrum
    IRSpectrum
    RamanSpectrum


ORCA output parsing
-------------------

.. currentmodule:: md.parsers

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    OrcaParser
    OrcaOutputParser
    OrcaFormatter
    OrcaPropertyParser
    OrcaMainFileParser
    OrcaHessianFileParser


MD utilities
------------

.. currentmodule:: md.utils

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    NormalModeTransformer

Utilities for thermostats

.. currentmodule:: md.utils.thermostat_utils

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    YSWeights
    GLEMatrixParser
    load_gle_matrices
    StableSinhDiv
