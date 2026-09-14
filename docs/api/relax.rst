schnetpack.relax
================
.. currentmodule:: relax

Batch-wise structure relaxation. A whole batch of structures is relaxed in parallel,
with one inverse Hessian approximation per structure, so that batches of differing
compositions can be optimized together.

Calculators
-----------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    BatchwiseCalculator
    BatchwiseEnsembleCalculator
    NNEnsemble


Optimizers
----------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    BatchwiseOptimizer
    BatchwiseLBFGS


Uncertainty
-----------

.. currentmodule:: uncertainty

How far the members of an ensemble disagree, one value per structure. Shared with
:class:`~schnetpack.interfaces.ase_interface.SpkEnsembleCalculator`.

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    Uncertainty
    AbsoluteUncertainty
    RelativeUncertainty


Observers
---------

.. currentmodule:: relax.observers

What a relaxation reports while it runs. See :mod:`schnetpack.relax.observers`.

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    RelaxationObserver
    RelaxationFrame
    Interval
    LogWriter
    TrajectoryRecorder
    FrameCollector


Trajectories
------------

.. currentmodule:: relax.batchwise_trajectory

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    BatchwiseTrajectoryWriter
    BatchwiseTrajectoryReader
