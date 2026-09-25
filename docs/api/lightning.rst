schnetpack.lightning
====================
.. currentmodule:: lightning

PyTorch Lightning integration used by ``spktrain``. Custom PyTorch training
loops do not need this module; see :mod:`schnetpack.objectives` instead.

Task
----

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    AtomisticTask

Data modules
------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    AtomsDataModule

Callbacks
---------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    ModelCheckpoint
    PredictionWriter
    ExponentialMovingAverage
