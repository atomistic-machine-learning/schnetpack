schnetpack.data
===============
.. currentmodule:: schnetpack.data

Atoms data
------------
.. autosummary::
    :toctree: generated
    :nosignatures:
    :recursive:
    :template: classtemplate.rst

    ASEAtomsData
    DownloadableASEAtomsData
    AtomsLoader
    StratifiedSampler

.. autosummary::
    :toctree: generated
    :nosignatures:

    prune_neighbors

Statistics
----------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    calculate_stats
    estimate_atomrefs
    StatsAtomrefProvider
    NumberOfAtomsCriterion
    PropertyCriterion

Data on the GPU
---------------

The arrays and padded batches behind ``AtomsDataModule(data_device=...)``, also for a
training loop of your own (see :doc:`/userguide/speed`).

.. currentmodule:: schnetpack.data.device

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    DeviceAtomsData
    DeviceAtomsLoader
    ArrayStats
    BudgetBatches
    ShuffledBatches
    OrderedBatches

.. autosummary::
    :toctree: generated
    :nosignatures:

    convert_to_arrays
    split_transforms
