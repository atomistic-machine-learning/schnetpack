schnetpack.data
===============
.. currentmodule:: schnetpack.data

Datasets and loaders
--------------------

:class:`AtomsDataset` reads a dataset structure by structure and
:class:`AtomsLoader` collates its structures into batches, with workers.
:class:`InMemoryDataset` holds the dataset in memory as flat arrays and returns
whole batches (``dataset[ids]``); :class:`InMemoryLoader` chooses the structures of
each batch. ``AtomsDataModule(in_memory=True)`` uses the second pair; for a training
loop of your own see :doc:`/guides/training_speed`.

.. autosummary::
    :toctree: generated
    :nosignatures:
    :recursive:
    :template: classtemplate.rst

    AtomsDataset
    AtomsLoader
    InMemoryDataset
    InMemoryLoader
    BudgetBatches
    OrderedBatches
    StratifiedSampler

Storage
-------

Where the structures of a dataset are held: an ASE database on disk (``*.db``),
which is also where new structures are written, or flat arrays converted from it (a
``*.arrays`` folder). A dataset takes a storage or its path.

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    AtomsStorage
    ASEDBStorage
    ArrayStorage

.. autosummary::
    :toctree: generated
    :nosignatures:

    open_storage

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

.. autosummary::
    :toctree: generated
    :nosignatures:

    structure_stats
    fit_atomrefs
