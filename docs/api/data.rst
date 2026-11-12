schnetpack.data
===============
.. currentmodule:: schnetpack.data

Datasets and loaders
--------------------

:class:`ASEAtomsData` reads a dataset from an ASE database, structure by structure,
with its transforms; :class:`AtomsLoader` collates the structures into batches, with
workers. :class:`AtomsCache` holds every structure of a dataset once read, as flat
tensors; :class:`AtomsLoader` with ``cache=`` then slices whole padded batches from
it. On both routes the loader moves each batch to ``device`` and applies its
``batch_transforms``. ``AtomsDataModule(cache=True)`` uses the cache; for a training
loop of your own see :doc:`/guides/training_speed`.

.. autosummary::
    :toctree: generated
    :nosignatures:
    :recursive:
    :template: classtemplate.rst

    ASEAtomsData
    DownloadableASEAtomsData
    AtomsLoader
    AtomsCache
    BudgetBatches
    OrderedBatches
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
    ArrayStats
    NumberOfAtomsCriterion
    PropertyCriterion

.. autosummary::
    :toctree: generated
    :nosignatures:

    structure_stats
    fit_atomrefs
