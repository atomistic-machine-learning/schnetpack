schnetpack.interfaces
=====================

ASE
---

Use SchNetPack models as ASE calculators and convert between ASE atoms and
SchNetPack batches.

.. currentmodule:: schnetpack.interfaces.ase_interface

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    SpkCalculator
    SpkEnsembleCalculator
    AtomsConverter
    AseInterface

.. autosummary::
    :toctree: generated
    :nosignatures:

    atoms_to_batch
    batch_to_atoms

MACE
----

Build, export and fine-tune MACE models. Needs the ``mace`` extra
(``pip install "schnetpack[mace]"``).

.. currentmodule:: schnetpack.interfaces.mace_interface

.. autosummary::
    :toctree: generated
    :nosignatures:

    build_mace
    export_mace
    export_mace_off
