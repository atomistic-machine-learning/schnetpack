Command line
============

Installing SchNetPack puts the ``spk`` command on your PATH, with one subcommand per
job, and ``spkmd`` for molecular dynamics.

.. list-table::
    :header-rows: 1
    :widths: 35 65

    * - Command
      - What it does
    * - ``spk train``
      - Train a model from Hydra configs; writes a run folder. See
        :doc:`/getting_started/cli` and :doc:`/guides/configs`.
    * - ``spk test run_dir=<run>``
      - Test a run's best checkpoint on its test split again (``data_dir=`` when the
        data moved since training).
    * - ``spk predict model_dir=<model> db=<db> out=<folder>``
      - Predict properties of the structures in an ASE db with a trained model.
    * - ``spk sample model_dir=<model>``
      - Sample structures with a generative model, with the compositions of its run's
        test split or of ``db=``.
    * - ``spk evaluate dbs=<dbs>``
      - Validity and uniqueness of generated molecules (RDKit), into ``eval.json``.
    * - ``spk deploy model_dir=<model> out=<package.pt2>``
      - Compile a trained potential with AOTInductor for the LAMMPS pair style
        (``device=``, ``dtype=``). See :doc:`/guides/lammps`.
    * - ``spk convert db=<db>``
      - Set the units of an ASE database, e.g. to convert a SchNetPack 1.0 dataset.
    * - ``spkmd``
      - Run molecular dynamics with a trained model, from Hydra configs. See
        :doc:`/guides/md_cli`.

Every command is a Hydra application: arguments are ``key=value``, and ``--help``
prints the command's config with every key. ``model_dir=`` takes a model folder or the run
folder that ``spk train`` wrote (its ``best_model``).
