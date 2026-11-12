Command line
============

Installing SchNetPack puts these commands on your PATH.

.. list-table::
    :header-rows: 1
    :widths: 20 80

    * - Command
      - What it does
    * - ``spktrain``
      - Train a model from Hydra configs. See :doc:`/getting_started/cli` and
        :doc:`/guides/configs`.
    * - ``spkpredict``
      - Predict properties of a dataset with a trained model, from Hydra configs.
    * - ``spkmd``
      - Run molecular dynamics with a trained model, from Hydra configs. See
        :doc:`/guides/md_cli`.
    * - ``spkdeploy``
      - Export a trained model for LAMMPS:
        ``spkdeploy <model_path> <deployed_model_path> [--device cpu]``. See
        :doc:`/guides/lammps`.
    * - ``spkconvert``
      - Set the units of an ASE database, e.g. to convert a SchNetPack 1.0 dataset.
        ``spkconvert --help`` lists the options.

``spktrain``, ``spkpredict`` and ``spkmd`` are Hydra applications: every config
value can be changed on the command line, and ``--help`` prints the full config.
