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
      - Compile a trained potential with AOTInductor for the LAMMPS pair style:
        ``spkdeploy <model_path> <package.pt2> [--device cpu] [--dtype float32]``. See
        :doc:`/guides/lammps`.
    * - ``spkconvert``
      - Set the units of an ASE database, e.g. to convert a SchNetPack 1.0 dataset.
        ``spkconvert --help`` lists the options.

``spktrain``, ``spkpredict`` and ``spkmd`` are Hydra applications: every config
value can be changed on the command line, and ``--help`` prints the full config.
