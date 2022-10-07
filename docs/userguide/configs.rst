=====================
Configuration and CLI
=====================
.. _configs:

SchNetPack models and tasks can be defined using hierarchical
`Hydra <https://hydra.cc/>`_ config files in YAML format and modified using command
line arguments. Here, we will introduce the structure and basic syntax of the
configuration. Please refer to the `Hydra documentation <https://hydra.cc/>`_ for more
extensive information.

We will explain the structure of the config at the example of training PaiNN on QM9
with the command::

   $ spktrain experiment=qm9_energy

Before going through the config step-by-step, we show the full config as printed by
the command::

    ⚙ Running with the following config:
    ├── run
    │   └── work_dir: ${hydra:runtime.cwd}
    │       data_dir: ${run.work_dir}/data
    │       path: runs
    │       id: ${uuid:1}
    │
    ├── globals
    │   └── model_path: best_model
    │       cutoff: 5.0
    │       lr: 0.0005
    │       property: energy_U0
    │
    ├── data
    │   └── _target_: schnetpack.datasets.QM9
    │       datapath: ${run.data_dir}/qm9.db
    │       data_workdir: null
    │       batch_size: 100
    │       num_train: 110000
    │       num_val: 10000
    │       num_test: null
    │       num_workers: 8
    │       num_val_workers: null
    │       num_test_workers: null
    │       remove_uncharacterized: false
    │       distance_unit: Ang
    │       property_units:
    │         energy_U0: eV
    │         energy_U: eV
    │         enthalpy_H: eV
    │         free_energy: eV
    │         homo: eV
    │         lumo: eV
    │         gap: eV
    │         zpve: eV
    │       transforms:
    │       - _target_: schnetpack.transform.SubtractCenterOfMass
    │       - _target_: schnetpack.transform.RemoveOffsets
    │         property: ${globals.property}
    │         remove_atomrefs: true
    │         remove_mean: true
    │       - _target_: schnetpack.transform.MatScipyNeighborList
    │         cutoff: ${globals.cutoff}
    │       - _target_: schnetpack.transform.CastTo32
    │
    ├── model
    │   └── representation:
    │         _target_: schnetpack.representation.PaiNN
    │         n_atom_basis: 128
    │         n_interactions: 3
    │         shared_interactions: false
    │         shared_filters: false
    │         radial_basis:
    │           _target_: schnetpack.nn.radial.GaussianRBF
    │           n_rbf: 20
    │           cutoff: ${globals.cutoff}
    │         cutoff_fn:
    │           _target_: schnetpack.nn.cutoff.CosineCutoff
    │           cutoff: ${globals.cutoff}
    │       _target_: schnetpack.model.NeuralNetworkPotential
    │       input_modules:
    │       - _target_: schnetpack.atomistic.PairwiseDistances
    │       output_modules:
    │       - _target_: schnetpack.atomistic.Atomwise
    │         output_key: ${globals.property}
    │         n_in: ${model.representation.n_atom_basis}
    │         aggregation_mode: sum
    │       postprocessors:
    │       - _target_: schnetpack.transform.CastTo64
    │       - _target_: schnetpack.transform.AddOffsets
    │         property: ${globals.property}
    │         add_mean: true
    │         add_atomrefs: true
    │
    ├── task
    │   └── optimizer_cls: torch.optim.AdamW
    │       optimizer_args:
    │         lr: ${globals.lr}
    │         weight_decay: 0.0
    │       scheduler_cls: schnetpack.train.ReduceLROnPlateau
    │       scheduler_monitor: val_loss
    │       scheduler_args:
    │         mode: min
    │         factor: 0.8
    │         patience: 80
    │         threshold: 0.0001
    │         threshold_mode: rel
    │         cooldown: 10
    │         min_lr: 0.0
    │         smoothing_factor: 0.0
    │       _target_: schnetpack.AtomisticTask
    │       outputs:
    │       - _target_: schnetpack.task.ModelOutput
    │         name: ${globals.property}
    │         loss_fn:
    │           _target_: torch.nn.MSELoss
    │         metrics:
    │           mae:
    │             _target_: torchmetrics.regression.MeanAbsoluteError
    │           mse:
    │             _target_: torchmetrics.regression.MeanSquaredError
    │         loss_weight: 1.0
    │       warmup_steps: 0
    │
    ├── trainer
    │   └── _target_: pytorch_lightning.Trainer
    │       devices: 1
    │       min_epochs: null
    │       max_epochs: 100000
    │       enable_model_summary: true
    │       profiler: null
    │       gradient_clip_val: 0
    │       accumulate_grad_batches: 1
    │       val_check_interval: 1.0
    │       check_val_every_n_epoch: 1
    │       num_sanity_val_steps: 0
    │       fast_dev_run: false
    │       overfit_batches: 0
    │       limit_train_batches: 1.0
    │       limit_val_batches: 1.0
    │       limit_test_batches: 1.0
    │       track_grad_norm: -1
    │       detect_anomaly: false
    │       amp_backend: native
    │       amp_level: null
    │       precision: 32
    │       accelerator: auto
    │       num_nodes: 1
    │       tpu_cores: null
    │       deterministic: false
    │       resume_from_checkpoint: null
    │
    ├── callbacks
    │   └── model_checkpoint:
    │         _target_: schnetpack.train.ModelCheckpoint
    │         monitor: val_loss
    │         save_top_k: 1
    │         save_last: true
    │         mode: min
    │         verbose: false
    │         dirpath: checkpoints/
    │         filename: '{epoch:02d}'
    │         model_path: ${globals.model_path}
    │       early_stopping:
    │         _target_: pytorch_lightning.callbacks.EarlyStopping
    │         monitor: val_loss
    │         patience: 1000
    │         mode: min
    │         min_delta: 0.0
    │       lr_monitor:
    │         _target_: pytorch_lightning.callbacks.LearningRateMonitor
    │         logging_interval: epoch
    │
    ├── logger
    │   └── tensorboard:
    │         _target_: pytorch_lightning.loggers.tensorboard.TensorBoardLogger
    │         save_dir: tensorboard/
    │         name: default
    │
    └── seed
        └── None

Train config and config groups
==============================

The config printed above is the flattened final config that the SchNetPack receives as
a dictionary. However, it is not necessary to write down the whole config in a file
when specifying a run.
Instead Hydra uses config groups that are hierarchically ordered in directories
and allow to predefine templates for parts of the config.

The default configs for SchNetPack are located in the directory ``src/schnetpack/configs``.
The config ``train.yaml`` is used as a basis for all training runs and sets the
following default config groups::

    defaults:
      - run: default_run
      - globals: default_globals
      - trainer: default_trainer
      - callbacks:
          - checkpoint
          - earlystopping
          - lrmonitor
      - task: default_task
      - model: null
      - data: custom
      - logger: tensorboard
      - experiment: null

Here is a description of the purpose of the different config groups:

* **run**: defines run-specific variables, such as the run ``id``, or working and data directories
* **globals**: defines custom variables that can be reused across the whole config by making use of the interpolation syntax ``${globals.variable}``
* **data**: defines the :class:`data.AtomsDataModule` to be used
* **model**: defines the :class:`model.AtomisticModel` to be used
* **task**: defines the :class:`task.AtomisticTask`
* **trainer**: configure the PyTorchLightning ``Trainer``
* **callbacks**: a list of callbacks for the PyTorchLightning ``Trainer``
* **logger**: a dictionary of training logger that is passed to the trainer
* **seed**: the random seed
* **experiment**: define experiment templates by overriding the train.yaml config


A special role plays the config group ``experiment``, which does not occur in the config
shown above. This is because ``experiment`` is used to overwrite the defaults of
``train.yaml`` and can be used to build pre-defined configs, such as for the QM9 case
shown above.

The config groups ``data``, ``model``, ``task``, ``trainer``, ``callback`` and
``logger`` directly define objects using the special key ``_target_``, which specifies
a class, while the remaining key-value pairs define the arguments passed to the
``__init__``.

Defining experiments
====================

We will now take a look at the QM9 experiment config::

    # @package _global_

    defaults:
      - override /model: nnp
      - override /data: qm9

The first line indicates that the experiment config should be placed at the base level
of the hierarchy, i.e. it directly overrides `train.yaml``.
Then, the defaults for the model and data config groups are overridden.

The configs for the :class:`model.NeuralNetworkPotential` (``nnp``) and the
:class:`dataset.AtomsDataModule` (``qm9``) are predefined in the respective directories
of their config groups. E.g., the data config loads the predefined
``AtomsDataModule`` for QM9 that automatically downloads the dataset and sets the
units that the property should be converted to::

    defaults:
      - custom

    _target_: schnetpack.datasets.QM9

    datapath: ${run.data_dir}/qm9.db  # data_dir is specified in train.yaml
    batch_size: 100
    num_train: 110000
    num_val: 10000
    remove_uncharacterized: False

    # convert to typically used units
    distance_unit: Ang
    property_units:
      energy_U0: eV
      energy_U: eV
      enthalpy_H: eV
      free_energy: eV
      homo: eV
      lumo: eV
      gap: eV
      zpve: eV


In the next section of the experiment config, the run path and global variables are set::

    run.path: runs/qm9_${globals.property}

    globals:
      cutoff: 5.
      lr: 5e-4
      property: energy_U0


These variables will be used in the following sections. There, the defaults for model
and data loaded above are overridden::

    data:
      transforms:
        - _target_: schnetpack.transform.SubtractCenterOfMass
        - _target_: schnetpack.transform.RemoveOffsets
          property: ${globals.property}
          remove_atomrefs: True
          remove_mean: True
        - _target_: schnetpack.transform.MatScipyNeighborList
          cutoff: ${globals.cutoff}
        - _target_: schnetpack.transform.CastTo32

    model:
      output_modules:
        - _target_: schnetpack.atomistic.Atomwise
          output_key: ${globals.property}
          n_in: ${model.representation.n_atom_basis}
          aggregation_mode: sum
      postprocessors:
        - _target_: schnetpack.transform.CastTo64
        - _target_: schnetpack.transform.AddOffsets
          property: ${globals.property}
          add_mean: True
          add_atomrefs: True

All parameters not set here are kept from the default configs.
The data config is altered by setting a custom list of pre-processing transforms.
These are suitable for prediction of the energy and similar extensive targets.
The list includes removing of the single atom reference and the mean energy per atom,
computing the neighborlist and finally casting to ``float32``.
The output-specific part of the model is set to predict the energy as a sum of atomwise
contributions. The key in the output dictionary ``output_key`` is set to the name of the
property to be predicted. Finally, the specified post-processor casts back to double and
adds the removed offsets to the prediction.

The missing part is to define the task that should be solved during the training::

    task:
      outputs:
        - _target_: schnetpack.task.ModelOutput
          name: ${globals.property}
          loss_fn:
            _target_: torch.nn.MSELoss
          metrics:
            mae:
              _target_: torchmetrics.regression.MeanAbsoluteError
            mse:
              _target_: torchmetrics.regression.MeanSquaredError
          loss_weight: 1.

This last section modifies the config of :class:`task.AtomisticTask` by setting a custom
list of model outputs. In this case, we use the mean squared error as a loss on
the predicted property. When the ``target_name`` is nor explicitly set, it is assumed to
be identical with the ``name`` of the prediction.

In conclusion, the hierarchical structure of the configuration allows us to prepare
templates for common use case of the individual components and then assemble and modify
them as required for a specific experiment. In the next section, we will show some
examples of how to override a given experiment with the command line.

Overriding arguments with the CLI
=================================

If you are running several variation of an experiment, it is convenient to make these
directly at the command line instead of creating a separate config file for each of
them. When changing a single value, such as the learning rate, you can use the following
notation::

   $ spktrain experiment=qm9_energy globals.lr=1e-4

Alternatively, one can also change a whole config group. The syntax for this is slightly
different::

   $ spktrain experiment=qm9_energy model/representation=schnet

The difference here is that ``schnet`` refers to a pre-defined subconfig, instead of a
single value. The config would be changed by this as follows::

    ...
    ├── model
    │   └── representation:
    │         _target_: schnetpack.representation.SchNet
    │         n_atom_basis: 128
    │         n_interactions: 6
    │         radial_basis:
    │           _target_: schnetpack.nn.radial.GaussianRBF
    │           n_rbf: 20
    │           cutoff: ${globals.cutoff}
    │         cutoff_fn:
    │           _target_: schnetpack.nn.cutoff.CosineCutoff
    │           cutoff: ${globals.cutoff}
    ...

Working with your own config
============================

Hydra is looking for config YAMLs in the config directory of SchNetPack, where
tha main config ``train.yaml`` is located, as well as in the paths specified in
``train.yaml``::

    hydra:
        searchpath:
          - file://${oc.env:PWD}
          - file://${oc.env:PWD}/configs

This means that Hydra is looking also in the current working directory as well as a
subdirectory ``./configs``.
This makes it convenient to create working directories for your projects where all
your custom configs and runs are located.

If you want to specify a different config location, this is possible with::

    $ spktrain  --config-dir=/path/to/configs experiment=my_config

Note, that in any case the config path points to the base directory of the hierarchical
config. The config files need to be placed in the respective directories
matching the config groups. This could look something like this::

    # create main config directory and subdirectory for experiment config group
    $ mkdir configs
    $ mkdir configs/experiment

    # create experiment and edit config
    $ vim my_configs/experiments/my_experiment.yaml

    # explicitly stating config dir is not required since it is in current working dir
    $ spktrain experiment=my_experiment

Within the configs you are not restricted to using SchNetPack modules, but you can also
set your own implementations as ``_target_``, as long as they conform with the required
interface, e.g. are subclasses of SchNetPack base classes. Unfortunately, this can not
be checked statically at the time, but will lead to errors when calling ``spktrain``.
This kind of flexibilty enables a convenient extension of SchNetPack with your own
config and code.
