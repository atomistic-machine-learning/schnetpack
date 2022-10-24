===================================
Running MD simulations with the CLI
===================================
.. _md:

Similar to the basic SchNetPack usage, it is also possible to quickly set up molecular dynamics (MD) simulations using
a combination of the `Hydra <https://hydra.cc/>`_  command line interface (CLI) and predefined config files. The latter
can be found in ``src/schnetpack/md/md_configs``.
In the following, we will give a short introduction on how to use the CLI and/or config files for performing
MD simulations with the ``spkmd`` script.

Basic command line input
========================

The inputs which need to be provided for every ``spkmd`` run are:

* a simulation directory (``simulation_dir``)
* the initial molecular geometry in an ASE readable format (``system.molecule_file``)
* the path to a trained ML model (``calculator.model_file``)
* and the cutoff used in the neighbor list (``calculator.neighbor_list.cutoff``)

Assuming the model and structure file are present in the local directory, the  command line call would be::

    spkmd simulation_dir=mdtut_cli system.molecule_file=md_ethanol.xyz calculator.model_file=md_ethanol.model calculator.neighbor_list.cutoff=5.0

This command would carry out a classical NVE simulation in the ``mdtut_cli`` directory, running on the GPU for 1000000
steps, using a time step of 0.5 fs (the device can be switched by appending ``device=cpu``). It would further
automatically set up checkpointing and logging to HDF5 and tensorboard as described above.

Running the command will print out the full config used for the simulation::

    ⚙ Running with the following config:
    ├── device
    │   └── cuda
    ├── precision
    │   └── 32
    ├── seed
    │   └── None
    ├── simulation_dir
    │   └── mdtut_cli
    ├── overwrite
    │   └── False
    ├── restart
    │   └── None
    ├── calculator
    │   └── neighbor_list:
    │         _target_: schnetpack.md.neighborlist_md.NeighborListMD
    │         cutoff: 5.0
    │         cutoff_shell: 2.0
    │         requires_triples: false
    │         base_nbl: schnetpack.transform.ASENeighborList
    │         collate_fn: schnetpack.data.loader._atoms_collate_fn
    │       _target_: schnetpack.md.calculators.SchNetPackCalculator
    │       required_properties:
    │       - energy
    │       - forces
    │       model_file: md_ethanol.model
    │       force_key: forces
    │       energy_unit: kcal / mol
    │       position_unit: Angstrom
    │       energy_key: energy
    │       stress_key: null
    │       script_model: false
    ├── system
    │   └── initializer:
    │         _target_: schnetpack.md.UniformInit
    │         temperature: 300
    │         remove_center_of_mass: true
    │         remove_translation: true
    │         remove_rotation: true
    │         wrap_positions: false
    │       molecule_file: md_ethanol.xyz
    │       load_system_state: null
    │       n_replicas: 1
    │       position_unit_input: Angstrom
    │       mass_unit_input: 1.0
    ├── dynamics
    │   └── integrator:
    │         _target_: schnetpack.md.integrators.VelocityVerlet
    │         time_step: 0.5
    │       n_steps: 1000000
    │       thermostat: null
    │       barostat: null
    │       progress: true
    │       simulation_hooks: []
    └── callbacks
        └── checkpoint:
              _target_: schnetpack.md.simulation_hooks.Checkpoint
              checkpoint_file: checkpoint.chk
              every_n_steps: 10
            hdf5:
              _target_: schnetpack.md.simulation_hooks.FileLogger
              filename: simulation.hdf5
              buffer_size: 100
              data_streams:
              - _target_: schnetpack.md.simulation_hooks.MoleculeStream
                store_velocities: true
              - _target_: schnetpack.md.simulation_hooks.PropertyStream
                target_properties:
                - energy
              every_n_steps: 1
              precision: 32
            tensorboard:
              _target_: schnetpack.md.simulation_hooks.TensorBoardLogger
              log_file: logs
              properties:
              - energy
              - temperature
              every_n_steps: 10

As can be seen, the config is structured into different blocks, e.g. ``calculator``, ``system``,
``dynamics`` and ``callbacks`` specifying the machine learning model, the system to be simulated, the settings for the
MD simulation and logging instructions.

Customizing the simulation
==========================

In the following, we will describe how to configure a simulation by overwriting existing configurations and loading
additional settings from predefined configs. As an example, we will carry out a MD run using a Langevin thermostat.

For this, we first need to change the number of simulation steps from 1000000 to 20000.
Since the corresponding config entry is ``n_steps`` in the ``dynamics`` block, this can be done by adding
``dynamics.n_steps=20000`` to the command line. Changing other existing config entries can be done in a similar manner.

We also need to add a thermostat to the simulation.
For convenience, several thermostats are preconfigured in ``src/schnetpack/md/md_configs/dynamics/thermostat``.
To load the Langevin thermostat (``langevin``), we add the ``+dynamics/thermostat=langevin`` option to the command line
call::

    spkmd simulation_dir=mdtut_cli system.molecule_file=md_ethanol.xyz calculator.model_file=md_ethanol.model calculator.neighbor_list.cutoff=5.0 dynamics.n_steps=20000 +dynamics/thermostat=langevin

The simulation config will now show a different entry for the ``thermostat`` option in the ``dynamics`` block::

    │       thermostat:
    │         _target_: schnetpack.md.simulation_hooks.LangevinThermostat
    │         temperature_bath: 300.0
    │         time_constant: 100.0

Here, the thermostat temperature is already set to the desired 300 K.
Similar to the simulation steps, it could e.g. be changed to 500 K with the option
``dynamics.thermostat.temperature_bath=500``

We could also easily use another preconfigured thermostat (e.g. Nosé-Hover chains, ``+dynamic/thermostat=nhc``)
or add a barostat if we wanted to perform a constant pressure simulation (e.g. an isotropic Nosé-Hoover barostat,
``+dynamic/barostat=nhc_iso``). A similar syntax can be used to modify the neighbor list in the calculator
(e.g. to use a torch based implementation add ``calculator/neighbor_list=torch``) You might have noticed, that some
modifications use a ``+`` where others do not. A general rule is, that the ``+`` is required if the corresponding entry
did not exists before or was empty (e.g. ``thermostat: null`` in the very first config).

Using the CLI, it is also possible to perform more extensive modifications to the simulation.
To carry out a ring polymer molecular dynamics (RPMD) simulation via the CLI for example, we have to:

* switch the integrator from Velocity Verlet to a suitable RPMD integrator (``dynamics/integrator=rpmd``)
* set the number of beads/replicas (``system.n_replicas=4``)
* add a suitable thermostat (``+dynamics/thermostat=pile_local``)
* and change the number of steps to 50000 (``dynamics.n_steps=50000``)

We should also change the simulation directory.
The corresponding command would be

    spkmd simulation_dir=mdtut_cli_rpmd system.molecule_file=md_ethanol.xyz calculator.model_file=md_ethanol.model calculator.neighbor_list.cutoff=5.0 dynamics/integrator=rpmd system.n_replicas=4 +dynamics/thermostat=pile_local dynamics.n_steps=50000

A quick look at the ``dynamics.integrator`` block confirms that it has changed and also uses reasonable defaults for the
time step and bead temperature::

    │   └── integrator:
    │         _target_: schnetpack.md.integrators.RingPolymer
    │         time_step: 0.2
    │         temperature: 300.0

Running simulations from config files
=====================================

In some cases, it can be useful to run simulations using config files as input.
These can for example be created using the ``spkmd`` CLI and then fine-tuned to suit one's needs.

Full configs for the MD can either be found in the simulation directories (``mdtut_cli/.hydra/config.yaml``) or
generated with ``spkmd`` by adding the ``--cfg job`` option and redirecting the output to a ``yaml`` file. This can then
be saved, modified and used to run simulations.

The config file for the MD with the Langevin thermostat would look something like this::

    calculator:
      neighbor_list:
        _target_: schnetpack.md.neighborlist_md.NeighborListMD
        cutoff: 5.0
        cutoff_shell: 2.0
        requires_triples: false
        base_nbl: schnetpack.transform.ASENeighborList
        collate_fn: schnetpack.data.loader._atoms_collate_fn
      _target_: schnetpack.md.calculators.SchNetPackCalculator
      required_properties:
      - energy
      - forces
      model_file: md_ethanol.model
      force_key: forces
      energy_unit: kcal / mol
      position_unit: Angstrom
      energy_key: energy
      stress_key: null
      script_model: false
    system:
      initializer:
        _target_: schnetpack.md.UniformInit
        temperature: 300
        remove_center_of_mass: true
        remove_translation: true
        remove_rotation: true
        wrap_positions: false
      molecule_file: md_ethanol.xyz
      load_system_state: null
      n_replicas: 1
      position_unit_input: Angstrom
      mass_unit_input: 1.0
    dynamics:
      integrator:
        _target_: schnetpack.md.integrators.VelocityVerlet
        time_step: 0.5
      n_steps: 20000
      thermostat:
        _target_: schnetpack.md.simulation_hooks.LangevinThermostat
        temperature_bath: 300.0
        time_constant: 100.0
      barostat: null
      progress: true
      simulation_hooks: []
    callbacks:
      checkpoint:
        _target_: schnetpack.md.simulation_hooks.Checkpoint
        checkpoint_file: checkpoint.chk
        every_n_steps: 10
      hdf5:
        _target_: schnetpack.md.simulation_hooks.FileLogger
        filename: simulation.hdf5
        buffer_size: 100
        data_streams:
        - _target_: schnetpack.md.simulation_hooks.MoleculeStream
          store_velocities: true
        - _target_: schnetpack.md.simulation_hooks.PropertyStream
          target_properties:
          - energy
        every_n_steps: 1
        precision: ${precision}
      tensorboard:
        _target_: schnetpack.md.simulation_hooks.TensorBoardLogger
        log_file: logs
        properties:
        - energy
        - temperature
        every_n_steps: 10
    device: cuda
    precision: 32
    seed: null
    simulation_dir: mdtut_cli
    overwrite: false
    restart: null

Settings can then be changed by modifying the corresponding entries.
E.g. to increase the simulation temperature to 500 K, the ``temperature_bath`` entry in the ``thermostat`` block can be
changed to 500.

Assuming the config file is e.g. stored in ``md_input_langevin.yaml``, it can be used to run the MD with the command::

    spkmd simulation_dir=md_from_config load_config=md_input_langevin.yaml

The ``simulation_dir`` option is still required, due to how hydra resolves configs.
Any ``simulation_dir`` entries in the provided config file will be ignored.

Since the ``hydra`` parser operates on classes from python modules, they can also be easily adapted to integrate external modules, e.g. custom calculators for simulations.

