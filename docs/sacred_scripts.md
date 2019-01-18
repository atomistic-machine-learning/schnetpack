# Training the Model and Running MD Simulations using Sacred Scripts

## Sacred Basics

### Running Scripts from Config-Files

The easiest way to run sacred experiments is to use the provided config-files. Theses config-files contain default
values for the different sacred-scripts. Every time a script is executed, the configuration is stored in the experiments
folder. In order to execute a script from the config-file run the script via the commandline with the sacred command and
the path to the config-file:

    my_script.py <command> with <path-to-config-file>
    
The configuration can easily be changed by modifying the config-file.

### Setting Parameters using the Commandline

In order to run sacred-scripts from the commandline without using the config-file, enter the desired parameters directly
in the commandline. The scripts consist of a sacred _Experiment_ with a default configuration, different named
configurations and commands. Usually the _Experiment_ is made out of different sacred _Ingredients_ which also consist
of a default configuration and named configurations. The following example will show the usage of the commandline
options:

    # my_scripy.py
    from sacred import Ingredient, Experiment
    
    ing = Ingredient('my_ingredient')
    
    @ing.config
    def config():
        i1 = 1
        i2 = 2
    
    @ing.named_config
    def other_config():
        i2 = 20
        i3 = 30
        
    ex = Experiment('my_experiment')
    
    @ex.config
    def config():
         e1 = 1
         
    @ex.named_config
    def exp_other_config():
        e1 = 10
    
    @ex.command
    def my_command():
        print(i1, i2, e1)
            
The configuration can now be changed by directly setting the parameters:

     my_script.py my_command with my_ingredient.i1=90 e1=90
     
This yields the output '90, 2, 90'. In order to use the named configurations, add the config name to the run arguments:

    my_script.py my_command with my_ingredient.other_config exp_other_config
    
This yields the output '1, 20, 10'.  

### Implementing new Configurations

If other configurations are needed for the experiments, it is possible to implement new parameters and new named
configurations. All implemented ingredients are located at _src/schnetpack/sacred/_. The experiment files are located at
_src/sacred_scripts_.

## Training Script

The training script is executed by running

    run_schnetpack.py train with ...
    
The script consists of three ingredients which define the configurations for the model, the trainer and the dataset.
The possible parameters for the _Experiment_ are:

- _loss_tradeoff (dict)_: define tradeoff for loss function and certain properties if required
- _overwrite (bool)_: overwrite the output folder if True
- _additional_outputs (list)_: additional model outputs that are not used for the loss calculation 
- _batch_size (int)_: batch size for for training
- _num_train (int or float)_: number of training points; use relative value of all datapoints if smaller than 1
- _num_val (int or float)_: number of validation points; use relative value of all datapoints if smaller than 1
- _num_workers (int)_: number of workers for the dataloader
- _device (str)_: device on which the model is trained
- _experiment_dir (str)_: path to experiments folder
- _training_dir (str)_: path to training folder 
- _properties (list)_: properties that the model uses for training
- _overwrite (bool)_: overwrite the output folder if True
- _additional_outputs (list)_: model outputs that are not used for loss calculation
- _batch_size (int)_: batch size

### Choosing the Dataset

The dataset ingredient comes with different implementations for datasets. Choose the dataset by calling the
named-configuration of the dataset. The possible datasets are QM9 (as _qm9_), ISO17 (as _iso17_), ANI1 (as _ani1_), MD17
(as _md17_) and Materials Project (as _matproj_). These datasets are downloaded automatically if they are not found at
the default location that is defined by the _dbpath_ parameter.  
In order to use other datasets change the _dbpath_ parameter to the path of the database and define the
property mapping between the dataset properties and the model properties. The possible parameters for the dataset
ingredient are:

- _dbpath (str)_: path to the database 
- _property_mapping (dict)_: defines a mapping between the model properties (as keys) and the properties of the database
                             (as values)
                             
Implement a new dataset that derives from the _src.schnetpack.data.AtomsData_ class in order to define a custom download
function for the new database. The __download()_ method must be overwritten and a new named-configuration should be
added to _src/schnetpach/sacred/dataset_ingredients.py_. Additionally the capture function _build_dataset()_ must be
modified in order to use the new dataset class.

### Model Ingredient

The model is selected by calling the named-configuration with the model settings. SchNet is at the moment the only
possible choice. In order to choose custom models, add a new named-configuration to the model ingredient and modify the
_build_model()_ method.

### Trainer Ingredient

The trainer ingredient uses the Adam-optimizer with a learning rate of 10<sup>-4</sup> and no logging hooks or schedules
by default. The following parameters can be modified: 

- _optimizer (str)_: name of the optimizer
- _learning_rate (float)_: initial learning rate
- _max_epochs (int)_: maximum number of epochs
- _logging_hooks (list)_: list of logging hooks
- _metrics (list)_: list of observables that are used for logging the training
- _max_steps (int)_: maximum number of steps

The hooks and schedules should be added by using their named-configurations. The implemented named_configs are:

- _base_hooks_: adds default logging hooks to the training procedure
- _sgdr_: adds the SGDR schedule to the training procedure
- _plateau_: adds the plateau schedule to the training procedure
- _early_stopping_: adds an early stopping hook to the training procedure

## MD Simulation Script

The script for molecular dynamics simulations is executed by running 

    run_md.py simulate with ...

The simulation script requires a trained model for the calculation of the molecular properties. The trained model can be
created using the training script. Similar to the training script, the simulation creates a folder for log-files and the
configuration-file. The simulation experiment consists of different ingredients for the simulator, the calculator, the
integrator, the system and the thermostat. The following parameters should be changed if needed:

- _experiment_dir (str)_: path to the experiments folder
- _simulation_steps (int)_: number of simulation steps
- _device (str)_: device to run the calculations on
- _path_to_molecules (str)_: path to the file with the molecule that should be simulated
- _simulation_dir (str)_: path to the simulation folder
- _training_dir (str)_: path to the simulation folder that is created by the training script
- _model_path (str)_: path to the trained model
- _overwrite (bool)_: overwrite the simulation folder if set to True

If the model is trained with the use of the provided training script use the same _experiment_dir_ as in the training
process. The trained model will be found automatically.

### Simulator Ingredient

The simulator ingredient provides the simulator parameters with several logging possibilities. The following parameters
can be set:

- _logging_hooks (list)_: list with names of logging hooks
- _data_streams (list)_: list with the data streams for the file logger
- _step (int)_: index of the initial simulation step
- _log_every_n_steps (int)_: update logging hooks after n steps
- _checkpoint_every_n_steps (int)_: store a checkpoint after n steps

In order to add basic loggers, call the named configuration _base_hooks_.

### Calculator Ingredient

The calculator ingredient provides the parameters for the calculator class. The following parameters can be set:
 
- _required_properties (list)_: list with properties that the calculator should calculate
- _force_handle (str)_: name of the force property in the model output
- _position_conversion (float)_: conversion factor for positions
- _force_conversion (float)_: conversion factor for forces
- _property_conversion (dict)_: dictionary with conversions for other properties

### Integrator Ingredient

The integrator ingredient defines the configuration for the integrator that is used for the simulation. The integrators
should be chosen by calling the named-configuration. Possible choices are _velocity_verlet_ and _ring_polymer_.

### System Ingredient

The system ingredient builds the system class. The only possible parameter that can be set is the number of replicas
_n_replicas_.

### Thermostat Ingredient

The thermostat ingredient defines the parameters for the thermostat. The choice of the thermostat should be made by
using the named-configurations. The possible thermostats are: _berendsen, gle, piglet, langevin, pile_local,
pile_global, nhc_ and _nhc_ring_polymer_.
