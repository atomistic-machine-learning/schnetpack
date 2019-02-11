Training Script
===============

The training script is executed by running

::

    run_schnetpack.py train with ...

The script consists of three ingredients which define the configurations
for the model, the trainer and the dataset. The possible parameters for
the *Experiment* are:

-  **loss\_tradeoff** (*dict*) - define tradeoff for loss function and
   certain properties if required
-  **overwrite** (*bool*) - overwrite the output folder if True
-  **additional\_outputs** (*list*) - additional model outputs that are not
   used for the loss calculation
-  **batch\_size** (*int*) - batch size for for training
-  **num\_train** (*int or float*) - number of training points; use relative
   value of all datapoints if smaller than 1
-  **num\_val** (*int or float*) - number of validation points; use relative
   value of all datapoints if smaller than 1
-  **num\_workers** (*int*) - number of workers for the dataloader
-  **device** (*str*) - device on which the model is trained
-  **experiment\_dir** (*str*) - path to experiments folder
-  **training\_dir** (*str*) - path to training folder
-  **properties** (*list*) - properties that the model uses for training
-  **overwrite** (*bool*) - overwrite the output folder if True
-  **additional\_outputs** (*list*) - model outputs that are not used for
   loss calculation
-  **batch\_size** (*int*) - batch size

Choosing the Dataset
--------------------

| The dataset ingredient comes with different implementations for
  datasets. Choose the dataset by calling the named-configuration of the
  dataset. The possible datasets are QM9 (as *qm9*), ISO17 (as *iso17*),
  ANI1 (as *ani1*), MD17 (as *md17*) and Materials Project (as
  *matproj*). These datasets are downloaded automatically if they are
  not found at the default location that is defined by the *dbpath*
  parameter.
| In order to use other datasets change the *dbpath* parameter to the
  path of the database and define the property mapping between the
  dataset properties and the model properties. The possible parameters
  for the dataset ingredient are:

-  **dbpath** (*str*) - path to the database
-  **property\_mapping** (*dict*) - defines a mapping between the model
   properties (as keys) and the properties of the database (as values)

Implement a new dataset that derives from the
*src.schnetpack.data.AtomsData* class in order to define a custom
download function for the new database. The *\_download()* method must
be overwritten and a new named-configuration should be added to
*src/schnetpach/sacred/dataset\_ingredients.py*. Additionally the
capture function *build\_dataset()* must be modified in order to use the
new dataset class.

Model Ingredient
----------------

The model is selected by calling the named-configuration with the model
settings. SchNet is at the moment the only possible choice. In order to
choose custom models, add a new named-configuration to the model
ingredient and modify the *build\_model()* method.

Trainer Ingredient
------------------

The trainer ingredient uses the Adam-optimizer with a learning rate of
10\ :sup:`-4` and no logging hooks or schedules by default. The following
parameters can be modified:

-  **optimizer** (*str*) - name of the optimizer
-  **learning\_rate** (*float*) - initial learning rate
-  **max\_epochs** (*int*) - maximum number of epochs
-  **logging\_hooks** (*list*) - list of logging hooks
-  **metrics** (*list*) - list of observables that are used for logging the
   training
-  **max\_steps** (*int*) - maximum number of steps

The hooks and schedules should be added by using their
named-configurations. The implemented named\_configs are:

-  **base\_hooks** - adds default logging hooks to the training procedure
-  **sgdr** - adds the SGDR schedule to the training procedure
-  **plateau** - adds the plateau schedule to the training procedure
-  **early\_stopping** - adds an early stopping hook to the training
   procedure
