Train a model with SchNetPack
=============================

Train a model using the provided scripts
----------------------------------------

The easiest way to train a sacred script on a dataset is to use the provided
sacred scripts. A general guide to sacred can be found at :ref:'sacred-basics'.
Use the training script by calling::

    run_schnetpack.py train with ...

The experiment will automatically create an experiment folder with a training
folder in order to store all outputs. In order to set the experiment
directory add ``experiment_dir=path/to/experiment_dir`` to your run arguments
. The training will be executed on you *CPU* by default. In order to run the
experiment on your *GPU*, add ``device=cuda`` to your run. Additional model
outputs can be defined with ``additional_outputs=['property1', 'property2', .
..]``.

Choosing the Model
------------------

The training method uses SchNet as the default model. In order to change any
model parameters, add ``model.param_name=new_value``. You can change the
number of interaction layers by setting the ``model.n_interactions`` parameter,
the cutoff type with ``model.cutoff_network`` and the cutoff with
``model.cutoff``.

Choosing the Dataset
--------------------

You can either choose from a number of pre-implemented datasets which are
downloaded automatically, or train the model on your own data. In order to
use the pre-implemented datasets, add ``dataset.name_of_the_dataset`` to your
run arguments. Available datasets are QM9 (as *qm9*), ISO17 (as *iso17*),
ANI1 (as *ani1*), MD17 (as *md17*) and Materials Project (as *matproj*).
In order to use your own data, add ``dataset.dbpath=path/to/your_db`` to the
run arguments. Additionally you will need to define a property_mapping in
order to connect the model properties to the dataset properties with ``dataset
.property_mapping="model_property1:data_property1,
model_propery2:data_property2``. If your data is a valid extended xyz-file,
with forces and/or energies, you could also pass an xyz-file instead of a
database to ``dataset.dbpath``. This will automatically create a valid
database for you. For more information visit section **todo add link**.


Training Settings
-----------------

Choosing the Optimizer
^^^^^^^^^^^^^^^^^^^^^^

The scripts uses the Adam optimizer for training with a default initial
learning rate of 0.01. The learning rate can be changed with ``optimizer
.learning_rate``. Other optimizers are currently not supported.

Choosing Learning Rate Schedules
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Learning rate schedules can be set through pre-implemented named configs.
Possible choices are **reduce_on_plateau**, **warm_restart** and
**exponential_decay**.

Early Stopping
^^^^^^^^^^^^^^

In order to abort the training by using some criterion you can add stopping
hooks to the training session. Therefore you need to define ``stopping_hooks
.max_steps``, ``stopping_hooks.max_epochs`` and/or ``stopping_hooks
.patience``. This will automatically activate the early stopping hooks.

Logging Hooks
^^^^^^^^^^^^^

The training session creates a CSV- and a TensorBoard-file by default. In
order to set the desired metrics that need to be logged, add ``metrics
.names=['metric1', ...]`` to your run arguments.

Dataloader Settings
-------------------

In order to change the settings for the dataloader, set the parameters by
adding them to ``dataloader.param_name=new_value``. You can set the number of
workers (**num_workers**), the batch size (**batch_size**) and the number of
training/validation-points (**num_train** and **num_val**). The default
values are 4 workers with a batch size of 128 and 80% of the dataset for
training and 10% for validation.

