.. _train model:

Train a model with SchNetPack
=============================


The easiest way to train a sacred script on a dataset is to use the provided
sacred scripts. A general guide to sacred can be found at section
:ref:`sacred basics`.
In order to train a model run::

    $ spk_train.py train with model_dir=path/to/store/model dataset.<qm9/...>
     device=<cpu/cuda> schedule_hooks.reduce_on_plateau

This will automatically download the dataset, train a SchNet model and use
80% of the data for training, 10% of the data for validation and the
remaining 10% for testing. Furthermore a new directory will be created at the
location that you have defined with the ``model_dir`` argument. The new
directory is used to store checkpoints, logging files and your best model of
the training session. By setting the ``schedule_hooks`` to
``reduce_on_plateau`` the learning rate will automatically decrease during
training, if the training does not improve any further.

Choosing the Dataset
--------------------

You can either choose from a number of pre-implemented datasets which are
downloaded automatically, or train the model on your own data. In order to
use the pre-implemented datasets, add ``dataset.<name>`` to your
run arguments. Available datasets are QM9 (as *qm9*), ISO17 (as *iso17*),
ANI1 (as *ani1*), MD17 (as *md17*) and Materials Project (as *matproj*).
In order to use your own data you must provide it as an ``ase.db``.
Visit :ref:`Prepare Data` for additional information on the requirements for
your dataset. After preparing your data use ``dataset.dbpath=<path>`` instead
of choosing a pre-implemented dataset. Additionally you will need to define a
property_mapping in order to connect the model properties to the dataset
properties. Therefore add
``dataset.property_mapping="model_property1:data_property1,..."`` to your run
arguments.


Monitoring Training with TensorBoard
------------------------------------

The default training session will store TensorBoard files for monitoring your
training session in *model_dir/log*. In order to use
TensorBoard open a new terminal and run::

    $ tensorboard --logdir model_dir/log

This will return url of your TensorBoard. Paste the url to your browser and
it will automatically show your training session.