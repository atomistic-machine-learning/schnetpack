.. _train model:

Train a model with SchNetPack
=============================

The easiest way to train a model on a dataset is to use the provided
sacred scripts. A general guide to sacred can be found at section
:ref:`sacred basics`.
In order to train a model on the the ethanol database that has been generated
according to :ref:`Prepare Data`, enter::

    $ spk_train.py with model_dir=tutorials/model_dir dataset.db_path=tutorials/ethanol.db device=cpu scheduling.reduce_on_plateau dataset.property_mapping='energy:energy'

This will train a SchNet model and use 80% of the data for training, 10% of
the data for validation and the remaining 10% for testing. Furthermore a new
directory will be created at the location that you have defined with the
``model_dir`` argument. The new directory is used to store checkpoints,
logging files and your best model of the training session. By setting the
``scheduling.reduce_on_plateau`` the learning rate will
automatically decrease during training, if the training does not improve any
further. When using your own datasets, you will need to define a
``property_mapping`` which connects the model properties to the properties of
your datasets, because properties in your datasets could have different
property names. For example the energy property could be called total_energy
in some datasets. In general, the property mapping should look like
``dataset.property_mapping="model_property1:data_property1,..."``. The
supported model properties are listed in
:class:`schnetpack.atomistic.Properties`.
For a detailed view on the possible parameters for the training script run::

    $ spk_train.py print_config

Choosing the Dataset
--------------------

You can either use your own datasets, as described in the example above, or
use one of the pre-implemented datasets. These datasets will be downloaded
and preprocessed automatically. In order to use the pre-implemented datasets,
replace ``dataset.db_path`` and ``dataset.property_mapping`` with
``dataset.<name>``. Available datasets are QM9 (as ``qm9``), ISO17 (as
``iso17``),
ANI1 (as ``ani1``), MD17 (as ``md17``) and Materials Project (as ``matproj``).


Monitoring Training Sessions
----------------------------

Every training session creates a CSV- and a TensorBoard-log per default. The
logging files can be used to monitor your training session. The CSV-log is
stored at *model_dir/log.csv* and the TensorBoard-log is stored
at *model_dir/log*. In order to use TensorBoard open a new terminal and run::

    $ tensorboard --logdir model_dir/log

This will return the url of your TensorBoard. Paste the url to your browser and
your training session will show up.