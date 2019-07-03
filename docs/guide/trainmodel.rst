.. _train model:

Train a model with SchNetPack
=============================

TODO new script example


Monitoring Training with TensorBoard
------------------------------------

The default training session will store TensorBoard files for monitoring your
training session in *model_dir/log*. In order to use
TensorBoard open a new terminal and run::

    $ tensorboard --logdir model_dir/log

This will return the url of your TensorBoard. Paste the url to your browser and
your training session will show up.


