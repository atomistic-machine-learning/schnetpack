Using SchNet to evaluate Datasets
=================================

In order to predict missing properties with the model that has been trained in
:ref:`train_model`, you can use the ``spk_eval.py`` script. For demonstration
purposes you can use an ethanol snippet without energies and forces. Store
the snippet in your tutorials folder and predict the labels by running::

   $ spk_eval.py with model_dir=tutorials/model_dir dataset.path=tutorials/ethanol_missing.xyz out_path=tutorials/ethanol_eval.db device=cpu

Select your trained model by setting the ``model_dir`` argument to the directory
of your trained model. If you want to evaluate your data on a GPU, set
``device=cuda``. The default device is set to CPU.

Selecting the Input-Data
------------------------
Select the input data that you want to evaluate by setting
``dataset.path``. Your data must be provided as ``.xyz``, ``.extxyt`` or as
``ase.db``.

Selecting the Output-Format
---------------------------
The location to the evaluated data is set with ``out_path``. You
can select between an ``ase.db`` and an ``.npz`` output format. Just add the
desired file-extension to your file path and the script will automatically
use the desired output format.