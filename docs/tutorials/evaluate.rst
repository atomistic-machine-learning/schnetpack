Using SchNet to evaluate Datasets
=================================

Im order to predict properties with a trained SchNet model run::

   $ spk_eval.py with model_dir=tutorials/model_dir in_path=tutorials/ethanol_missing.xyz out_path=tutorials/ethanol_eval.db device=cpu

Select your trained model by setting the ``model_dir`` argument to the directory
of your trained model. If you want to evaluate your data on a GPU, set
``device=cuda``. The default device is set to CPU.

Selecting the Input-Data
------------------------
Select the input data that you want to evaluate by setting
``dataset.path``. Your data must be provided as ``.xyz``, ``.extxyt`` or as
``ase.db``. If you want to use the test split for the evaluation, add
``test_set`` to your run arguments. This will automatically get the IDs of
the test split from your model directory.

Selecting the Output-Format
---------------------------
The location to the evaluated data is set with ``out_path``. You
can select between an ``ase.db`` and an ``.npz`` output format. Just add the
desired file-extension to your file path and the script will automatically
use the desired output format.