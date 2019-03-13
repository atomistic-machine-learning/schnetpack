Using SchNet to evaluate Datasets
=================================

Im order to predict properties with a trained SchNet-model run::

   $ spk_eval evaluate with model_path=<path> dataset.path=<input_data>
     out_path=<output_path> device=<cuda/cpu>

Select your trained model by setting the ``model_path`` argument to the path
of your trained model. In order to generate a trained model visit
:ref:`train model`. If you want to evaluate your data on a GPU, set
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