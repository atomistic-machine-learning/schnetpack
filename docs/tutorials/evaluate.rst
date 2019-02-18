Using SchNet to evaluate datasets
=================================

Im order to predict energies and forces with SchNet run::

   $ evaluate_model evaluate with ...

Selecting the model
-------------------

During every training session the best model is saved as *best_model* by
default. In order to select a model you need to add the path to the desired
model by adding ``model_path=path/to/best_model`` to the run arguments.

Requirements for the input data
-------------------------------

The input data for the evaluation of the model needs to be a valid extended
xyz-file as described in **add section** or an ``ase`` database. If the
selected input data is provided as extended xyz file, it will automatically
be transformed into an ``ase`` database, since SchNet is meant to be used
with these databases. The database will be stored next to the xyz-file. In
order to select the input data file, add ``dataset.path=path/to/file`` to
your run arguments. The input file needs to be a valid extended xyz file or
an ase database.

Selecting the output format
---------------------------

By default, the evaluation of the input file will be stored as ase database
in your evaluation folder. It is also possible to return the evaluated data
as .npz file. Just add ``evaluator.out_file=filename.ext`` to your run
arguments. The file-type of your evaluated data will be detected
automatically according to the extension of your out_file.

Selecting the device for evaluation
-----------------------------------

By default the evaluation will be performed on your *CPU*. If you want to use
a *GPU* instead, add ``device=cuda`` to your run arguments.
