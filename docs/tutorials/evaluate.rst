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
with these databases. In order to select the extended xyz file as input you
need to add ``evaluation_data.data_type=xyz evaluation_data
.data_path=path/to/xyz_file.xyz``. If the input data is provided as database
file just add ``evaluation_data.data_path=path/to/database.db`` to your run
arguments.

Selecting the output format
---------------------------

The default output format is to overwrite the existing input database by
adding the predicted properties. If you want to receive the predictions as
npz file, add ``evaluator.output=to_npz evaluator.out_file=file_name.npz`` to
your run arguments. This will automatically create the folder *evaluate*
inside of your *experiment* directory. The directory for the output data and
also for the *experiment* directory can be changed by adding
``experiment_dir=...`` and ``output_dir=...``.

Selecting the device for evaluation
-----------------------------------

By default the evaluation will be performed on your *CPU*. If you want to use
a *GPU* instead, add ``device=cuda`` to your run arguments.
