# Sacred Guide
## Running SchNetPack Experiments

The sacred scripts allow to train both SchNet and wACSF neural networks on 
different datasets. The training is started via the following command:
    
    run_schnetpack.py train with default_config dataset.<qm9/iso17/ani1/matproj/md17> model.<schnet/wacsf>

This will start the training session for the chosen model based on the 
dataset. If the dataset is not found at the default location, it will be 
downloaded automatically.

## Custom Settings

There are several ways to customize the configuration.

### Using the Config-File

In order to change configurations it is possible to save the whole 
configuration to a JSON-file by running:

    run_schnetpack.py save_config with ...
    
This creates a config-file at _./config.json_. The JSON-file can be modified 
and used to provide new settings by running:

    run_schnetpack.py with config.json

### Adding new Configurations

Another possible way of changing the configuration is to directly overwrite 
the configurations in the code or by adding some run arguments.
The sacred experiment consists of 3 different ingredients (model, 
trainer, dataset), which can be found at _src/schnetpack/sacred/_. The experiment 
and every ingredient have some configuration settings that can be overwritten 
using run arguments. In order to overwrite experiment variables just add
`variable_name=<new_value>` to the run arguments. In order to set ingredient 
variables just add `ingredient_name.variable_name=<new_value>`.  
If different variables should be overwritten it might also be useful to 
define a new _named_config_ for an ingredient or the experiment. This could 
be done according to:

   
    
    # creating the ingredient
    my_ing = Ingredient('ingredient_name')
    
    # base configuration
    @my_ing.config
    def config():
        var1 = 3.0
        var2 = 'test'
    
    # custom settings
    @my_ing.named_config
    def my_config():
        var1 = 50.0
        var2 = 'test2'

In order to use the custom config just add it to the run arguments:

    run_experiment.py with ingredient_name.my_config    


### Logging Hooks

In order to observe the progress of the training session, one might want to 
use the logging hooks. The script is able to write a CSV log and a 
TensorBoard log while training. The observables for the loggers can be chosen
 via the metrics settings. For a basic logging one needs to add the 
 _named_config_ `trainer.base_hooks` to the run parameters. This adds the CSV
  and the TensorBoard logging to the training process. For every predicted 
  property the RMSE and the MAE are tracked.  
 In order to choose more specific settings for the loggers it is possible to 
 define the logging settings via `trainer.logging_hooks=[...]` for the 
 loggers and `trainer.metrics=[...]` for the metrics settings. It is also 
 possible to just add a new _named_config_ with the logging settings to the 
 trainer ingredient.
 
 ### Setting a Schedule
 
 The trainer experiment comes with two different schedules that can be used 
 during training. In order to use these schedules one needs to add `trainer
 .<plateau/sgdr>` to the run arguments.
 
 ### Using a different Dataset

It is also possible to use the models with different datasets. Therefore one 
could store the custom dataset as a sqlite database and add the path to the run 
commands by replacing `dataset.<qm9/...>` with `dataset.dbpath=<path-to-your-.db-file>`. 
Furthermore it is necessary to add a property mapping to the commands that 
maps the model properties to the corresponding dataset properties. This is 
achieved by adding `dataset.property_mapping={<model_property>: 
<dataset_property>}`. This could for example look like this:

    run_schnetpack.py train with dataset.dbpath=path.db dataset.property_mapping={'energy': 'db_energy'} model.schnet

It is also possible to just add a new _named_config_ to the 
dataset_ingredient. Therefore one needs to add a new config with the 
name='CUSTOM' to the ingredient:

    @dataset_ingredient.named_config
    def my_dataset():
        dbpath = <path-to-your-db>
        property_mapping = {<model_property>: <property-of-db>}

If an automatic download is desired, one needs to write a new class that 
derives from the AtomsData class in _src/schnetpack/data/_. The new class 
just needs a new implementation of the __download()_ method and possibly also
 a new _create_subset()_ method. Afterwards one just needs to add the 
 _named_config_ that also overwrites the name variable and add it to the 
 _build_dataset()_ method.
