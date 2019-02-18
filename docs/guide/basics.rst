.. _sacred-basics:

=============
Sacred Basics
=============

Running Scripts from Config-Files
---------------------------------

The easiest way to run sacred experiments is to use the provided
config-files. Theses config-files contain default values for the
different sacred-scripts. Every time a script is executed, the
configuration is stored in the experiments folder. In order to execute a
script from the config-file run the script via the commandline with the
sacred command and the path to the config-file:

::

    my_script.py <command> with <path-to-config-file>

The configuration can easily be changed by modifying the config-file.

Setting Parameters using the Commandline
----------------------------------------

In order to run sacred-scripts from the commandline without using the
config-file, enter the desired parameters directly in the commandline.
The scripts consist of a sacred *Experiment* with a default
configuration, different named configurations and commands. Usually the
*Experiment* is made out of different sacred *Ingredients* which also
consist of a default configuration and named configurations. The
following example will show the usage of the commandline options:

::

    # my_scripy.py
    from sacred import Ingredient, Experiment

    ing = Ingredient('my_ingredient')

    @ing.config
    def config():
        i1 = 1
        i2 = 2

    @ing.named_config
    def other_config():
        i2 = 20
        i3 = 30
        
    ex = Experiment('my_experiment')

    @ex.config
    def config():
         e1 = 1
         
    @ex.named_config
    def exp_other_config():
        e1 = 10

    @ex.command
    def my_command():
        print(i1, i2, e1)
            

The configuration can now be changed by directly setting the parameters:

::

     my_script.py my_command with my_ingredient.i1=90 e1=90
     

This yields the output ``90, 2, 90``. In order to use the named
configurations, add the config name to the run arguments:

::

    my_script.py my_command with my_ingredient.other_config exp_other_config

This yields the output ``1, 20, 10``.

Implementing new Configurations
-------------------------------

If other configurations are needed for the experiments, it is possible
to implement new parameters and new named configurations. All
implemented ingredients are located at *src/schnetpack/sacred/*. The
experiment files are located at *src/sacred\_scripts*.
