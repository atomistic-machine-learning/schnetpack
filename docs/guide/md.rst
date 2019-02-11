MD Simulation Script
====================

The script for molecular dynamics simulations is executed by running

::

    run_md.py simulate with ...

The simulation script requires a trained model for the calculation of
the molecular properties. The trained model can be created using the
training script. Similar to the training script, the simulation creates
a folder for log-files and the configuration-file. The simulation
experiment consists of different ingredients for the simulator, the
calculator, the integrator, the system and the thermostat. The following
parameters should be changed if needed:

-  **experiment\_dir (*str*) - path to the experiments folder
-  **simulation\_steps (*int*) - number of simulation steps
-  **device (*str*) - device to run the calculations on
-  **path\_to\_molecules (*str*) - path to the file with the molecule that
   should be simulated
-  **simulation\_dir (*str*) - path to the simulation folder
-  **training\_dir (*str*) - path to the simulation folder that is created
   by the training script
-  **model\_path (*str*) - path to the trained model
-  **overwrite (*bool*) - overwrite the simulation folder if set to True

If the model is trained with the use of the provided training script use
the same *experiment\_dir* as in the training process. The trained model
will be found automatically.

Simulator Ingredient
--------------------

The simulator ingredient provides the simulator parameters with several
logging possibilities. The following parameters can be set:

-  **logging\_hooks (*list*) - list with names of logging hooks
-  **data\_streams (*list*) - list with the data streams for the file
   logger
-  **step (*int*) - index of the initial simulation step
-  **log\_every\_n\_steps (*int*) - update logging hooks after n steps
-  **checkpoint\_every\_n\_steps (*int*) - store a checkpoint after n steps

In order to add basic loggers, call the named configuration
*base\_hooks*.

Calculator Ingredient
---------------------

The calculator ingredient provides the parameters for the calculator
class. The following parameters can be set:

-  **required\_properties (*list*) - list with properties that the
   calculator should calculate
-  **force\_handle (*str*) - name of the force property in the model output
-  **position\_conversion (*float*) - conversion factor for positions
-  **force\_conversion (*float*) - conversion factor for forces
-  **property\_conversion (*dict*) - dictionary with conversions for other
   properties

Integrator Ingredient
---------------------

The integrator ingredient defines the configuration for the integrator
that is used for the simulation. The integrators should be chosen by
calling the named-configuration. Possible choices are *velocity\_verlet*
and *ring\_polymer*.

System Ingredient
-----------------

The system ingredient builds the system class. The only possible
parameter that can be set is the number of replicas *n\_replicas*.

Thermostat Ingredient
---------------------

The thermostat ingredient defines the parameters for the thermostat. The
choice of the thermostat should be made by using the
named-configurations. The possible thermostats are: *berendsen, gle,
piglet, langevin, pile\_local, pile\_global, nhc* and
*nhc\_ring\_polymer*.
