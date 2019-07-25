"""
Calculators for molecular dynamics simulations in SchNetPack. These calculators take the current structures
in the :obj:`schnetpack.md.System` class and compute the associated forces. Other properties can be returned,
but molecular forces are the bare minimum for driving a simulations. All calculators should be derived from
the base classes :obj:`MDCalculator` (if no external code is called) or :obj:`QMCalculator`
(if external codes are required).

Currently implemented machine learning calculators include the :obj:`SchnetPackCalculator` for all models
generated with SchNetPack and the :obj:`SGDMLCalculator` for sGDML models [#sgdml5]_ .
In addition, an :obj:`OrcaCalculator` class can be used to carry out molecular dynamics using the ORCA
electronic structure code [#orca2]_ .

References
----------
.. [#sgdml5] Chmiela, Sauceda, Müller, Tkatchenko:
   Towards Exact Molecular Dynamics Simulations with Machine-Learned Force Fields.
   Nature Communications, 9 (1), 3887. 2018.
.. [#orca2] Neese:
   The ORCA program system.
   WIREs Comput Mol Sci, 2 (1), 73-78. 2012.
"""
from .basic_calculators import *
from .orca_calculator import *
from .schnet_calculator import *
from .sgdml_calculator import *
