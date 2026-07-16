"""
Models: everything that maps a batch of atoms to predictions.

Three layers, and the split between them is what the sub-packages name:

- :mod:`~schnetpack.model.representation` — the backbones (SchNet, PaiNN,
  SO3net, FieldSchNet). Each turns a geometry into per-atom features and knows
  nothing about what is predicted from them.
- :mod:`~schnetpack.model.atomistic` — the input and output modules that read
  those features: heads (``Atomwise``, ``DipoleMoment``, ``Forces``, ...),
  geometry preprocessing (``PairwiseDistances``) and physics terms
  (``EnergyEwald``, ``ZBLRepulsionEnergy``, ...).
- :mod:`~schnetpack.model.base` — the container that runs them in order
  (``NeuralNetworkPotential``), plus the postprocessing and dtype handling every
  model shares (``AtomisticModel``).

Every public name is re-exported here, so ``spk.model.PaiNN`` and
``spk.model.Atomwise`` both work and the sub-package path is only needed when it
aids the reader. The sub-modules stay importable under their own names for that::

    from schnetpack.model import PaiNN, Atomwise, NeuralNetworkPotential
    from schnetpack.model.representation import PaiNN      # equivalent
"""

from schnetpack.model import atomistic
from schnetpack.model import representation

from schnetpack.model.base import *
from schnetpack.model.atomistic import *
from schnetpack.model.representation import *
