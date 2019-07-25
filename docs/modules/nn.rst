:mod:`schnetpack.nn`
====================


.. automodule:: schnetpack.nn


Basic layers
------------

.. automodule:: schnetpack.nn.base

.. autoclass:: schnetpack.nn.Dense
   :members:

.. autoclass:: schnetpack.nn.GetItem
   :members:

.. autoclass:: schnetpack.nn.ScaleShift
   :members:

.. autoclass:: schnetpack.nn.Standardize
   :members:

.. autoclass:: schnetpack.nn.Aggregate
   :members:

Blocks
------

.. automodule:: schnetpack.nn.blocks

.. autoclass:: schnetpack.nn.MLP
   :members:

.. autoclass:: schnetpack.nn.TiledMultiLayerNN
   :members:

.. autoclass:: schnetpack.nn.ElementalGate
   :members:

.. autoclass:: schnetpack.nn.GatedNetwork
   :members:


Convolutions
------------

.. automodule:: schnetpack.nn.cfconv

.. autoclass:: schnetpack.nn.CFConv
   :members:


Cutoff
------

.. automodule:: schnetpack.nn.cutoff

.. autoclass:: schnetpack.nn.CosineCutoff
   :members:

.. autoclass:: schnetpack.nn.MollifierCutoff
   :members:

.. autoclass:: schnetpack.nn.HardCutoff
   :members:


Neighbors
---------

.. automodule:: schnetpack.nn.neighbors

.. autoclass:: schnetpack.nn.AtomDistances
   :members:

.. autofunction:: schnetpack.nn.atom_distances

.. autoclass:: schnetpack.nn.TriplesDistances
   :members:

.. autoclass:: schnetpack.nn.NeighborElements
   :members:


ACSF
----

.. automodule:: schnetpack.nn.acsf

.. autoclass:: schnetpack.nn.AngularDistribution
   :members:

.. autoclass:: schnetpack.nn.BehlerAngular
   :members:

.. autoclass:: schnetpack.nn.GaussianSmearing
   :members:

.. autoclass:: schnetpack.nn.RadialDistribution
   :members:


Activation functions
--------------------

.. automodule:: schnetpack.nn.activations

.. autofunction:: schnetpack.nn.shifted_softplus


