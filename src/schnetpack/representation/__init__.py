"""
Classes for constructing the different representations available in SchnetPack. This encompasses SchNet [#schnet4]_,
Behler-type atom centered symmetry functions (ACSF) [#acsf2]_ and a weighted variant thereof (wACSF) [#wacsf2]_.

References
----------
.. [#schnet4] Schütt, Arbabzadah, Chmiela, Müller, Tkatchenko:
   Quantum-chemical insights from deep tensor neural networks.
   Nature Communications, 8, 13890. 2017.
.. [#acsf2] Behler:
   Atom-centered symmetry functions for constructing high-dimensional neural network potentials.
   The Journal of Chemical Physics 134. 074106. 2011.
.. [#wacsf2] Gastegger, Schwiedrzik, Bittermann, Berzsenyi, Marquetand:
   wACSF -- Weighted atom-centered symmetry functions as descriptors in machine learning potentials.
   The Journal of Chemical Physics 148 (24), 241709. 2018.
"""

from schnetpack.representation.schnet import SchNet, SchNetInteraction
from schnetpack.representation.hdnn import (
    BehlerSFBlock,
    StandardizeSF,
    SymmetryFunctions,
)
