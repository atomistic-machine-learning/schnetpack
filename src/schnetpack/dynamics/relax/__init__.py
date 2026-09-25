"""
Relaxation: drive structures downhill on a force-like field until they stop.

For a force field that field is the forces; for GPFF it is the pseudo-force,
whose relaxer is :class:`DirectDenoising`. The batch-wise optimizers
(L-BFGS and friends) join this package when the batch-wise optimizer port
lands.
"""

from schnetpack.dynamics.relax.direct_denoising import *
