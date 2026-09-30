"""
Relaxation: drive structures downhill on a force-like field.

For a force field that field is the forces, and the driver is
:class:`Relaxer`, stepping with a step rule from
:mod:`~schnetpack.dynamics.integrators` (L-BFGS by default, Euler for
steepest descent). For GPFF it is the pseudo-force and the relaxer is
:class:`DirectDenoising`.
"""

from schnetpack.dynamics.relax.direct_denoising import *
from schnetpack.dynamics.relax.relaxer import *
