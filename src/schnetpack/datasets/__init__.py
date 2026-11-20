# The dataset modules used to star-import schnetpack.data, which passed all of
# its names on, e.g. schnetpack.datasets.ASEAtomsData. Kept for backward
# compatibility.
from schnetpack.data import *
from schnetpack.data import __getattr__  # the deprecated AtomsDataModule alias

from .ani1 import *
from .iso17 import *
from .materials_project import *
from .md17 import *
from .md22 import *
from .mptraj import *
from .omdb import *
from .qm7x import *
from .qm9 import *
from .rmd17 import *
