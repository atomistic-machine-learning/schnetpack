"""
The ``spk*`` commands, one module each (``spkmd`` lives in :mod:`schnetpack.md.cli`):

- ``spktrain`` (:mod:`.train`), ``spkpredict`` (:mod:`.predict`): Hydra commands on
  the configs in ``schnetpack/configs``.
- ``spkconvert`` (:mod:`.convert`), ``spkdeploy`` (:mod:`.deploy`): standalone tools.

Importing the package registers the config resolvers ``uuid``, ``petname`` and
``tmpdir``.
"""

import tempfile
import uuid

import petname
from omegaconf import OmegaConf

OmegaConf.register_new_resolver("uuid", lambda x: str(uuid.uuid1()), use_cache=True)
OmegaConf.register_new_resolver("petname", lambda: petname.generate())
OmegaConf.register_new_resolver("tmpdir", tempfile.mkdtemp, use_cache=True)
