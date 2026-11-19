"""
The ``spk*`` commands, one module each (``spkmd`` lives in :mod:`schnetpack.md.cli`):

- ``spktrain`` (:mod:`.train`), ``spkpredict`` (:mod:`.predict`): Hydra commands on
  the configs in ``schnetpack/configs``.
- ``spktest`` (:mod:`.test`), ``spksample`` (:mod:`.sample`): on a finished spktrain
  run, from its ``config.yaml``.
- ``spkevaluate`` (:mod:`.evaluate`), ``spkconvert`` (:mod:`.convert`),
  ``spkdeploy`` (:mod:`.deploy`): standalone tools.

Importing the package registers the config resolvers ``uuid``, ``petname`` and
``tmpdir``.
"""

import os
import tempfile
import uuid

import petname
from omegaconf import DictConfig, OmegaConf

OmegaConf.register_new_resolver("uuid", lambda x: str(uuid.uuid1()), use_cache=True)
OmegaConf.register_new_resolver("petname", lambda: petname.generate())
OmegaConf.register_new_resolver("tmpdir", tempfile.mkdtemp, use_cache=True)


def _run_config(
    run_dir: str, overrides: list[str], defaults: DictConfig | None = None
) -> DictConfig:
    """A run's config.yaml, merged over ``defaults`` and under the key=value
    ``overrides``. run.path and run.id point at run_dir (and with them a split file
    in it); run.work_dir, Hydra's ${hydra:runtime.cwd}, is the current folder."""
    run_dir = os.path.abspath(run_dir)
    config = OmegaConf.load(os.path.join(run_dir, "config.yaml"))
    config.run.work_dir = os.getcwd()
    config.run.path, config.run.id = os.path.split(run_dir)
    return OmegaConf.merge(defaults or {}, config, OmegaConf.from_dotlist(overrides))
