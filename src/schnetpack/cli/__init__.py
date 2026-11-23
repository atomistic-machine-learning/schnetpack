"""
The ``spk`` command: ``spk <command> key=value ...``, one module per command, each
a Hydra app on its config in ``schnetpack/configs`` (``spkmd`` lives in
:mod:`schnetpack.md.cli`):

- ``spk train`` (:mod:`.train`): train a model; writes a run folder.
- ``spk test`` (:mod:`.test`): test a run's best checkpoint again (``run_dir=``).
- ``spk predict`` (:mod:`.predict`), ``spk sample`` (:mod:`.sample`),
  ``spk deploy`` (:mod:`.deploy`): use a trained model (``model_dir=``, a model folder
  or a run folder).
- ``spk evaluate`` (:mod:`.evaluate`), ``spk convert`` (:mod:`.convert`): tools for
  ASE dbs.

Every command but ``train`` writes only its own results, no Hydra output folder.
Importing the package registers the config resolvers ``uuid``, ``petname`` and
``tmpdir``.
"""

import importlib
import os
import sys
import tempfile
import uuid

import petname
from omegaconf import DictConfig, OmegaConf

OmegaConf.register_new_resolver("uuid", lambda x: str(uuid.uuid1()), use_cache=True)
OmegaConf.register_new_resolver("petname", lambda: petname.generate())
OmegaConf.register_new_resolver("tmpdir", tempfile.mkdtemp, use_cache=True)

COMMANDS = ("train", "test", "predict", "sample", "evaluate", "convert", "deploy")


def main():
    """``spk <command> key=value ...``: runs the command's Hydra app."""
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(
            "usage: spk <command> key=value ...\n\ncommands: "
            + ", ".join(COMMANDS)
            + "\n\n`spk <command> --help` prints the command's config."
        )
        sys.exit(0 if sys.argv[1:] in ([], ["-h"], ["--help"]) else 2)
    command = sys.argv.pop(1)
    sys.argv[0] = f"spk {command}"
    importlib.import_module(f"schnetpack.cli.{command}").main()


def _run_config(run_dir: str, data_dir: str | None = None) -> DictConfig:
    """A run's config.yaml. run.path and run.id point at run_dir (and with them a
    split file in it); run.work_dir, Hydra's ${hydra:runtime.cwd}, is the current
    folder; run.data_dir is ``data_dir`` when given."""
    run_dir = os.path.abspath(run_dir)
    config = OmegaConf.load(os.path.join(run_dir, "config.yaml"))
    config.run.work_dir = os.getcwd()
    config.run.path, config.run.id = os.path.split(run_dir)
    if data_dir is not None:
        config.run.data_dir = os.path.abspath(data_dir)
    return config
