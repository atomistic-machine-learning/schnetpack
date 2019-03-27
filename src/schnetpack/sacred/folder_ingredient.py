import os
import yaml
from shutil import rmtree
from sacred import Ingredient


folder_ing = Ingredient("folder")


@folder_ing.config
def config():
    overwrite = False


@folder_ing.capture
def save_config(_config, output_dir):
    """
    Save the configuration to the model directory.

    Args:
        _config (dict): configuration of the experiment
        output_dir (str): path to the training directory

    """
    with open(os.path.join(output_dir, "config.yaml"), "w") as f:
        yaml.dump(_config, f, default_flow_style=False)


@folder_ing.capture
def create_dirs(_log, output_dir, overwrite):
    """
    Create the directory for the experiment.

    Args:
        output_dir (str): path to the output directory
        overwrite (bool): overwrites the model directory if True
    """
    _log.info("Create model directory")

    if output_dir is None:
        raise ValueError("Config `output_dir` has to be set!")

    if os.path.exists(output_dir) and not overwrite:
        raise ValueError(
            "Output directory already exists (set overwrite flag?):", output_dir
        )

    if os.path.exists(output_dir) and overwrite:
        rmtree(output_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
