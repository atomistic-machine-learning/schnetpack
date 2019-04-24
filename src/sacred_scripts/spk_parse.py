import os
from sacred import Experiment
from schnetpack.data import generate_db
from shutil import rmtree

parsing = Experiment("parsing")


@parsing.config
def config():
    """
    Settings for the db parser.
    """
    file_path = None  # path to the input file
    db_path = None  # path to the output db
    atomic_properties = (
        "Properties=species:S:1:pos:R:3"
    )  # atomic properties of the input file
    molecular_properties = ["energy"]  # molecular properties of the input file
    overwrite = False


@parsing.named_config
def forces():
    """
    Adds forces to the atomic property string.
    """
    atomic_properties = "Properties=species:S:1:pos:R:3:forces:R:3"


@parsing.capture
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

    if not os.path.exists(output_dir) and output_dir not in ["", "."]:
        os.makedirs(output_dir)


@parsing.command
def parse(_log, _config, file_path, db_path, atomic_properties, molecular_properties):
    """
    Runs the data parsing.

    Args:
        file_path (str): path to input file
        db_path (str): path to output file
        atomic_properties (str): property string for .xyz files
        molecular_properties (list): list with molecular properties in
            comment section
    """
    output_dir = os.path.dirname(db_path)
    create_dirs(_log=_log, output_dir=output_dir)
    generate_db(file_path, db_path, atomic_properties, molecular_properties)


@parsing.automain
def main():
    parse()
