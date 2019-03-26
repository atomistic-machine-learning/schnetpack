import yaml
import os
from sacred import Experiment
from schnetpack.data import generate_db

parsing = Experiment('parsing')


@parsing.config
def config():
    """
    Settings for the db parser.
    """
    file_path = None                    # path to the input file
    db_path = None                      # path to the output db
    atomic_properties =\
        'Properties=species:S:1:pos:R:3'# atomic properties of the input file
    molecular_properties = ['energy']   # molecular properties of the input file


@parsing.named_config
def forces():
    """
    Adds forces to the atomic property string.
    """
    atomic_properties = 'Properties=species:S:1:pos:R:3:forces:R:3'


@parsing.capture
def save_config(_config, cfg_dir):
    """
    Save the evaluation configuration.

    Args:
        _config (dict): configuration of the experiment
        cfg_dir (str): path to the config directory

    """
    os.makedirs(cfg_dir, exist_ok=True)
    with open(os.path.join(cfg_dir, 'parse_config.yaml'), 'w') as f:
        yaml.dump(_config, f, default_flow_style=False)


@parsing.command
def parse(file_path, db_path, atomic_properties, molecular_properties):
    """
    Runs the data parsing.

    Args:
        file_path (str): path to input file
        db_path (str): path to output file
        atomic_properties (str): property string for .xyz files
        molecular_properties (list): list with molecular properties in
            comment section
    """
    save_config(cfg_dir=os.path.dirname(db_path))
    generate_db(file_path, db_path, atomic_properties, molecular_properties)


@parsing.automain
def main(_log):
    parse()
