import os
from sacred import Experiment
from schnetpack.data import generate_db
from schnetpack.sacred.folder_ingredient import save_config, create_dirs,\
    folder_ing

parsing = Experiment('parsing', ingredients=[folder_ing])


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


@parsing.command
def parse(_log, _config, file_path, db_path, atomic_properties,
          molecular_properties):
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
    save_config(_config=_config, output_dir=output_dir)
    generate_db(file_path, db_path, atomic_properties, molecular_properties)


@parsing.automain
def main():
    parse()
