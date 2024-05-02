import jax
import argparse

from pathlib import Path

from ..data.dataloader import AseDataLoader


class StoreDictKeyPair(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        my_dict = {}
        for kv in values.split(","):
            k, v = kv.split("=")
            my_dict[k] = v
        setattr(namespace, self.dest, my_dict)


def to_schnet_input():
    # Create the parser
    parser = argparse.ArgumentParser(description='Convert ASE digestible format to ASE Database that can be used as input'
                                                 'argument to `train_so3krates --data_file $FILE`')

    # Add the arguments
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=False, default=None,
                        help='Defaults to the directory of DB and name of the input file.')

    kwargs = {'load_stress': False,
              'load_energy_and_forces': True}

    parser.add_argument("--kwargs", action=StoreDictKeyPair, default=None)

    args = parser.parse_args()
    if args.kwargs is not None:
        kwargs.update(args.kwargs)

        def to_bool(u):
            if type(u) == str:
                if u.lower() == 'true':
                    return True
                if u.lower() == 'false':
                    return False
                else:
                    raise RuntimeError('--kwargs only accepts true/false (True/False) as values for booleans.')
            elif type(u) == bool:
                return u
            else:
                raise RuntimeError(f'Unknown data type {type(u)} in --kwargs.')

        kwargs = {k: to_bool(v) for k, v in kwargs.items()}

    input_file = Path(args.input_file).absolute()
    output_file = Path(args.output_file).absolute() if args.output_file is not None else None

    if output_file is None:
        output_file_db_path = f"{input_file.stem}.db"
        save_dir = Path(input_file.parent, output_file_db_path).resolve()
    else:
        save_dir = Path(output_file).absolute()
        if save_dir.suffix != '.db':
            save_dir = f'{save_dir}.db'

    data_loader = AseDataLoader(input_file=input_file,
                                db_path=save_dir,
                                **kwargs)

    _ = data_loader.load_all()  # call to load all saves .npz file into output file


if __name__ == '__main__':
    to_schnet_input()
