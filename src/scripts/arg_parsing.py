import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Mini-batch size for training and prediction (default: %(default)s)",
        default=100,
    )
    parser.add_argument(
        "--device",
        type=str,
        help="device to train the model on",
        default='cuda',
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        help="path to model_dir",
        default='training',
    )
    parser.add_argument(
        "--data_path",
        type=str,
        help="path to dataset",
        default='data',
    )
    parser.add_argument(
        "--split",
        help="Split into [train] [validation] and use remaining for testing",
        type=float,
        nargs=2,
        default=[0.8, 0.1],
    )
    return parser
