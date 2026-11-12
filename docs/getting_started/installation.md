# Installation

## Requirements

- [Python](https://www.python.org/) 3.12 or newer
- [PyTorch](https://pytorch.org/) 2.5 or newer
- [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) 2.2 or newer
- [NumPy](https://numpy.org/) 2.0 or newer
- [Hydra](https://hydra.cc/) 1.3
- [ASE](https://wiki.fysik.dtu.dk/ase/) 3.23 or newer

pip installs all of them with SchNetPack. Install PyTorch first if you need a
specific CUDA build.

## Install with pip

```bash
pip install --upgrade schnetpack
```

## Install from source

```bash
git clone https://github.com/atomistic-machine-learning/schnetpack.git
cd schnetpack
pip install .
```

## Optional extras

| Extra | Adds |
| --- | --- |
| `mace` | MACE models as teachers and for fine-tuning (`schnetpack.interfaces.mace_interface`) |
| `test` | pytest and the test helpers |
| `doc` | everything needed to build this documentation |
| `dev` | pre-commit |
| `all` | all of the above |

```bash
pip install "schnetpack[mace]"
```

## TensorBoard

SchNetPack logs training runs through PyTorch Lightning. The default logger is
TensorBoard, which is installed with SchNetPack. See
[Training from the command line](cli.md#logging) for how to open the logs.
