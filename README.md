# SchNetPack - Deep Neural Networks for Atomistic Systems
[![Build Status](https://travis-ci.com/atomistic-machine-learning/schnetpack.svg?branch=master)](https://travis-ci.com/atomistic-machine-learning/schnetpack)
[![codecov](https://codecov.io/gh/atomistic-machine-learning/schnetpack/branch/master/graph/badge.svg)](https://codecov.io/gh/atomistic-machine-learning/schnetpack)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)


SchNetPack aims to provide accessible atomistic neural networks
that can be trained and applied out-of-the-box, while still being
extensible to custom atomistic architectures. 

##### Currently provided models:

- SchNet - an end-to-end continuous-filter CNN for molecules and materials [1-3]
- wACSF - weighted atom-centered symmetry functions [4,5]

_**Note: We will keep working on improving the documentation, 
supporting more architectures and datasets and many more features.**_

##### Requirements:
- python 3
- ASE
- numpy
- PyTorch (>=0.4.1)
- h5py
- Optional: tensorboardX

_**Note: We recommend using a GPU for training the neural networks.**_

## Installation

### Install with pip

```
pip install schnetpack
```

### Install from source

#### Clone the repository

```
git clone https://github.com/atomistic-machine-learning/schnetpack.git
cd schnetpack
```

#### Install requirements

```
pip install -r requirements.txt
```

#### Install SchNetPack

```
pip install .
```

You're ready to go!

## Getting started

The best place to start is training a SchNetPack model on a common benchmark dataset. 
The example scripts provided by SchNetPack are inserted into your PATH during installation. 

### QM9 example

The QM9 example scripts allows to train and evaluate both SchNet and wACSF neural networks.
The training can be started using:

```
spk_run.py train <schnet/wacsf> qm9 <dbpath> <modeldir> --split num_train num_val [--cuda]
```

where num_train and num_val need to be replaced by the number of training and validation datapoints respectively.

You can choose between SchNet and wACSF networks and have to provide a path to the database file and a path to a directory which will be used to store the model. If the database path does not exist, the data is downloaded and stored there. Please note that the database path must include the file extension .db.
With the `--cuda` flag, you can activate GPU training.
The default hyper-parameters should work fine, however, you can change them through command-line arguments. 
Please refer to the help at `spk_run.py train <schnet/wacsf> --help`. 

The training progress will be logged in `<modeldir>/log`, either as CSV 
(default) or as TensorBoard event files. For the latter, TensorBoard needs to be installed to view the event files.
This can be done by installing the version included in TensorFlow 

```
pip install tensorflow
```

or the [standalone version](https://github.com/dmlc/tensorboard).

To evaluate the trained model with the best validation error, call

```
spk_run.py eval <modeldir> --split test [--cuda]
```

which will run on the specified `--split` and write a result file `evaluation.txt` into the model directory.

## Documentation

For the full API reference, visit our [documentation](https://schnetpack.readthedocs.io).

If you are using SchNetPack in you research, please cite:

K.T. Schütt, P. Kessel, M. Gastegger, K. Nicoli, A. Tkatchenko, K.-R. Müller.
SchNetPack: A Deep Learning Toolbox For Atomistic Systems.
J. Chem. Theory Comput.
[10.1021/acs.jctc.8b00908](http://dx.doi.org/10.1021/acs.jctc.8b00908)
[arXiv:1809.01072](https://arxiv.org/abs/1809.01072). (2018)


## References

* [1] K.T. Schütt. F. Arbabzadah. S. Chmiela, K.-R. Müller, A. Tkatchenko.  
*Quantum-chemical insights from deep tensor neural networks.*
Nature Communications **8**. 13890 (2017)   
[10.1038/ncomms13890](http://dx.doi.org/10.1038/ncomms13890)

* [2] K.T. Schütt. P.-J. Kindermans, H. E. Sauceda, S. Chmiela, A. Tkatchenko, K.-R. Müller.  
*SchNet: A continuous-filter convolutional neural network for modeling quantum interactions.*
Advances in Neural Information Processing Systems 30, pp. 992-1002 (2017) [link](http://papers.nips.cc/paper/6700-schnet-a-continuous-filter-convolutional-neural-network-for-modeling-quantum-interactions)

* [3] K.T. Schütt. P.-J. Kindermans, H. E. Sauceda, S. Chmiela, A. Tkatchenko, K.-R. Müller.  
*SchNet - a deep learning architecture for molecules and materials.* 
The Journal of Chemical Physics 148(24), 241722 (2018) [10.1063/1.5019779](https://doi.org/10.1063/1.5019779)

* [4] M. Gastegger, L. Schwiedrzik, M. Bittermann, F. Berzsenyi, P. Marquetand.
*wACSF—Weighted atom-centered symmetry functions as descriptors in machine learning potentials.*
The Journal of Chemical Physics, 148(24), 241709. (2018) [10.1063/1.5019667](https://doi.org/10.1063/1.5019667)

* [5] J. Behler, M. Parrinello. 
*Generalized neural-network representation of high-dimensional potential-energy surfaces.*
Physical Review Letters, 98(14), 146401. (2007) [10.1103/PhysRevLett.98.146401](https://doi.org/10.1103/PhysRevLett.98.146401)
