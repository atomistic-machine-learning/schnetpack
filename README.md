# SchNetPack - Deep Neural Networks for Atomistic Systems

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
- PyTorch (>=0.4)
- Optional: tensorboardX, h5py

_**Note: We recommend using a GPU for training the neural networks.**_

## Installation

### Install with pip

`pip install schnetpack`

### Install from source

#### Clone the repository

`git clone https://github.com/atomistic-machine-learning/schnetpack.git`

`cd schnetpack`

#### Install requirements

`pip install -r requirements.txt`

#### Install SchNetPack

`python setup.py install`

`cd ..`

You're ready to go!
 
## Documentation

For the full API reference, visit our [documentation](https://schnetpack.readthedocs.io).

## Getting started

The best place to start is training a SchNetPack model on a common benchmark dataset. 
The example scripts provided by SchNetPack are inserted into your PATH during installation. 

### QM9 example

The QM9 example scripts allows to train and evaluate both SchNet and wACSF neural networks.
The training can be started using:

`schnetpack_qm9.py train <schnet/wacsf> <datadir> <modeldir> --split num_train num_val [--cuda]`

where num_train and num_val need to be replaced by the number of training and validation datapoints respectively.

You can choose between SchNet and wACSF networks and have to provide directories to store the model and the QM9 dataset 
(will be downloaded if not in `<datadir>`). With the `--cuda` flag, you can activate GPU training.
The default hyper-parameters should work fine, however, you can change them through command-line arguments. 
Please refer to the help at 

`schnetpack_qm9.py train <schnet/wacsf> --help`. 

The training progress will be logged in `<modeldir>/log`, either as CSV 
(default) or as TensorBoard event files. For the latter, TensorBoard needs to be installed to view the event files.
This can be done by installing the version included in TensorFlow 

`pip install tensorflow` 

or the [standalone version](https://github.com/dmlc/tensorboard).

To evaluate the trained model with the best validation error, call

`schnetpack_qm9.py eval <schnet/wacsf> <datadir> <modeldir> [--split train val test] [--cuda]`

which will write a result file `evaluation.txt` into the model directory.


## References

* [1] K.T. Schütt. F. Arbabzadah. S. Chmiela, K.-R. Müller, A. Tkatchenko.  
Quantum-chemical insights from deep tensor neural networks.*
Nature Communications **8**. 13890 (2017)   
[10.1038/ncomms13890](http://dx.doi.org/10.1038/ncomms13890)

* [2] K.T. Schütt. P.-J. Kindermans, H. E. Sauceda, S. Chmiela, A. Tkatchenko, K.-R. Müller.  
SchNet: A continuous-filter convolutional neural network for modeling quantum interactions.  
Advances in Neural Information Processing Systems 30, pp. 992-1002 (2017) [link](http://papers.nips.cc/paper/6700-schnet-a-continuous-filter-convolutional-neural-network-for-modeling-quantum-interactions)

* [3] K.T. Schütt. P.-J. Kindermans, H. E. Sauceda, S. Chmiela, A. Tkatchenko, K.-R. Müller.  
SchNet - a deep learning architecture for molecules and materials. 
The Journal of Chemical Physics 148(24), 241722 (2018) [10.1063/1.5019779](https://doi.org/10.1063/1.5019779)

* [4] M. Gastegger, L. Schwiedrzik, M. Bittermann, F. Berzsenyi, P. Marquetand.
wACSF—Weighted atom-centered symmetry functions as descriptors in machine learning potentials. 
The Journal of Chemical Physics, 148(24), 241709. (2018) [10.1063/1.5019667](https://doi.org/10.1063/1.5019667)

* [5] J. Behler, M. Parrinello. 
Generalized neural-network representation of high-dimensional potential-energy surfaces. 
Physical Review Letters, 98(14), 146401. (2007) [10.1103/PhysRevLett.98.146401](https://doi.org/10.1103/PhysRevLett.98.146401)
