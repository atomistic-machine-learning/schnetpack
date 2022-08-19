# SchNetPack - Deep Neural Networks for Atomistic Systems
[![Build Status](https://travis-ci.com/atomistic-machine-learning/schnetpack.svg?branch=master)](https://travis-ci.com/atomistic-machine-learning/schnetpack)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)
[![](https://shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=303030)](https://github.com/hobogalaxy/lightning-hydra-template)



SchNetPack aims to provide accessible atomistic neural networks
that can be trained and applied out-of-the-box, while still being
extensible to custom atomistic architectures. 

##### *Major update! Breaking changes!*
*You can find the old SchNetPack 1.0 in the **schnetpack1.0** branch*

##### Features

- SchNet - an end-to-end continuous-filter CNN for molecules and materials [1-3]
- PaiNN - equivariant message-passing for molecules and materials [4]
- Output modules for dipole moments, polarizability, stress, and general response properties
- Modules for electrostatics, Ewald summation, ZBL repulsion
- GPU-accelerated molecular dynamics code incl. path-integral MD, thermostats, barostats

##### Requirements:
- python 3.8
- ASE
- numpy
- PyTorch 1.9
- hydra

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
 

### QM9 example

Under construction. For a first test, use:

```
spktrain experiment=qm9 model/representation=painn
```

## Documentation

For the full API reference, visit our [documentation](https://schnetpack.readthedocs.io).

If you are using SchNetPack in your research, please cite:

K.T. Schütt, P. Kessel, M. Gastegger, K. Nicoli, A. Tkatchenko, K.-R. Müller.
SchNetPack: A Deep Learning Toolbox For Atomistic Systems.
J. Chem. Theory Comput.
[10.1021/acs.jctc.8b00908](http://dx.doi.org/10.1021/acs.jctc.8b00908)
[arXiv:1809.01072](https://arxiv.org/abs/1809.01072). (2018)


## Acknowledgements

CLI and hydra configs for PyTorch Lightning are adapted from this template: [![](https://shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=303030)](https://github.com/hobogalaxy/lightning-hydra-template)


## References

* [1] K.T. Schütt. F. Arbabzadah. S. Chmiela, K.-R. Müller, A. Tkatchenko.  
*Quantum-chemical insights from deep tensor neural networks.*
Nature Communications **8**. 13890 (2017)   
[10.1038/ncomms13890](http://dx.doi.org/10.1038/ncomms13890)

* [2] K.T. Schütt. P.-J. Kindermans, H. E. Sauceda, S. Chmiela, A. Tkatchenko, K.-R. Müller.  
*SchNet: A continuous-filter convolutional neural network for modeling quantum interactions.*
Advances in Neural Information Processing Systems 30, pp. 992-1002 (2017) [Paper](http://papers.nips.cc/paper/6700-schnet-a-continuous-filter-convolutional-neural-network-for-modeling-quantum-interactions)

* [3] K.T. Schütt. P.-J. Kindermans, H. E. Sauceda, S. Chmiela, A. Tkatchenko, K.-R. Müller.  
*SchNet - a deep learning architecture for molecules and materials.* 
The Journal of Chemical Physics 148(24), 241722 (2018) [10.1063/1.5019779](https://doi.org/10.1063/1.5019779)

* [4] K. T. Schütt, O. T. Unke, M. Gastegger  
*Equivariant message passing for the prediction of tensorial properties and molecular spectra.* 
International Conference on Machine Learning (pp. 9377-9388). PMLR, [Paper](https://proceedings.mlr.press/v139/schutt21a.html).
