================
LAMMPS Interface
================
.. _lammps:

The interface is designed according to the custom pair style approach for
LAMMPs https://docs.lammps.org/Modify_pair.html [1] and it is adapted from the
pair_nequip github repository https://github.com/mir-group/pair_nequip [2].


Requirements
============
For the installation of the LAMMPS interface we need the following pre-requisites. Different versions for CUDA
might cause unknown errors during the installation:

* **CUDA** 11.7
* **cuDNN**
* **python** 3.9 with **schnetpack** 2.0, **pytorch** 1.13, and **mkl-include**

In this installation guide we use CUDA 11.7 and pytorch 1.13. If you want to use different
versions, make sure that the cuda versions of standalone CUDA and pytorch-CUDA are matching! This installation guide
will focus on the installation within a conda environment, but pip environments should generally also work.

The installation of standalone CUDA can be done according to this installation guide: https://developer.nvidia.com/cuda-11-7-0-download-archive.

Afterwards, install cuDNN with the help of this installation guide: https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html.
cuDNN can be downloaded from: https://developer.nvidia.com/rdp/cudnn-archive.

We recommend to create a new environment to install the matching version of pytorch-CUDA together with schnetpack and all dependencies.
For example, the following commands will set up a corresponding conda environment called :code:`spk_lammps`::

    conda create -n spk_lammps python=3.9 cuda-toolkit=11.7 pytorch mkl-include numpy -c pytorch -c nvidia
    conda activate spk_lammps
    pip install schnetpack
    

Downloading LAMMPS
==================
Please download LAMMPS directly from Github::

    git clone --depth 1 git@github.com:lammps/lammps

Patching SchNetPack into LAMMPS
===============================
We provide a simple patching script for including our interface into LAMMPS.

If you have downloaded the schnetpack repository from Github, move to::

    cd <path/to/schnetpack/interfaces/lammps>

**Or** if you do not know where the schnetpack repository is located, download the files directly::

    mkdir spk_lammps
    cd spk_lammps
    wget https://raw.githubusercontent.com/atomistic-machine-learning/schnetpack/master/interfaces/lammps/pair_schnetpack.cpp
    wget https://raw.githubusercontent.com/atomistic-machine-learning/schnetpack/master/interfaces/lammps/pair_schnetpack.h
    wget https://raw.githubusercontent.com/atomistic-machine-learning/schnetpack/master/interfaces/lammps/patch_lammps.sh
    chmod u+x patch_lammps.sh

Now we can run the patching script::

    ./patch_lammps.sh <path/to/lammps>

Configure LAMMPS
================
In order to configure and build LAMMPS, we need to move to the location of our LAMMPS folder::

    cd <path/to/lammps>

Next we create the build folder and :code:`cd` into it::

    mkdir build
    cd build

Now the build-files can be created.
With conda (`recommended`)::

    cmake ../cmake -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'` -DMKL_INCLUDE_DIR="$CONDA_PREFIX/include"

**Or** with pip::

    cmake ../cmake -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'`
    -DMKL_INCLUDE_DIR=python -c "import sysconfig;from pathlib import Path;print(Path(sysconfig.get_paths()[\"include\"]).parent)"

Build LAMMPS
============
Finally we can install our patched LAMMPS with::

    make -j$(nproc)

This will create a runfile called `lmp` in the build folder. By calling this runfile we can now start experiments in LAMMPS.

Creating a deployed Model
=========================
Since standard :code:`pytroch` models cannot directly be used within LAMMPS, we need to deploy our trained model first. For
this purpose, we provide a script, that has already been installed with :code:`schnetpack`. A model trained on the rMD17 dataset
for Aspirin can be found in the SchNetPack repository.
If you have downloaded the schnetpack repository from Github, move to the Aspirin examples folder::

        cd <path/to/schnetpack>/interfaces/lammps/examples/aspirin

**Or** if you do not know where the SchNetPack folder is located, create an empty folder and download the example files
with::

    mkdir aspirin-example
    cd aspirin-example
    wget https://raw.githubusercontent.com/atomistic-machine-learning/schnetpack/master/interfaces/lammps/examples/aspirin/aspirin_md.in
    wget https://raw.githubusercontent.com/atomistic-machine-learning/schnetpack/master/interfaces/lammps/examples/aspirin/aspirin.data
    wget https://raw.githubusercontent.com/atomistic-machine-learning/schnetpack/master/interfaces/lammps/examples/aspirin/best_model

Next we can run the deploy script::

    spkdeploy ./best_model ./deployed_model

:code:`./best_model` denotes the path to the trained SchNetPack model and :code:`./deployed_model` is the target path of the deployed model

Running LAMMPS with SchNetPack Models
=====================================
After installing LAMMPS and deploying the trained model, we are ready to run some experiments. For this we have prepared
an input file and an input structure in the examples folder. The input file is configured to run a small MD simulation
starting with the aspirin structure, that is defined in `aspirin.data`. The new :code:`schnetpack` interface can be used
by setting the :code:`pair_style` and the :code:`pair_coeff` in the input file::

    pair_style	schnetpack
    pair_coeff	* * deployed_model 6 1 8

The :code:`pair_style` argument tells LAMMPS to use the new :code:`schnetpack` interface and with :code:`pair_coeff` we
can define the settings for the interface. :code:`deployed_model` indicates the path to our deployed model. The
arguments after the model path  indicate, in order, the atomic numbers corresponding to the LAMMPS atom types defined in
`aspirin.data`. We need to provide exactly as many atomic numbers, as we have atom types in the structure input file.
For the example of `aspirin.data` we match atom type 1 to carbon, atom type 2 to hydrogen and atom type 3 to oxygen.
The order of atom types in the input file must be known by the user, that runs the experiment. Finally we can run our
first MD simulation in LAMMPS with the use of the :code:`schnetpack` interface::

    <path/to/lmp> -in ./aspirin_md.in

References
==========
* [1] A. P. Thompson, H. M. Aktulga, R. Berger. et. al. LAMMPS - a flexible simulation tool for particle-based materials modeling at the atomic, meso, and continuum scales. Comp. Phys. Comm. **271**. 108171 (2022).
* [2] Batzner, S., Musaelian, A., Sun, L. et al. E(3)-equivariant graph neural networks for data-efficient and accurate interatomic potentials. Nat Commun **13**. 2453 (2022). https://doi.org/10.1038/s41467-022-29939-5
