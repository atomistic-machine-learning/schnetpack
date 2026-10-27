================
LAMMPS Interface
================
.. _lammps:

SchNetPack potentials run in LAMMPS [1] through the pair style :code:`schnetpack`.
:code:`spkdeploy` compiles a trained model with PyTorch's AOTInductor into a package
that holds the energy, the per-atom energies, the forces and the virial, and the pair
style runs that package without Python. Since the virial is computed, NPT simulations
work, and the pair style runs on any number of MPI ranks.

The pair style lives in :code:`interfaces/lammps` in the SchNetPack repository. Its
first version was adapted from pair_nequip [2].


Requirements
============

* **torch** >= 2.14, the **same version** for :code:`spkdeploy` and for building
  the pair style: a package runs only with the libtorch version that compiled it.
* A C++20 compiler (torch >= 2.14 needs C++20), e.g. GCC 13, and **cmake** >= 3.20.
* LAMMPS with the PLUGIN package (:code:`-D PKG_PLUGIN=on`), built with the same
  compiler and MPI as the pair style.
* For GPU runs: a CUDA toolkit that matches the CUDA version of torch, and a GPU
  with compute capability 7.0 or higher, which AOTInductor's Triton kernels need.

A CPU-only torch works as well. One environment for SchNetPack and the build is
simplest::

    conda create -n spk_lammps python=3.12
    conda activate spk_lammps
    pip install "schnetpack[compile]" cmake


Building LAMMPS and the plugin
==============================
Build LAMMPS from a stable release with the PLUGIN package, plus MPI and whatever
other packages you need::

    git clone --depth 1 --branch stable https://github.com/lammps/lammps.git
    cmake -S lammps/cmake -B lammps/build -D CMAKE_BUILD_TYPE=Release \
        -D PKG_PLUGIN=on -D BUILD_MPI=on
    cmake --build lammps/build -j$(nproc)

Then build the pair style as a plugin, against the LAMMPS sources and the torch of
your environment::

    cd <path/to/schnetpack>/interfaces/lammps
    cmake -S . -B build -D CMAKE_BUILD_TYPE=Release \
        -D LAMMPS_HEADER_DIR=<path/to/lammps>/src \
        -D CMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')"
    cmake --build build

This writes :code:`build/schnetpackplugin.so`. LAMMPS loads every plugin in the
directories of :code:`LAMMPS_PLUGIN_PATH` at startup::

    export LAMMPS_PLUGIN_PATH=<path/to/schnetpack>/interfaces/lammps/build

or load it in the input file with :code:`plugin load /path/to/schnetpackplugin.so`.
After an update of torch, rebuild the plugin and deploy the models again.

**Alternatively**, :code:`patch_lammps.sh` builds the pair style into LAMMPS itself,
e.g. for static builds. It copies the sources into :code:`<path/to/lammps>/src` and
links libtorch; configure LAMMPS with :code:`-D CMAKE_CXX_STANDARD=20` and
:code:`-D CMAKE_PREFIX_PATH` as above::

    cd <path/to/schnetpack>/interfaces/lammps
    ./patch_lammps.sh <path/to/lammps>


Deploying a model
=================
:code:`spkdeploy` compiles a trained model for one device. The package runs only with
the torch version it was compiled with and, on GPU, on the GPU architecture it was
compiled for, so run it **on the machine that runs LAMMPS**::

    spkdeploy best_model deployed_model.pt2                  # CPU, float32
    spkdeploy best_model deployed_model.pt2 --device cuda    # GPU
    spkdeploy best_model deployed_model.pt2 --dtype float64

Compiling takes a few tens of seconds. Before the package is written, it is checked
against the model.

Supported are :code:`NeuralNetworkPotential` models with a SchNet, PaiNN or SO3net
representation, an :code:`Atomwise` energy head and :code:`Forces`, and with
:code:`AddOffsets` postprocessing. Models whose energy has other terms (electrostatics,
ZBL), that embed a total charge or spin, or that postprocess the energy otherwise are
rejected with an error, as are ensembles.


Running LAMMPS
==============
The pair style takes the package and the atomic number of each LAMMPS atom type, in
the order of the types::

    pair_style  schnetpack
    pair_coeff  * * deployed_model.pt2 6 1 8

The aspirin example in :code:`interfaces/lammps/examples/aspirin` runs a short MD
simulation with a model trained on rMD17::

    cd <path/to/schnetpack>/interfaces/lammps/examples/aspirin
    spkdeploy best_model deployed_model.pt2
    <path/to/lmp> -in aspirin_md.in

Per-atom energies are available through :code:`compute pe/atom`. With the virial,
pressure and NPT work, e.g. for a periodic system::

    fix  1 all npt temp 300.0 300.0 $(100.0*dt) iso 1.0 1.0 $(1000.0*dt)

Per-atom virials (:code:`compute stress/atom`) are not supported.

Parallel runs
-------------
By default the pair style decomposes the domain like any LAMMPS potential and runs on
any number of MPI ranks::

    mpirun -np 4 <path/to/lmp> -in input.in

The ghost atoms of each rank are part of its graph. A message-passing model reaches
:code:`n_interactions` cutoffs, so the pair style raises the ghost cutoff to that
receptive field plus the neighbor skin, and the energies and forces are those of the
whole system on any number of ranks. This needs :code:`newton pair on`, the LAMMPS
default.

For small periodic boxes, the ghost shell can be larger than the box itself. Then the
pair style can instead run on a single rank with only the atoms of the box and
periodic shifts, which is cheaper::

    pair_style  schnetpack no_domain_decomposition

On GPU, the ranks on a node take turns over its GPUs. On CPU, each rank runs with
:code:`OMP_NUM_THREADS` threads; set it so that the ranks do not oversubscribe the cores.


References
==========
* [1] A. P. Thompson, H. M. Aktulga, R. Berger. et. al. LAMMPS - a flexible simulation tool for particle-based materials modeling at the atomic, meso, and continuum scales. Comp. Phys. Comm. **271**. 108171 (2022).
* [2] Batzner, S., Musaelian, A., Sun, L. et al. E(3)-equivariant graph neural networks for data-efficient and accurate interatomic potentials. Nat Commun **13**. 2453 (2022). https://doi.org/10.1038/s41467-022-29939-5
