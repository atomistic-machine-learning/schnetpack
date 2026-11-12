# LAMMPS Interface

The interface is designed according to the custom pair style approach for
[LAMMPS](https://docs.lammps.org/Modify_pair.html) [1] and it is adapted from the
[pair_nequip](https://github.com/mir-group/pair_nequip) repository [2].

## Requirements

For the installation of the LAMMPS interface we need the following pre-requisites:

- **python** >= 3.12 with **schnetpack** >= 2.0, **pytorch** >= 2.5 and **mkl-include**
- a C++17 compiler (e.g. GCC >= 9), **cmake** >= 3.16 and **make** or **ninja**
- for GPU runs: a **CUDA** toolkit and **cuDNN** whose version matches the CUDA version
  that your PyTorch build was compiled against

Make sure that standalone CUDA toolkit and the CUDA version of pytorch match!
A CPU-only PyTorch build works as well and needs neither CUDA nor cuDNN.

Standalone CUDA can be installed according to <https://developer.nvidia.com/cuda-downloads>,
cuDNN according to <https://docs.nvidia.com/deeplearning/cudnn/installation/latest/>.

We recommend to create a new environment for the matching version of pytorch-CUDA together
with schnetpack and all dependencies. The following commands set up a corresponding conda
environment called `spk_lammps` (replace the CUDA version with the one you need):

```bash
conda create -n spk_lammps python=3.12 cuda-toolkit=12.4 pytorch pytorch-cuda=12.4 mkl-include numpy -c pytorch -c nvidia
conda activate spk_lammps
pip install schnetpack
```

The interface is built against LAMMPS stable releases. The most recently reported combination is

> LAMMPS `stable_22Jul2025_update5` / PyTorch 2.13 (CUDA 13.2) / GCC 15 / C++17

## Downloading LAMMPS

Please download LAMMPS directly from Github. We recommend to check out a stable release
rather than the development branch:

```bash
git clone --depth 1 --branch stable_22Jul2025_update5 https://github.com/lammps/lammps.git
```

## Patching SchNetPack into LAMMPS

We provide a simple patching script for including our interface into LAMMPS.

If you have downloaded the schnetpack repository from Github, move to:

```bash
cd <path/to/schnetpack/interfaces/lammps>
```

**Or** if you do not know where the schnetpack repository is located, download the files directly:

```bash
mkdir spk_lammps
cd spk_lammps
wget https://raw.githubusercontent.com/atomistic-machine-learning/schnetpack/master/interfaces/lammps/pair_schnetpack.cpp
wget https://raw.githubusercontent.com/atomistic-machine-learning/schnetpack/master/interfaces/lammps/pair_schnetpack.h
wget https://raw.githubusercontent.com/atomistic-machine-learning/schnetpack/master/interfaces/lammps/patch_lammps.sh
chmod u+x patch_lammps.sh
```

Now we can run the patching script:

```bash
./patch_lammps.sh <path/to/lammps>
```

## Configure LAMMPS

In order to configure and build LAMMPS, we need to move to the location of our LAMMPS folder:

```bash
cd <path/to/lammps>
```

Next we create the build folder and `cd` into it:

```bash
mkdir build
cd build
```

Now the build-files can be created. libtorch requires C++17, so the standard is set explicitly.
With conda (`recommended`):

```bash
cmake ../cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=17 \
    -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'` \
    -DMKL_INCLUDE_DIR="$CONDA_PREFIX/include"
```

**Or** with pip:

```bash
cmake ../cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=17 \
    -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'` \
    -DMKL_INCLUDE_DIR=`python -c 'import sysconfig;from pathlib import Path;print(Path(sysconfig.get_paths()["include"]).parent)'`
```

Additional LAMMPS packages can be enabled as usual, they do not interfere with the interface,
e.g.:

```bash
cmake ../cmake ... -DPKG_KOKKOS=yes -DPKG_MEAM=yes -DKokkos_ENABLE_SERIAL=yes -DKokkos_ENABLE_OPENMP=yes
```

## Build LAMMPS

Finally we can build our patched LAMMPS with:

```bash
cmake --build . -j$(nproc)
```

This will create a runfile called `lmp` in the build folder. By calling this runfile we can now
start experiments in LAMMPS.

## Creating a deployed Model

Since standard `pytroch` models cannot directly be used within LAMMPS, we need to deploy our trained model first. For
this purpose, we provide a script, that has already been installed with `schnetpack`. A model trained on the rMD17 dataset
for Aspirin can be found in the SchNetPack repository.
If you have downloaded the schnetpack repository from Github, move to the Aspirin examples folder:

```bash
cd <path/to/schnetpack>/interfaces/lammps/examples/aspirin
```

**Or** if you do not know where the SchNetPack folder is located, create an empty folder and download the example files
with:

```bash
mkdir aspirin-example
cd aspirin-example
wget https://raw.githubusercontent.com/atomistic-machine-learning/schnetpack/master/interfaces/lammps/examples/aspirin/aspirin_md.in
wget https://raw.githubusercontent.com/atomistic-machine-learning/schnetpack/master/interfaces/lammps/examples/aspirin/aspirin.data
wget https://raw.githubusercontent.com/atomistic-machine-learning/schnetpack/master/interfaces/lammps/examples/aspirin/best_model
```

Next we can run the deploy script:

```bash
spkdeploy ./best_model ./deployed_model
```

`./best_model` denotes the path to the trained SchNetPack model and `./deployed_model` is the target path of the deployed model

## Running LAMMPS with SchNetPack Models

After installing LAMMPS and deploying the trained model, we are ready to run some experiments. For this we have prepared
an input file and an input structure in the examples folder. The input file is configured to run a small MD simulation
starting with the aspirin structure, that is defined in `aspirin.data`. The new `schnetpack` interface can be used
by setting the `pair_style` and the `pair_coeff` in the input file:

```
pair_style  schnetpack
pair_coeff  * * deployed_model 6 1 8
```

The `pair_style` argument tells LAMMPS to use the new `schnetpack` interface and with `pair_coeff` we
can define the settings for the interface. `deployed_model` indicates the path to our deployed model. The
arguments after the model path indicate, in order, the atomic numbers corresponding to the LAMMPS atom types defined in
`aspirin.data`. We need to provide exactly as many atomic numbers, as we have atom types in the structure input file.
For the example of `aspirin.data` we match atom type 1 to carbon, atom type 2 to hydrogen and atom type 3 to oxygen.
The order of atom types in the input file must be known by the user, that runs the experiment. Finally we can run our
first MD simulation in LAMMPS with the use of the `schnetpack` interface:

```
<path/to/lmp> -in ./aspirin_md.in
```

## References

- [1] A. P. Thompson, H. M. Aktulga, R. Berger. et. al. LAMMPS - a flexible simulation tool for particle-based materials modeling at the atomic, meso, and continuum scales. Comp. Phys. Comm. **271**. 108171 (2022).
- [2] Batzner, S., Musaelian, A., Sun, L. et al. E(3)-equivariant graph neural networks for data-efficient and accurate interatomic potentials. Nat Commun **13**. 2453 (2022). <https://doi.org/10.1038/s41467-022-29939-5>
