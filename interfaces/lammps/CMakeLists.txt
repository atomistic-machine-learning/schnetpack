# Builds pair_style schnetpack as a LAMMPS plugin, loaded with
# `plugin load schnetpackplugin.so`:
#
#   cmake -S . -B build -D LAMMPS_HEADER_DIR=/path/to/lammps/src \
#     -D CMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')"
#   cmake --build build
#
# LAMMPS must be built with -D PKG_PLUGIN=on, with the same compiler, MPI and
# LAMMPS_SIZES. libtorch must be the version of the torch that runs spkdeploy;
# torch >= 2.14 needs C++20.

cmake_minimum_required(VERSION 3.20)
project(schnetpackplugin LANGUAGES CXX)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 20)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(LAMMPS_HEADER_DIR "" CACHE PATH "The src directory of the LAMMPS sources")
if(NOT EXISTS "${LAMMPS_HEADER_DIR}/lammpsplugin.h")
  message(FATAL_ERROR "Set LAMMPS_HEADER_DIR to the src directory of the LAMMPS sources")
endif()

# LAMMPS' own settings for plugins: MPI, integer sizes, the `lammps` target
list(APPEND CMAKE_MODULE_PATH "${LAMMPS_HEADER_DIR}/../examples/plugins")
include(LAMMPSInterfaceCXX)

find_package(Torch 2.14 REQUIRED)

add_library(schnetpackplugin MODULE
  schnetpackplugin.cpp pair_schnetpack.cpp schnetpack_model.cpp)
target_link_libraries(schnetpackplugin PRIVATE lammps ${TORCH_LIBRARIES})
set_target_properties(schnetpackplugin PROPERTIES PREFIX "" SUFFIX ".so")
