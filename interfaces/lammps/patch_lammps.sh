#!/bin/bash
# usage: patch_lammps.sh [-e] /path/to/lammps/
#
# Builds pair_style schnetpack into LAMMPS itself, e.g. for static builds.
# Without patching LAMMPS, build it as a plugin instead (see CMakeLists.txt).
# -e links the sources instead of copying them.
#
# References:
#
#    .. [#pair_nequip] https://github.com/mir-group/pair_nequip

FILES="pair_schnetpack.cpp pair_schnetpack.h schnetpack_model.cpp schnetpack_model.h"

do_e_mode=false

while getopts "he" option; do
   case $option in
      e)
         do_e_mode=true;;
      h) # display Help
         echo "patch_lammps.sh [-e] /path/to/lammps/"
         exit;;
   esac
done

# https://stackoverflow.com/a/9472919
shift $(($OPTIND - 1))
lammps_dir=$1

if [ "$lammps_dir" = "" ];
then
    echo "lammps_dir must be provided"
    exit 1
fi

if [ ! -d "$lammps_dir" ]
then
    echo "$lammps_dir doesn't exist"
    exit 1
fi

if [ ! -d "$lammps_dir/cmake" ]
then
    echo "$lammps_dir doesn't look like a LAMMPS source directory"
    exit 1
fi

# Check if root directory is correct
if [ ! -f pair_schnetpack.cpp ]; then
    echo "Please run patch_lammps.sh from the directory that holds pair_schnetpack.cpp."
    exit 1
fi

echo "Updating CMakeLists.txt..."
# Check for double-patch
if grep -q "find_package(Torch" $lammps_dir/cmake/CMakeLists.txt
then
    echo "This LAMMPS installation _seems_ to already have been patched. CMakeLists.txt file not modified."
else
    # Add libtorch
    cat >> $lammps_dir/cmake/CMakeLists.txt << "EOF2"

find_package(Torch 2.14 REQUIRED)
target_link_libraries(lammps PUBLIC "${TORCH_LIBRARIES}")
EOF2

fi

for file in $FILES; do
    if [ -e $lammps_dir/src/$file ]; then
        echo "$file already exists in $lammps_dir/src, not copied."
    elif [ "$do_e_mode" = true ]; then
        ln -s `realpath -s $file` $lammps_dir/src/$file
    else
        cp $file $lammps_dir/src/$file
    fi
done

echo "Done! Configure LAMMPS with -D CMAKE_CXX_STANDARD=20 (torch >= 2.14 needs C++20)."
