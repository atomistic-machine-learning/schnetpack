#!/bin/bash -l
#SBATCH --partition=gpu-test
#SBATCH --gres=gpu:80gb:1
# Check the H100 GRES name with: sinfo -o "%P %G %f" (gpu:80gb may also match an A100)
# Whole node, so other jobs do not distort the wall times
#SBATCH --exclusive
# Standard output and error:
#SBATCH -o job_%j.out        # Standard output, %j = job ID
#SBATCH -e job_%j.err        # Standard error, %j = job ID
# Initial working directory (submit from this folder):
#SBATCH -D ./
# Job Name:
#SBATCH -J relax_benchmark
# Number of nodes
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4

# Time batch-wise dynamics.LBFGS against sequential ase LBFGS.
# Submit from this folder: sbatch submit_benchmark.sh
# Plot afterwards with plot_benchmark.py.

CONTAINER=/home/Jonas/container/schnetpack/schnetpack_container.sif
OUTPUT=$SLURM_SUBMIT_DIR/results_$SLURM_JOB_ID

# pin the torch CPU threads to the allocated cores
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

apptainer exec --nv $CONTAINER python $SLURM_SUBMIT_DIR/run_benchmark.py \
  --device cuda \
  --batch-sizes 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 \
  --repeats 5 \
  --sequential-max-batch-size 32 \
  --output $OUTPUT
