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
#SBATCH -J distill_inference
# Number of nodes
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4

# Time the energy and force inference of a teacher against its distilled students.
# Set the paths below, then submit from this folder: sbatch submit_inference_benchmark.sh

CONTAINER=/home/Jonas/container/schnetpack/schnetpack_container.sif
OUTPUT=$SLURM_SUBMIT_DIR/results_$SLURM_JOB_ID

# a MACE TorchScript archive, from schnetpack.interfaces.mace_interface.compile_mace_off
TEACHER=/path/to/MACE-OFF23_medium.pt
# the best_model of each distillation run, as LABEL=PATH
STUDENTS=(
  student=/path/to/runs/<run_id>/best_model
)
# structures with the elements of the teacher, cycled to fill a batch
STRUCTURES=/path/to/structures.xyz

# pin the torch CPU threads to the allocated cores
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

apptainer exec --nv $CONTAINER python $SLURM_SUBMIT_DIR/run_inference_benchmark.py \
  --device cuda \
  --teacher $TEACHER \
  --teacher-format mace \
  --students "${STUDENTS[@]}" \
  --structures $STRUCTURES \
  --batch-sizes 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 \
  --repeats 20 \
  --warmup 5 \
  --output $OUTPUT
