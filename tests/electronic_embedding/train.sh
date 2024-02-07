#!/usr/bin/bash

#SBATCH --job-name=ag3ion
#SBATCH --partition=gpu-2d
#SBATCH --gpus-per-node=1
#SBATCH --mem=40G
#SBATCH --output=/home/e.pens/projects/qmml/logs/job-%j.out
#SBATCH --error=/home/e.pens/projects/qmml/logs/error-job-%j.out
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=e.pens@tu-berlin.de
#SBATCH --array=1-1


# =================================================================
# CHECKLIST (Options y for "yes" or n for "no" or specific comment)
# =================================================================

# [y] Test run started via gpu-test ?
# []
# []
# []
# []

# =================================================================
# Needed Updates (Options x for "done" or specific comment)
# =================================================================

# [] --output generatet via user input (maybe via bash script to build slurm input file which is then submitted)
# [] --error generatet via user input (maybe via bash script to build slurm input file which is then submitted)
# [] --job-name generatet via user input (maybe via bash script to build slurm input file which is then submitted)
# [] --provide python script via user input (maybe via bash script to build slurm input file which is then submitted)

# =================================================================
# USAGE
# =================================================================

# INPUT_FILE: Is path to a specific input file.
#   V01: path to a json file of structure see /home/e.pens/projects/qmml/scripts/bash_scripts/ag3-ion_debug_run.json
#   sbatch main.sh carbene.json 

# Use your own API KEY
export WANDB_API_KEY=TBD
INPUT_FILE="$1"


echo "Start job with following specs..."
echo "Input file used ${INPUT_FILE}"
echo "Slurm array task id ${SLURM_ARRAY_TASK_ID}"
echo "slurm job name ${SLURM_JOB_NAME}"

# calling the appropiate python script
apptainer \
run --nv /home/e.pens/container/qmml.sif \
python /home/e.pens/projects/qmml/scripts/python_scripts/train_V01.py $INPUT_FILE

echo "Job finished"