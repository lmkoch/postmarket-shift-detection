#!/bin/bash
#SBATCH --ntasks=1                # Number of tasks (see below)
#SBATCH --cpus-per-task=8         # Number of CPU cores per task
#SBATCH --nodes=1                 # Ensure that all cores are on one machine
#SBATCH --time=2-00:00            # Runtime in D-HH:MM
#SBATCH --gres=gpu:1              # optionally type and number of gpus
#SBATCH --mem=60G                 # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --output=slurm_logs/job_%A_%a.out  # File to which STDOUT will be written
#SBATCH --error=slurm_logs/job_%A_%a.out   # File to which STDERR will be written
#SBATCH --mail-type=ALL           # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=lisa.koch@uni-tuebingen.de  # Email to which notifications will be sent
#SBATCH --array=0,1,2,3,4,5,6,7,8,9

# print info about current job
echo "---------- JOB INFOS ------------"
scontrol show job $SLURM_JOB_ID 
echo -e "---------------------------------\n"

# Due to a potential bug, we need to manually load our bash configurations first
# source $HOME/.bashrc

# TODO mount eyepacs data

# Next activate the conda environment 
conda activate subgroup

# Run our code
echo "-------- PYTHON OUTPUT ----------"
python train.py --random-seed ${SLURM_ARRAY_TASK_ID}
echo "---------------------------------"

# Deactivate environment again
conda deactivate