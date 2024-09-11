#!/bin/bash
#SBATCH --job-name=16639595222974174     
#SBATCH --output=./slurmlog/1663959522.2974174/16639595222974174-%j.out
#SBATCH --error=./slurmlog/1663959522.2974174/16639595222974174-%j.err

#SBATCH --partition=gpu-2080ti
#SBATCH --ntasks=1
#SBATCH --time=3-0
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=END
#SBATCH --mail-user=lisa.koch@uni-tuebingen.de

# adapt job-name, output, error

scontrol show job ${SLURM_JOB_ID}

sudo /opt/eyepacs/start_mount.sh  
source ~/.bashrc 
which conda
conda env list
nvidia-smi

ls -l

conda activate subgroup
which python

python scripts/train_lightning_task.py --config_file ./config/lightning/eyepacs_comorb_1_c2st.yaml --method c2st --data_frac 1.0 --exp_dir ./experiments/endspurt/eyepacs_comorb
sudo /opt/eyepacs/stop_mount.sh  
