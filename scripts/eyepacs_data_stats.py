#!/usr/bin/python3

import argparse
import os
import sys
import time

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from core.dataset import dataset_fn

# TODO move to core data location
from core.model import (
    DataModule,
    DomainClassifier,
    EyepacsClassifier,
    MaxKernel,
    TaskClassifier,
)
from utils.config import hash_dict, load_config, save_config


def sbatch_build_submit(job_name, slurm_log_dir, argv):

    os.makedirs(slurm_log_dir, exist_ok=True)

    output_file = os.path.join(slurm_log_dir, f"{job_name}.out")
    error_file = os.path.join(slurm_log_dir, f"{job_name}.err")
    job_file = os.path.join(slurm_log_dir, "deploy.sh")

    with open(job_file, "w") as sbatch_file:
        sbatch_file.writelines("#!/bin/bash\n")

        # TODO insert job-name, output, error
        sbatch_file.writelines(f"#SBATCH --job-name={job_name}\n")
        sbatch_file.writelines(f"#SBATCH --output={output_file}\n")
        sbatch_file.writelines(f"#SBATCH --error={error_file}\n")

        # slurm resources

        sbatch_file.writelines("#SBATCH --partition=gpu-2080ti\n")
        sbatch_file.writelines("#SBATCH --ntasks=1\n")
        sbatch_file.writelines("#SBATCH --time=3-0\n")
        sbatch_file.writelines("#SBATCH --gres=gpu:1\n")
        sbatch_file.writelines("#SBATCH --mem=120G\n")
        sbatch_file.writelines("#SBATCH --cpus-per-task=8\n")
        sbatch_file.writelines("#SBATCH --mail-type=END\n")
        sbatch_file.writelines("#SBATCH --mail-user=lisa.koch@uni-tuebingen.de\n")

        # Body
        sbatch_file.writelines("\n")
        sbatch_file.writelines("scontrol show job ${SLURM_JOB_ID}\n\n")
        sbatch_file.writelines(
            [
                "sudo /opt/eyepacs/start_mount.sh  \n",
                "source ~/.bashrc \n",
                "which conda\n" "conda env list\n" "nvidia-smi\n\n",
                "ls -l\n\n",
                "conda activate subgroup \n",
                "which python\n\n",
                "python "
                + " ".join(argv)
                + " --no-slurm"
                + "\n",  # execute locally on compute node
                "sudo /opt/eyepacs/stop_mount.sh  \n",
            ]
        )

    cmd_args = ["sbatch"]
    cmd_args.append("--verbose")
    cmd_args.append(job_file)

    # Ensure everything is a string
    cmd_args = [str(s) for s in cmd_args]

    cmd = " ".join(cmd_args)

    os.system(cmd)


#####################

# Datasets:

# Task:

# TODO checkpointing stuff, how to load model in different trainer instance

# FYI. it’s now
# lightning_logs/version_{version number}/epoch_{epoch number}-step_{global_step}.ckpt

# and to access them:

# version_number -> trainer.logger.version
# epoch_number -> trainer.current_epoch
# global_step -> trainer.global_step


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Run single experiment", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--config_file",
        action="store",
        type=str,
        help="config file",
        # default="./config/eyepacs_quality_ood.yaml"
        default="./config/lightning/eyepacs.yaml"
        # default="./config/lightning/mnist.yaml"
        # default='./experiments/hypothesis-tests/mnist/5c3010e7e9f5de06c7d55ecbed422251/config.yaml'
    )
    parser.add_argument(
        "--seed",
        action="store",
        default=1000,
        type=int,
        help="random seed",
    )

    parser.add_argument("--no-slurm", dest="slurm", action="store_false", default=False)
    args = parser.parse_args()

    params = load_config(args.config_file)

    ###############################################################################################################################

    # TODO check for weird seed stuff
    pl.seed_everything(args.seed, workers=True)

    dataloader = dataset_fn(params_dict=params["dataset"])

    for k, v in dataloader.items():

        # print(f"split: {k}: {dataloader[k]['p'].dataset}")

        dset = dataloader[k]["p"].dataset

        print("----------------------------------------")
        print(f"Split: {k}")
        print("----------------------------------------")

        dset.print_summary()

    # TODO all stats regarding subgroups...
    # Age: mean/std per split
    # Gender: f/m/other per split
    print("done")
