#!/usr/bin/python3

import argparse
import os
import sys
import time

import pytorch_lightning as pl
from core.dataset import dataset_fn

# TODO move to core data location
from core.model import (
    DataModule,
    DomainClassifier,
    EyepacsClassifier,
    MaxKernel,
    TaskClassifier,
)
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
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
        # sbatch_file.writelines("#SBATCH --mem=120G\n")
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
        "--exp_dir",
        action="store",
        type=str,
        help="experiment folder",
        default="./experiments/tmp",
    )
    parser.add_argument(
        "--config_file",
        action="store",
        type=str,
        help="config file",
        # default="./config/eyepacs_quality_ood.yaml"
        # default="./config/lightning/eyepacs_quality_ood.yaml"
        # default="./config/lightning/mnist.yaml"
        default="./config/lightning/eyepacs.yaml"
        # default='./experiments/hypothesis-tests/mnist/5c3010e7e9f5de06c7d55ecbed422251/config.yaml'
    )
    parser.add_argument(
        "--seed",
        action="store",
        default=1000,
        type=int,
        help="random seed",
    )
    parser.add_argument(
        "--method",
        dest="test_method",
        action="store",
        help="Test Method (mmdd, c2st, muks)",
        default="muks",
    )
    parser.add_argument(
        "--data_frac",
        action="store",
        default=0.01,
        type=float,
        help="Fraction of data to use (for debug purposes)",
    )
    parser.add_argument(
        "--slurm",
        action="store_true",
        default=False,
        help="Prepare sbatch script and submit instead of executing locally",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        default=False,
        help="Run on CPU",
    )
    parser.add_argument(
        "--train_data_frac",
        action="store",
        type=float,
        help="Fraction of data to use (for ablation). Different from debug option!",
    )
    parser.add_argument("--no-slurm", dest="slurm", action="store_false", default=False)
    args = parser.parse_args()

    params = load_config(args.config_file)

    # TODO make sure this works
    params["dataset"]["ds"]["data_frac"] = args.train_data_frac

    # mapping of correct config file chapter:
    param_category = {"mmdd": "mmd", "c2st": "domain_classifier", "muks": "task_classifier"}

    artifacts_dir = os.path.join(args.exp_dir, args.test_method)
    model_config = {k: params[k] for k in ["dataset", param_category[args.test_method]]}
    hash_string = hash_dict(model_config)

    save_config(model_config, artifacts_dir)

    if args.slurm:
        # submit slurm job instead of executing locally

        dataset_type = params["dataset"]["ds"]["p"]["dataset"]
        timestamp = time.time()
        job_name = f"{args.test_method}_{dataset_type}_{timestamp}"

        slurm_log_dir = os.path.join(artifacts_dir, hash_string, "slurm")

        sbatch_build_submit(job_name, slurm_log_dir, sys.argv)

        print("submitted job, do not execute locally")
        sys.exit(0)
    #

    print("executing locally now")
    # run experiments:

    # MNIST: p=0.5 vs p = {0, 1}
    # Camelyon: all vs each site s = {0, ..., 4}
    # Eyepacs: all vs. comorbidity = {False, True}

    # TODO:
    # 1. add specific options to parser, these should then override the config file.
    # 2. call this script with SLURM_TASK_ARRAY

    ###############################################################################################################################
    # Preparation
    ###############################################################################################################################

    # TODO check for weird seed stuff
    pl.seed_everything(args.seed, workers=True)

    dataloader = dataset_fn(params_dict=params["dataset"])

    data_module = DataModule(dataloader)

    logger = TensorBoardLogger(save_dir=artifacts_dir, name=hash_string)

    ###############################################################################################################################
    # Run training: separate exp hashes for all three methods (matched with dataset config)
    ###############################################################################################################################

    dataset_type = params["dataset"]["ds"]["p"]["dataset"]

    if dataset_type == "eyepacs":
        module = EyepacsClassifier
    else:
        module = TaskClassifier

    checkpoint_callbacks = [
        ModelCheckpoint(
            monitor="val/loss",
            filename="best-loss-{epoch}-{step}",
        ),
        ModelCheckpoint(
            monitor="val/acc",
            filename="best-acc-{epoch}-{step}",
        ),
    ]

    # Train model
    gpus = 0 if args.cpu else 1

    trainer = pl.Trainer(
        max_epochs=params[param_category[args.test_method]]["trainer"]["epochs"],
        log_every_n_steps=100,
        limit_train_batches=args.data_frac,
        limit_val_batches=args.data_frac,
        logger=logger,
        callbacks=checkpoint_callbacks,
        gpus=gpus,
        num_sanity_val_steps=0,
    )

    model = module(**params[param_category[args.test_method]]["model"])
    trainer.fit(model, datamodule=data_module)

    ###############################################################################################################################
    # Run Eval:
    #
    # No replacement -> useful for subgroup performance analysis
    ###############################################################################################################################

    ckpt_path = "best"

    trainer.test(model=model, ckpt_path=ckpt_path, datamodule=DataModule(dataloader))

    # MNIST only: report subgroup performances here (as corruptness subgroups are not configured in Dataset class)
    if dataset_type == "mnist":

        for p_erase in [0, 1]:

            params["dataset"]["ds"]["p"]["subset_params"]["p_erase"] = p_erase
            dataloader = dataset_fn(params_dict=params["dataset"])
            data_module = DataModule(dataloader)

            logger = TensorBoardLogger(save_dir=artifacts_dir, name=f"{hash_string}_{p_erase}")

            trainer = pl.Trainer(
                max_epochs=params[param_category[args.test_method]]["trainer"]["epochs"],
                logger=logger,
                callbacks=checkpoint_callbacks,
                gpus=gpus,
            )

            model = module(**params[param_category[args.test_method]]["model"])

            trainer.test(model=model, ckpt_path=ckpt_path, datamodule=data_module)

    print("done")
