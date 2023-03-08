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
        # sbatch_file.writelines("#SBATCH --mem=60G\n")
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
        default="./config/lightning/mnist.yaml"
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
        "--train_data_frac",
        action="store",
        default=1.0,
        type=float,
        help="Fraction of data to use (for ablation). Different from debug option!",
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
        "--img_size",
        action="store",
        type=int,
        help="Input image size. Data will be resized to this resolution",
    )
    parser.add_argument(
        "--mmd_feature_extractor",
        action="store",
        type=str,
        help="MMD backbone (liu or resnet50)",
    )
    parser.add_argument(
        "--c2st_arch",
        action="store",
        type=str,
        help="C2ST backbone (shallow or resnet50)",
    )
    parser.add_argument(
        "--data_aug",
        action="store_true",
        default=None,
        help="override data augmentation settings (yes/no)",
    )
    parser.add_argument(
        "--no_data_aug",
        action="store_false",
        default=None,
        dest="data_aug",
        help="override data augmentation settings (yes/no)",
    )
    parser.add_argument(
        "--batch_size",
        action="store",
        type=int,
        help="Batch size",
    )
    parser.add_argument(
        "--subset_p_erase",
        action="store",
        type=float,
        help="List of subgroups to include",
    )
    parser.add_argument(
        "--subset_comorbid",
        nargs="+",
        type=bool,
        help="List of subgroups to include",
    )
    parser.add_argument(
        "--subset_qual",
        nargs="+",
        type=str,
        help="List of subgroups to include",
    )
    parser.add_argument(
        "--subset_sex",
        nargs="+",
        type=str,
        help="List of subgroups to include",
    )
    parser.add_argument(
        "--subset_ethnicity",
        nargs="+",
        type=str,
        help="List of subgroups to include",
    )
    parser.add_argument(
        "--subset_center",
        nargs="+",
        type=int,
        help="List of subgroups to include",
    )
    parser.add_argument(
        "--subset_qual_gradual",
        action="store",
        type=int,
        help="Overrepresentation factor used for Adequate image quality",
    )

    # parser.add_argument(
    #     "--subset_comorbid",
    #     nargs="+",
    #     type=bool,
    #     help="List of subgroups to include",
    # )
    parser.add_argument("--no-slurm", dest="slurm", action="store_false", default=False)
    args = parser.parse_args()

    params = load_config(args.config_file)

    # allow overriding params from command line
    if args.img_size is not None:
        params["dataset"]["ds"]["basic_preproc"]["img_size"] = args.img_size
        params["mmd"]["model"]["img_size"] = args.img_size

    if args.batch_size is not None:
        params["dataset"]["dl"]["batch_size"] = args.batch_size

    params["dataset"]["ds"]["data_frac"] = args.train_data_frac

    if args.mmd_feature_extractor is not None:
        params["mmd"]["model"]["feature_extractor"] = args.mmd_feature_extractor

    if args.c2st_arch is not None:
        params["domain_classifier"]["model"]["arch"] = args.c2st_arch

    if args.data_aug:
        params["dataset"]["ds"]["data_augmentation"] = [
            "random_crop",
            "horizontal_flip",
            "vertical_flip",
            "color_distortion",
            "rotation",
        ]
    elif args.data_aug == False:
        params["dataset"]["ds"]["data_augmentation"] = []

    if args.subset_p_erase is not None:
        params["dataset"]["ds"]["q"]["subset_params"]["p_erase"] = args.subset_p_erase

    if args.subset_comorbid is not None:
        params["dataset"]["ds"]["q"]["subset_params"][
            "diagnoses_comorbidities"
        ] = args.subset_comorbid

    if args.subset_qual is not None:
        params["dataset"]["ds"]["q"]["subset_params"]["session_image_quality"] = args.subset_qual

    if args.subset_sex is not None:
        params["dataset"]["ds"]["q"]["subset_params"]["patient_gender"] = args.subset_sex

    if args.subset_ethnicity is not None:
        params["dataset"]["ds"]["q"]["subset_params"]["patient_ethnicity"] = args.subset_ethnicity

    if args.subset_center is not None:
        params["dataset"]["ds"]["q"]["subset_params"]["center"] = args.subset_center

    if args.subset_qual_gradual is not None:
        params["dataset"]["dl"]["q"]["sampling_weights"][1] = args.subset_qual_gradual

    # TODO: eyepacs remaining categories

    # mapping of correct config file chapter:
    param_category = {"mmdd": "mmd", "c2st": "domain_classifier", "muks": "task_classifier"}

    artifacts_dir = os.path.join(args.exp_dir, args.test_method)
    model_config = {k: params[k] for k in ["dataset", param_category[args.test_method]]}
    hash_string = hash_dict(model_config)

    save_config(model_config, artifacts_dir)

    dataset_type = params["dataset"]["ds"]["p"]["dataset"]

    if args.slurm:
        # submit slurm job instead of executing locally

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

    if args.test_method == "mmdd":
        checkpoint_callbacks = [
            ModelCheckpoint(
                monitor="val/loss",
                filename="best-loss-{epoch}-{step}",
            ),
            ModelCheckpoint(
                monitor="val/loss",
                filename="best-loss",
            ),
            ModelCheckpoint(
                monitor="val/power",
                filename="best-power-{epoch}-{step}",
            ),
        ]

        module = MaxKernel

    elif args.test_method == "c2st":
        checkpoint_callbacks = [
            ModelCheckpoint(
                monitor="val/loss",
                filename="best-loss-{epoch}-{step}",
            ),
            ModelCheckpoint(
                monitor="val/loss",
                filename="best-loss",
            ),
            ModelCheckpoint(
                monitor="val/acc",
                filename="best-acc-{epoch}-{step}",
            ),
        ]

        module = DomainClassifier

    # 3. MUKS
    elif args.test_method == "muks":

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
                monitor="val/loss",
                filename="best-loss",
            ),
            ModelCheckpoint(
                monitor="val/acc",
                filename="best-acc-{epoch}-{step}",
            ),
        ]

    # Train model

    # training
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

    if args.test_method in ["c2st", "mmdd"]:

        # import glob

        # log_dir = os.path.join(
        #     artifacts_dir,
        #     hash_string,
        # )

        # versions = os.listdir(log_dir)
        # checkpoint_dir = os.path.join(log_dir, versions[-1], "checkpoints")

        # checkpoint_path = glob.glob(f"{checkpoint_dir}/best-loss-*.ckpt")[0]

        # if len(checkpoint_path) > 0:
        #     checkpoint_path = checkpoint_path[0]
        # else:
        #     print(f"No checkpoint available for exp: {log_dir}")

        # TODO: if train, fit the model. Else: load latest checkpoint

        model = module(**params[param_category[args.test_method]]["model"])

        trainer.fit(model, datamodule=data_module)
    elif args.test_method == "muks":

        artifacts_path = {
            "eyepacs": "experiments/task/eyepacs/muks/ed38371b2586ab224c4c55642b443b9b/version_0/checkpoints/best-loss-epoch=20-step=102522.ckpt",
            # "eyepacs": "experiments/archive/endspurt/.../ed38371b2586ab224c4c55642b443b9b/version_0/checkpoints/best-loss-epoch=15-step=78112.ckpt",
            # "camelyon": "experiments/endspurt/camelyon/task_smallevents/task_classifier/1bd08d2856418bd6056d24f58671ec86/version_0/checkpoints/best-loss-epoch=13-step=248682.ckpt",
            "camelyon": "experiments/nov/task/camelyon/muks/ae9d742a0d9fd635b06dd96a1afd35fe/version_0/checkpoints/best-loss-epoch=17-step=319734.ckpt",
            "mnist": "experiments/oct/task/mnist/muks/0f2ab3ce9f01eb2f5c230e2c2aa2f99f/version_0/checkpoints/best-loss-epoch=13-step=10934.ckpt",
        }

        # TODO TEST

        eyepacs_smaller_train_data = {
            "1": artifacts_path["eyepacs"],
            "0.5": "experiments/eyepacs/task_eyepacs_data_frac/muks/20bf3822ef0dedf683189f4d10376047/version_0/checkpoints/best-loss-epoch=10-step=26851.ckpt",
            "0.1": "experiments/eyepacs/task_eyepacs_data_frac/muks/7acfe952ca2b44df74a70db987ffe90e/version_0/checkpoints/best-loss-epoch=17-step=8784.ckpt",
            "0.01": "experiments/eyepacs/task_eyepacs_data_frac/muks/d478a4d44fcb7f2f19e37fd3cdf3a0af/version_0/checkpoints/best-loss-epoch=24-step=1200.ckpt",
        }

        if dataset_type == "eyepacs" and params["dataset"]["ds"]["data_frac"] < 1:
            model_path = eyepacs_smaller_train_data[str(params["dataset"]["ds"]["data_frac"])]
            print(f"Load task model from: {model_path}")
        else:
            model_path = artifacts_path[dataset_type]

        model = module.load_from_checkpoint(model_path, strict=False)

    ###############################################################################################################################
    # Run Eval:
    ###############################################################################################################################
    # Eval
    from core.eval import eval

    if args.test_method in ["c2st", "mmdd"]:
        ckpt_path = "best"
    else:
        ckpt_path = model_path
        # ckpt_path = None

    eval(trainer, model, ckpt_path, params, sample_sizes=[10, 30, 50, 100, 200, 500])

    print("done")
