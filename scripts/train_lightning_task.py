#!/usr/bin/python3

import argparse
import os

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


def main(args, params):

    ###############################################################################################################################
    # Preparation
    ###############################################################################################################################

    # TODO check for weird seed stuff
    pl.seed_everything(args.seed, workers=True)

    dataloader = dataset_fn(params_dict=params["dataset"])

    ###############################################################################################################################
    # Run training: separate exp hashes for all three methods (matched with dataset config)
    ###############################################################################################################################

    def train_model(checkpoint_callbacks, module, param_category, data_frac=1):

        artifacts_dir = os.path.join(args.exp_dir, param_category)

        model_config = {k: params[k] for k in ["dataset", param_category]}
        hash_string = hash_dict(model_config)
        save_config(model_config, artifacts_dir)

        # model
        model = module(**params[param_category]["model"])

        # training
        logger = TensorBoardLogger(save_dir=artifacts_dir, name=hash_string)

        data = DataModule(dataloader)

        trainer = pl.Trainer(
            max_epochs=params[param_category]["trainer"]["epochs"],
            log_every_n_steps=10,
            limit_train_batches=data_frac,  # TODO increase to 1.0 after debugging
            limit_val_batches=data_frac,  # TODO increase to 1.0 after debugging
            logger=logger,
            callbacks=checkpoint_callbacks,
            gpus=0,
        )

        trainer.fit(model, datamodule=data)

        return trainer

    if args.test_method == "mmdd":
        checkpoint_callbacks = [
            ModelCheckpoint(
                monitor="val/loss",
                filename="best-loss-{epoch}-{step}",
            ),
            ModelCheckpoint(
                monitor="val/power",
                filename="best-power-{epoch}-{step}",
            ),
        ]

        specs = {
            "param_category": "mmd",
            "module": MaxKernel,
            "checkpoint_callbacks": checkpoint_callbacks,
        }

    elif args.test_method == "c2st":
        checkpoint_callbacks = [
            ModelCheckpoint(
                monitor="val/acc",
                filename="best-acc-{epoch}-{step}",
            ),
            ModelCheckpoint(
                monitor="val/loss",
                filename="best-loss-{epoch}-{step}",
            ),
        ]

        specs = {
            "param_category": "domain_classifier",
            "module": DomainClassifier,
            "checkpoint_callbacks": checkpoint_callbacks,
        }

    # 3. MUKS
    elif args.test_method == "muks":

        if params["dataset"]["ds"]["p"]["dataset"] == "eyepacs":
            model_class = EyepacsClassifier
        else:
            model_class = TaskClassifier

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

        specs = {
            "param_category": "task_classifier",
            "module": model_class,
            "checkpoint_callbacks": checkpoint_callbacks,
        }

    # Train model

    trainer = train_model(**specs, data_frac=args.data_frac)

    # Eval

    #

    res = trainer.test(datamodule=DataModule(dataloader))

    # res [{'val/loss_epoch': 0.7246412634849548, 'val/acc_epoch': 0.5092648267745972, 'test/power': 0.42, 'test/type_1err': 0.08}]


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
        default="c2st",
    )
    parser.add_argument(
        "--data_frac",
        action="store",
        default=0.01,
        type=float,
        help="Fraction of data to use (for debug purposes)",
    )

    args = parser.parse_args()

    params = load_config(args.config_file)

    # run experiments:

    # MNIST: p=0.5 vs p = {0, 1}
    # Camelyon: all vs each site s = {0, ..., 4}
    # Eyepacs: all vs. quality = {0, 1, 2}

    # Epochs: 10
    #

    # TODO:
    # 1. add specific options to parser, these should then override the config file.
    # 2. call this script with SLURM_TASK_ARRAY

    main(args, params)

    print("done")
