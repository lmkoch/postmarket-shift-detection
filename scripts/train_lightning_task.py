#!/usr/bin/python3

import argparse
import os

import pytorch_lightning as pl
from core.dataset import dataset_fn
from core.eval import eval

# TODO move to core data location
from core.model import DataModule, DomainClassifier, TaskClassifier
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from utils.config import hash_dict, load_config, save_config

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Run single experiment", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--exp_dir",
        action="store",
        type=str,
        help="experiment folder",
        default="./experiments/tmi"
        # default='./experiments/hypothesis-tests/mnist'
    )
    parser.add_argument(
        "--artifacts_dir",
        action="store",
        type=str,
        help="artifacts folder",
        default="./experiments/tmi/artifacts",
    )
    parser.add_argument(
        "--config_file",
        action="store",
        type=str,
        help="config file",
        # default="./config/eyepacs_quality_ood.yaml"
        default="./config/mnist_subgroups.yaml"
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
        "--eval_splits",
        action="store",
        default=["test"],
        nargs="+",
        help="List of splits to be evaluated, e.g. --eval_splits validation test",
    )

    args = parser.parse_args()

    params = load_config(args.config_file)

    ###############################################################################################################################
    # Preparation
    ###############################################################################################################################

    dataloader = dataset_fn(seed=args.seed, params_dict=params["dataset"])

    ###############################################################################################################################
    # Run training: separate exp hashes for all three methods (matched with dataset config)
    ###############################################################################################################################

    pl.seed_everything(args.seed, workers=True)

    # TODO check for weird seed stuff

    def train_model(specs):

        artifacts_dir = os.path.join(args.exp_dir, specs["dir_name"])

        model_config = {k: params[k] for k in ["dataset", specs["param_category"]]}
        hash_string = hash_dict(model_config)
        save_config(model_config, artifacts_dir)

        log_dir = os.path.join(artifacts_dir, hash_string)

        # model
        model = specs["module"](**params[specs["param_category"]]["model"])

        # training
        logger = TensorBoardLogger(save_dir=artifacts_dir, name=hash_string)
        checkpoint_callbacks = [
            ModelCheckpoint(
                monitor="val/loss",
                filename="best-loss-{epoch}-{step}",
            ),
            ModelCheckpoint(monitor="val/acc", filename="best-acc-{epoch}-{step}"),
        ]

        data = DataModule(dataloader, domains=specs["domains"])

        trainer = pl.Trainer(
            max_epochs=params[specs["param_category"]]["trainer"]["epochs"],
            log_every_n_steps=10,
            limit_train_batches=0.2,  # TODO increase to 1.0 after debugging
            limit_val_batches=0.2,  # TODO increase to 1.0 after debugging
            logger=logger,
            callbacks=checkpoint_callbacks,
        )
        trainer.fit(model, datamodule=data)

    # 2. C2ST

    specs = {
        "dir_name": "domain_classifier",
        "param_category": "domain_classifier",
        "module": DomainClassifier,
        "domains": ["p", "q"],
    }

    train_model(specs)

    # 3. MUKS

    specs = {
        "dir_name": "task_classifier",
        "param_category": "task_classifier",
        "module": TaskClassifier,
        "domains": ["p"],
    }

    train_model(specs)

    # 3. MUKS: train task classifier

    # artifacts_dir = os.path.join(args.exp_dir, "task_classif")

    # # TODO: hash could be done only on ['ds']['p'] and ['ds']['q]
    # model_config = {k: params[k] for k in ["dataset", "task_classifier"]}
    # hash_string = hash_dict(model_config)
    # save_config(model_config, artifacts_dir)

    # log_dir = os.path.join(artifacts_dir, hash_string)

    # # model
    # model = TaskClassifier(**params["task_classifier"]["model"])

    # # training
    # logger = TensorBoardLogger(save_dir=artifacts_dir, name=hash_string, default_hp_metric=False)
    # checkpoint_callbacks = [
    #     ModelCheckpoint(
    #         monitor="val/loss",
    #         filename="best-loss-{epoch}-{step}",
    #     ),
    #     ModelCheckpoint(monitor="val/acc", filename="best-acc-{epoch}-{step}"),
    # ]

    # trainer = pl.Trainer(
    #     max_epochs=params["task_lightning"]["trainer"]["epochs"],
    #     log_every_n_steps=10,
    #     limit_train_batches=0.05,
    #     limit_val_batches=0.05,
    #     logger=logger,
    #     callbacks=checkpoint_callbacks,
    # )
    # trainer.fit(model, dataloader["train"]["p"], dataloader["validation"]["q"])

    # TODO checkpointing stuff: what's version_0?

    # FYI. it’s now
    # lightning_logs/version_{version number}/epoch_{epoch number}-step_{global_step}.ckpt

    # and to access them:

    # version_number -> trainer.logger.version
    # epoch_number -> trainer.current_epoch
    # global_step -> trainer.global_step

    # eval:

    print("done")
