#!/usr/bin/python3

import argparse
import os

import pytorch_lightning as pl
from core.dataset import dataset_fn
from core.eval import eval
from core.model import TaskClassifier
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

    # 3. MUKS: train task classifier

    artifacts_dir = os.path.join(args.exp_dir, "task_classif")

    # TODO: hash could be done only on ['ds']['p'] and ['ds']['q]
    task_classifier_config = {k: params[k] for k in ["dataset", "task_lightning"]}
    hash_string = hash_dict(task_classifier_config)
    save_config(task_classifier_config, artifacts_dir)

    log_dir = os.path.join(artifacts_dir, hash_string)

    # data
    train_loader = dataloader["train"]["p"]
    val_loader = dataloader["validation"]["q"]

    # model
    model = TaskClassifier(
        model_params=params["task_lightning"]["model"],
    )

    # training
    logger = TensorBoardLogger(save_dir=artifacts_dir, name=hash_string, default_hp_metric=False)
    checkpoint_callbacks = [
        ModelCheckpoint(
            monitor="val/loss",
            filename="best-loss-{epoch}-{step}",
        ),
        ModelCheckpoint(monitor="val/acc", filename="best-acc-{epoch}-{step}"),
    ]

    trainer = pl.Trainer(
        default_root_dir=log_dir,
        max_epochs=params["task_lightning"]["trainer"]["epochs"],
        log_every_n_steps=10,
        limit_train_batches=0.05,
        limit_val_batches=0.05,
        logger=logger,
        callbacks=checkpoint_callbacks,
    )
    trainer.fit(model, train_loader, val_loader)

    # TODO checkpointing stuff: what's version_0?

    # eval:

    print("done")
