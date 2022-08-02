#!/usr/bin/python3

import argparse
import os
import pickle
from mimetypes import init

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torchvision
from core.dataset import dataset_fn
from core.eval import eval
from core.mmdd import trainer_object_fn
from core.model import model_fn
from scipy.stats import permutation_test
from sklearn.metrics import auc, confusion_matrix, roc_curve
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import helpers
from utils.config import create_exp_from_config, hash_dict, load_config, save_config
from utils.helpers import stat_C2ST


class DomainClassifier:
    def forward_pass(self, test_loader, num_items=None, keep_images=False, witness_type="logits"):
        """_summary_

        Args:
            test_loader (_type_): _description_
            num_items (_type_, optional): _description_. Defaults to None.
            keep_images (bool, optional): _description_. Defaults to False.
            witness_type (str, optional): _description_. Defaults to "logits".

        Raises:
            argparse.ArgumentError: _description_

        Returns:
            pd.DataFrame: dataframe with columns: sm output of positive class, domain prediction, witness
            torch tensor: images or None
        """

        self.model.eval()

        self.model = self.model.to(self.device)

        if num_items is None:
            num_items = len(test_loader)

        with torch.no_grad():

            images = []
            logits = []
            for idx, (x, *_) in tqdm(enumerate(test_loader)):

                batch_size = x.shape[0]

                if idx * batch_size >= num_items:
                    break

                x = x.to(self.device)
                logits.append(self.model(x))

                batch_size = x.shape[0]

                if keep_images:
                    images.append(x)

        logits = torch.cat(logits, 0)
        y_sm = torch.softmax(logits, dim=1)[:, 1]
        y_bin = torch.argmax(logits, dim=1)

        if keep_images:
            images = torch.cat(images, 0)
        else:
            images = None

        if witness_type == "logits":
            witness = (logits[:, 0] - logits[:, 1]).detach().cpu().numpy()
        else:
            raise argparse.ArgumentError(f"witness type not supported: {witness_type}")

        sample = {
            "y_sm": y_sm.detach().cpu().numpy(),  # softmax output for positive class (score)
            "y_bin": y_bin.detach().cpu().numpy(),  # binarised prediction
            "witness": witness,
        }

        return pd.DataFrame(sample), images

    def eval_general(self, test_loader):
        """Evaluation that does not dependn on dataset specifics

        Args:
            test_loader (_type_): _description_
        """

        self.load_best_val_results_and_checkpoint()

        self.model.eval()

        ############################################################################################
        # Forward pass through whole test set (P and Q)

        num_items = 100

        sample = {}

        for p_q in ["p", "q"]:

            sample[p_q], _ = self.forward_pass(
                test_loader[p_q], num_items=num_items, keep_images=False
            )

        results_file = os.path.join(self.log_dir, "test_predictions.pickle")

        with open(results_file, "wb") as f:
            pickle.dump(sample, f)

        ############################################################################################
        # Test power and type I error

        num_reps = self.eval_config["n_test_reps"]
        num_reps = 10

        # sample_sizes = [10, 30, 50, 100, 500, 1000]
        sample_sizes = [10, 30]
        powers = []
        type_1_errs = []
        for sample_size in sample_sizes:
            print(f"sample size: {sample_size}")

            power = 0
            type_1_err = 0

            for _ in range(num_reps):

                x = np.random.choice(sample["p"]["witness"], size=sample_size, replace=True)
                x2 = np.random.choice(sample["p"]["witness"], size=sample_size, replace=True)
                y = np.random.choice(sample["q"]["witness"], size=sample_size, replace=True)

                res_power = permutation_test((x, y), stat_C2ST)
                res_type_1_err = permutation_test((x, x2), stat_C2ST)

                power += res_power.pvalue < 0.05
                type_1_err += res_type_1_err.pvalue < 0.05

            powers.append(power / num_reps)
            type_1_errs.append(type_1_err / num_reps)

            print(powers)
            print(type_1_errs)

        ############################################################################################
        # Interpretability:
        # Witness function histogram + subgroup / witness correspondence

        sample["p"]["y"] = 1
        sample["q"]["y"] = 0

        df = pd.concat((sample["p"], sample["q"])).reset_index(drop=True)

        fig, ax = plt.subplots(figsize=(4, 4))

        sns.histplot(data=df, x="witness", hue="y", kde=True, bins=20)

        out_hists = os.path.join(self.log_dir, "witness_hists.pdf")
        fig.savefig(out_hists)

        fpr, tpr, _ = roc_curve(df["y"], df["y_sm"])
        roc_auc = auc(fpr, tpr)
        ii = np.where(tpr > 0.95)[0][0]

        results = {
            "accuracy": np.sum(df["y"] == df["y_bin"]) / df.shape[0],
            "roc_auc": roc_auc,
            "fpr95": fpr[ii],
            "power_c2st_l": powers,
            "type_1_err__c2st_l": type_1_errs,
            "sample_size": sample_sizes,
        }

        print(results)

        res_file = os.path.join(self.log_dir, "test_results.csv")
        res_df = pd.DataFrame.from_dict(results, orient="index")
        res_df.to_csv(res_file)

        # TODO: just do a forward pass of a few batches or so, and sort them by witness, and show them?
        # Plot some example images with low and high witness function values

        num_items = 100

        sample = {}
        img = {}

        for p_q in ["p", "q"]:

            sample[p_q], img[p_q] = self.forward_pass(
                test_loader[p_q], num_items=num_items, keep_images=True
            )

        sample["p"]["y"] = 1
        sample["q"]["y"] = 0

        df = pd.concat((sample["p"], sample["q"])).reset_index(drop=True)
        images = torch.cat([img["p"], img["q"]], 0)

        sorted = df["witness"].argsort()
        num_examples = 8
        idx_10 = int(len(sorted) * 0.1)
        idx_90 = int(len(sorted) * 0.9)

        example_idx = {
            "like_p": sorted[idx_10 : idx_10 + num_examples],
            "like_q": sorted[idx_90 : idx_90 + num_examples],
        }

        print(df["witness"][example_idx["like_p"]])
        print(df["witness"][example_idx["like_q"]])

        for p_q in ["p", "q"]:
            out_fig = os.path.join(self.log_dir, f"panel_like_{p_q}.png")

            imgs = []

            # get images correponding to specific indices back from dataset
            for ele in example_idx[f"like_{p_q}"]:

                x = images[ele]

                imgs.append(x)

            img_grid = torchvision.utils.make_grid(imgs, normalize=True)
            img_grid = np.transpose(img_grid, (1, 2, 0)).numpy()
            plt.imsave(out_fig, img_grid)

    def eval(self, test_loader):

        self.load_best_val_results_and_checkpoint()

        self.model.eval()

        debug = True

        # FIXME already in the forward pass, I assume I have metadata -> not the case for MNIST.

        with torch.no_grad():

            sample = {}

            for p_q in ["p", "q"]:

                logits = []
                metadata = []
                indices = []
                print(f"forward pass through dataset {p_q}:")
                for idx, (x, _, m, inds) in tqdm(enumerate(test_loader[p_q])):

                    if debug and idx == 10:
                        break

                    x = x.to(self.device)
                    logits.append(self.model(x))
                    metadata.append(m)
                    indices.append(inds)

                    y = torch.ones((x.shape[0], 1))
                    if p_q == "q":
                        y = y * 0

                logits = torch.cat(logits, 0)
                y_sm = torch.softmax(logits, dim=1)
                y_bin = torch.argmax(logits, dim=1)

                y = torch.cat(y, 0)

                metadata = np.concatenate(metadata, 0)
                indices = np.concatenate(indices, 0)

                sample[p_q] = {
                    "logits": logits.detach().cpu().numpy(),  # logits for each class
                    "y_sm": y_sm.detach().cpu().numpy(),  # softmax output for each class
                    "y_bin": y_bin.detach().cpu().numpy(),  # binarised prediction
                    "m": metadata,
                    "inds": indices,
                    "witness": (logits[:, 0] - logits[:, 1]).detach().cpu().numpy(),
                }

        results_file = os.path.join(self.log_dir, "test_predictions.pickle")

        with open(results_file, "wb") as f:
            pickle.dump(sample, f)

        dataset = {"p": test_loader["p"].dataset, "q": test_loader["q"].dataset}

        sample_p = sample["p"]["y_sm"]
        sample_q = sample["q"]["y_sm"]

        labels_p = np.ones((sample_p.shape[0], 1))
        labels_q = np.zeros((sample_q.shape[0], 1))

        y = np.concatenate([labels_p, labels_q], axis=None)
        y_pred = np.concatenate([sample["p"]["y_bin"], sample["q"]["y_bin"]], axis=None)
        logits = np.concatenate([sample["p"]["logits"], sample["q"]["logits"]], axis=0)
        y_sm = np.concatenate([sample["p"]["y_sm"], sample["q"]["y_sm"]], 0)
        dset_ind = np.concatenate([sample["p"]["inds"], sample["q"]["inds"]], axis=None)

        # Test power at various sample sizes

        # Interpretability:
        # Witness function histogram + subgroup / witness correspondence

        # Theoretical performance limits (based on subgroup proportions)

        # witness is the softmax output for the *positive* class (1, not 0)

        # df = pd.DataFrame(
        #     {"y": y[:, 0], "witness": y_sm[:, 1], "y_pred": y_pred, "dset_ind": dset_ind}
        # )

        df = pd.DataFrame(
            {
                "y": y,
                "witness": logits[:, 0] - logits[:, 1],
                "y_pred": y_pred,
                "dset_ind": dset_ind,
            }
        )

        # Plot some example images with low and high witness function values
        sorted = df["witness"].argsort()
        num_examples = 5
        idx_10 = int(len(sorted) * 0.1)
        idx_90 = int(len(sorted) * 0.9)

        example_idx = {
            "like_p": sorted[idx_10 : idx_10 + num_examples],
            "like_q": sorted[idx_90 : idx_90 + num_examples],
        }

        print(df["witness"][example_idx["like_p"]])
        print(df["witness"][example_idx["like_q"]])

        for p_q in ["p", "q"]:
            out_fig = os.path.join(self.log_dir, f"panel_like_{p_q}.png")

            imgs = []

            # get images correponding to specific indices back from dataset
            for ele in example_idx[f"like_{p_q}"]:
                if df["y"][ele] == 1:
                    dset = dataset["p"]
                else:
                    dset = dataset["q"]

                idx = df["dset_ind"][ele]

                x, *_ = dset[idx]

                imgs.append(x)

            img_grid = torchvision.utils.make_grid(imgs, normalize=True)
            img_grid = np.transpose(img_grid, (1, 2, 0)).numpy()
            plt.imsave(out_fig, img_grid)

        # FIXME: from here on, there is code specific to eyepacs -> should keep this general!

        metadata_fields = ["side", "field", "gender", "age", "quality", "ethnicity", "y_meta"]

        metadata = np.concatenate([sample["p"]["m"], sample["q"]["m"]], 0)
        meta = pd.DataFrame.from_records(metadata, columns=metadata_fields)

        df = df.join(meta)

        fig, ax = plt.subplots(1, 2, figsize=(8, 4))

        sns.histplot(data=df, x="witness", hue="y", ax=ax[0], kde=True, bins=20)
        sns.histplot(
            data=df, x="witness", hue="quality", multiple="dodge", ax=ax[1], kde=True, bins=20
        )

        out_hists = os.path.join(self.log_dir, "witness_hists.pdf")
        fig.savefig(out_hists)

        num_reps = self.eval_config["n_test_reps"]
        num_reps = 10

        sample_sizes = [10, 30, 50, 100, 500, 1000]
        powers = []
        type_1_errs = []
        for sample_size in sample_sizes:
            print(f"sample size: {sample_size}")

            power = 0
            type_1_err = 0

            for _ in range(num_reps):

                x = np.random.choice(sample["p"]["witness"], size=sample_size, replace=True)
                x2 = np.random.choice(sample["p"]["witness"], size=sample_size, replace=True)
                y = np.random.choice(sample["q"]["witness"], size=sample_size, replace=True)

                res_power = permutation_test((x, y), stat_C2ST)
                res_type_1_err = permutation_test((x, x2), stat_C2ST)

                power += res_power.pvalue < 0.05
                type_1_err += res_type_1_err.pvalue < 0.05

            powers.append(power / num_reps)
            type_1_errs.append(type_1_err / num_reps)

            print(powers)
            print(type_1_errs)

        # # conf_mat = confusion_matrix(y_all.cpu().numpy(), y_bin.cpu().numpy())

        fpr, tpr, _ = roc_curve(df["y"], -df["witness"])
        roc_auc = auc(fpr, tpr)
        ii = np.where(tpr > 0.95)[0][0]

        results = {
            "accuracy": np.sum(df["y"] == df["y_pred"]) / df.shape[0],
            "roc_auc": roc_auc,
            "fpr95": fpr[ii],
            "power_c2st_l": powers,
            "type_1_err__c2st_l": type_1_errs,
            "sample_size": sample_size,
        }

        print(results)

        res_file = os.path.join(self.log_dir, "test_results.csv")
        res_df = pd.DataFrame.from_dict(results, orient="index")
        res_df.to_csv(res_file)


def load_or_train_task_classifier():
    pass


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

    dataloader = dataset_fn(params_dict=params["dataset"])

    ###############################################################################################################################
    # Run training: separate exp hashes for all three methods (matched with dataset config)
    ###############################################################################################################################

    # 1. MMD: train kernel

    artifacts_dir = os.path.join(args.exp_dir, "mmdd")

    mmd_config = {k: params[k] for k in ["dataset", "trainer"]}
    hash_string = hash_dict(mmd_config)
    save_config(mmd_config, artifacts_dir)

    log_dir = os.path.join(artifacts_dir, hash_string)

    # FIXME mmd config should all be in one section (merge "trainer" and relevant parts of "model")
    model = model_fn(seed=args.seed, params=params["model"])
    trainer = trainer_object_fn(
        model=model, dataloaders=dataloader, seed=args.seed, log_dir=log_dir, **params["trainer"]
    )

    trainer.train()

    # 2. C2ST: train domain classifier

    artifacts_dir = os.path.join(args.exp_dir, "domain_classif")

    # extract hash from config
    domain_classifier_config = {k: params[k] for k in ["dataset", "domain_classifier"]}
    hash_string = hash_dict(domain_classifier_config)
    save_config(domain_classifier_config, artifacts_dir)

    log_dir = os.path.join(artifacts_dir, hash_string)

    domain_classifier = DomainClassifier(
        dataloader,
        log_dir,
        model_params=params["domain_classifier"]["model"],
        train_params=params["domain_classifier"]["train"],
        eval_params=params["domain_classifier"]["eval"],
    )

    domain_classifier.train()

    # 3. MUKS: train task classifier

    artifacts_dir = os.path.join(args.exp_dir, "task_classif")

    # TODO: hash could be done only on ['ds']['p'] and ['ds']['q]
    task_classifier_config = {k: params[k] for k in ["dataset", "task_classifier"]}
    hash_string = hash_dict(task_classifier_config)
    save_config(task_classifier_config, args.artifacts_dir)

    log_dir = os.path.join(artifacts_dir, hash_string)

    from core.model import get_classification_model

    task_classifier = get_classification_model(
        task_classifier_config["task_classifier"]["model"], log_dir=log_dir, download=False
    )

    task_classifier.train_model(dataloader_train=dataloader["train"]["p"])
    acc = task_classifier.eval_model(dataloader=dataloader["validation"]["p"])

    print(f"task acc: {acc}")

    ###############################################################################################################################
    # Eval MMD-D and MUKS on various sample sizes
    ###############################################################################################################################

    # all three models:
    #    - Power vs sample size
    #    - Type 1 err vs sample size

    # 1. MMD-D
    #    - ** witness interpretability stuff -> CAVEAT: depends on dataset!

    # 2. C2ST
    #    - ** witness interpretability stuff -> CAVEAT: depends on dataset!
    #    - domain classification accuracy / auc / fpr95

    # 3. MUKS
    #    - subgroup task performance -> -> CAVEAT: depends on dataset! maybe separate script?

    # 1. MMD-D

    # 2. C2ST

    # EVAL:
    # FIXME the eval code below is eyepacs specific. Need this to work for MNIST and camelyon too
    # domain_classifier.eval(test_loader=dataloader["test"])

    # domain_classifier.eval_general(test_loader=dataloader["test"])

    # 3. MUKS

    eval_dir = os.path.join(args.exp_dir, "eval")

    hash_string = hash_dict(params)
    save_config(params, eval_dir)

    log_dir = os.path.join(eval_dir, hash_string)

    models = {"mmdd": trainer.model, "muks": task_classifier, "c2st": domain_classifier}

    for split in args.eval_splits:
        eval(
            eval_dir,
            hash_string,
            params,
            args.seed,
            split,
            models,
            sample_sizes=[10, 30, 50, 100, 200, 500],
            # sample_sizes=[5, 10, 30],
            num_reps=100,
            num_permutations=1000,
            force_run=True,
        )

    print("done")
