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


def initialize_lr_scheduler(optimizer, scheduler_args):
    """define learning rate scheduler

    Args:
        optimizer (_type_): _description_
        scheduler_args (dict): _description_

    Returns:
        _type_: _description_
    """
    return torch.optim.lr_scheduler.ExponentialLR(optimizer, **scheduler_args)


# define optmizer
def initialize_optimizer(
    model, optimizer_strategy, learning_rate, weight_decay, momentum, nesterov
):
    if optimizer_strategy == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=weight_decay,
        )
    elif optimizer_strategy == "ADAM":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    else:
        raise NotImplementedError("Not implemented optimizer.")

    return optimizer


def stat_C2ST(sample_a, sample_b):
    """callable to calculate statistic from two samples for C2ST-S

    Args:
        sample_a (_type_): binary (C2ST-S) or softmax (C2ST-L) predictions for sample a
        sample_b (_type_): binary (C2ST-S) or softmax (C2ST-L) predictions for sample b
    """

    stat = abs(sample_a.mean() - sample_b.mean())

    return stat


# Define the deep network for C2ST-S and C2ST-L
class Discriminator(nn.Module):
    def __init__(self, n_channels, img_size):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [
                nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0),
            ]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(n_channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = img_size // 2**4
        self.adv_layer = nn.Sequential(
            nn.Linear(128 * ds_size**2, 300), nn.ReLU(), nn.Linear(300, 2), nn.Softmax()
        )

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity


class DomainClassifier:
    def __init__(
        self,
        dataloaders,
        artifacts_dir,
        hash_string,
        train_params,
        eval_params,
        seed=1000,
    ):

        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        # check if model exists
        self.log_dir = os.path.join(artifacts_dir, hash_string)

        os.makedirs(self.log_dir, exist_ok=True)

        # otherwise train domain classifier

        self.trainloader_p = dataloaders["train"]["p"]
        self.trainloader_q = dataloaders["train"]["q"]
        self.valloader_p = dataloaders["validation"]["p"]
        self.valloader_q = dataloaders["validation"]["q"]
        self.seed = seed

        self.loss_fn = torch.nn.CrossEntropyLoss()

        # eval config
        self.use_tensorboard = eval_params.get("use_tensorboard", True)
        self.eval_interval_globalstep = eval_params.get("eval_interval_globalstep", 10)
        self.eval_config = eval_params

        if self.use_tensorboard:
            self.writer = SummaryWriter(self.log_dir, flush_secs=10)

        self.model = torchvision.models.resnet50(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)

        # self.model = Discriminator(dataset_params['ds']['n_channels'], dataset_params['ds']['img_size'][0])
        self.model = self.model.to(self.device)

        self.best_val_acc = 0

        self.epochs = train_params["epochs"]

        self.optimizer = initialize_optimizer(
            self.model,
            train_params["optimizer"],
            train_params["learning_rate"],
            train_params["weight_decay"],
            train_params["momentum"],
            train_params["nesterov"],
        )

        if train_params["lr_scheduler"]:
            self.lr_scheduler = initialize_lr_scheduler(self.optimizer, {"gamma": 0.6})
        else:
            self.lr_scheduler = None

        if self.check_if_already_trained():
            self.load_last_results_and_checkpoint()
        else:
            print("Does not exists - train it now.")

    @property
    def checkpoint_path(self):
        return os.path.join(self.log_dir, "model.pt")

    @property
    def checkpoint_path_best_val(self):
        return os.path.join(self.log_dir, "model_best_val_acc.pt")

    def check_if_already_trained(self):
        """Check if model is already trained by checking for results file

        Returns:
            Bool: true if trained
        """

        return os.path.exists(self.checkpoint_path)

    def save_results(self, epoch, batch_loss, checkpoint_path):

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "loss": batch_loss.item(),
                "best_val_acc": self.best_val_acc,
            },
            checkpoint_path,
        )

    def load_results_and_checkpoint(self, checkpoint_path):

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint["epoch"]
        batch_loss = checkpoint["loss"]
        self.best_val_acc = checkpoint["best_val_acc"]

        return epoch, batch_loss

    def load_last_results_and_checkpoint(self):
        return self.load_results_and_checkpoint(self.checkpoint_path)

    def load_best_val_results_and_checkpoint(self):
        return self.load_results_and_checkpoint(self.checkpoint_path_best_val)

    def prepare_batch_data(self, x_p: torch.Tensor, x_q: torch.Tensor):

        labels_p = torch.ones((x_p.shape[0], 1))
        labels_q = torch.zeros((x_q.shape[0], 1))

        x = torch.cat([x_p, x_q], 0)
        y = torch.cat([labels_p, labels_q], 0).squeeze().long().to(self.device)

        return x, y

    def train_step(self, x_p: torch.Tensor, x_q: torch.Tensor) -> None:

        self.optimizer.zero_grad()
        self.model.train()

        x, y = self.prepare_batch_data(x_p, x_q)

        y_pred = self.model(x)

        loss = self.loss_fn(y_pred, y)
        loss.backward()
        self.optimizer.step()

        return loss

    def train(self):

        torch.manual_seed(self.seed)

        if self.check_if_already_trained():
            self.load_last_results_and_checkpoint()
            print("model already trained - do not train again.")
            return

        global_step = 0

        self.model.train()

        for epoch in range(self.epochs):

            dl_tr_f_enumerator = enumerate(self.trainloader_q)

            progress = tqdm(enumerate(self.trainloader_p))
            for batch_idx, batch_data in progress:

                (imgs_p, *_) = batch_data

                try:
                    _, (imgs_q, *_) = next(dl_tr_f_enumerator)
                except StopIteration:
                    break

                imgs_p = imgs_p.to(self.device)
                imgs_q = imgs_q.to(self.device)

                batch_loss = self.train_step(imgs_p, imgs_q)

                progress.set_description(
                    f"batch / epoch [{batch_idx} / {epoch}], loss: {batch_loss}"
                )

                if global_step % self.eval_interval_globalstep == 0:

                    generate_img_grid = True
                    if self.eval_config["plot_images_first_only"] and global_step > 0:
                        generate_img_grid = False

                    results = self.val_step(global_step, generate_img_grid=generate_img_grid)

                    val_acc = results["val"]["accuracy"]

                    if val_acc > self.best_val_acc:
                        print(f"New best val accuracy: {val_acc}")
                        self.best_val_acc = val_acc
                        self.save_results(epoch, batch_loss, self.checkpoint_path_best_val)

                global_step += 1

            # update learning rate
            if self.lr_scheduler:
                self.lr_scheduler.step()

        if self.use_tensorboard:
            self.writer.close()

        self.save_results(epoch, batch_loss, self.checkpoint_path)

    def performance_measures(self, dataloader_p, dataloader_q, num_batches, num_permutations=100):
        """Calculate a bunch of performance measures

        Returns:
            Dictionary containing all calculated measures as well as fpr and tpr arrays
        """

        iterator_p = enumerate(dataloader_p)
        iterator_q = enumerate(dataloader_q)

        running_rejects = 0.0
        running_rejects_h0 = 0.0
        running_loss = 0.0

        # Results

        # classifier performance (FPR95, )

        # classifier-based test

        # class proportions (zerorate)

        running_y = []
        running_y_pred = []

        with torch.no_grad():

            # for idx, (x, _) in enumerate(iterator_p):

            for idx in range(num_batches):
                try:
                    _, (x_p, *_) = next(iterator_p)
                    _, (x_p2, *_) = next(iterator_p)
                    batch_idx, (x_q, *_) = next(iterator_q)
                except:
                    self.logger.info(
                        f"{num_batches} larger than dataset size. \
                        Wrap around after {batch_idx + 1} batches."
                    )
                    iterator_p = enumerate(dataloader_p)
                    iterator_q = enumerate(dataloader_q)
                    _, (x_p, *_) = next(iterator_p)
                    _, (x_p2, *_) = next(iterator_p)
                    batch_idx, (x_q, *_) = next(iterator_q)

                x_p, x_q, x_p2 = x_p.to(self.device), x_q.to(self.device), x_p2.to(self.device)

                x, y = self.prepare_batch_data(x_p, x_q)

                y_pred = self.model(x)
                softmax_all = torch.softmax(y_pred, dim=1)

                y_bin = torch.argmax(y_pred, dim=1)

                running_loss += self.loss_fn(y_pred, y)

                running_y.append(y)
                running_y_pred.append(y_pred)

                # Gather results
                # running_rejects += h_u
                # running_rejects_h0 += h_u_h0
                # running_loss += loss.item()

        # reject_rate = running_rejects / (idx+1)
        # reject_rate_h0 = running_rejects_h0 / (idx+1)

        y_all = torch.cat(running_y, 0)
        y_pred_all = torch.cat(running_y_pred, 0)
        softmax_all = torch.softmax(y_pred_all, dim=1)
        y_bin = torch.argmax(y_pred_all, dim=1)

        sample_p = softmax_all[y_all == 1, 1].detach().cpu().numpy()
        sample_q = softmax_all[y_all == 0, 1].detach().cpu().numpy()

        power = 0
        type_1_err = 0
        num_reps = self.eval_config["n_test_reps"]

        for _ in range(num_reps):
            sample_size = 100

            x = np.random.choice(sample_p, size=sample_size, replace=True)
            x2 = np.random.choice(sample_p, size=sample_size, replace=True)
            y = np.random.choice(sample_q, size=sample_size, replace=True)

            res_power = permutation_test((x, y), stat_C2ST)
            res_type_1_err = permutation_test((x, x2), stat_C2ST)

            power += res_power.pvalue < self.eval_config["alpha"]
            type_1_err += res_type_1_err.pvalue < self.eval_config["alpha"]

        power = power / num_reps
        type_1_err = type_1_err / num_reps

        # conf_mat = confusion_matrix(y_all.cpu().numpy(), y_bin.cpu().numpy())

        fpr, tpr, _ = roc_curve(y_all.cpu().numpy(), softmax_all[:, 1].detach().cpu().numpy())

        roc_auc = auc(fpr, tpr)
        ii = np.where(tpr > 0.95)[0][0]

        # print(conf_mat)

        # TODO check sampling - am I getting the correct data??

        results = {
            "loss": running_loss / (idx + 1),
            "accuracy": torch.sum(y_bin == y_all) / y_bin.shape[0],
            "roc_auc": roc_auc,
            "fpr95": fpr[ii],
            "power_c2st_l": power,
            "type_1_err__c2st_l": type_1_err,
        }

        return results

    def val_step(self, global_step, generate_img_grid=True):

        self.model.eval()

        num_val_batches = self.eval_config["num_eval_batches"]
        num_permutations = self.eval_config["n_permute"]

        # Results

        # classifier performance (accuracy, FPR95, )

        # classifier-based test

        #

        results = {
            "train": self.performance_measures(
                self.trainloader_p, self.trainloader_q, num_val_batches, num_permutations
            ),
            "val": self.performance_measures(
                self.valloader_p, self.valloader_q, num_val_batches, num_permutations
            ),
        }

        if self.use_tensorboard:

            for split in ["train", "val"]:
                for k, v in results[split].items():
                    self.writer.add_scalar(f"{split}_{k}", v, global_step)

            if generate_img_grid:
                self.writer.add_image(
                    "train_images_p",
                    helpers.make_image_grid(self.trainloader_p, num_images=8),
                    global_step,
                )
                self.writer.add_image(
                    "train_images_q",
                    helpers.make_image_grid(self.trainloader_q, num_images=8),
                    global_step,
                )

            self.writer.flush()

        self.model.train()

        return results

    def eval(self, test_loader):

        self.load_best_val_results_and_checkpoint()

        self.model.eval()

        debug = True

        with torch.no_grad():

            sample = {}

            for p_q in ["p", "q"]:

                y_pred = []
                metadata = []
                indices = []
                print(f"forward pass through dataset {p_q}:")
                for idx, (x, _, m, inds) in tqdm(enumerate(test_loader[p_q])):

                    if debug and idx == 10:
                        break

                    x = x.to(self.device)
                    y_pred.append(self.model(x))
                    metadata.append(m)
                    indices.append(inds)

                y_pred = torch.cat(y_pred, 0)
                y_sm = torch.softmax(y_pred, dim=1).detach().cpu().numpy()
                y_bin = torch.argmax(y_pred, dim=1).detach().cpu().numpy()

                metadata = np.concatenate(metadata, 0)
                indices = np.concatenate(indices, 0)

                sample[p_q] = {
                    "y_sm": y_sm,  # softmax output for each class
                    "y_bin": y_bin,  # binarised prediction
                    "m": metadata,
                    "inds": indices,
                }

        results_file = os.path.join(self.log_dir, "test_predictions.pickle")

        with open(results_file, "wb") as f:
            pickle.dump(sample, f)

        dataset = {"p": test_loader["p"].dataset, "q": test_loader["q"].dataset}

        metadata_fields = ["side", "field", "gender", "age", "quality", "ethnicity", "y_meta"]

        sample_p = sample["p"]["y_sm"]
        sample_q = sample["q"]["y_sm"]

        labels_p = np.ones((sample_p.shape[0], 1))
        labels_q = np.zeros((sample_q.shape[0], 1))

        y = np.concatenate([labels_p, labels_q], axis=None)
        y_pred = np.concatenate([sample["p"]["y_bin"], sample["q"]["y_bin"]], axis=None)
        y_sm = np.concatenate([sample["p"]["y_sm"], sample["q"]["y_sm"]], 0)
        dset_ind = np.concatenate([sample["p"]["inds"], sample["q"]["inds"]], axis=None)

        metadata = np.concatenate([sample["p"]["m"], sample["q"]["m"]], 0)
        meta = pd.DataFrame.from_records(metadata, columns=metadata_fields)

        # Test power at various sample sizes

        # Interpretability:
        # Witness function histogram + subgroup / witness correspondence

        # Theoretical performance limits (based on subgroup proportions)

        # witness is the softmax output for the *positive* class (1, not 0)

        # df = pd.DataFrame(
        #     {"y": y[:, 0], "witness": y_sm[:, 1], "y_pred": y_pred, "dset_ind": dset_ind}
        # )

        df = pd.DataFrame({"y": y, "witness": y_sm[:, 1], "y_pred": y_pred, "dset_ind": dset_ind})

        df = df.join(meta)

        fig, ax = plt.subplots(1, 2, figsize=(8, 4))

        sns.histplot(data=df, x="witness", hue="y", ax=ax[0], kde=True, bins=20)
        sns.histplot(
            data=df, x="witness", hue="quality", multiple="dodge", ax=ax[1], kde=True, bins=20
        )

        out_hists = os.path.join(self.log_dir, "witness_hists.pdf")
        fig.savefig(out_hists)

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
            metadata = []

            # get images correponding to specific indices back from dataset
            for ele in example_idx[f"like_{p_q}"]:
                if df["y"][ele] == 1:
                    dset = dataset["p"]
                else:
                    dset = dataset["q"]

                idx = df["dset_ind"][ele]

                x, _, m, *_ = dset[idx]

                imgs.append(x)
                metadata.append(m)

            # x, _, m = dataset[example_idx[f"like_{p_q}"]]
            # print(m[4])
            print(f"m: {[ele[4] for ele in metadata]}")
            print(f'quality: {df["quality"][example_idx[f"like_{p_q}"]]}')

            img_grid = torchvision.utils.make_grid(imgs, normalize=True)
            img_grid = np.transpose(img_grid, (1, 2, 0)).numpy()
            plt.imsave(out_fig, img_grid)

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

                x = np.random.choice(sample["p"]["y_sm"][:, 1], size=sample_size, replace=True)
                x2 = np.random.choice(sample["p"]["y_sm"][:, 1], size=sample_size, replace=True)
                y = np.random.choice(sample["q"]["y_sm"][:, 1], size=sample_size, replace=True)

                res_power = permutation_test((x, y), stat_C2ST)
                res_type_1_err = permutation_test((x, x2), stat_C2ST)

                power += res_power.pvalue < 0.05
                type_1_err += res_type_1_err.pvalue < 0.05

            powers.append(power / num_reps)
            type_1_errs.append(type_1_err / num_reps)

            print(powers)
            print(type_1_errs)

        # # conf_mat = confusion_matrix(y_all.cpu().numpy(), y_bin.cpu().numpy())

        fpr, tpr, _ = roc_curve(df["y"], df["witness"])
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
        default="./config/eyepacs_quality_ood.yaml"
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
        default=[],
        nargs="+",
        help="List of splits to be evaluated, e.g. --eval_splits validation test",
    )

    args = parser.parse_args()

    params = load_config(args.config_file)

    ###############################################################################################################################
    # Preparation
    ###############################################################################################################################

    # 1. Load artifacts
    dataloader = dataset_fn(seed=args.seed, params_dict=params["dataset"])

    # extract hash from config
    domain_classifier_config = {k: params[k] for k in ["dataset", "domain_classifier"]}
    hash_string = hash_dict(domain_classifier_config)
    save_config(domain_classifier_config, args.artifacts_dir)

    domain_classifier = DomainClassifier(
        dataloader,
        args.artifacts_dir,
        hash_string,
        train_params=params["domain_classifier"]["train"],
        eval_params=params["domain_classifier"]["eval"],
    )

    domain_classifier.train()

    # EVAL:
    domain_classifier.eval(test_loader=dataloader["test"])

    task_classifier = load_or_train_task_classifier()

    ###############################################################################################################################
    # Run training
    ###############################################################################################################################

    # Creates experiment folder and places config file inside
    # (overwrites, if already there)
    # exp_name = create_exp_from_config(args.config_file, args.exp_dir)
    # log_dir = os.path.join(args.exp_dir, exp_name)

    # TODO figure out config and hash sitch again once I get to the MMD parts...

    # 2. Train shift detectors on P/Q train/val folds

    # MMD-D

    model = model_fn(seed=args.seed, params=params["model"])
    trainer = trainer_object_fn(
        model=model, dataloaders=dataloader, seed=args.seed, log_dir=log_dir, **params["trainer"]
    )

    # ODIN

    #

    trainer.train()

    ###############################################################################################################################
    # Eval MMD-D and MUKS on various sample sizes
    ###############################################################################################################################

    for split in args.eval_splits:
        eval(
            args.exp_dir,
            exp_name,
            params,
            args.seed,
            split,
            sample_sizes=[10, 30, 50, 100, 200, 500],
            num_reps=100,
            num_permutations=1000,
        )

    print("done")
