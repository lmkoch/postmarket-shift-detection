import os
from argparse import ArgumentError
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
import torch
import torch.nn as nn
import torchmetrics
import torchvision
from openTSNE import TSNE
from pytorch_lightning.trainer.supporters import CombinedLoader
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from utils.helpers import (
    consistency_analysis,
    permutation_test_c2st,
    quadratic_weighted_kappa,
    repeated_test,
)
from utils.tsne_utils import plot

from core.mmdd import assemble_loss, mmd_test
from core.muks import mass_ks_test


# TODO maybe move to utils?
def initialize_lr_scheduler(optimizer, scheduler_args):
    """define learning rate scheduler

    Args:
        optimizer (_type_): _description_
        scheduler_args (dict): _description_

    Returns:
        _type_: _description_
    """
    return torch.optim.lr_scheduler.ExponentialLR(optimizer, **scheduler_args)


# TODO maybe move to utils?
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


class DataModule(pl.LightningDataModule):
    def __init__(self, dataloader):
        self.train_loaders = dataloader["train"]
        self.val_loaders = dataloader["validation"]
        self.test_loaders = dataloader["test"]

        super().__init__()

    def train_dataloader(self):
        return CombinedLoader(self.train_loaders, mode="max_size_cycle")

    def val_dataloader(self):
        return CombinedLoader(self.val_loaders, mode="max_size_cycle")

    def test_dataloader(self):
        return CombinedLoader(self.test_loaders, mode="max_size_cycle")


class BaseClassifier(pl.LightningModule):
    def __init__(self, arch, in_channels, n_outputs, loss_type, optim_config):
        super().__init__()

        self.save_hyperparameters()

        self.n_classes = n_outputs

        if loss_type == "cross_entropy":
            self.loss_fn = torch.nn.CrossEntropyLoss()
        # elif loss_type == "mean_squared_error":
        #     self.loss_fn = torch.nn.MSELoss()
        #     # FIXME would need to do a lot of changes to architecture etc to allow for MSE loss
        else:
            raise NotImplementedError(f"loss not configured: {loss_type}")

        if arch == "resnet50":
            self.model = torchvision.models.resnet50(pretrained=True)
            self.model.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            self.model.fc = nn.Linear(self.model.fc.in_features, n_outputs)

        elif arch == "resnet18":
            self.model = torchvision.models.resnet18(pretrained=True)
            self.model.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            self.model.fc = nn.Linear(self.model.fc.in_features, n_outputs)

        elif arch == "shallow":

            self.model = ShallowNet(n_channels=in_channels, n_outputs=n_outputs)

        else:
            raise NotImplementedError(f"Model not implemented: {arch}")

        # need this for visualising penultimate layer later
        # layers = list(self.model.children())[:-1]
        # self.feature_extractor = nn.Sequential(*layers)

        self.optim_params = optim_config

        self.val_acc = torchmetrics.Accuracy()
        self.train_acc = torchmetrics.Accuracy()

    def forward(self, x):
        logits = self.model(x)
        return logits

    # def get_features(self, x):
    #     x = self.feature_extractor(x).flatten(1)
    #     return x

    def configure_optimizers(self):
        optimizer = initialize_optimizer(
            self.model,
            self.optim_params["optimizer"],
            self.optim_params["learning_rate"],
            self.optim_params["weight_decay"],
            self.optim_params["momentum"],
            self.optim_params["nesterov"],
        )

        if self.optim_params["lr_scheduler"]:
            scheduler = initialize_lr_scheduler(optimizer, {"gamma": 0.6})
            return [optimizer], [scheduler]

        else:
            return optimizer

    def training_step(self, batch, batch_idx) -> None:

        x, y = batch
        y_pred = self.model(x)
        loss = self.loss_fn(y_pred, y)
        self.train_acc(y_pred, y)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log(
            "train/acc",
            self.train_acc,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        y_pred = torch.argmax(y_pred, dim=1)

        return {"loss": loss, "y": y, "y_pred": y_pred}

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        y_pred = self.model(x)
        loss = self.loss_fn(y_pred, y)
        self.val_acc(y_pred, y)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        y_pred = torch.argmax(y_pred, dim=1)

        return {"loss": loss, "y": y, "y_pred": y_pred}


class MaxKernel(BaseClassifier):
    def __init__(
        self,
        in_channels,
        img_size,
        optim_config,
        loss_type,
        loss_lambda=10**-6,
        feature_extractor="liu",
    ):

        # def __init__(self, arch, in_channels, n_outputs, loss_type, optim_config):
        #     super().__init__()

        super(BaseClassifier, self).__init__()

        self.save_hyperparameters()

        self.loss_fn = assemble_loss(loss_type=loss_type, loss_lambda=loss_lambda)

        if feature_extractor == "liu":
            self.model = Featurizer(n_channels=in_channels, img_size=img_size)
        elif feature_extractor == "resnet50":
            self.model = torchvision.models.resnet50(pretrained=True)
            self.model.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            self.model.fc = nn.Linear(self.model.fc.in_features, 128)
        else:
            raise ValueError(f"Feature extractor not configured: {feature_extractor}")

        # Other parameters
        self.epsilonOPT = nn.Parameter(torch.log(torch.rand(1) * 10 ** (-10)))
        self.sigmaOPT = nn.Parameter(torch.ones(1) * np.sqrt(2 * 32 * 32))
        self.sigma0OPT = nn.Parameter(torch.ones(1) * np.sqrt(0.005))

        self.optim_params = optim_config

    def get_features(self, x):
        x = self.model(x).flatten(1)
        return x

    def forward(self, x):
        logits = self.model(x)
        return logits

    def training_step(self, batch, batch_idx) -> None:

        x_p, *_ = batch["p"]
        x_q, *_ = batch["q"]

        feat_p = self.model(x_p)
        feat_q = self.model(x_q)

        ep = self.ep
        sigma = self.sigma_sq
        sigma0_u = self.sigma0_sq

        loss = self.loss_fn(feat_p, feat_q, x_p, x_q, sigma, sigma0_u, ep)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx) -> None:

        outputs = self._shared_test_step(batch, batch_idx, num_permutations=100)

        loss = outputs["loss"]

        x_p, *_ = batch["p"]
        x_q, *_ = batch["q"]

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        img_p = torchvision.utils.make_grid(x_p[:8], normalize=True)
        img_q = torchvision.utils.make_grid(x_q[:8], normalize=True)

        self.logger.experiment.add_image("val/img_p", img_p, self.trainer.global_step)
        self.logger.experiment.add_image("val/img_q", img_q, self.trainer.global_step)

        return outputs

    def test_step(self, batch, batch_idx) -> None:

        outputs = self._shared_test_step(batch, batch_idx, num_permutations=1000)

        return outputs

    def _shared_test_step(self, batch, batch_idx, num_permutations=100) -> None:

        x_p, *_ = batch["p"]
        x_q, *_ = batch["q"]

        feat_p = self.model(x_p)
        feat_q = self.model(x_q)

        ep = self.ep
        sigma = self.sigma_sq
        sigma0_u = self.sigma0_sq

        alpha = 0.05

        h_u = mmd_test(
            x_p,
            x_q,
            self.model,
            ep,
            sigma,
            sigma0_u,
            num_permutations=num_permutations,
            alpha=alpha,
        )

        loss = self.loss_fn(feat_p, feat_q, x_p, x_q, sigma, sigma0_u, ep)

        outputs = {"loss": loss, "hypothesis_rejected": h_u}

        # feature extraction for t-sne
        outputs["feat_p"] = self.get_features(x_p)
        outputs["feat_q"] = self.get_features(x_q)

        return outputs

    def validation_epoch_end(self, outputs):

        power = self._shared_epoch_end(outputs, "val")
        self.log("val/power", power, on_epoch=True, prog_bar=True, logger=True)

    def test_epoch_end(self, outputs):

        power = self._shared_epoch_end(outputs, "test")
        self.log("test/power", power, on_epoch=True, prog_bar=True, logger=True)

    def _shared_epoch_end(self, outputs, split):
        loss = torch.stack([x["loss"] for x in outputs])
        hypothesis_rejected = np.stack([x["hypothesis_rejected"] for x in outputs])

        loss = torch.mean(loss)

        if torch.isnan(loss):
            power = np.nan
        else:
            power = np.mean(hypothesis_rejected)

        log_dir = self.logger.log_dir

        # TODO stack feature reps into one array
        feat_p = torch.cat([x["feat_p"] for x in outputs]).detach().cpu().numpy()
        feat_q = torch.cat([x["feat_q"] for x in outputs]).detach().cpu().numpy()

        out_arr_p = os.path.join(log_dir, f"{split}_feat_p.npy")
        with open(out_arr_p, "wb") as f:
            np.save(f, feat_p)

        feat = np.concatenate((feat_p, feat_q))
        out_arr_p = os.path.join(log_dir, f"{split}_feat.npy")
        with open(out_arr_p, "wb") as f:
            np.save(f, feat)

        # make labels (p, q)

        # labels_p = np.ones(feat_p.shape[0])
        # labels_q = np.zeros(feat_q.shape[0])
        # y = np.concatenate([labels_p, labels_q], 0).squeeze()

        labels_p = ["P"] * feat_p.shape[0]
        labels_q = ["Q"] * feat_q.shape[0]
        y = labels_p + labels_q

        # # TODO t-sne
        # from openTSNE import TSNE
        # from utils.tsne_utils import plot

        # # TODO useful tsne parameters
        # tsne = TSNE()
        # embedding = tsne.fit(feat)

        # # TODO meaningful labels (P, Q)
        # fig, ax = plot(embedding, y)

        # self.logger.experiment.add_figure(f"{split}/tsne", fig, self.trainer.global_step)

        # out_fig = os.path.join(log_dir, f"{split}_tsne.pdf")

        # fig.savefig(out_fig)

        return power

    @property
    def ep(self):
        return torch.exp(self.epsilonOPT) / (1 + torch.exp(self.epsilonOPT))

    @property
    def sigma_sq(self):
        return self.sigmaOPT**2

    @property
    def sigma0_sq(self):
        return self.sigma0OPT**2


class TaskClassifier(BaseClassifier):
    def __init__(self, *args, **kwargs):

        self._test_sample_sizes = [10, 30, 50, 100, 500]
        super().__init__(*args, **kwargs)

    def training_step(self, batch, batch_idx) -> None:

        # for the case batch consists of more than x, y (e.g. x, y, m), pass only x, y
        return super().training_step(batch["p"][:2], batch_idx)

    def validation_step(self, batch, batch_idx):

        outputs = self._shared_test_step(batch, batch_idx)

        x_p, *_ = batch["p"]
        x_q, *_ = batch["q"]

        img_p = torchvision.utils.make_grid(x_p[:8], normalize=True)
        img_q = torchvision.utils.make_grid(x_q[:8], normalize=True)

        self.logger.experiment.add_image("val/img_p", img_p, self.trainer.global_step)
        self.logger.experiment.add_image("val/img_q", img_q, self.trainer.global_step)

        return outputs

    def test_step(self, batch, batch_idx) -> None:

        outputs = self._shared_test_step(batch, batch_idx)
        return outputs

    # def training_epoch_end(self, outputs):
    #     self._shared_epoch_end(outputs, "train")

    def validation_epoch_end(self, outputs):

        y_p_sm, y_q_sm = self._shared_epoch_end(outputs, "val")

        # TODO here default replace=True is used, should this be False?
        # log power for sample size=100
        power, type_1_err = repeated_test(
            y_p_sm, y_q_sm, mass_ks_test, num_reps=100, alpha=0.05, sample_size=100, replace=False
        )

        self.log(f"val/power", power, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"val/type_1err", type_1_err, on_epoch=True, prog_bar=True, logger=True)

    @property
    def test_sample_sizes(self):
        return self._test_sample_sizes

    @test_sample_sizes.setter
    def test_sample_sizes(self, sample_sizes):
        self._test_sample_sizes = sample_sizes

    def test_epoch_end(self, outputs):

        y_p_sm, y_q_sm = self._shared_epoch_end(outputs, "test")

        out_csv = os.path.join(self.logger.log_dir, f"test_consistency_analysis.csv")

        consistency_analysis(y_p_sm, y_q_sm, mass_ks_test, self.test_sample_sizes, out_csv)

    def _shared_epoch_end(self, outputs, split):

        y_p_sm = torch.cat([x["y_p_sm"] for x in outputs]).detach().cpu().numpy()
        y_q_sm = torch.cat([x["y_q_sm"] for x in outputs]).detach().cpu().numpy()

        self._shared_subgroup_analysis(outputs, split)

        log_dir = self.logger.log_dir

        # # TODO stack feature reps into one array
        # feat_p = torch.cat([x["feat_p"] for x in outputs]).detach().cpu().numpy()
        # feat_q = torch.cat([x["feat_q"] for x in outputs]).detach().cpu().numpy()

        # out_arr_p = os.path.join(log_dir, f"{split}_feat_p.npy")
        # with open(out_arr_p, "wb") as f:
        #     np.save(f, feat_p)

        # feat = np.concatenate((feat_p, feat_q))
        # out_arr_p = os.path.join(log_dir, f"{split}_feat.npy")
        # with open(out_arr_p, "wb") as f:
        #     np.save(f, feat)

        # # make labels (p, q)

        # # labels_p = np.ones(feat_p.shape[0])
        # # labels_q = np.zeros(feat_q.shape[0])
        # # y = np.concatenate([labels_p, labels_q], 0).squeeze()

        # labels_p = ["P"] * feat_p.shape[0]
        # labels_q = ["Q"] * feat_q.shape[0]
        # y = labels_p + labels_q

        # TODO t-sne

        # # TODO useful tsne parameters
        # tsne = TSNE()
        # embedding = tsne.fit(feat)

        # # TODO meaningful labels (P, Q)
        # fig, ax = plot(embedding, y)

        # self.logger.experiment.add_figure(f"{split}/tsne", fig, self.trainer.global_step)

        # out_fig = os.path.join(log_dir, f"{split}_tsne.pdf")

        # fig.savefig(out_fig)

        return y_p_sm, y_q_sm

    def _shared_test_step(self, batch, batch_idx):

        # Task accuracy

        outputs = super().validation_step(batch["p"][:2], batch_idx)

        # TODO: subgroup analysis
        #       Need to access meta information, which is dataset dependent.. child classes?

        x_p, *_ = batch["p"]
        x_q, *_ = batch["q"]

        sm = nn.Softmax(dim=1)

        y_p_sm = sm(self.model(x_p))
        y_q_sm = sm(self.model(x_q))

        outputs["y_p_sm"] = y_p_sm
        outputs["y_q_sm"] = y_q_sm

        outputs["m_p"] = batch["p"][2]

        # feature extraction for t-sne
        # outputs["feat_p"] = self.get_features(x_p)
        # outputs["feat_q"] = self.get_features(x_q)

        return outputs

    def _shared_subgroup_analysis(self, outputs, split):

        y = torch.cat([x["y"] for x in outputs]).detach().cpu().numpy()
        y_pred = torch.cat([x["y_pred"] for x in outputs]).detach().cpu().numpy()
        metadata = torch.cat([x["m_p"] for x in outputs]).detach().cpu().numpy()

        df = pd.DataFrame()
        metrics_df = pd.DataFrame()

        cols = ["y", "y_pred"] + [val for val in range(metadata.shape[1])]

        matrix = np.concatenate((y[:, np.newaxis], y_pred[:, np.newaxis], metadata), axis=1)
        df = pd.DataFrame(data=matrix, columns=cols)

        # whole dataset
        group_specs = {"criterion": "all", "group": None, "n": len(df)}
        results = self._performance_metrics(df)
        metrics_df = metrics_df.append(group_specs | results, ignore_index=True)

        # subgroups
        for col_idx in range(metadata.shape[1]):

            column = metadata[:, col_idx]

            for group in np.unique(column):

                subset = df[df[col_idx] == group]

                group_specs = {"criterion": col_idx, "group": group, "n": len(subset)}

                print("========================================")
                print(f"Group: {col_idx}={group} (n={len(subset)})")

                results = self._performance_metrics(subset)
                print("========================================")

                metrics_df = metrics_df.append(group_specs | results, ignore_index=True)

        log_dir = self.logger.log_dir
        out_csv = os.path.join(
            log_dir, f"{split}_subgroup_performance_epoch:{self.current_epoch}.csv"
        )

        metrics_df.to_csv(out_csv)

    def _performance_metrics(self, subset):

        conf_mat = confusion_matrix(subset["y"], subset["y_pred"], labels=range(self.n_classes))

        results = {
            "acc": accuracy_score(subset["y"], subset["y_pred"]),
            "bal_acc": balanced_accuracy_score(subset["y"], subset["y_pred"]),
            "binary_acc_onset_1": accuracy_score(subset["y"] >= 1.0, subset["y_pred"] >= 1.0),
            "binary_acc_onset_2": accuracy_score(subset["y"] >= 2.0, subset["y_pred"] >= 2.0),
            "quadratic_kappa": quadratic_weighted_kappa(conf_mat),
        }

        print(f"Metrics:")
        print(results)
        print("Confusion Matrix:")
        print(conf_mat)

        return results


class EyepacsClassifier(TaskClassifier):
    def training_epoch_end(self, outputs):

        super().training_epoch_end(outputs)

        # TODO (for train and val end epoch): acc, bal_acc, kappa
        # TODO: for eyepacs only: binary onset acc

        y = torch.cat([x["y"] for x in outputs]).detach().cpu().numpy()
        y_pred = torch.cat([x["y_pred"] for x in outputs]).detach().cpu().numpy()

        conf_mat = confusion_matrix(y, y_pred, labels=range(5))

        self.log(
            "train/bal_acc",
            balanced_accuracy_score(y, y_pred),
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        self.log(
            "train/bin_acc_onset_1",
            accuracy_score(y >= 1.0, y_pred >= 1.0),
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        self.log(
            "train/bin_acc_onset_2",
            accuracy_score(y >= 2.0, y_pred >= 2.0),
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        self.log(
            "train/kappa",
            quadratic_weighted_kappa(conf_mat),
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

    def validation_epoch_end(self, outputs):

        super().validation_epoch_end(outputs)

        y = torch.cat([x["y"] for x in outputs]).detach().cpu().numpy()
        y_pred = torch.cat([x["y_pred"] for x in outputs]).detach().cpu().numpy()

        conf_mat = confusion_matrix(y, y_pred, labels=range(5))

        self.log(
            "val/bal_acc",
            balanced_accuracy_score(y, y_pred),
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        self.log(
            "val/bin_acc_onset_1",
            accuracy_score(y >= 1.0, y_pred >= 1.0),
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        self.log(
            "val/bin_acc_onset_2",
            accuracy_score(y >= 2.0, y_pred >= 2.0),
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        self.log(
            "val/kappa",
            quadratic_weighted_kappa(conf_mat),
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )


# TODO debug - witness hists should have correct colors depending on number of chnnels


class DomainClassifier(BaseClassifier):
    def prepare_batch_data(self, x_p: torch.Tensor, x_q: torch.Tensor):

        labels_p = torch.ones((x_p.shape[0], 1))
        labels_q = torch.zeros((x_q.shape[0], 1))

        x = torch.cat([x_p, x_q], 0)
        y = torch.cat([labels_p, labels_q], 0).squeeze().long().to(self.device)

        return x, y

    def training_step(self, batch, batch_idx) -> None:

        # print(f"Training batch size: {batch['p'][0].shape}")

        prepped_batch = self.prepare_batch_data(batch["p"][0], batch["q"][0])

        return super().training_step(prepped_batch, batch_idx)

    def validation_step(self, batch, batch_idx) -> None:

        print(f"Validation batch size: {batch['p'][0].shape}")

        outputs = self._shared_test_step(batch, batch_idx)

        self.val_x_p, *_ = batch["p"]
        self.val_x_q, *_ = batch["q"]

        # img_p = torchvision.utils.make_grid(x_p[:8], normalize=True)
        # img_q = torchvision.utils.make_grid(x_q[:8], normalize=True)

        # self.logger.experiment.add_image("val/img_p", img_p, self.trainer.global_step)
        # self.logger.experiment.add_image("val/img_q", img_q, self.trainer.global_step)

        return outputs

    def validation_epoch_end(self, outputs):

        sample_p, sample_q, witness_fig = self._shared_epoch_end(outputs, "val")

        # TODO report both binary and logits result

        # TODO why is this only size 32?? 2 batches??
        print(sample_p.shape)
        print(sample_q.shape)

        power, type_1_err = repeated_test(
            sample_p,
            sample_q,
            permutation_test_c2st,
            num_reps=100,
            alpha=0.05,
            sample_size=100,
            replace=False,  # this requires replacement in dataloder!!!
        )

        self.log("val/power", power, on_epoch=True, prog_bar=True, logger=True)
        self.log("val/type_1err", type_1_err, on_epoch=True, prog_bar=True, logger=True)

        # TODO change in other classes as well: plot images only once per epoch!
        img_p = torchvision.utils.make_grid(self.val_x_p[:8], normalize=True)
        img_q = torchvision.utils.make_grid(self.val_x_q[:8], normalize=True)

        self.logger.experiment.add_image("val/img_p", img_p, self.trainer.global_step)
        self.logger.experiment.add_image("val/img_q", img_q, self.trainer.global_step)

        self.logger.experiment.add_figure(
            "val/witness_examples", witness_fig, self.trainer.global_step
        )

        log_dir = self.logger.log_dir
        out_fig = os.path.join(log_dir, f"val_witness_examples.pdf")

        witness_fig.savefig(out_fig)

    def test_step(self, batch, batch_idx) -> None:

        outputs = self._shared_test_step(batch, batch_idx)
        return outputs

    def test_epoch_end(self, outputs):

        sample_p, sample_q, witness_fig = self._shared_epoch_end(outputs, "test")

        self.logger.experiment.add_figure(
            "test/witness_examples", witness_fig, self.trainer.global_step
        )

        log_dir = self.logger.log_dir
        out_fig = os.path.join(log_dir, f"test_witness_examples.pdf")

        witness_fig.savefig(out_fig)

        # power analysis
        out_csv = os.path.join(self.logger.log_dir, f"test_consistency_analysis.csv")

        # TODO report both binary and logits result

        consistency_analysis(
            sample_p, sample_q, permutation_test_c2st, self.test_sample_sizes, out_csv
        )

    def _shared_test_step(self, batch, batch_idx) -> None:

        x_p = batch["p"][0]
        x_q = batch["q"][0]

        prepped_batch = self.prepare_batch_data(x_p, x_q)

        x, y = prepped_batch
        y_logits = self.model(x)

        # Remember validation images for visualisation later
        self.val_x = x
        self.y_logits = y_logits

        outputs = super().validation_step(prepped_batch, batch_idx)

        outputs["y"] = y
        outputs["y_logits"] = y_logits

        # # for t-sne visualisation
        # outputs["feat_p"] = self.get_features(x_p)
        # outputs["feat_q"] = self.get_features(x_q)

        return outputs

    def _shared_epoch_end(self, outputs, split):

        batch_size = outputs[0]["y"].shape[0]

        y = torch.cat([x["y"] for x in outputs]).detach().cpu().numpy()
        y_logits = torch.cat([x["y_logits"] for x in outputs])
        y_pred = torch.argmax(y_logits, dim=1)
        y_sm = torch.softmax(y_logits, dim=1)

        # witness is difference in logits for soft C2ST
        witness = (y_logits[:, 0] - y_logits[:, 1]).detach().cpu().numpy()

        sample_p = witness[y == 1]
        sample_q = witness[y == 0]

        df = pd.DataFrame({"witness": witness, "y": y})
        fig, ax = plt.subplots()
        sns.stripplot(data=df, x="witness", y="y", orient="h")

        indices = np.random.choice(batch_size // 2, 4)

        indices = np.concatenate((indices, indices + batch_size // 2))

        # normalise
        self.val_x -= self.val_x.min()
        self.val_x /= self.val_x.max()

        for idx in indices:

            img = np.squeeze(self.val_x[idx].permute(1, 2, 0).detach().cpu().numpy())
            example_witness = (
                (self.y_logits[:, 0] - self.y_logits[:, 1]).detach().cpu().numpy()[idx]
            )
            example_y = idx < batch_size / 2

            example_coord = [example_witness, example_y]

            coord_img = ax.transData.transform(example_coord)

            figimg = ax.figure.figimage(img, *coord_img)

        log_dir = self.logger.log_dir

        # # TODO stack feature reps into one array
        # feat_p = torch.cat([x["feat_p"] for x in outputs]).detach().cpu().numpy()
        # feat_q = torch.cat([x["feat_q"] for x in outputs]).detach().cpu().numpy()

        # out_arr_p = os.path.join(log_dir, f"{split}_feat_p.npy")
        # with open(out_arr_p, "wb") as f:
        #     np.save(f, feat_p)

        # feat = np.concatenate((feat_p, feat_q))
        # out_arr_p = os.path.join(log_dir, f"{split}_feat.npy")
        # with open(out_arr_p, "wb") as f:
        #     np.save(f, feat)

        # # make labels (p, q)

        # # labels_p = np.ones(feat_p.shape[0])
        # # labels_q = np.zeros(feat_q.shape[0])
        # # y = np.concatenate([labels_p, labels_q], 0).squeeze()

        # labels_p = ["P"] * feat_p.shape[0]
        # labels_q = ["Q"] * feat_q.shape[0]
        # y = labels_p + labels_q

        # # TODO t-sne

        # # # TODO useful tsne parameters
        # # tsne = TSNE()
        # # embedding = tsne.fit(feat)

        # # fig_tsne, ax = plot(embedding, y)

        # # self.logger.experiment.add_figure(f"{split}/tsne", fig_tsne, self.trainer.global_step)

        # # out_fig = os.path.join(log_dir, f"{split}_tsne.pdf")

        # # fig_tsne.savefig(out_fig)

        return sample_p, sample_q, fig


class ShallowNet(nn.Module):
    def __init__(self, n_channels=1, n_outputs=128):
        super(ShallowNet, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=False):
            block = [
                nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0),
            ]  # 0.25
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(n_channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        self.adv_layer = nn.Sequential(nn.Conv2d(128, n_outputs, 1), nn.AdaptiveAvgPool2d((1, 1)))

    def forward(self, img):
        out = self.model(img)
        out = self.adv_layer(out)
        logits = out.view(out.shape[0], -1)

        print(logits.shape)

        return logits


# Define the deep network for MMD-D
class FeaturizerLiu(nn.Module):
    def __init__(self, n_channels=1, img_size=32):
        super(FeaturizerLiu, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=False):
            block = [
                nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0),
            ]  # 0.25
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
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size**2, 128))

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        feature = self.adv_layer(out)

        return feature


class Featurizer(FeaturizerLiu):
    """Featurizer class used for old code (including ep etc member vars)

    Args:
        FeaturizerLiu (_type_): _description_
    """

    def __init__(self, n_channels=1, img_size=32):
        super(Featurizer, self).__init__(n_channels=n_channels, img_size=img_size)

        # Other parameters
        self.epsilonOPT = nn.Parameter(torch.log(torch.rand(1) * 10 ** (-10)))
        self.sigmaOPT = nn.Parameter(torch.ones(1) * np.sqrt(2 * 32 * 32))
        self.sigma0OPT = nn.Parameter(torch.ones(1) * np.sqrt(0.005))

    @property
    def ep(self):
        return torch.exp(self.epsilonOPT) / (1 + torch.exp(self.epsilonOPT))

    @property
    def sigma_sq(self):
        return self.sigmaOPT**2

    @property
    def sigma0_sq(self):
        return self.sigma0OPT**2
