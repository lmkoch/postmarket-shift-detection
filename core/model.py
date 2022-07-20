import logging
from argparse import ArgumentError
from typing import Dict

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
import torchvision
from pytorch_lightning.trainer.supporters import CombinedLoader
from scipy.stats import permutation_test
from utils.helpers import stat_C2ST

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
        return CombinedLoader(self.train_loaders)

    def val_dataloader(self):
        return CombinedLoader(self.val_loaders)

    def test_dataloader(self):
        return CombinedLoader(self.test_loaders)


class BaseClassifier(pl.LightningModule):
    def __init__(self, arch, in_channels, n_outputs, optim_config):
        super().__init__()

        self.loss_fn = torch.nn.CrossEntropyLoss()

        if arch == "resnet50":
            self.model = torchvision.models.resnet50(pretrained=True)
            self.model.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            self.model.fc = nn.Linear(self.model.fc.in_features, n_outputs)

        else:
            raise NotImplementedError(f"Model not implemented: {arch}")

        self.optim_params = optim_config

        self.val_acc = torchmetrics.Accuracy()
        self.train_acc = torchmetrics.Accuracy()

    def forward(self, x):
        logits = self.model(x)
        return logits

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

        return {"loss": loss}

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        y_pred = self.model(x)
        loss = self.loss_fn(y_pred, y)
        self.val_acc(y_pred, y)

        self.log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val/acc", self.val_acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return {"loss": loss}


class MaxKernel(BaseClassifier):
    def __init__(self, in_channels, img_size, optim_config, loss_type, loss_lambda=10**-6):
        super(BaseClassifier, self).__init__()

        self.loss_fn = assemble_loss(loss_type=loss_type, loss_lambda=loss_lambda)

        model = Featurizer(n_channels=in_channels, img_size=img_size)

        self.model = model

        self.optim_params = optim_config

    def forward(self, x):
        logits = self.model(x)
        return logits

    def training_step(self, batch, batch_idx) -> None:

        x_p, *_ = batch["p"]
        x_q, *_ = batch["q"]

        feat_p = self.model(x_p)
        feat_q = self.model(x_q)

        ep = self.model.ep
        sigma = self.model.sigma_sq
        sigma0_u = self.model.sigma0_sq

        loss = self.loss_fn(feat_p, feat_q, x_p, x_q, sigma, sigma0_u, ep)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx) -> None:

        x_p, *_ = batch["p"]
        x_q, *_ = batch["q"]

        feat_p = self.model(x_p)
        feat_q = self.model(x_q)

        ep = self.model.ep
        sigma = self.model.sigma_sq
        sigma0_u = self.model.sigma0_sq

        loss = self.loss_fn(feat_p, feat_q, x_p, x_q, sigma, sigma0_u, ep)

        self.log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        alpha = 0.05
        num_permutations = 100

        h_u = mmd_test(
            x_p,
            x_q,
            self.model,
            num_permutations=num_permutations,
            alpha=alpha,
        )

        img_p = torchvision.utils.make_grid(x_p[:8], normalize=True)
        img_q = torchvision.utils.make_grid(x_q[:8], normalize=True)

        self.logger.experiment.add_image("val/img_p", img_p, self.trainer.global_step)
        self.logger.experiment.add_image("val/img_q", img_q, self.trainer.global_step)

        return {"loss": loss, "hypothesis_rejected": h_u}

    def validation_epoch_end(self, outputs):

        loss = torch.stack([x["loss"] for x in outputs])
        hypothesis_rejected = np.stack([x["hypothesis_rejected"] for x in outputs])

        # TODO figure out how to calculate power for specific sample size..

        loss = torch.mean(loss)

        if torch.isnan(loss):
            power = np.nan
        else:
            power = np.mean(hypothesis_rejected)

        self.log("val/power", power, on_epoch=True, prog_bar=True, logger=True)


def repeated_test(x_pop, y_pop, test_fun, num_reps=100, alpha=0.05, sample_size=100):

    power = 0
    type_1_err = 0

    for _ in range(num_reps):

        x = x_pop[np.random.choice(x_pop.shape[0], size=sample_size, replace=True)]
        x2 = x_pop[np.random.choice(x_pop.shape[0], size=sample_size, replace=True)]
        y = y_pop[np.random.choice(y_pop.shape[0], size=sample_size, replace=True)]

        power += test_fun(x, y) < alpha
        type_1_err += test_fun(x, x2) < alpha

    power = power / num_reps
    type_1_err = type_1_err / num_reps

    return power, type_1_err


class TaskClassifier(BaseClassifier):
    def training_step(self, batch, batch_idx) -> None:

        # for the case batch consists of more than x, y (e.g. x, y, m), pass only x, y
        return super().training_step(batch["p"][:2], batch_idx)

    def validation_step(self, batch, batch_idx):

        # Task accuracy

        outputs = super().validation_step(batch["p"][:2], batch_idx)

        # TODO: subgroup analysis
        #       Need to access meta information, which is dataset dependent.. child classes?

        x_p, *_ = batch["p"]
        x_q, *_ = batch["q"]

        img_p = torchvision.utils.make_grid(x_p[:8], normalize=True)
        img_q = torchvision.utils.make_grid(x_q[:8], normalize=True)

        self.logger.experiment.add_image("val/img_p", img_p, self.trainer.global_step)
        self.logger.experiment.add_image("val/img_q", img_q, self.trainer.global_step)

        sm = nn.Softmax(dim=1)

        y_p_sm = sm(self.model(x_p))
        y_q_sm = sm(self.model(x_q))

        outputs["y_p_sm"] = y_p_sm
        outputs["y_q_sm"] = y_q_sm

        return outputs

    def validation_epoch_end(self, outputs):

        y_p_sm = torch.cat([x["y_p_sm"] for x in outputs]).detach().cpu().numpy()
        y_q_sm = torch.cat([x["y_q_sm"] for x in outputs]).detach().cpu().numpy()

        # TODO: check if correct: draw multi-D sample (multiple classes)

        power, type_1_err = repeated_test(
            y_p_sm, y_q_sm, mass_ks_test, num_reps=10, alpha=0.05, sample_size=100
        )

        self.log("val/power", power, on_epoch=True, prog_bar=True, logger=True)
        self.log("val/type_1err", type_1_err, on_epoch=True, prog_bar=True, logger=True)


class DomainClassifier(BaseClassifier):
    def prepare_batch_data(self, x_p: torch.Tensor, x_q: torch.Tensor):

        labels_p = torch.ones((x_p.shape[0], 1))
        labels_q = torch.zeros((x_q.shape[0], 1))

        x = torch.cat([x_p, x_q], 0)
        y = torch.cat([labels_p, labels_q], 0).squeeze().long().to(self.device)

        return x, y

    def training_step(self, batch, batch_idx) -> None:

        prepped_batch = self.prepare_batch_data(batch["p"][0], batch["q"][0])

        return super().training_step(prepped_batch, batch_idx)

    def validation_step(self, batch, batch_idx) -> None:

        prepped_batch = self.prepare_batch_data(batch["p"][0], batch["q"][0])

        x, y = prepped_batch
        y_logits = self.model(x)

        # Remember validation images for visualisation later
        self.val_y = y
        self.y_logits = y_logits

        outputs = super().validation_step(prepped_batch, batch_idx)

        x_p, *_ = batch["p"]
        x_q, *_ = batch["q"]

        img_p = torchvision.utils.make_grid(x_p[:8], normalize=True)
        img_q = torchvision.utils.make_grid(x_q[:8], normalize=True)

        self.logger.experiment.add_image("val/img_p", img_p, self.trainer.global_step)
        self.logger.experiment.add_image("val/img_q", img_q, self.trainer.global_step)

        outputs["y"] = y
        outputs["y_logits"] = y_logits

        return outputs

    def validation_epoch_end(self, outputs):

        y = torch.cat([x["y"] for x in outputs])
        y_logits = torch.cat([x["y_logits"] for x in outputs])
        y_pred = torch.argmax(y_logits, dim=1)
        y_sm = torch.softmax(y_logits, dim=1)

        sample_p = y_sm[y == 1, 1].detach().cpu().numpy()
        sample_q = y_sm[y == 0, 1].detach().cpu().numpy()

        def permutation_test_wrapper(x, y):
            res = permutation_test((x, y), stat_C2ST)
            return res.pvalue

        power, type_1_err = repeated_test(
            sample_p, sample_q, permutation_test_wrapper, num_reps=10, alpha=0.05, sample_size=100
        )

        self.log("val/power", power, on_epoch=True, prog_bar=True, logger=True)
        self.log("val/type_1err", type_1_err, on_epoch=True, prog_bar=True, logger=True)


# Define the deep network for MMD-D
class Featurizer(nn.Module):
    def __init__(self, n_channels=1, img_size=32):
        super(Featurizer, self).__init__()

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

        # Other parameters
        self.epsilonOPT = nn.Parameter(torch.log(torch.rand(1) * 10 ** (-10)))
        self.sigmaOPT = nn.Parameter(torch.ones(1) * np.sqrt(2 * 32 * 32))
        self.sigma0OPT = nn.Parameter(torch.ones(1) * np.sqrt(0.005))

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        feature = self.adv_layer(out)

        return feature

    @property
    def ep(self):
        return torch.exp(self.epsilonOPT) / (1 + torch.exp(self.epsilonOPT))

    @property
    def sigma_sq(self):
        return self.sigmaOPT**2

    @property
    def sigma0_sq(self):
        return self.sigma0OPT**2
