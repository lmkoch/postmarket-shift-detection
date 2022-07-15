import logging
import os
from argparse import ArgumentError
from typing import Dict

import gdown
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from scipy.stats import permutation_test
from torch.nn.functional import one_hot
from utils.helpers import stat_C2ST
from wilds.datasets.download_utils import extract_archive


def get_classification_model(params, log_dir=None, download=True):

    checkpoint_path = params["task_classifier_path"]

    if log_dir is not None:
        checkpoint_path = os.path.join(log_dir, "model.pt")

    # FIXME make sure in the end, all trained models are loadable. Consistency issue with self and self.model

    if params["task_classifier_type"] == "mnist":
        model = MNISTNet(
            checkpoint_path=checkpoint_path, n_outputs=params["n_outputs"], download=download
        )
    elif params["task_classifier_type"] == "camelyon":
        # TODO: remove_idx config for camelyon as well
        outer_model = CamelyonDensenet(checkpoint_path=checkpoint_path)
        model = outer_model.model
    else:
        raise NotImplementedError

    return model


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


import pytorch_lightning as pl
import torchmetrics
import torchvision
from pytorch_lightning.trainer.supporters import CombinedLoader


class DataModule(pl.LightningDataModule):
    def __init__(self, dataloader, domains=["p", "q"]):
        self.train_loaders = {domain: dataloader["train"][domain] for domain in domains}
        self.val_loaders = {domain: dataloader["validation"][domain] for domain in domains}
        self.test_loaders = {domain: dataloader["test"][domain] for domain in domains}

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

        img_grid = torchvision.utils.make_grid(x[:8], normalize=True)

        self.logger.experiment.add_image("val/img", img_grid, self.trainer.global_step)

        return {"loss": loss}


class TaskClassifier(BaseClassifier):
    def training_step(self, batch, batch_idx) -> None:
        return super().training_step(batch["p"], batch_idx)

    def validation_step(self, batch, batch_idx) -> None:
        return super().validation_step(batch["p"], batch_idx)


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

        outputs["y"] = y
        outputs["y_logits"] = y_logits

        return outputs

    def validation_epoch_end(self, outputs):

        # Log outputs
        # losses = [x["loss"] for x in outputs]
        y = torch.cat([x["y"] for x in outputs])
        y_logits = torch.cat([x["y_logits"] for x in outputs])
        # y = torch.stack(outputs["y"])
        # y_logits = torch.stack(outputs["y_logits"])
        # y, y_logits = self.val_y, self.y_logits
        y_pred = torch.argmax(y_logits, dim=1)
        y_sm = torch.softmax(y_logits, dim=1)

        sample_p = y_sm[y == 1, 1].detach().cpu().numpy()
        sample_q = y_sm[y == 0, 1].detach().cpu().numpy()

        power = 0
        type_1_err = 0

        # TODO all of below configurable
        num_reps = 10
        alpha = 0.05
        sample_size = 100

        for _ in range(num_reps):

            x = np.random.choice(sample_p, size=sample_size, replace=True)
            x2 = np.random.choice(sample_p, size=sample_size, replace=True)
            y = np.random.choice(sample_q, size=sample_size, replace=True)

            res_power = permutation_test((x, y), stat_C2ST)
            res_type_1_err = permutation_test((x, x2), stat_C2ST)

            power += res_power.pvalue < alpha
            type_1_err += res_type_1_err.pvalue < alpha

        power = power / num_reps
        type_1_err = type_1_err / num_reps

        self.log("val/power", power, on_epoch=True, prog_bar=True, logger=True)
        self.log("val/type_1err", type_1_err, on_epoch=True, prog_bar=True, logger=True)


class CamelyonDensenet(nn.Module):
    """This is a wrapper around densenet to allow loading trained
    Camelyon ERM models.
    """

    def __init__(self, checkpoint_path=None, download=True):
        super().__init__()
        self.model = models.densenet121(num_classes=2)

        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        out_dir = os.path.dirname(checkpoint_path)
        os.makedirs(out_dir, exist_ok=True)

        self.logger = logging.getLogger("camelyon_classification")
        self.logger.addHandler(logging.FileHandler(os.path.join(out_dir, "camelyon.log")))
        self.logger.setLevel(logging.INFO)

        self.checkpoint_path = checkpoint_path

        trained_models_url = "https://drive.google.com/uc?id=1fYmNcgvm91YnMRX4isGAXVors9I17oWD"
        archive = os.path.join(out_dir, "archive.tar.gz")

        if os.path.exists(checkpoint_path):
            self.load_checkpoint()
        else:
            if download:
                gdown.download(trained_models_url, archive, quiet=False)

                self.logger.info("Extracting {} to {}".format(archive, out_dir))
                extract_archive(archive, out_dir, True)

                self.load_checkpoint()
            else:
                self.logger.warning("Dataset does not exist. Download it or train the model")

    def load_checkpoint(self):

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.load_state_dict(checkpoint["algorithm"])

        self.model = self.model.to(self.device)

        return self.model


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


def model_fn(seed: int, params: Dict) -> torch.nn.Module:
    """
    Builds a model object for the given config
    Args:
        data_loaders: a dictionary of data loaders
        seed: random seed (e.g. for model initialization)
    Returns:
        Instance of torch.nn.Module
    """

    required_arguments = ["channels", "img_size"]
    for ele in required_arguments:
        assert ele in params

    # just for safety: remove any potential unexpected items
    params = {k: v for k, v in params.items() if k in required_arguments}

    torch.manual_seed(seed)  # for reproducibility (almost)

    model = Featurizer(n_channels=params["channels"], img_size=params["img_size"])

    return model
