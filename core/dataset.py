import os
from argparse import ArgumentError
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from multi_level_split.util import train_test_split as multilevel_split
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from torchvision import datasets
from torchvision.datasets import VisionDataset
from utils.helpers import balanced_weights
from utils.transforms import data_transforms
from wilds.datasets.camelyon17_dataset import Camelyon17Dataset
from wilds.datasets.wilds_dataset import WILDSDataset


def dataset_fn(params_dict, boot_strap_test=False) -> Dict:
    """
    Returns data loaders for the given config
    Args:
    Returns:
        data_loaders: containing "train", "validation" and "test" data loaders
    """

    required_keys = ["ds", "dl"]

    for ele in required_keys:
        assert ele in params_dict

    # TODO input validation - check that params_dict contains all the right keys

    params_ds = params_dict["ds"]
    params_dl = params_dict["dl"]

    dataloader = {"train": {}, "val": {}, "test": {}}
    for p_q in ["p", "q"]:

        dataset = get_dataset(
            params_ds[p_q]["dataset"],
            params_ds[p_q]["data_root"],
            params_ds[p_q].get("subset_params"),
            params_ds["basic_preproc"],
            params_ds["data_augmentation"],
            params_ds["data_augmentation_args"],
        )

        for split in ["train", "val", "test"]:

            boot_strap = False

            if split in ["val", "test"] and boot_strap_test:
                boot_strap = True

            dataloader[split][p_q] = get_dataloader(
                dataset[split],
                batch_size=params_dl["batch_size"],
                use_sampling=params_dl[p_q]["use_sampling"],
                sampling_by_variable=params_dl[p_q]["sampling_by_variable"],
                sampling_weights=params_dl[p_q]["sampling_weights"],
                num_workers=params_dl["num_workers"],
                pin_memory=params_dl["pin_memory"],
                boot_strapping=boot_strap,
            )

    return {
        "train": dataloader["train"],
        "validation": dataloader["val"],
        "test": dataloader["test"],
    }


def get_dataset(
    dataset_type: str,
    data_root: str,
    subset_params,
    basic_preproc_config,
    augmentations,
    augmentation_config,
):

    train_transform, test_transform = data_transforms(
        basic_preproc_config, augmentations, augmentation_config
    )

    dataset = {}

    if dataset_type == "mnist":

        scale_erase = subset_params["scale_erase"]

        random_erase = transforms.RandomErasing(
            p=subset_params["p_erase"], scale=(scale_erase, scale_erase), ratio=(1, 1)
        )

        train_transform = transforms.Compose(
            [transforms.ToTensor(), random_erase, transforms.ToPILImage(), train_transform]
        )
        test_transform = transforms.Compose(
            [transforms.ToTensor(), random_erase, transforms.ToPILImage(), test_transform]
        )

        mnist_train = datasets.MNIST(
            data_root, transform=train_transform, download=True, train=True
        )
        mnist_val = datasets.MNIST(data_root, transform=test_transform, download=True, train=True)

        dataset["test"] = datasets.MNIST(
            data_root, transform=test_transform, download=True, train=False
        )

        train_indices, val_indices, _, _ = train_test_split(
            range(len(mnist_train)),
            mnist_train.targets,
            stratify=mnist_train.targets,
            test_size=1.0 / 6,
        )

        # generate subset based on indices
        dataset["train"] = Subset(mnist_train, train_indices)
        dataset["val"] = Subset(mnist_val, val_indices)

    elif dataset_type == "camelyon":

        full_dataset = LisaCamelyon17Dataset(
            root_dir=data_root, split_scheme="vanilla", download=True
        )

        dataset["train"] = full_dataset.get_subset("train", transform=train_transform)
        dataset["val"] = full_dataset.get_subset("val", transform=test_transform)
        dataset["test"] = full_dataset.get_subset("test", transform=test_transform)

    elif dataset_type == "eyepacs":

        # TODO fix data_root relative to both image_dir and csv
        # data_root = '/home/lkoch/mnt/cin_root/data/eyepacs/'

        for split in ["train", "val", "test"]:

            transform = train_transform if split == "train" else test_transform

            dataset[split] = EyepacsDataset(
                data_root=data_root, split=split, transform=transform, subset_params=subset_params
            )

    else:
        raise NotImplementedError(f"Dataset not implemented: {dataset_type}")

    return dataset


def get_dataloader(
    dataset,
    batch_size: int,
    use_sampling: bool,
    sampling_by_variable: str,
    sampling_weights: list,
    num_workers: int = 4,
    pin_memory: bool = True,
    boot_strapping: bool = False,
):
    """Get dataloader based on a dataset and minibatch sampling strategy

    Args:
        dataset (VisionDataset):
        data_params (Dict): Minibatch sampling strategy
        batch_size (int):

    Returns:
        Dataloader:
    """

    if boot_strapping:
        num_samples = batch_size * 100
        replacement = True
    else:
        num_samples = len(dataset)
        replacement = False

    if use_sampling:
        # weights for balanced minibatches
        weights_train = balanced_weights(
            dataset, rebalance_weights=sampling_weights, balance_variable=sampling_by_variable
        )
        sampler = torch.utils.data.sampler.WeightedRandomSampler(
            weights_train, num_samples=num_samples
        )
        dataloader_kwargs = {"sampler": sampler}

    else:
        sampler = torch.utils.data.sampler.RandomSampler(
            dataset, replacement=replacement, num_samples=num_samples
        )
        dataloader_kwargs = {"sampler": sampler}

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        **dataloader_kwargs,
    )

    return dataloader


class SubgroupSampler(torch.utils.data.sampler.Sampler):
    """Sample only specific labels"""

    def __init__(self, data_source, label=5, subgroup="targets"):

        subgroup = getattr(data_source, subgroup).clone().detach()

        self.mask = subgroup == label
        self.indices = torch.nonzero(self.mask)
        self.data_source = data_source

    def __iter__(self):

        return iter([self.indices[i].item() for i in torch.randperm(len(self.indices))])

    def __len__(self):
        return len(self.indices)


class LisaCamelyon17Dataset(Camelyon17Dataset):
    def get_subset(self, split, frac=1.0, transform=None):
        """
        Args:
            - split (str): Split identifier, e.g., 'train', 'val', 'test'.
                           Must be in self.split_dict.
            - frac (float): What fraction of the split to randomly sample.
                            Used for fast development on a small dataset.
            - transform (function): Any data transformations to be applied to the input x.
        Output:
            - subset (WILDSSubset): A (potentially subsampled) subset of the WILDSDataset.
        """
        if split not in self.split_dict:
            raise ValueError(f"Split {split} not found in dataset's split_dict.")
        split_mask = self.split_array == self.split_dict[split]
        split_idx = np.where(split_mask)[0]
        if frac < 1.0:
            num_to_retain = int(np.round(float(len(split_idx)) * frac))
            split_idx = np.sort(np.random.permutation(split_idx)[:num_to_retain])
        subset = LisaWILDSSubset(self, split_idx, transform)
        return subset


class LisaWILDSSubset(WILDSDataset):
    def __init__(self, dataset, indices, transform, do_transform_y=False):
        """
        This acts like `torch.utils.data.Subset`, but on `WILDSDatasets`.
        We pass in `transform` (which is used for data augmentation) explicitly
        because it can potentially vary on the training vs. test subsets.

        `do_transform_y` (bool): When this is false (the default),
                                 `self.transform ` acts only on  `x`.
                                 Set this to true if `self.transform` should
                                 operate on `(x,y)` instead of just `x`.
        """
        self.dataset = dataset
        self.indices = indices
        inherited_attrs = [
            "_dataset_name",
            "_data_dir",
            "_collate",
            "_split_scheme",
            "_split_dict",
            "_split_names",
            "_y_size",
            "_n_classes",
            "_metadata_fields",
            "_metadata_map",
        ]
        for attr_name in inherited_attrs:
            if hasattr(dataset, attr_name):
                setattr(self, attr_name, getattr(dataset, attr_name))
        self.transform = transform
        self.do_transform_y = do_transform_y

    def __getitem__(self, idx):
        x, y, _ = self.dataset[self.indices[idx]]
        if self.transform is not None:
            if self.do_transform_y:
                x, y = self.transform(x, y)
            else:
                x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.indices)

    @property
    def split_array(self):
        return self.dataset._split_array[self.indices]

    @property
    def y_array(self):
        return self.dataset._y_array[self.indices]

    @property
    def metadata_array(self):
        return self.dataset.metadata_array[self.indices]

    def eval(self, y_pred, y_true, metadata):
        return self.dataset.eval(y_pred, y_true, metadata)

    @property
    def hospitals(self):
        return self.metadata_array[:, 0]


class EyepacsDataset(VisionDataset):
    def __init__(
        self,
        data_root,
        root="",
        transform=None,
        target_transform=None,
        split="train",
        subset_params=None,
        use_prepared_splits=True,
    ):
        super(EyepacsDataset, self).__init__(
            root, transform=transform, target_transform=target_transform
        )

        self.image_path = os.path.join(data_root, "data_processed", "images")
        meta_csv = os.path.join(
            data_root, "data_processed", "metadata", "metadata_image_circular_crop.csv"
        )
        metadata_df = pd.read_csv(meta_csv)

        self._split_dict = {"train": 0, "test": 1, "val": 2}
        self._split_names = {
            "train": "Train",
            "test": "Test",
            "val": "Validation",
        }

        if split not in self._split_dict:
            raise ArgumentError(f"split not recognised: {split}")

        dev, test = multilevel_split(
            metadata_df, "image_id", split_by="patient_id", test_split=0.2, seed=12345
        )

        train, val = multilevel_split(
            dev, "image_id", split_by="patient_id", test_split=0.25, seed=12345
        )

        data = {"train": train, "val": val, "test": test}

        self._metadata_df = data[split]

        # declutter: keep only images with the following characteristics
        sides = {"left": 1, "right": 0}
        fields = {"field 1": 1, "field 2": 2, "field 3": 3}
        genders = {"Male": 0, "Female": 1, "Other": 2}
        image_qualities = {
            "Insufficient for Full Interpretation": 0,
            "Adequate": 1,
            "Good": 2,
            "Excellent": 3,
        }
        ethnicities = {
            "Latin American": 0,
            "Caucasian": 1,
            "African Descent": 2,
            "Asian": 3,
            "Indian subcontinent origin": 4,
            "Native American": 5,
            "Multi-racial": 6,
        }
        dr_levels = [0.0, 1.0, 2.0, 3.0, 4.0]

        # filter
        keep_fields = ["field 1"]
        keep_quality = ["Adequate", "Good", "Excellent"]

        self._metadata_df = self._metadata_df.query(f"image_side in {list(sides)}")
        self._metadata_df = self._metadata_df.query(f"image_field in {list(keep_fields)}")
        self._metadata_df = self._metadata_df.query(f"patient_gender in {list(genders)}")
        self._metadata_df = self._metadata_df.query(
            f"session_image_quality in {list(keep_quality)}"
        )
        self._metadata_df = self._metadata_df.query(f"patient_ethnicity in {list(ethnicities)}")
        self._metadata_df = self._metadata_df.query(f"diagnosis_image_dr_level in {dr_levels}")

        # Age correction (due to wrong data export from Eyepacs Inc):
        export_year = 2022
        self._metadata_df["patient_age"] = self._metadata_df["patient_age"] - (
            export_year - self._metadata_df["clinical_encounterDate"]
        )

        # co-morbidity: at least one diagnosis other than DR
        co_diagnoses = [
            "diagnosis_cataract",
            "diagnosis_dme",
            "diagnosis_glaucoma",
            "diagnosis_maculopathy",
            "diagnosis_occlusion",
            "diagnosis_other_referrable",
            "diagnosis_unspecified_complications",
        ]

        self._metadata_df["diagnoses_comorbidities"] = (
            self._metadata_df[co_diagnoses].sum(axis=1) > 0
        )

        # allow subset query here

        if subset_params is not None:
            # TODO check that this is a dict with keys in valid_keys and values are list of attributes to keep
            valid_keys = [
                "patient_gender",
                "patient_ethnicity",
                "session_image_quality",
                "diagnoses_comorbidities",
            ]

            for k, v in subset_params.items():
                print(k, v)
                self._metadata_df = self._metadata_df.query(f"{k} in {v}")

        # Get the y values
        self._y_array = torch.LongTensor(self._metadata_df["diagnosis_image_dr_level"].values)
        self._n_classes = 5

        # Get filenames
        self._input_array = [
            os.path.join(self.image_path, ele) for ele in self._metadata_df["image_path"].values
        ]

        self._side_array = torch.LongTensor(
            [sides[ele] for ele in self._metadata_df["image_side"]]
        )
        self._field_array = torch.LongTensor(
            [fields[ele] for ele in self._metadata_df["image_field"]]
        )
        self._gender_array = torch.LongTensor(
            [genders[ele] for ele in self._metadata_df["patient_gender"]]
        )
        self._age_array = torch.FloatTensor([ele for ele in self._metadata_df["patient_age"]])
        self._quality_array = torch.LongTensor(
            [image_qualities[ele] for ele in self._metadata_df["session_image_quality"]]
        )
        self._ethnicity_array = torch.LongTensor(
            [ethnicities[ele] for ele in self._metadata_df["patient_ethnicity"]]
        )

        self._metadata_array = torch.stack(
            (
                self._side_array,
                self._field_array,
                self._gender_array,
                self._age_array,
                self._quality_array,
                self._ethnicity_array,
                self._y_array,
            ),
            dim=1,
        )
        self._metadata_fields = ["side", "field", "gender", "age", "quality", "ethnicity", "y"]

        self.targets = list(self._y_array)
        self.classes = sorted(list(set(self.targets)))

        self._metadata_df = self._metadata_df.reset_index(drop=True)

    def __len__(self):
        return len(self.y_array)

    def __getitem__(self, idx):
        x = self.get_input(idx)
        y = self.y_array[idx]
        m = self._metadata_array[idx]

        return x, y, m, idx

    def get_input(self, idx):
        """
        Args:
            - idx (int): Index of a data point
        Output:
            - x (Tensor): Input features of the idx-th data point
        """

        img_filename = os.path.join(self.image_path, self._input_array[idx])
        x = Image.open(img_filename).convert("RGB")

        if self.transform is not None:
            x = self.transform(x)

        return x

    @property
    def side(self):
        return self._side_array

    @property
    def field(self):
        return self._field_array

    @property
    def gender(self):
        return self._gender_array

    @property
    def quality(self):
        return self._quality_array

    @property
    def ethnicity(self):
        return self._ethnicity_array

    @property
    def y_array(self):
        """
        A Tensor of targets (e.g., labels for classification tasks),
        with y_array[i] representing the target of the i-th data point.
        y_array[i] can contain multiple elements.
        """
        return self._y_array

    @property
    def n_classes(self):
        """
        Number of classes for single-task classification datasets.
        Used for logging and to configure models to produce appropriately-sized output.
        None by default.
        Leave as None if not applicable (e.g., regression or multi-task classification).
        """
        return getattr(self, "_n_classes", None)

    @property
    def metadata_fields(self):
        """
        A list of strings naming each column of the metadata table, e.g., ['hospital', 'y'].
        Must include 'y'.
        """
        return self._metadata_fields

    @property
    def metadata_array(self):
        """
        A Tensor of metadata, with the i-th row representing the metadata associated with
        the i-th data point. The columns correspond to the metadata_fields defined above.
        """
        return self._metadata_array
