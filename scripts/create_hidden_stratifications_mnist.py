#!/usr/bin/python3

import argparse
import logging
import os

import torch
import torchvision.transforms as transforms
import yaml
from core.dataset import dataset_fn
from core.model import MNISTNet
from multi_level_split.util import train_test_split as multilevel_split
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from torchvision import datasets
from utils.config import load_config


def compose_transform(
    p_blur, p_erase=0.5, blur_kernel_size=(9, 9), blur_sigma=(10, 10), scale_erase=0.15
):

    random_erase = transforms.RandomErasing(
        p=p_erase, scale=(scale_erase, scale_erase), ratio=(1, 1)
    )

    random_blur = transforms.RandomApply(
        transforms=[transforms.GaussianBlur(kernel_size=blur_kernel_size, sigma=blur_sigma)],
        p=p_blur,
    )
    return transforms.Compose(
        [
            transforms.ToTensor(),
            random_blur,
            random_erase,
            transforms.Normalize([0.1307], [0.3081]),
        ]
    )


# class SubgroupMNIST(VisionDataset):
#     def __init__(
#         self,
#         data_root,
#         root="",
#         transform=None,
#         target_transform=None,
#         split="train",
#         subset_params=None,
#         use_prepared_splits=True,
#     ):
#         super(EyepacsDataset, self).__init__(
#             root, transform=transform, target_transform=target_transform
#         )

#         self.image_path = os.path.join(data_root, "data_raw", "images")
#         meta_csv = os.path.join(data_root, "data_processed", "metadata", "metadata_image.csv")
#         metadata_df = pd.read_csv(meta_csv)

#         self._split_dict = {"train": 0, "test": 1, "val": 2}
#         self._split_names = {
#             "train": "Train",
#             "test": "Test",
#             "val": "Validation",
#         }

#         if split not in self._split_dict:
#             raise ArgumentError(f"split not recognised: {split}")

#         dev, test = multilevel_split(
#             metadata_df, "image_id", split_by="patient_id", test_split=0.2, seed=12345
#         )

#         train, val = multilevel_split(
#             dev, "image_id", split_by="patient_id", test_split=0.25, seed=12345
#         )

#         data = {"train": train, "val": val, "test": test}

#         self._metadata_df = data[split]

#         # declutter: keep only images with the following characteristics
#         sides = {"left": 1, "right": 0}
#         fields = {"field 1": 1, "field 2": 2, "field 3": 3}
#         genders = {"Male": 0, "Female": 1, "Other": 2}
#         image_qualities = {
#             "Insufficient for Full Interpretation": 0,
#             "Adequate": 1,
#             "Good": 2,
#             "Excellent": 3,
#         }
#         ethnicities = {
#             "Latin American": 0,
#             "Caucasian": 1,
#             "African Descent": 2,
#             "Asian": 3,
#             "Indian subcontinent origin": 4,
#             "Native American": 5,
#             "Multi-racial": 6,
#         }

#         # filter
#         keep_fields = ["field 1"]
#         keep_quality = ["Adequate", "Good", "Excellent"]

#         self._metadata_df = self._metadata_df.query(f"image_side in {list(sides)}")
#         self._metadata_df = self._metadata_df.query(f"image_field in {list(keep_fields)}")
#         self._metadata_df = self._metadata_df.query(f"patient_gender in {list(genders)}")
#         self._metadata_df = self._metadata_df.query(
#             f"session_image_quality in {list(keep_quality)}"
#         )
#         self._metadata_df = self._metadata_df.query(f"patient_ethnicity in {list(ethnicities)}")

#         # TODO: allow subset query here

#         if subset_params is not None:
#             # TODO check that this is a dict with keys in valid_keys and values are list of attributes to keep
#             valid_keys = ["patient_gender", "patient_ethnicity", "session_image_quality"]

#             for k, v in subset_params.items():
#                 print(k, v)
#                 self._metadata_df = self._metadata_df.query(f"{k} in {v}")

#         # Get the y values
#         self._y_array = torch.LongTensor(self._metadata_df["diagnosis_image_dr_level"].values)
#         self._n_classes = 5

#         # Get filenames
#         self._input_array = [
#             os.path.join(self.image_path, ele) for ele in self._metadata_df["image_path"].values
#         ]

#         self._side_array = torch.LongTensor(
#             [sides[ele] for ele in self._metadata_df["image_side"]]
#         )
#         self._field_array = torch.LongTensor(
#             [fields[ele] for ele in self._metadata_df["image_field"]]
#         )
#         self._gender_array = torch.LongTensor(
#             [genders[ele] for ele in self._metadata_df["patient_gender"]]
#         )
#         self._age_array = torch.FloatTensor([ele for ele in self._metadata_df["patient_age"]])
#         self._quality_array = torch.LongTensor(
#             [image_qualities[ele] for ele in self._metadata_df["session_image_quality"]]
#         )
#         self._ethnicity_array = torch.LongTensor(
#             [ethnicities[ele] for ele in self._metadata_df["patient_ethnicity"]]
#         )

#         self._metadata_array = torch.stack(
#             (
#                 self._side_array,
#                 self._field_array,
#                 self._gender_array,
#                 self._age_array,
#                 self._quality_array,
#                 self._ethnicity_array,
#                 self._y_array,
#             ),
#             dim=1,
#         )
#         self._metadata_fields = ["side", "field", "gender", "age", "quality", "ethnicity", "y"]

#         self.targets = list(self._y_array)
#         self.classes = sorted(list(set(self.targets)))

#         self._metadata_df = self._metadata_df.reset_index(drop=True)

#     def __len__(self):
#         return len(self.y_array)

#     def __getitem__(self, idx):
#         x = self.get_input(idx)
#         y = self.y_array[idx]
#         m = self._metadata_array[idx]

#         return x, y, m, idx

#     def get_input(self, idx):
#         """
#         Args:
#             - idx (int): Index of a data point
#         Output:
#             - x (Tensor): Input features of the idx-th data point
#         """

#         img_filename = os.path.join(self.image_path, self._input_array[idx])
#         x = Image.open(img_filename).convert("RGB")

#         if self.transform is not None:
#             x = self.transform(x)

#         return x

#     @property
#     def side(self):
#         return self._side_array

#     @property
#     def field(self):
#         return self._field_array

#     @property
#     def gender(self):
#         return self._gender_array

#     @property
#     def quality(self):
#         return self._quality_array

#     @property
#     def ethnicity(self):
#         return self._ethnicity_array

#     @property
#     def y_array(self):
#         """
#         A Tensor of targets (e.g., labels for classification tasks),
#         with y_array[i] representing the target of the i-th data point.
#         y_array[i] can contain multiple elements.
#         """
#         return self._y_array

#     @property
#     def n_classes(self):
#         """
#         Number of classes for single-task classification datasets.
#         Used for logging and to configure models to produce appropriately-sized output.
#         None by default.
#         Leave as None if not applicable (e.g., regression or multi-task classification).
#         """
#         return getattr(self, "_n_classes", None)

#     @property
#     def metadata_fields(self):
#         """
#         A list of strings naming each column of the metadata table, e.g., ['hospital', 'y'].
#         Must include 'y'.
#         """
#         return self._metadata_fields

#     @property
#     def metadata_array(self):
#         """
#         A Tensor of metadata, with the i-th row representing the metadata associated with
#         the i-th data point. The columns correspond to the metadata_fields defined above.
#         """
#         return self._metadata_array


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
        "--config_file",
        action="store",
        type=str,
        help="config file",
        default="./config/classification_mnist_subgroups.yaml"
        # default='./experiments/hypothesis-tests/mnist/5c3010e7e9f5de06c7d55ecbed422251/config.yaml'
    )
    parser.add_argument(
        "--seed",
        action="store",
        default=1000,
        type=int,
        help="random seed",
    )

    args = parser.parse_args()

    params = load_config(args.config_file)

    ###############################################################################################################################
    # Preparation
    ###############################################################################################################################

    # TODO provide transform params to dataset creator

    dataloader = dataset_fn(params_dict=params["dataset"])

    ###############################################################################################################################
    # Prepare model and training
    ###############################################################################################################################

    if params["model"]["task_classifier_type"] == "mnist":
        model = MNISTNet(
            n_outputs=params["model"]["n_outputs"],
            checkpoint_path=params["model"]["task_classifier_path"],
            download=False,
        )
    else:
        raise NotImplementedError

    model.train_model(dataloader_train=dataloader["train"]["p"])
    acc = model.eval_model(dataloader=dataloader["validation"]["p"])

    logging.info(f"Val acc: {acc}")

    subgroup_dataloaders = {}

    p_erase = [0, 0.5, 1]

    for ele in p_erase:
        params["dataset"]["ds"]["q"]["subgroup"]["p_erase"] = ele

        dataloader = dataset_fn(params_dict=params["dataset"])

        acc = model.eval_model(dataloader=dataloader["test"]["q"])

        print(f"acc on test fold for p_erase {ele}: {acc}")
