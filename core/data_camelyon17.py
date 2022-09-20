import os
from argparse import ArgumentError

import numpy as np
import pandas as pd
import torch
from PIL import Image
from wilds.datasets.wilds_dataset import WILDSDataset


class Camelyon17Dataset(WILDSDataset):
    """
    The CAMELYON17-WILDS histopathology dataset.
    This is a modified version of the original CAMELYON17 dataset.

    Supported `split_scheme`:
        - 'vanilla'
        - 'vanilla_no3'

    Input (x):
        96x96 image patches extracted from histopathology slides.

    Label (y):
        y is binary. It is 1 if the central 32x32 region contains any tumor tissue, and 0 otherwise.

    Metadata:
        Each patch is annotated with the ID of the hospital it came from (integer from 0 to 4)
        and the slide it came from (integer from 0 to 49).

    Website:
        https://camelyon17.grand-challenge.org/

    Original publication:
        @article{bandi2018detection,
          title={From detection of individual metastases to classification of lymph node status at the patient level: the camelyon17 challenge},
          author={Bandi, Peter and Geessink, Oscar and Manson, Quirine and Van Dijk, Marcory and Balkenhol, Maschenka and Hermsen, Meyke and Bejnordi, Babak Ehteshami and Lee, Byungjae and Paeng, Kyunghyun and Zhong, Aoxiao and others},
          journal={IEEE transactions on medical imaging},
          volume={38},
          number={2},
          pages={550--560},
          year={2018},
          publisher={IEEE}
        }

    License:
        This dataset is in the public domain and is distributed under CC0.
        https://creativecommons.org/publicdomain/zero/1.0/
    """

    _dataset_name = "camelyon17"
    _versions_dict = {
        "1.0": {
            "download_url": "https://worksheets.codalab.org/rest/bundles/0xe45e15f39fb54e9d9e919556af67aabe/contents/blob/",
            "compressed_size": 10_658_709_504,
        }
    }

    def __init__(
        self,
        version=None,
        root_dir="data",
        download=False,
        transform=None,
        split_scheme="vanilla",
        split="train",
        subset_params=None,
    ):
        self._version = version
        self._data_dir = self.initialize_data_dir(root_dir, download)
        self._original_resolution = (96, 96)

        self.transform = transform

        # Read in metadata
        self._metadata_df = pd.read_csv(
            os.path.join(self._data_dir, "metadata.csv"), index_col=0, dtype={"patient": "str"}
        )

        self._split_dict = {"train": 0, "test": 2, "val": 3}
        self._split_names = {
            "train": "Train",
            "test": "Test",
            "val": "Validation",
        }

        # Extract splits

        self._split_scheme = split_scheme
        if self._split_scheme == "vanilla":
            # train-val-test with equal representation of hospitals.
            #
            # constraints:
            # - equal representation of hospitals.
            # - train/val and test set must be separated by slides
            # - large and small slides are approximately equally distributed
            # - stratified by tumour presence
            # note: separating by patient is not necessary, as patient IDs were anyway assigned arbitrarily

            rng = np.random.default_rng(12345)

            train_mask = rng.random(len(self._metadata_df)) < 0.8
            self._metadata_df.loc[train_mask, "split"] = self.split_dict["train"]

            val_mask = train_mask == False
            self._metadata_df.loc[val_mask, "split"] = self.split_dict["val"]

            # these slides were selected to ensure equal label prevalence and approximately 20% of overall patches per hospital
            test_slides = [4, 5, 7, 14, 16, 19, 22, 23, 25, 32, 33, 39, 47, 49]
            for slide in test_slides:
                slide_mask = self._metadata_df["slide"] == slide
                self._metadata_df.loc[slide_mask, "split"] = self.split_dict["test"]

        elif self._split_scheme == "vanilla_no3":
            # train-val-test with equal representation of hospitals.
            #
            # constraints:
            # - equal representation of hospitals.
            # - train/val and test set must be separated by slides
            # - large and small slides are approximately equally distributed
            # - stratified by tumour presence
            # note: separating by patient is not necessary, as patient IDs were anyway assigned arbitrarily

            rng = np.random.default_rng(12345)

            train_mask = rng.random(len(self._metadata_df)) < 0.8
            self._metadata_df.loc[train_mask, "split"] = self.split_dict["train"]

            val_mask = train_mask == False
            self._metadata_df.loc[val_mask, "split"] = self.split_dict["val"]

            # these slides were selected to ensure equal label prevalence and approximately 20% of overall patches per hospital
            test_slides = [4, 5, 7, 14, 16, 19, 22, 23, 25, 32, 33, 39, 47, 49]
            for slide in test_slides:
                slide_mask = self._metadata_df["slide"] == slide
                self._metadata_df.loc[slide_mask, "split"] = self.split_dict["test"]

        else:
            raise ValueError(f"Split scheme {self._split_scheme} not recognized")

        # get train val test splits by splitting metadata_df
        if split not in ["train", "val", "test"]:
            raise ArgumentError(f"Invalid split: {split}")

        self._metadata_df = self._metadata_df.query(f"split in [{self.split_dict[split]}]")

        # TODO get subset (from subset_params) by splitting metadata_df again
        if subset_params is not None:
            # TODO check that this is a dict with keys in valid_keys and values are list of attributes to keep
            valid_keys = [
                "center",
            ]

            for k, v in subset_params.items():
                print(k, v)
                self._metadata_df = self._metadata_df.query(f"{k} in {v}")

        # TODO check if reset_index is necessary (probably yes)
        self._metadata_df = self._metadata_df.reset_index(drop=True)

        # TODO now I shouldn't need the Subset class anymore if all went well

        self._split_array = self._metadata_df["split"].values

        # Get the y values
        self._y_array = torch.LongTensor(self._metadata_df["tumor"].values)
        self._y_size = 1
        self._n_classes = 2

        # Get filenames
        self._input_array = [
            f"patches/patient_{patient}_node_{node}/patch_patient_{patient}_node_{node}_x_{x}_y_{y}.png"
            for patient, node, x, y in self._metadata_df.loc[
                :, ["patient", "node", "x_coord", "y_coord"]
            ].itertuples(index=False, name=None)
        ]

        self._metadata_array = torch.stack(
            (
                torch.LongTensor(self._metadata_df["center"].values.astype("long")),
                torch.LongTensor(self._metadata_df["slide"].values),
                self._y_array,
            ),
            dim=1,
        )
        self._metadata_fields = ["hospital", "slide", "y"]

        super().__init__(root_dir, download, split_scheme)

    def get_input(self, idx):
        """
        Returns x for a given idx.
        """
        img_filename = os.path.join(self.data_dir, self._input_array[idx])
        x = Image.open(img_filename).convert("RGB")

        if self.transform is not None:
            x = self.transform(x)

        return x

    @property
    def hospitals(self):
        return self.metadata_array[:, 0]

    def __getitem__(self, idx):
        x = self.get_input(idx)
        y = self.y_array[idx]
        m = self._metadata_array[idx]

        return x, y, m
