import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from utils.helpers import set_rcParams

import core.muks
from core.dataset import dataset_fn
from core.mmdd import trainer_object_fn
from core.model import get_classification_model, model_fn
from core.muks import c2st, mmdd, muks


def stderr_proportion(p, n):
    return np.sqrt(p * (1 - p) / n)


def eval(
    trainer,
    data_params,
    exp_dir,
    exp_name,
    params,
    split,
    sample_sizes=[10, 30, 50, 100, 500],
    num_reps=100,
    num_permutations=1000,
):
    """Analysis of test power vs sample size for both MMD-D and MUKS

    Args:
        exp_dir ([type]): exp base directory
        exp_name ([type]): experiment name (hashed config)
        params (Dict): [description]
        split (str): fold to evaluate, e.g. 'validation' or 'test
        sample_sizes (list, optional): Defaults to [10, 30, 50, 100, 500].
        num_reps (int, optional): for calculation rejection rates. Defaults to 100.
        num_permutations (int, optional): for MMD-D permutation test. Defaults to 1000.
    """

    log_dir = os.path.join(exp_dir, exp_name)
    out_csv = os.path.join(log_dir, f"{split}_consistency_analysis.csv")

    df = pd.DataFrame(columns=["sample_size", "power", "type_1err", "method"])

    for batch_size in sample_sizes:

        params["dataset"]["dl"]["batch_size"] = batch_size

        # FIXME boot_strapping should be enabled for MMD only
        boot_strap_test = True

        # FIXME: for MMD, should have another dataloader with two ['p']s for type 1 err

        dataloader = dataset_fn(params_dict=params["dataset"], boot_strap_test=boot_strap_test)

        # TODO implement test_step: MMD
        # TODO for MMD: dataloaders with
        #      - batch_size = sample_size
        #      - num_samples = batch_size * num_repetitions
        #      - sampling with replacement (already implemented for camelyon, careful not to destroy)
        #      - additional dataloader for type I error
        res = trainer.test(dataloaders=dataloader[split])[0]

        # TODO test returns power only forMMD -> adapt interface. Hacky, but what can you do
        reject_rate = res["test/power"]
        type_1_err = res["test/type_1err"]

        res = {
            "sample_size": batch_size,
            "power": reject_rate,
            "type_1err": type_1_err,
        }

        df = df.append(pd.DataFrame(res, index=[""]), ignore_index=True)

        print(res)

    df["exp_hash"] = exp_name
    df["power_stderr"] = stderr_proportion(df["power"], df["sample_size"].astype(float))
    df["type_1err_stderr"] = stderr_proportion(df["type_1err"], df["sample_size"].astype(float))

    df.to_csv(out_csv)
