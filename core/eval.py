import os

import numpy as np
import pandas as pd

import core.muks
from core.dataset import dataset_fn
from core.mmdd import trainer_object_fn
from core.model import get_classification_model, model_fn
from core.muks import c2st, mmdd, muks


def stderr_proportion(p, n):
    return np.sqrt(p * (1 - p) / n)


def eval(
    exp_dir,
    exp_name,
    params,
    seed,
    split,
    model_dict,
    sample_sizes=[10, 30, 50, 100, 500],
    num_reps=100,
    num_permutations=1000,
    force_run=False,
):
    """Analysis of test power vs sample size for both MMD-D and MUKS

    Args:
        exp_dir ([type]): exp base directory
        exp_name ([type]): experiment name (hashed config)
        params (Dict): [description]
        seed (int): random seed
        split (str): fold to evaluate, e.g. 'validation' or 'test
        sample_sizes (list, optional): Defaults to [10, 30, 50, 100, 500].
        num_reps (int, optional): for calculation rejection rates. Defaults to 100.
        num_permutations (int, optional): for MMD-D permutation test. Defaults to 1000.
    """

    log_dir = os.path.join(exp_dir, exp_name)
    out_csv = os.path.join(log_dir, f"{split}_consistency_analysis.csv")

    if os.path.exists(out_csv) and not force_run:
        print(f"Results exists - do not calculate again.")
        return

    df = pd.DataFrame(columns=["sample_size", "power", "type_1err", "method"])

    for batch_size in sample_sizes:

        params["dataset"]["dl"]["batch_size"] = batch_size
        dataloader = dataset_fn(seed=seed, params_dict=params["dataset"])

        # MMD-D
        for method in ["mmdd", "muks", "c2st"]:

            print(f"evaluate: {method}")

            func = getattr(core.muks, method)

            reject_rate, type_1_err = func(
                dataloader[split]["p"],
                dataloader[split]["q"],
                num_reps,
                model=model_dict[method],
                alpha=0.05,
                num_permutations=num_permutations,
            )

            res = {
                "sample_size": batch_size,
                "power": reject_rate,
                "type_1err": type_1_err,
                "method": method,
            }

            df = df.append(pd.DataFrame(res, index=[""]), ignore_index=True)

            print(res)

    df["exp_hash"] = exp_name
    df["power_stderr"] = stderr_proportion(df["power"], df["sample_size"].astype(float))
    df["type_1err_stderr"] = stderr_proportion(df["type_1err"], df["sample_size"].astype(float))

    df.to_csv(out_csv)

    col_width = 3.5

    methods = ["mmdd", "c2st", "muks"]
    legend_labels = ["MMD-D", "C2ST", "MUKS"]

    import matplotlib.pyplot as plt
    import seaborn as sns
    from utils.helpers import set_rcParams

    set_rcParams()

    out_fig = os.path.join(log_dir, f"{split}_power_err.pdf")

    fig, ax = plt.subplots(1, 2, figsize=(3.5, 2))

    sns.lineplot(
        data=df,
        x="sample_size",
        y="power",
        hue="method",
        hue_order=methods,
        markers=True,
        dashes=False,
        ax=ax[0],
    )
    sns.lineplot(
        data=df,
        x="sample_size",
        y="type_1err",
        hue="method",
        hue_order=methods,
        markers=True,
        dashes=False,
        ax=ax[1],
    )

    colors = sns.color_palette()

    for cidx, hue in enumerate(methods):
        subset = df[df["method"] == hue]
        ax[0].fill_between(
            subset["sample_size"].astype(float),
            subset["power"].astype(float) - subset["power_stderr"].astype(float),
            subset["power"].astype(float) + subset["power_stderr"].astype(float),
            color=colors[cidx],
            alpha=0.5,
        )
        ax[1].fill_between(
            subset["sample_size"].astype(float),
            subset["type_1err"].astype(float) - subset["type_1err_stderr"].astype(float),
            subset["type_1err"].astype(float) + subset["type_1err_stderr"].astype(float),
            color=colors[cidx],
            alpha=0.5,
        )

    ax[0].set_ylim(0, 1.05)
    ax[1].set_ylim(0, 0.4)

    for idx in [0, 1]:
        ax[idx].set(xscale="log")
        ax[idx].grid(axis="x")

    ax[0].set_ylabel("Test power")
    ax[0].set_xlabel(r"Sample size m")
    ax[1].set_ylabel("Type I Error")
    ax[1].set_xlabel(r"Sample size m")

    ax[1].get_legend().remove()
    ax[0].legend(legend_labels, loc="best", frameon=False)

    plt.tight_layout()
    fig.savefig(out_fig)
