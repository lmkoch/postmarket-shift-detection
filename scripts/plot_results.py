import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torchvision
import yaml

from core.dataset import dataset_fn
from utils.helpers import flatten_dict, set_rcParams


def save_example_images(exp_dir, exp_name, seed=1000):

    config_file = os.path.join(exp_dir, exp_name, "config.yaml")
    with open(config_file) as fhandle:
        params = yaml.safe_load(fhandle)

    params["dataset"]["dl"]["batch_size"] = 64
    dataloader = dataset_fn(seed=seed, params_dict=params["dataset"])

    log_dir = os.path.join(exp_dir, exp_name)

    for dist in ["p", "q"]:
        out_fig = os.path.join(log_dir, f"panel_{dist}.png")

        _, (img, _) = next(enumerate(dataloader["train"][dist]))
        img_grid = torchvision.utils.make_grid(img, normalize=True)
        img_grid = np.transpose(img_grid, (1, 2, 0)).numpy()
        plt.imsave(out_fig, img_grid)


def load_all_results(hashes, exp_dir, split="test"):

    all_configs = {}
    all_results = []

    for exp_name in hashes:
        config_file = os.path.join(exp_dir, exp_name, "config.yaml")
        with open(config_file) as fhandle:
            params = yaml.safe_load(fhandle)

        all_configs[exp_name] = flatten_dict(params, sep="_")

        # FIXME version should maybe be "latest"?

        versions = os.listdir(os.path.join(exp_dir, exp_name))

        results_file = os.path.join(
            exp_dir, exp_name, versions[-1], f"{split}_consistency_analysis.csv"
        )

        print(f"Loading {results_file}")

        if os.path.exists(results_file):
            result = pd.read_csv(results_file)
            result["exp_hash"] = exp_name

            all_results.append(result)
        else:
            print(f"Warning: file not available: {results_file}. Continue without")
            # raise ValueError('results are not available. run consistency analysis first')

    configs = pd.DataFrame.from_dict(all_configs, orient="index")
    configs.rename_axis("exp_hash", inplace=True)

    if len(all_results) > 0:
        results = pd.concat(all_results)
        results = results.set_index(["exp_hash", "sample_size"], drop=False)
        df = configs.join(results)
    else:
        df = configs

    return df


def load_ood_results(hashes, exp_dir, split="test"):

    all_configs = {}
    all_results = []

    for exp_name in hashes:
        config_file = os.path.join(exp_dir, exp_name, "config.yaml")
        with open(config_file) as fhandle:
            params = yaml.safe_load(fhandle)

        all_configs[exp_name] = flatten_dict(params, sep="_")

        results_file = os.path.join(exp_dir, exp_name, f"ood_{split}.csv")

        if os.path.exists(results_file):
            df = pd.read_csv(results_file)
            df["exp_hash"] = exp_name
            all_results.append(df)
        else:
            print(f"Warning: file not available: {results_file}. Continue without")
            # raise ValueError('results are not available. run consistency analysis first')

    configs = pd.DataFrame.from_dict(all_configs, orient="index")
    results = pd.concat(all_results)

    configs.rename_axis("exp_hash", inplace=True)
    results = results.set_index(["exp_hash"], drop=False)
    df = configs.join(results)

    return df


def mmd_model_selection(exp_dir, eval_dir):
    """Compare validation performance for different hyperparameter and model choices for MMD-D
    Requires consistency analysis to have been run before.

    Raises:
        ValueError: [description]
    """

    set_rcParams()

    hashes = os.listdir(exp_dir)
    df = load_all_results(hashes, exp_dir, split="validation")

    # line plot of val rejection rate vs lamba param
    loss_types = ["original"]
    loss_labels = ["MMD ratio", "MMD"]

    lambdas = [10**-4, 10**-2, 10**0, 10**2]

    out_fig_lambda_hyperparam = os.path.join(eval_dir, "lambda_hyperparams.pdf")
    fig, ax = plt.subplots(figsize=(3, 2))

    df = df[df["method"] == "mmd"]
    tmp = df[df["sample_size"] == 50]
    mmd = tmp.loc[tmp["trainer_loss_type"] == "mmd2", "power"].values[0]

    tmp = tmp[tmp["trainer_loss_type"] == "original"]
    ax.plot(tmp["trainer_loss_lambda"], tmp["power"], marker="o")

    ax.axhline(mmd, color="k", ls="--", label="hi")

    for hue in loss_types:
        subset = tmp[tmp["trainer_loss_type"] == hue]
        ax.fill_between(
            subset["trainer_loss_lambda"],
            subset["power"] - subset["power_stderr"],
            subset["power"] + subset["power_stderr"],
            alpha=0.5,
        )
    ax.set_ylim(0, 1.05)
    ax.set(xscale="log")
    ax.set_xticks(lambdas)

    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"Test power")
    ax.legend(loss_labels, loc="lower right", frameon=False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(out_fig_lambda_hyperparam)
    fig.show()
    print("plotted model selection.")


def plot_subgroup_results_mnist_camelyon(eval_dir, exp_dir_mnist, exp_dir_camelyon):

    os.makedirs(eval_dir, exist_ok=True)

    set_rcParams()

    exp_dir = {"mnist": exp_dir_mnist, "camelyon": exp_dir_camelyon}

    subgroup_idx = {"mnist": 5, "camelyon": 2}

    df = {"mnist": None, "camelyon": None}
    for dataset in ["mnist", "camelyon"]:

        hashes = os.listdir(exp_dir[dataset])

        df[dataset] = load_all_results(hashes, exp_dir[dataset], split="test")
        df[dataset]["mixing_proportions"] = np.array(
            df[dataset]["dataset_dl_q_sampling_weights"].tolist()
        )[:, subgroup_idx[dataset]]

    out_fig_weights = os.path.join(eval_dir, "panel_complete_rebuttal.png")

    weights = [1, 5, 10, 100]

    methods = ["mmd", "rabanser"]
    legend_labels = ["MMD-D (MNIST)", "MMD-D (Camelyon)", "MUKS (MNIST)", "MUKS (Camelyon)"]

    fig, ax = plt.subplots(3, len(weights), figsize=(6, 5), sharey="row")

    for idx, weight in enumerate(weights):
        row_ax = ax[0, :]

        df["mnist"]["dataset"] = "mnist"
        df["camelyon"]["dataset"] = "camelyon"

        whole_df = pd.concat([df["mnist"], df["camelyon"]])
        tmp = whole_df[whole_df["mixing_proportions"] == weight].reset_index(drop=True)
        mnist = df["mnist"][df["mnist"]["mixing_proportions"] == weight].reset_index(drop=True)
        cam = df["camelyon"][df["camelyon"]["mixing_proportions"] == weight].reset_index(drop=True)

        row_ax[idx].set_title(f"w={weight}")
        sns.lineplot(
            data=tmp,
            x="sample_size",
            y="power",
            hue="method",
            hue_order=methods,
            style="dataset",
            markers=True,
            dashes=False,
            ax=row_ax[idx],
        )

        colors = sns.color_palette()

        for cidx, hue in enumerate(methods):
            subset = mnist[mnist["method"] == hue]
            row_ax[idx].fill_between(
                subset["sample_size"],
                subset["power"] - subset["power_stderr"],
                subset["power"] + subset["power_stderr"],
                color=colors[cidx],
                alpha=0.5,
            )

        for cidx, hue in enumerate(methods):
            subset = cam[cam["method"] == hue]
            row_ax[idx].fill_between(
                subset["sample_size"],
                subset["power"] - subset["power_stderr"],
                subset["power"] + subset["power_stderr"],
                color=colors[cidx],
                alpha=0.5,
            )

        row_ax[idx].set_ylim(0, 1.05)
        row_ax[idx].set(xscale="log")
        row_ax[idx].grid(axis="x")

        row_ax[idx].set_ylabel("Test power")
        row_ax[idx].set_xlabel(r"Sample size m")

        if idx > 0:
            row_ax[idx].get_legend().remove()
        else:
            row_ax[idx].legend(legend_labels, loc="best", frameon=False)

        # images
        # MNIST
        exp_name = mnist.iloc[0]["exp_hash"]

        img_file = os.path.join(exp_dir["mnist"], exp_name, "panel_q.png")
        image = plt.imread(img_file)
        ax[1, idx].imshow(image)
        ax[1, idx].axis("off")

        # Camelyon
        exp_name = cam.iloc[0]["exp_hash"]

        img_file = os.path.join(exp_dir["camelyon"], exp_name, "panel_q.png")
        image = plt.imread(img_file)
        ax[2, idx].imshow(image)
        ax[2, idx].axis("off")
        ax[2, idx].set_ylabel("Camelyon17 examples")

    plt.tight_layout()
    fig.savefig(out_fig_weights)


def plot_subgroup_results_mnist_camelyon_appendix(eval_dir, exp_dir_mnist, exp_dir_camelyon):

    os.makedirs(eval_dir, exist_ok=True)

    set_rcParams()

    datasets = ["mnist", "camelyon"]

    exp_dir = {"mnist": exp_dir_mnist, "camelyon": exp_dir_camelyon}

    subgroup_idx = {"mnist": 5, "camelyon": 2}

    df = {"mnist": None, "camelyon": None}
    for dataset in datasets:

        hashes = os.listdir(exp_dir[dataset])

        df[dataset] = load_all_results(hashes, exp_dir[dataset], split="test")

        sampling_weights = np.array(df[dataset]["dataset_dl_q_sampling_weights"].tolist())

        w, idx = np.max(sampling_weights, axis=1), np.argmax(sampling_weights, axis=1)

        df[dataset]["mixing_proportions"] = w
        df[dataset]["subgroup_idx"] = idx

    out_fig_weights = os.path.join(eval_dir, "panel_complete_appendix_rebuttal.png")
    out_fig_scatter = os.path.join(eval_dir, "muks_vs_mmdd_appendix_rebuttal.png")

    weights = [1, 5, 10, 100]

    methods = ["mmd", "rabanser"]
    legend_labels = ["MMD-D (MNIST)", "MMD-D (Camelyon)", "MUKS (MNIST)", "MUKS (Camelyon)"]

    fig, ax = plt.subplots(1, len(weights), figsize=(6, 2), sharey="row")

    for idx, weight in enumerate(weights):
        row_ax = ax

        df["mnist"]["dataset"] = "mnist"
        dset = "mnist"

        tmp = df[dset][df[dset]["mixing_proportions"] == weight].reset_index(drop=True)

        row_ax[idx].set_title(f"w={weight}")
        sns.lineplot(
            data=tmp,
            x="sample_size",
            y="power",
            hue="method",
            hue_order=methods,
            style="subgroup_idx",
            markers=True,
            dashes=False,
            ax=row_ax[idx],
        )

        row_ax[idx].set_ylim(0, 1.05)
        row_ax[idx].set(xscale="log")
        row_ax[idx].grid(axis="x")

        row_ax[idx].set_ylabel("Test power")
        row_ax[idx].set_xlabel(r"Sample size m")

        # if idx < 3:
        row_ax[idx].get_legend().remove()
        # else:
        #     row_ax[idx].legend(legend_labels, loc='best', frameon=False)

    plt.tight_layout()
    fig.savefig(out_fig_weights)

    pal = sns.color_palette(n_colors=2)
    print(pal)

    small = df["mnist"][["sample_size", "power", "mixing_proportions", "method", "subgroup_idx"]]
    tmp = small

    tmp_mmd = tmp[tmp["method"] == "mmd"]
    tmp_muks = tmp[tmp["method"] == "rabanser"]

    tmp_mmd = tmp_mmd.rename(columns={"power": "power_mmd"})
    tmp_muks = tmp_muks.rename(columns={"power": "power_muks"})

    joined = tmp_mmd.combine_first(tmp_muks)

    fig, ax = plt.subplots(
        figsize=(3, 3),
    )

    sns.scatterplot(data=joined, y="power_mmd", x="power_muks", hue="subgroup_idx", ax=ax)

    ax.set_ylabel("Test power (MMD-D)")
    ax.set_xlabel("Test power (MUKS)")

    ax.set_aspect("equal")
    ax.legend(loc="lower right", frameon=False)

    plt.tight_layout()
    plt.savefig(out_fig_scatter)

    # fig,ax = plt.subplots()

    # joined['avg'] = 0.5* (joined['power_mmd'] + joined['power_muks'])
    # joined['diff'] = joined['power_mmd'] - joined['power_muks']

    # sns.scatterplot(data=joined, x='avg', y='diff', hue='subgroup_idx', ax=ax)


def generate_table_ood(eval_dir, exp_dir):

    os.makedirs(eval_dir, exist_ok=True)

    set_rcParams()

    df = {"mnist": None, "camelyon": None}

    hashes = os.listdir(exp_dir)

    df = load_ood_results(hashes, exp_dir, split="test")

    shift_type = np.array(df["dataset_dl_p_sampling_weights"].tolist()).sum(axis=1) == 9

    sampling_weights = np.array(df["dataset_dl_q_sampling_weights"].tolist())

    idx = np.argmax(sampling_weights, axis=1)

    shift_label = ["OOD" if ele else "subgroup" for ele in shift_type]

    df["subgroup_idx"] = idx
    df["shift_type"] = shift_label  # True for OOD, False for subgroup

    shift_types = ["OOD", "subgroup"]

    digits = range(10)

    df_small = df[["subgroup_idx", "shift_type", "fpr95", "rocauc", "method"]]
    df_small["tnr95"] = 1 - df_small["fpr95"]

    df_small = df_small.sort_values(by=["subgroup_idx", "shift_type", "method"])

    table_rows = []

    title_row_1 = "P & Q & AUC & Detection rate \n"
    table_rows.append(title_row_1)

    for shift_type in shift_types:

        for digit in digits:

            p_label = f"MNIST-no-{digit}" if shift_type == "OOD" else "MNIST-all"
            q_label = f"MNIST-{digit}"

            row_sel = df_small[
                (df_small["subgroup_idx"] == digit) & (df_small["shift_type"] == shift_type)
            ]

            auc_str = " / ".join(f"{x:.2f}" for x in row_sel["rocauc"].tolist())
            tnr_str = " / ".join(f"{x:.2f}" for x in row_sel["tnr95"].tolist())

            row = f"{p_label} & {q_label} & {auc_str} & {tnr_str} \\\\ \n"

            table_rows.append(row)

        table_rows.append("\hline \n")

    table_file = os.path.join(eval_dir, "ood_table.txt")
    with open(table_file, "w") as f:
        f.writelines(table_rows)


def plot_results(eval_dir, exp_dirs, subset_spec_column, experiment_name):

    # exp_dirs = {
    #     "muks": "./experiments/endspurt/camelyon/muks",
    #     "c2st": "./experiments/endspurt/camelyon/domain_classifier",
    #     "mmdd": "./experiments/endspurt/camelyon/mmdd",
    # }

    # # specific to camelyon
    # subset_spec_column = "dataset_ds_q_subset_params_center"

    os.makedirs(eval_dir, exist_ok=True)

    set_rcParams()

    methods = ["c2st", "mmdd", "muks"]

    df = {ele: None for ele in methods}

    for method in methods:

        hashes = os.listdir(exp_dirs[method])

        df[method] = load_all_results(hashes, exp_dirs[method], split="test")

        df[method]["method"] = method

    # method-specific columns are not compatible - but doesn't matter
    results = pd.concat([v for k, v in df.items()])

    # TODO: identify image size here as well

    # TODO: convert subset specification to homogeneous form (ideally string), from:
    #       - list
    #       - scalar
    import collections.abc

    if isinstance(results[subset_spec_column].values[0], collections.abc.Sequence):
        results["subgroup_id"] = [ele[0] for ele in results[subset_spec_column].values]
    else:
        results["subgroup_id"] = [ele for ele in results[subset_spec_column].values]

    out_fig = os.path.join(eval_dir, f"{experiment_name}_power_type_1_err.pdf")

    fig, ax = plt.subplots(1, 2, figsize=(3.5, 1.6))

    row_ax = ax[0]

    sns.lineplot(
        data=results,
        x="sample_size",
        y="power",
        hue="method",
        hue_order=methods,
        # style="subgroup_id",
        style="dataset_ds_basic_preproc_img_size",
        markers=True,
        dashes=False,
        ax=row_ax,
        legend="brief",
    )

    sns.lineplot(
        data=results,
        x="sample_size",
        y="type_1err",
        hue="method",
        hue_order=methods,
        # style="subgroup_id",
        style="dataset_ds_basic_preproc_img_size",
        markers=True,
        dashes=False,
        ax=ax[1],
    )

    ax[0].set_ylim(0, 1.05)
    ax[0].set(xscale="log")
    ax[0].grid(axis="x")

    ax[0].set_ylabel("Test power")
    ax[0].set_xlabel(r"Sample size m")

    ax[1].set_ylim(0, 0.2)
    ax[1].set(xscale="log")
    ax[1].grid(axis="x")

    ax[1].set_ylabel("Type I error")
    ax[1].set_xlabel(r"Sample size m")

    handles, labels = ax[1].get_legend_handles_labels()
    ax[1].legend(handles=handles[1:4], labels=labels[1:4], frameon=False)
    ax[0].get_legend().remove()

    plt.tight_layout()
    fig.savefig(out_fig)
    plt.close(fig)


def generate_latex_subgroup_table(eval_dir):

    artifacts_path = {
        "eyepacs": "experiments/endspurt/eyepacs_comorb/task_classifier/ed38371b2586ab224c4c55642b443b9b/version_0/checkpoints/best-loss-epoch=15-step=78112.ckpt",
        "camelyon": "experiments/endspurt/camelyon/task_smallevents/task_classifier/1bd08d2856418bd6056d24f58671ec86/version_0/checkpoints/best-loss-epoch=13-step=248682.ckpt",
        # "mnist": "experiments/oct/task/mnist/muks/0f2ab3ce9f01eb2f5c230e2c2aa2f99f/version_0/checkpoints/best-loss-epoch=13-step=10934.ckpt",
    }

    datasets = ["eyepacs", "camelyon", "mnist"]

    out_csv = {ele: os.path.join(eval_dir, f"{ele}_subgroup_performance.csv") for ele in datasets}

    eyepacs_categories = {
        "2.0": "sex",
        "4.0": "quality",
        "5.0": "ethnicity",
        "6.0": "co-morbidity",
    }

    camelyon_categories = {"0.0": "hospital"}

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

    quality_map = dict((float(v), k) for k, v in image_qualities.items())
    ethnicity_map = dict((float(v), k) for k, v in ethnicities.items())
    sex_map = dict((float(v), k) for k, v in genders.items())

    for k, v in artifacts_path.items():

        path_bits = v.split("/")

        base_path = "/".join(path_bits[:-2])
        epoch_num = path_bits[-1].split("epoch=")[1].split("-step")[0]

        if k == "eyepacs":
            # FIXME this is the subgroup evaluation I did post-hoc (original one does not contain co-morbidities)
            # TODO use re-trained model once complete, then I don't need this hack anymore
            base_path = "./experiments/endspurt/eyepacs_comorbid/subgroup_eval/muks/ed38371b2586ab224c4c55642b443b9b/version_2"

        # TODO change back after debuggin
        # base_path = "../experiments/oct/debug_latex_table"

        subgroup_file = os.path.join(base_path, f"test_subgroup_performance_epoch:{epoch_num}.csv")

        df = pd.read_csv(subgroup_file)

        if k == "eyepacs":

            df["criterion"].replace(eyepacs_categories, inplace=True)

            df["group"].loc[df["criterion"] == "sex"] = (
                df["group"].loc[df["criterion"] == "sex"].replace(sex_map, inplace=False)
            )
            df["group"].loc[df["criterion"] == "quality"] = (
                df["group"].loc[df["criterion"] == "quality"].replace(quality_map, inplace=False)
            )
            df["group"].loc[df["criterion"] == "ethnicity"] = (
                df["group"]
                .loc[df["criterion"] == "ethnicity"]
                .replace(ethnicity_map, inplace=False)
            )
            # df[df["criterion"] == "quality"]["group"].replace(quality_map, inplace=True)
            # df[df["criterion"] == "ethnicity"]["group"].replace(ethnicity_map, inplace=True)

            # TODO limit decimal digits of acc
            # TODO save only some criteria of interest?

            # TODO generalise to other datasets
            keep_rows = ["all"] + list(eyepacs_categories.values())

        elif k == "camelyon":
            df["criterion"].replace(camelyon_categories, inplace=True)
            keep_rows = ["all"] + list(camelyon_categories.values())

        df = df.query(f"criterion in {keep_rows}")

        df["n"] = df["n"].astype(int)

        # out_csv = "../experiments/oct/debug_latex_table/out.csv"
        # df[["criterion", "group", "n", "acc"]].to_csv(out_csv, float_format="%.3f")
        df[["criterion", "group", "n", "acc"]].to_csv(out_csv[k], float_format="%.3f")

    pass


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Plot results", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--eval_dir",
        action="store",
        type=str,
        help="Results directory (will be created if it does not exist)",
        default="./experiments/jan23/eval",
    )
    args = parser.parse_args()

    # generate_latex_subgroup_table(args.eval_dir)

    test_types = ["muks", "c2st", "mmdd"]

    # #####################################################################################3
    # # MNIST

    # details = {
    #     "exp_type": "mnist",
    #     "subset_spec_column": "dataset_ds_q_subset_params_p_erase",
    #     "eval_string": "mnist_corrupt",
    # }

    # exp_type = "mnist"

    # exp_dirs = {ele: f"./experiments/oct/{exp_type}/{ele}" for ele in test_types}

    # # specific to camelyon
    # subset_spec_column = "dataset_ds_q_subset_params_p_erase"

    # plot_results(args.eval_dir, exp_dirs, subset_spec_column, "mnist_corrupt")

    # #####################################################################################3
    # # Camelyon

    # exp_type = "camelyon"

    # exp_dirs = {ele: f"./experiments/oct/{exp_type}/{ele}" for ele in test_types}
    # exp_dirs = {ele: f"./experiments/nov/{exp_type}/{ele}" for ele in test_types}

    # # specific to camelyon
    # subset_spec_column = "dataset_ds_q_subset_params_center"

    # plot_results(args.eval_dir, exp_dirs, subset_spec_column, "camelyon_hospitals")

    #####################################################################################3
    # Eyepacs

    # exp_type = "eyepacs_comorb"
    # # exp_dirs = {ele: f"./experiments/oct/{exp_type}/{ele}" for ele in test_types}
    # exp_dirs = {
    #     ele: f"/home/lkoch/mnt/slurm_work/exp_lightning/oct/{exp_type}/{ele}" for ele in test_types
    # }
    # subset_spec_column = "dataset_ds_q_subset_params_diagnoses_comorbidities"

    # plot_results(args.eval_dir, exp_dirs, subset_spec_column, "eyepacs_comorbidities")

    # exp_type = "eyepacs_ethnicity"
    # exp_dirs = {ele: f"./experiments/oct/{exp_type}/{ele}" for ele in test_types}
    # subset_spec_column = "dataset_ds_q_subset_params_patient_ethnicity"

    # plot_results(args.eval_dir, exp_dirs, subset_spec_column, "eyepacs_ethnicity")

    exp_type = "eyepacs_gender"
    # exp_dirs = {ele: f"./experiments/oct/{exp_type}/{ele}" for ele in test_types}
    exp_dirs = {
        ele: f"/home/lkoch/mnt/slurm_work/exp_lightning/oct/{exp_type}/{ele}" for ele in test_types
    }
    subset_spec_column = "dataset_ds_q_subset_params_patient_gender"

    plot_results(args.eval_dir, exp_dirs, subset_spec_column, "eyepacs_gender")

    # exp_type = "eyepacs_quality"
    # exp_dirs = {ele: f"./experiments/oct/{exp_type}/{ele}" for ele in test_types}
    # subset_spec_column = "dataset_ds_q_subset_params_session_image_quality"

    # plot_results(args.eval_dir, exp_dirs, subset_spec_column, "eyepacs_quality")

    print("done")
