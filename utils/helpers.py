import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torchvision


def set_rcParams():

    sns.set_context("paper")
    sns.set_style("whitegrid")
    sns.color_palette("dark")

    plt.rcParams["axes.linewidth"] = 0.7
    plt.rcParams["xtick.major.width"] = 0.5
    plt.rcParams["ytick.major.width"] = 0.5
    plt.rcParams["xtick.minor.width"] = 0.5
    plt.rcParams["ytick.minor.width"] = 0.5
    plt.rcParams["xtick.major.size"] = 2
    plt.rcParams["ytick.major.size"] = 2
    plt.rcParams["xtick.minor.size"] = 1
    plt.rcParams["ytick.minor.size"] = 1
    plt.rcParams["font.size"] = 8
    plt.rcParams["axes.titlesize"] = 10
    plt.rcParams["axes.labelsize"] = 10
    plt.rcParams["legend.fontsize"] = 6.5
    plt.rcParams["xtick.labelsize"] = 7
    plt.rcParams["ytick.labelsize"] = 7
    plt.rcParams["figure.facecolor"] = (0.5, 0.5, 0.5, 0.5)  # only affects the notebook
    plt.rcParams["savefig.facecolor"] = (1, 1, 1, 0)
    plt.rcParams["savefig.dpi"] = 600
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["figure.dpi"] = 120  # only affects the notebook

    # plt.rcParams['text.usetex'] = True
    # plt.rcParams['text.latex.preamble'] = r'\usepackage{amsfonts}'
    # plt.rcParams['font.family'] = 'sans-serif'
    # plt.rcParams['font.sans-serif'] = ["Helvetica"]


def flatten_dict(nested_dict, sep="_"):
    """Flattens nested config dict.
       Example: d = {'a': 1,
                     'c': {'a': 2, 'b': {'x': 5, 'y' : 10}},
                     'd': [1, 2, 3]}

                is flattened to:
                flat_d = {'a': 1, 'd': [1, 2, 3], 'c.a': 2, 'c.b.x': 5, 'c.b.y': 10}

    Args:
        nested_dict ([type]): [description]
        sep (str, optional): [description]. Defaults to '.'.

    Returns:
        [type]: [description]
    """

    df = pd.json_normalize(nested_dict, sep=sep)

    return df.to_dict(orient="records")[0]


def make_image_grid(dataloader, num_images):

    img_count = 0
    batches = []
    for _, (x, _) in enumerate(dataloader):

        batches.append(x)
        img_count += len(x)

        if img_count >= num_images:
            break

    images = torch.cat(batches, 0)

    img_grid = torchvision.utils.make_grid(images[:num_images], normalize=True)
    matplotlib_imshow(img_grid, one_channel=True)
    return img_grid


def matplotlib_imshow(img, one_channel=False):
    """helper function to show an image. From:
    https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html

    """
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


def matplotlib_roccurve(fpr_tpr_tuples, labels, points=None, point_labels=None):
    """helper function to plot a roc curve

    fpr_tpr_tuples: list of tuples [(fpr, tpr), (fpr, tpr)]
    """

    if not len(fpr_tpr_tuples) == len(labels):
        raise ValueError("both inputs must have same length.")

    if points is not None:
        if not len(points) == len(point_labels):
            raise ValueError("both inputs must have same length.")

    fig, ax = plt.subplots()

    for ii in range(len(fpr_tpr_tuples)):
        plt.plot(fpr_tpr_tuples[ii][0], fpr_tpr_tuples[ii][1], label=labels[ii])

    if points is not None:
        for ii in range(len(points)):
            plt.plot(points[ii][0], points[ii][1], "o", label=point_labels[ii])

    plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend(loc="lower right")
    ax.set_aspect("equal")

    return fig


def get_attribute(dataset, class_variable):

    if hasattr(dataset, class_variable):
        return getattr(dataset, class_variable)

    # TODO: does not work for nested subsets
    elif hasattr(dataset, "dataset"):

        attr = getattr(dataset.dataset, class_variable)
        return attr[dataset.indices]


def class_proportions(dataset, class_variable="targets"):

    class_var = get_attribute(dataset, class_variable)

    n_classes = torch.max(class_var) + 1
    y = class_var.view(-1, 1)

    targets_onehot = (y == torch.arange(n_classes).reshape(1, n_classes)).float()
    proportions = torch.div(torch.sum(targets_onehot, dim=0), targets_onehot.shape[0])
    return proportions


def balanced_weights(dataset, rebalance_weights=None, balance_variable="targets"):

    class_weights = class_proportions(dataset, class_variable=balance_variable)

    if rebalance_weights is not None:
        class_weights = [
            rebalance_weights[idx] / (val + 0.01) for idx, val in enumerate(class_weights)
        ]

    class_var = get_attribute(dataset, balance_variable)

    weights = [0] * len(dataset)
    for idx, val in enumerate(class_var):
        weights[idx] = class_weights[val]

    return weights


def stat_C2ST(sample_a, sample_b):
    """callable to calculate statistic from two samples for C2ST-S

    Args:
        sample_a (_type_): binary (C2ST-S) or softmax (C2ST-L) predictions for sample a
        sample_b (_type_): binary (C2ST-S) or softmax (C2ST-L) predictions for sample b
    """

    stat = abs(sample_a.mean() - sample_b.mean())

    return stat


def quadratic_weighted_kappa(conf_mat):
    assert conf_mat.shape[0] == conf_mat.shape[1]
    cate_num = conf_mat.shape[0]

    # Quadratic weighted matrix
    weighted_matrix = np.zeros((cate_num, cate_num))
    for i in range(cate_num):
        for j in range(cate_num):
            weighted_matrix[i][j] = 1 - float(((i - j) ** 2) / ((cate_num - 1) ** 2))

    # Expected matrix
    ground_truth_count = np.sum(conf_mat, axis=1)
    pred_count = np.sum(conf_mat, axis=0)
    expected_matrix = np.outer(ground_truth_count, pred_count)

    # Normalization
    conf_mat = conf_mat / conf_mat.sum()
    expected_matrix = expected_matrix / expected_matrix.sum()

    observed = (conf_mat * weighted_matrix).sum()
    expected = (expected_matrix * weighted_matrix).sum()
    return (observed - expected) / (1 - expected)
