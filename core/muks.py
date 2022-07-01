import logging

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import ks_2samp, permutation_test
from utils.helpers import stat_C2ST
from utils.utils_HD import TST_MMD_u

# FIXME c2st, mmdd, muks should be in some sort of "power" module or class,
# not under "muks"


def mass_ks_test(x, y):
    """mass-univariate two-sample kolmogorov-smirnov test

    Args:
        x ([type]): 1D numpy array
        y ([type]): 1D numpy array
    """

    pvals = []
    num_dims = x.shape[1]
    for dim in range(num_dims):
        _, p_val = ks_2samp(x[:, dim], y[:, dim])
        pvals.append(p_val)

    corrected_pval = np.min(np.array(pvals)) * num_dims

    return np.minimum(corrected_pval, 1.0)


def c2st(dataloader_p, dataloader_q, num_repetitions, model, alpha=0.05, num_permutations=100):
    """Evaluation of MMD-D test

    Returns:
        power and type 1 err
    """

    """Evaluation that does not dependn on dataset specifics

    Args:
        test_loader (_type_): _description_
    """

    model.load_best_val_results_and_checkpoint()

    num_reps = num_repetitions

    power = 0
    type_1_err = 0

    for _ in range(num_reps):

        num_items = dataloader_p.batch_size
        x, _ = model.forward_pass(dataloader_p, num_items=num_items, keep_images=False)
        x2, _ = model.forward_pass(dataloader_p, num_items=num_items, keep_images=False)
        y, _ = model.forward_pass(dataloader_q, num_items=num_items, keep_images=False)

        res_power = permutation_test((x["witness"], y["witness"]), stat_C2ST)
        res_type_1_err = permutation_test((x["witness"], x2["witness"]), stat_C2ST)

        power += res_power.pvalue < alpha
        type_1_err += res_type_1_err.pvalue < alpha

    return power / num_reps, type_1_err / num_reps


def mmdd(dataloader_p, dataloader_q, num_repetitions, model, alpha=0.05, num_permutations=100):
    """Evaluation of MMD-D test

    Returns:
        power and type 1 err
    """

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    model = model.to(device)

    running_rejects = 0.0
    running_rejects_h0 = 0.0

    for idx in range(num_repetitions):
        _, (x_p, _) = next(enumerate(dataloader_p))
        _, (x_p2, _) = next(enumerate(dataloader_p))
        _, (x_q, _) = next(enumerate(dataloader_q))

        x_p, x_q, x_p2 = x_p.to(device), x_q.to(device), x_p2.to(device)

        ep = model.ep
        sigma = model.sigma_sq
        sigma0_u = model.sigma0_sq

        # power
        S = torch.cat([x_p, x_q], 0)
        batch_size = x_p.shape[0]
        dtype = x_p.dtype
        Sv = S.view(2 * batch_size, -1)

        h_u, _, _ = TST_MMD_u(
            model(S),
            num_permutations,
            batch_size,
            Sv,
            sigma,
            sigma0_u,
            ep,
            alpha,
            device,
            dtype,
        )

        # type I error
        S = torch.cat([x_p, x_p2], 0)
        Sv = S.view(2 * batch_size, -1)
        h_u_h0, _, _ = TST_MMD_u(
            model(S),
            num_permutations,
            batch_size,
            Sv,
            sigma,
            sigma0_u,
            ep,
            alpha,
            device,
            dtype,
        )

        # Gather results
        running_rejects += h_u
        running_rejects_h0 += h_u_h0

    reject_rate = running_rejects / (idx + 1)
    reject_rate_h0 = running_rejects_h0 / (idx + 1)

    return reject_rate, reject_rate_h0


def muks(dataloader_p, dataloader_q, num_repetitions, model, alpha=0.05, num_permutations=None):
    """Evaluation of MUKS test

    Returns:
        power and type 1 err
    """

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    model = model.to(device)

    sm = nn.Softmax(dim=1)

    count_rejects = 0
    count_rejects_h0 = 0

    model.eval()
    with torch.no_grad():
        for idx in range(num_repetitions):

            _, (x_p, _) = next(enumerate(dataloader_p))
            _, (x_p2, _) = next(enumerate(dataloader_p))
            _, (x_q, _) = next(enumerate(dataloader_q))

            x_p, x_q, x_p2 = x_p.to(device), x_q.to(device), x_p2.to(device)

            outputs = model(x_p)
            softmax_p = sm(outputs)

            outputs_p2 = model(x_p2)
            softmax_p2 = sm(outputs_p2)

            outputs_q = model(x_q)
            softmax_q = sm(outputs_q)

            pval = mass_ks_test(softmax_p.cpu().numpy(), softmax_q.cpu().numpy())
            pval0 = mass_ks_test(softmax_p.cpu().numpy(), softmax_p2.cpu().numpy())

            count_rejects += pval < alpha
            count_rejects_h0 += pval0 < alpha

    reject_rate = count_rejects / (idx + 1)
    reject_rate_h0 = count_rejects_h0 / (idx + 1)

    return reject_rate, reject_rate_h0


if __name__ == "__main__":
    pass
