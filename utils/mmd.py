"""
Helper functions for calculating MMD

Refactored implementations w.r.t. utils_HD.py

Copyright (c) 2022 Lisa Koch
"""

import torch


def _mmd2_and_variance_sutherland(K_XX, K_XY, K_YY, biased=False):
    """Implementaiton of variance estimator as in Sutherland 2017 ICRL

    Original code from https://github.com/djsutherland/opt-mmd/blob/master/two_sample/mmd.py,
    translated from theano to PyTorch, and removed diagonal option

    """

    m = K_XX.shape[0]  # Assumes X, Y are same shape

    ### Get the various sums of kernels that we'll use
    # Kts drop the diagonal, but we don't need to compute them explicitly

    diag_X = torch.diagonal(K_XX)
    diag_Y = torch.diagonal(K_YY)

    sum_diag_X = diag_X.sum()
    sum_diag_Y = diag_Y.sum()

    sum_diag2_X = torch.dot(diag_X, diag_X)
    sum_diag2_Y = torch.dot(diag_Y, diag_Y)

    Kt_XX_sums = K_XX.sum(axis=1) - diag_X
    Kt_YY_sums = K_YY.sum(axis=1) - diag_Y
    K_XY_sums_0 = K_XY.sum(axis=0)
    K_XY_sums_1 = K_XY.sum(axis=1)

    Kt_XX_sum = Kt_XX_sums.sum()
    Kt_YY_sum = Kt_YY_sums.sum()
    K_XY_sum = K_XY_sums_0.sum()

    # TODO: turn these into dot products?
    # should figure out if that's faster or not on GPU / with theano...
    Kt_XX_2_sum = (K_XX**2).sum() - sum_diag2_X
    Kt_YY_2_sum = (K_YY**2).sum() - sum_diag2_Y
    K_XY_2_sum = (K_XY**2).sum()

    if biased:
        mmd2 = (
            (Kt_XX_sum + sum_diag_X) / (m * m)
            + (Kt_YY_sum + sum_diag_Y) / (m * m)
            - 2 * K_XY_sum / (m * m)
        )
    else:
        mmd2 = Kt_XX_sum / (m * (m - 1)) + Kt_YY_sum / (m * (m - 1)) - 2 * K_XY_sum / (m * m)

    torch.dot(Kt_YY_sums, K_XY_sums_0)

    var_est = (
        2
        / (m**2 * (m - 1) ** 2)
        * (
            2 * torch.dot(Kt_XX_sums, Kt_XX_sums)
            - Kt_XX_2_sum
            + 2 * torch.dot(Kt_YY_sums, Kt_YY_sums)
            - Kt_YY_2_sum
        )
        - (4 * m - 6) / (m**3 * (m - 1) ** 3) * (Kt_XX_sum**2 + Kt_YY_sum**2)
        + 4
        * (m - 2)
        / (m**3 * (m - 1) ** 2)
        * (torch.dot(K_XY_sums_1, K_XY_sums_1) + torch.dot(K_XY_sums_0, K_XY_sums_0))
        - 4 * (m - 3) / (m**3 * (m - 1) ** 2) * K_XY_2_sum
        - (8 * m - 12) / (m**5 * (m - 1)) * K_XY_sum**2
        + 8
        / (m**3 * (m - 1))
        * (
            1 / m * (Kt_XX_sum + Kt_YY_sum) * K_XY_sum
            - torch.dot(Kt_XX_sums, K_XY_sums_1)
            - torch.dot(Kt_YY_sums, K_XY_sums_0)
        )
    )

    return mmd2, var_est


def _mmd2_and_variance(K_XX, K_XY, K_YY, biased=False):
    """Implementation of variance estimator as in Liu 2020 ICML"""
    m = K_XX.shape[0]  # Assumes X, Y are same shape

    hh = K_XX + K_YY - K_XY - K_XY.transpose(0, 1)
    hh_diag = torch.diagonal(hh)

    if biased:
        mmd2 = hh.sum() / (m * m)
    else:
        mmd2 = (hh.sum() - hh_diag.sum()) / (m * (m - 1))

    V1 = torch.dot(hh.sum(1) / m, hh.sum(1) / m) / m
    V2 = hh.sum() / m**2
    var_est = 4 * (V1 - V2**2)

    return mmd2, var_est, None


def mmd_and_var(f_x, f_y, x, y, f_sigma, sigma, epsilon=10**-10, biased=False, sutherland=False):

    K_xx = kernel_liu(f_x, f_x, x, x, f_sigma, sigma, epsilon)
    K_yy = kernel_liu(f_y, f_y, y, y, f_sigma, sigma, epsilon)
    K_xy = kernel_liu(f_x, f_y, x, y, f_sigma, sigma, epsilon)

    if sutherland:
        return _mmd2_and_variance_sutherland(K_xx, K_xy, K_yy, biased=biased)
    else:
        return _mmd2_and_variance(K_xx, K_xy, K_yy, biased=biased)


def kernel_liu(f_x, f_y, x, y, f_sigma, sigma, epsilon=10**-10):
    kernel = (1 - epsilon) * gaussian_kernel(f_x, f_y, f_sigma) * gaussian_kernel(
        x, y, sigma
    ) + epsilon * gaussian_kernel(x, y, sigma)
    return kernel


def gaussian_kernel(x, y, sigma):
    """Gaussian kernel. Note that sigma = (2*sig**2)"""
    ret_val = torch.exp(-pdist(x, y) ** 2 / sigma)
    return ret_val


def pdist(x, y):
    """calculate pairwise distances on tensors without batch dimension

    Args:
        x ([type]): tensor of shape [n, d]
        y ([type]): tensor of shape [m, d]

    Returns:
        [type]: tensor of shape [n, m]. normed distance!
    """

    # create batch dimension to use torch batched pairwise distance calculation
    b_x = x.view(1, *x.shape)
    b_y = y.view(1, *y.shape)
    b_dist = torch.cdist(b_x, b_y, p=2)

    # remove batch dimension
    dist = b_dist.view(*b_dist.shape[1:])

    return dist


if __name__ == "__main__":
    pass
