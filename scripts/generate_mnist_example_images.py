import os

import matplotlib.pyplot as plt
import numpy as np
import torchvision
from core.dataset import dataset_fn
from utils.config import load_config

# # MNIST

# config_file = "./config/lightning/mnist.yaml"
# params = load_config(config_file)

# out_dir = "/home/lkoch/Dropbox/BerensLab/Papers/2022_tmi_subgroups/figures"

# for p in [0, 0.5, 1]:

#     params["dataset"]["ds"]["p"]["subset_params"]["p_erase"] = p
#     dataloader = dataset_fn(params_dict=params["dataset"])

#     _, (x, *_) = next(enumerate(dataloader["train"]["p"]))

#     img_grid = torchvision.utils.make_grid(x, normalize=True)
#     img_grid = np.transpose(img_grid, (1, 2, 0)).numpy()

#     out_fig = os.path.join(out_dir, f"mnist_noise_prob_{p}.png")
#     plt.imshow(img_grid)
#     plt.imsave(out_fig, img_grid)

# Camelyon

config_file = "./config/lightning/camelyon.yaml"
params = load_config(config_file)

# out_dir = "/home/lkoch/Dropbox/BerensLab/Papers/2022_tmi_subgroups/figures"
out_dir = "."
for center in [[ele] for ele in range(5)]:

    params["dataset"]["ds"]["q"]["subset_params"]["center"] = center
    dataloader = dataset_fn(params_dict=params["dataset"])

    _, (x, *_) = next(enumerate(dataloader["train"]["p"]))
    _, (y, *_) = next(enumerate(dataloader["train"]["q"]))

    img_grid = torchvision.utils.make_grid(x, normalize=True)
    img_grid = np.transpose(img_grid, (1, 2, 0)).numpy()
    out_fig = os.path.join(out_dir, f"camelyon_p.png")
    plt.imshow(img_grid)
    plt.imsave(out_fig, img_grid)

    img_grid = torchvision.utils.make_grid(y, normalize=True)
    img_grid = np.transpose(img_grid, (1, 2, 0)).numpy()
    out_fig = os.path.join(out_dir, f"camelyon_{center}.png")
    plt.imshow(img_grid)
    plt.imsave(out_fig, img_grid)
