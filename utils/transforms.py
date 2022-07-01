from typing import Dict, List

import torch
from munch import munchify
from packaging import version
from torchvision import transforms


def data_transforms(basic_preproc_config: Dict, augmentations: List, augmentation_config: Dict):
    """compose augmentation and preprocessing transformations

    Args:
        basic_preproc_config (List): configuration of basic preprocessing (resize, normalize)
        augmentations (List): list of operations
        augmentation_config (Dict): detailed augmentation parameters

    Returns:
        _type_: train and test transforms
    """

    # Args:
    #     cfg (Dict): config dictionary (munchified or original dict)

    # Returns:
    #     _type_: Train and test pytorch transforms

    basic_preproc_config = munchify(basic_preproc_config)
    augmentation_config = munchify(augmentation_config)

    operations = {
        "random_crop": random_apply(
            transforms.RandomResizedCrop(
                size=(basic_preproc_config.img_size, basic_preproc_config.img_size),
                scale=augmentation_config.random_crop.scale,
                ratio=augmentation_config.random_crop.ratio,
            ),
            p=augmentation_config.random_crop.prob,
        ),
        "horizontal_flip": transforms.RandomHorizontalFlip(
            p=augmentation_config.horizontal_flip.prob
        ),
        "vertical_flip": transforms.RandomVerticalFlip(p=augmentation_config.vertical_flip.prob),
        "color_distortion": random_apply(
            transforms.ColorJitter(
                brightness=augmentation_config.color_distortion.brightness,
                contrast=augmentation_config.color_distortion.contrast,
                saturation=augmentation_config.color_distortion.saturation,
                hue=augmentation_config.color_distortion.hue,
            ),
            p=augmentation_config.color_distortion.prob,
        ),
        "rotation": random_apply(
            transforms.RandomRotation(
                degrees=augmentation_config.rotation.degrees, fill=augmentation_config.value_fill
            ),
            p=augmentation_config.rotation.prob,
        ),
        "translation": random_apply(
            transforms.RandomAffine(
                degrees=0,
                translate=augmentation_config.translation.range,
                fillcolor=augmentation_config.value_fill,
            ),
            p=augmentation_config.translation.prob,
        ),
        "grayscale": transforms.RandomGrayscale(p=augmentation_config.grayscale.prob),
    }

    if version.parse(torch.__version__) >= version.parse("1.7.1"):
        operations["gaussian_blur"] = random_apply(
            transforms.GaussianBlur(
                kernel_size=augmentation_config.gaussian_blur.kernel_size,
                sigma=augmentation_config.gaussian_blur.sigma,
            ),
            p=augmentation_config.gaussian_blur.prob,
        )

    augmentation_ops = []
    for op in augmentations:
        if op not in operations:
            raise NotImplementedError(
                "Not implemented data augmentation operations: {}".format(op)
            )
        augmentation_ops.append(operations[op])

    normalization_1 = [
        transforms.Resize(basic_preproc_config.img_size),
        transforms.ToTensor(),
        transforms.CenterCrop(basic_preproc_config.img_size),
    ]
    normalization_2 = [
        transforms.Normalize(basic_preproc_config.mean, basic_preproc_config.std),
    ]

    train_preprocess = transforms.Compose([*normalization_1, *augmentation_ops, *normalization_2])
    test_preprocess = transforms.Compose([*normalization_1, *normalization_2])

    return train_preprocess, test_preprocess


def random_apply(op, p):
    return transforms.RandomApply([op], p=p)
