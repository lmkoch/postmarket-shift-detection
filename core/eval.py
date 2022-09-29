import copy
import os

import pandas as pd
from utils.helpers import stderr_proportion

from core.dataset import dataset_fn
from core.model import DomainClassifier, MaxKernel, TaskClassifier


def eval(
    trainer,
    model,
    ckpt_path,
    params,
    split="test",  # FIXME need additional logic with trainer.validate
    sample_sizes=[10, 30, 50, 100, 500],
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

    log_dir = trainer.logger.log_dir
    out_csv = os.path.join(log_dir, f"{split}_consistency_analysis.csv")

    df = pd.DataFrame(columns=["sample_size", "power", "type_1err", "method"])

    from core.model import DataModule

    if isinstance(model, TaskClassifier) or isinstance(model, DomainClassifier):

        model.test_sample_sizes = sample_sizes

        # dataloader_length is the number of samples (with replacement) that will be drawn all-together.
        # From these, samples are drawn for repeated testing.
        dataloader_length = 20000  # maybe set an appropriate number :)
        dataloader = dataset_fn(
            params_dict=params["dataset"],
            replacement=True,
            num_samples=dataloader_length,
        )

        dataloader = dataset_fn(params_dict=params["dataset"])
        trainer.test(model=model, ckpt_path=ckpt_path, datamodule=DataModule(dataloader))
        return

    elif isinstance(model, MaxKernel):

        # This could at some point be refactored:
        # At the moment, the testing is implemented differently from C2ST/MUKS,
        # samples for hypothesis testing are actual minibatches, so need to get different dataloaders
        # for every sample_size in consistency analysis

        for batch_size in sample_sizes:

            params["dataset"]["dl"]["batch_size"] = batch_size

            num_repeated_tests = 100
            dataloader_length = batch_size * num_repeated_tests
            dataloader = dataset_fn(
                params_dict=params["dataset"],
                replacement=True,
                num_samples=dataloader_length,
            )

            res = trainer.test(
                model=model, ckpt_path=ckpt_path, datamodule=DataModule(dataloader)
            )[0]
            reject_rate = res["test/power"]

            # MMD-D lightning model cannot natively calculate type 1 error - need to
            # calculate power on same distribution (P=Q)

            type_1_err_params = copy.deepcopy(params)
            type_1_err_params["dataset"]["ds"]["q"] = type_1_err_params["dataset"]["ds"]["p"]

            dataloader = dataset_fn(
                params_dict=type_1_err_params["dataset"],
                replacement=True,
                num_samples=dataloader_length,
            )

            res = trainer.test(
                model=model, ckpt_path=ckpt_path, datamodule=DataModule(dataloader)
            )[0]

            type_1_err = res["test/power"]

            res = {
                "sample_size": batch_size,
                "power": reject_rate,
                "type_1err": type_1_err,
            }

            df = df.append(pd.DataFrame(res, index=[""]), ignore_index=True)

            print(res)

        df["power_stderr"] = stderr_proportion(df["power"], df["sample_size"].astype(float))
        df["type_1err_stderr"] = stderr_proportion(
            df["type_1err"], df["sample_size"].astype(float)
        )

        df.to_csv(out_csv)

    else:
        raise ValueError(f"Unsupported model: {trainer.model}")
