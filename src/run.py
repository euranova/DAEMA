""" Entry point to run the experiments """

import os

import numpy as np
import torch

import pipeline.utils as U
from pipeline import argument_parser, datasets, metrics
from models import MODELS


def get_preds(model, samples, masks, seed):
    """ Computes the predictions of the model for the given samples without modifying the random state.

    :param model: Any; model to use for the imputation
    :param samples: np.ndarray(Float); samples to impute
    :param masks: np.ndarray(Float); corresponding mask matrix
    :param seed: Integer; seed to use
    :return: np.ndarray(Float); imputed samples
    """
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    U.fix_seed(seed)

    ret = model.test(samples, masks)

    np.random.set_state(np_state)
    torch.set_rng_state(torch_state)

    return ret


def main(args):
    """ Entry point of the program; prints results.

    :param args: ArgumentParser; arguments of the program
    """
    seeds = ([(dataset_seed, seed) for dataset_seed in args.dataset_seeds for seed in args.seeds]
             if len(args.dataset_seeds) > 0 else
             [(seed, seed) for seed in args.seeds])

    for dataset in args.datasets:
        res = {metric: [] for metric in args.metrics}
        for dataset_seed, seed in seeds:
            train_set, test_set, classif = datasets.get_dataset(
                dataset, scaler=args.scaler, ms_prop=args.ms_prop, ms_setting=args.ms_setting,
                ms_method=args.ms_method, train_size=args.train_size, seed=dataset_seed)
            train_samples, train_masks, train_targets = train_set
            train_samples_missing = train_samples * (1 - train_masks) if args.model != "Real" else train_samples
            test_samples, test_masks, test_targets = test_set
            test_samples_missing = test_samples * (1 - test_masks) if args.model != "Real" else test_samples

            U.fix_seed(seed)
            model = MODELS[args.model](train_samples_missing, train_masks, args)
            for step in model.train_generator(train_samples_missing, train_masks, args):
                if step in args.metric_steps:
                    imputed = get_preds(model, test_samples_missing, test_masks, seed=step)
                    if "NRMS" in args.metrics:
                        res["NRMS"].append(metrics.nrms(test_samples, imputed, test_masks))
                    if "RF" in args.metrics:
                        imputed_train = get_preds(model, train_samples_missing, train_masks, seed=step)
                        res["RF"].append(
                            metrics.dml_metric(imputed_train, imputed, train_targets, test_targets, classif))
            print(f"Seed {dataset_seed}.{seed} (dataset_seed, model_seed) results for the {dataset} dataset"
                  f"and the {args.model} model:")
            for metric, vals in res.items():
                print(f"{metric}: Mean {np.mean(res[metric][-len(args.metric_steps):])} "
                      f"from values {res[metric][-len(args.metric_steps):]}")
        print(f"Final results for the {dataset} dataset and the {args.model} model:")
        for metric, vals in res.items():
            print(f"{metric}: Mean {np.mean(vals)}, Std {np.std(vals, ddof=1)}")


if __name__ == "__main__":
    main(argument_parser.get_args())
