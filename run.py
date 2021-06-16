""" Entry point to run the experiments """

import numpy as np
import torch

import pipeline.utils as U
from pipeline import argument_parser, datasets, metrics
from models import MODELS


def get_preds(model, dataset, mask, step):
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    U.fix_seed(step)

    ret = model.test(dataset, mask)

    np.random.set_state(np_state)
    torch.set_rng_state(torch_state)

    return ret


def main(args):
    seeds = ([(dataset_seed, seed) for dataset_seed in args.dataset_seeds for seed in args.seeds]
             if len(args.dataset_seeds) > 0 else
             [(seed, seed) for seed in args.seeds])

    for dataset in args.datasets:
        res = {metric: [] for metric in args.metrics}
        for dataset_seed, seed in seeds:
            train_set, test_set, classif = datasets.get_dataset(dataset, scaler=args.scaler, ms_prop=args.ms_prop,
                                                                ms_setting=args.ms_setting, ms_method=args.ms_method,
                                                                train_size=args.train_size, seed=dataset_seed)
            train_dataset, train_mask, train_target = train_set
            train_dataset_missing = train_dataset * (1 - train_mask) if args.model != "Real" else train_dataset
            test_dataset, test_mask, test_target = test_set
            test_dataset_missing = test_dataset * (1 - test_mask) if args.model != "Real" else test_dataset

            U.fix_seed(seed)
            model = MODELS[args.model](train_dataset_missing, train_mask, args)
            for step in model.train_generator(train_dataset_missing, train_mask, args):
                if step in args.metric_steps:
                    imputed = get_preds(model, test_dataset_missing, test_mask, step)
                    if "NRMS" in args.metrics:
                        res["NRMS"].append(metrics.nrms(test_dataset, imputed, test_mask))
                    if "RF" in args.metrics:
                        imputed_train = get_preds(model, train_dataset_missing, train_mask, step)
                        res["RF"].append(metrics.dml_metric(imputed_train, imputed, train_target, test_target, classif))
            print("Seed {}.{} (dataset_seed, model_seed) results for the {} dataset and the {} model:".format(
                dataset_seed, seed, dataset, args.model))
            for metric, vals in res.items():
                print("{}: Mean {} from values {}".format(
                    metric, np.mean(res[metric][-len(args.metric_steps):]), res[metric][-len(args.metric_steps):]))
        print("Final results for the {} dataset and the {} model:".format(dataset, args.model))
        for metric, vals in res.items():
            print("{}: Mean {}, Std {}".format(metric, np.mean(res[metric]), np.std(res[metric], ddof=1)))


if __name__ == "__main__":
    main(argument_parser.get_args())
