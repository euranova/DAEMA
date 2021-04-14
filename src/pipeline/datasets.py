""" Contains all the dataset creation and preprocessing. """

import operator
import os
import time
import traceback
import urllib.request
import zipfile
import io

import sklearn.datasets
import pandas as pd
import numpy as np
import unlzw3
from sklearn import preprocessing
from sklearn.preprocessing import OrdinalEncoder
from torch.utils.data import DataLoader

import pipeline.utils as U


def _get_archive_url(path):
    return os.path.join("https://archive.ics.uci.edu/ml/machine-learning-databases", path)


def _download_dataset(access, path_to):
    if not isinstance(access, dict):
        access = {"csv": access}
    string_ds = b""
    for type, link in access.items():
        if type == "sklearn":
            ds = getattr(sklearn.datasets, link)()
            df = pd.DataFrame(ds.data)
            df["target"] = pd.DataFrame(ds.target)
            string_ds += bytes(df.to_csv(index=False), encoding='utf-8')
        elif type in ["csv", "z"]:
            u = urllib.request.urlopen(link)
            string_ds += u.read() if type == "csv" else unlzw3.unlzw(u.read())
            u.close()
        else:
            raise NotImplementedError(
                "Unknown type entered: {}. Please use only 'csv', 'sklearn' or 'zip'.".format(type))
    with open(path_to, "wb") as f:
        f.write(string_ds)


def _create_mask(data, setting="mcar", method="uniform", thresh=0.2):
    """ Creates a random missingness mask following the parameters.

    :param data: pd.DataFrame; data to create a mask matrix for
    :param setting: 'mcar' or 'mnar'; type of missingness
    :param method: 'uniform' or 'random'; either to apply on all columns or only half of these
    :param thresh: Float; proportion of missingness in the samples having missing values
    :return: np.ndarray(Bool); a missingness-mask matrix
    """
    if method not in ['uniform', 'random']:
        raise ValueError("method should be in ['uniform', 'random'].")
    if setting not in ['mcar', 'mnar']:
        raise ValueError("mechanism should be in ['mcar', 'mnar'].")

    rows, cols = data.shape
    # uniform random vector
    v = np.random.uniform(size=(rows, cols))
    mask = v <= thresh  # mcar uniform mask, subsequently modified for the other settings

    # Should be put all in the if, but would change randomness (and prevent reproducibility).
    c = np.zeros(cols, dtype=bool)
    missing_cols = np.random.choice(cols, cols // 2, replace=False)
    c[missing_cols] = True
    if method == 'random':
        mask *= c

    if setting == "mnar":
        # unmask some values in an mnar fashion: a value can be missing only
        # if the two randomly selected features are above the mean.
        sample_cols = np.random.choice(cols, 2, replace=False)
        col_data1 = data.iloc[:, sample_cols[0]]
        m1 = col_data1 > np.median(col_data1, axis=0)
        col_data2 = data.iloc[:, sample_cols[1]]
        m2 = col_data2 < np.median(col_data2, axis=0)
        mask *= np.array((~m1 | ~m2))[:, np.newaxis]

    return mask


def _normalise(df, train_ids, mask, scaler='Standard'):
    if scaler not in ["Standard", "MinMax"]:
        raise NotImplementedError("scaler should be 'Standard' or 'MinMax', not {}.".format(scaler))
    scaler = preprocessing.StandardScaler() if scaler == "Standard" else preprocessing.MinMaxScaler()
    df2 = df.copy()
    df2[mask] = np.NaN
    scaler.fit(df2.loc[train_ids, :])
    return scaler.transform(df)


DATASETS = {
    "EEG": {"access": _get_archive_url("00264/EEG%20Eye%20State.arff"),
            "target": 14, "task": "classification",
            "csv_params": {"na_values": ["309231", "362564", "642564", "1030.77"],  # removing 4 extreme outliers
                           "skiprows": 19, "header": None}},

    "Glass": {"access": _get_archive_url("glass/glass.data"),
              "target": 10, "task": "classification",
              "csv_params": {"usecols": list(range(1, 11)), "header": None}},

    "Breast": {"access": _get_archive_url("breast-cancer-wisconsin/breast-cancer-wisconsin.data"),
               "target": 10, "task": "classification",
               "csv_params": {"usecols": list(range(1, 11)), "header": None, "na_values": "?"}},

    "Ionosphere": {"access": _get_archive_url("ionosphere/ionosphere.data"),
                   "target": 34, "task": "classification",
                   "csv_params": {"header": None}},

    "Shuttle": {"access": {"z": _get_archive_url("statlog/shuttle/shuttle.trn.Z"),
                           "csv": _get_archive_url("statlog/shuttle/shuttle.tst")},
                "target": 9, "task": "classification",
                "csv_params": {"header": None, "sep": " "}},

    "Boston": {"access": {"sklearn": "load_boston"},
               "target": "target", "task": "regression",
               "csv_params": {}},

    "CASP": {"access": _get_archive_url("00265/CASP.csv"),
             "target": "RMSD", "task": "regression",
             "csv_params": {}},
}


def get_dataset(name, scaler="Standard", ms_prop=0.2, ms_setting='mcar', ms_method='uniform', train_size=0.7, seed=0):
    U.fix_seed(seed)
    path = U.data_path("{}.csv".format(name))
    if not os.path.exists(path):
        _download_dataset(DATASETS[name]["access"], path_to=path)

    df = pd.read_csv(U.data_path("{}.csv".format(name)), **DATASETS[name]["csv_params"])
    df = df[~df.isna().any(axis=1)]

    target = df[[DATASETS[name]["target"]]].values
    df = df.drop(DATASETS[name]["target"], axis=1)
    if DATASETS[name]["task"] == "classification":
        encoder = OrdinalEncoder()
        target = encoder.fit_transform(target)

    mask = _create_mask(df, method=ms_method, setting=ms_setting, thresh=ms_prop)
    train_ids = np.random.permutation(len(df)) < len(df) * train_size
    perm = np.random.permutation(np.sum(train_ids))
    df[train_ids] = df[train_ids].iloc[perm, :].values
    target[train_ids] = target[train_ids][perm]
    #TODO: mask?

    df = _normalise(df.copy(), train_ids, mask, scaler=scaler)
    train_set = tuple((val[train_ids].astype(np.float32) for val in [df, mask, target]))
    test_set = tuple((val[~train_ids].astype(np.float32) for val in [df, mask, target]))

    return train_set, test_set, DATASETS[name]["task"] == "classification"


if __name__ == "__main__":
    for ds_name in DATASETS:
        train_set, test_set, classif = get_dataset(ds_name, seed=42)

        dl = DataLoader(list(zip(*train_set)), batch_size=5)
        for i, (data, missing_data, mask) in enumerate(dl):
            print(data, missing_data, mask, sep="\n")
            if i == 1:
                break
        dl = DataLoader(list(zip(*test_set)), batch_size=5)
        for i, (data, missing_data, mask) in enumerate(dl):
            print(data, missing_data, mask, sep="\n")
            if i == 1:
                break

        print(test_set[2][:5])
        print(train_set[2][:5])
