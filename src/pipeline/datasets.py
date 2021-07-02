""" Contains all the dataset creation and preprocessing parts. """

import os
import urllib.request

import sklearn.datasets
import pandas as pd
import numpy as np
import unlzw3
from sklearn import preprocessing
from sklearn.preprocessing import OrdinalEncoder
from torch.utils.data import DataLoader

import pipeline.utils as U


def _get_archive_url(path):
    """ Helper to get the UCI path

    :param path: String; end of the path
    :return: String; complete path
    """
    return os.path.join("https://archive.ics.uci.edu/ml/machine-learning-databases", path)


def _download_dataset(access, path_to):
    """ Downloads the datasets and saves it.

    :param access: dict["csv" or "z" or "sklearn", String] or String; path(s) to access the data,
        keys being the format of the data.
    :param path_to: String; path to save the data
    """
    if not isinstance(access, dict):
        access = {"csv": access}
    string_ds = b""
    for type_, link in access.items():
        if type_ == "sklearn":
            ds = getattr(sklearn.datasets, link)()
            df = pd.DataFrame(ds.data)
            df["target"] = pd.DataFrame(ds.target)
            string_ds += bytes(df.to_csv(index=False), encoding='utf-8')
        elif type_ in ["csv", "z"]:
            u = urllib.request.urlopen(link)
            string_ds += u.read() if type_ == "csv" else unlzw3.unlzw(u.read())
            u.close()
        else:
            raise NotImplementedError(
                f"Unknown type entered: {type_}. Please use only 'csv', 'sklearn' or 'zip'.")
    with open(path_to, "wb") as f:
        f.write(string_ds)


def _create_mask(samples, setting="mcar", method="uniform", thresh=0.2):
    """ Creates a random missingness mask following the parameters.

    :param samples: pd.DataFrame; samples to create a mask matrix for
    :param setting: 'mcar' or 'mnar'; type of missingness
    :param method: 'uniform' or 'random'; either to apply on all columns or only half of these
    :param thresh: Float; proportion of missingness in the samples having missing values
    :return: np.ndarray(Bool); a missingness-mask matrix
    """
    if method not in ['uniform', 'random']:
        raise ValueError("method should be in ['uniform', 'random'].")
    if setting not in ['mcar', 'mnar']:
        raise ValueError("mechanism should be in ['mcar', 'mnar'].")

    rows, cols = samples.shape
    # uniform random vector
    v = np.random.uniform(size=(rows, cols))
    mask = v <= thresh  # mcar uniform mask, subsequently modified for the other settings

    # It could be all inside the if clause, but it would not reproduce the paper results as faithfully
    c = np.zeros(cols, dtype=bool)
    missing_cols = np.random.choice(cols, cols // 2, replace=False)
    c[missing_cols] = True
    if method == 'random':
        mask *= c

    if setting == "mnar":
        # unmask some values in an mnar fashion: a value can be missing only
        # if the first and second randomly selected features are respectively below and above the median value.
        sample_cols = np.random.choice(cols, 2, replace=False)
        col1 = samples.iloc[:, sample_cols[0]]
        m1 = col1 > np.median(col1, axis=0)
        col2 = samples.iloc[:, sample_cols[1]]
        m2 = col2 < np.median(col2, axis=0)
        mask *= np.array((~m1 | ~m2))[:, np.newaxis]

    return mask


def _normalise(samples, mask, train_ids, scaler='Standard'):
    """ Normalises the samples based on the available values of the train set.

    :param samples: np.ndarray(Float); samples to normalise
    :param mask: np.ndarray(Bool); corresponding masks
    :param train_ids: np.ndarray(Bool); training indices (True for included)
    :param scaler: "Standard" or "MinMax"; scaler to use
    :return: np.ndarray(Float); normalised samples
    """
    if scaler not in ["Standard", "MinMax"]:
        raise NotImplementedError(f"scaler should be 'Standard' or 'MinMax', not {scaler}.")
    scaler = preprocessing.StandardScaler() if scaler == "Standard" else preprocessing.MinMaxScaler()
    fit_samples = samples.copy()
    fit_samples[mask] = np.NaN
    scaler.fit(fit_samples.loc[train_ids, :])
    return scaler.transform(samples)


DATASETS = {
    # "Name": {"access": <see _download_dataset documentation>,
    # "target": <target-column number or name>, "task": <"classification" or "regression">,
    # "csv_params: <dict with the params to be given to the pd.read_csv function>}

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
    """ Downloads and returns the preprocessed dataset.

    :param name: String; name of the dataset (has to be in DATASETS)
    :param scaler: "Standard" or "MinMax"; scaler to use
    :param ms_prop: Float; proportion of missingness in the samples having missing values
    :param ms_setting: 'mcar' or 'mnar'; type of missingness
    :param ms_method: 'uniform' or 'random'; either to apply on all columns or only half of these
    :param train_size: Float in [0, 1]; proportion of training samples
    :param seed: Integer; seed to use for the preprocessing steps
    :return:
        - (np.ndarray(Float), np.ndarray(Bool), np.ndarray(Float)); (train_samples, train_masks, train_targets)
        - (np.ndarray(Float), np.ndarray(Bool), np.ndarray(Float)); (test_samples, test_masks, test_targets)
        - Bool; True if it is a classification dataset
    """
    U.fix_seed(seed)
    path = os.path.join(U.DATA_PATH, f"{name}.csv")
    if not os.path.exists(path):
        _download_dataset(DATASETS[name]["access"], path_to=path)

    df = pd.read_csv(path, **DATASETS[name]["csv_params"])
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
    if ms_setting == "mnar":  # It could be done for mcar, but it would not reproduce the paper results as faithfully
        mask[train_ids] = mask[train_ids][perm]

    df = _normalise(df.copy(), mask, train_ids, scaler=scaler)
    train_set = tuple((val[train_ids].astype(np.float32) for val in [df, mask, target]))
    test_set = tuple((val[~train_ids].astype(np.float32) for val in [df, mask, target]))

    return train_set, test_set, DATASETS[name]["task"] == "classification"


def test():
    for ds_name in DATASETS:
        train_set, test_set, _ = get_dataset(ds_name, seed=42)

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


if __name__ == "__main__":
    test()
