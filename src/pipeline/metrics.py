""" Contains all the metrics. """

import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score


def nrms(ground_truth_samples, imputed_samples, masks=None, sigma2=None):
    """ Computes the NRMS score measured only on missing values.

    :param ground_truth_samples: np.ndarray(Float); ground_truth samples
    :param imputed_samples: np.ndarray(Float); imputed samples to be evaluated
    :param masks: np.ndarray(Bool); corresponding mask matrix
    :param sigma2: variances of columns for NRMS computation (default to real_data variances)
    :return: Float; nrms of the imputation
    """
    if sigma2 is None:
        sigma2 = np.var(ground_truth_samples, axis=0, ddof=1)
    if masks is None:
        masks = np.ones(ground_truth_samples.shape)
    assert np.all(sigma2 >= 0), sigma2
    nrms_cols = [i for i in range(ground_truth_samples.shape[1]) if sigma2[i] != 0]
    return np.sqrt(np.average((ground_truth_samples[:, nrms_cols] - imputed_samples[:, nrms_cols]) ** 2
                              / sigma2[np.newaxis, nrms_cols], weights=masks[:, nrms_cols]))


def dml_metric(imputed_samples_train, imputed_samples_test, targets_train, targets_test, classif, measures=10):
    """ Computes the downstream machine-learning metric (NRMS for regression tasks, Accuracy for classification tasks)
    using random forests.

    :param imputed_samples_train: np.ndarray(Float); imputed train-set samples
    :param imputed_samples_test: np.ndarray(Float); imputed test-set samples
    :param targets_train: np.ndarray(Float); targets of the train set
    :param targets_test: np.ndarray(Float); targets of the test set
    :param classif: Bool; True for a classification task
    :param measures: Integer; number of random forests to run for more precision
    :return: Float; downstream result using the imputation
    """
    score = 0
    metric = accuracy_score if classif else nrms

    scaler = preprocessing.StandardScaler()
    imputed_data_train = scaler.fit_transform(imputed_samples_train)
    imputed_data_test = scaler.transform(imputed_samples_test)

    for seed in range(measures):
        model = (RandomForestClassifier(n_estimators=100, max_leaf_nodes=1000, random_state=seed)
                 if classif else
                 RandomForestRegressor(n_estimators=100, max_leaf_nodes=1000, random_state=seed))
        model.fit(imputed_data_train, targets_train[:, 0])
        pred = np.reshape(model.predict(imputed_data_test), (-1, 1))
        score += metric(targets_test, pred)
    return score / measures
