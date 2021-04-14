""" Contains all the metrics. """
from copy import deepcopy

import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score


def nrms(real_data, imputed_data, mask=None, sigma2=None):
    """ Measures the NRMS scores.

    :param real_data: np.array; ground_truth
    :param imputed_data: np.array; imputation to be evaluated
    :param mask: np.array; mask of the missing data
    :param sigma2: variances of columns for NRMS computation (default to real_data variances)
    """
    if sigma2 is None:
        sigma2 = np.var(real_data, axis=0, ddof=1)
    if mask is None:
        mask = np.ones(real_data.shape)
    assert np.all(sigma2 >= 0), sigma2
    nrms_cols = [i for i in range(real_data.shape[1]) if sigma2[i] != 0]
    return np.sqrt(np.average((real_data[:, nrms_cols] - imputed_data[:, nrms_cols]) ** 2
                              / sigma2[np.newaxis, nrms_cols], weights=mask[:, nrms_cols]))


def dml_metric(imputed_data_train, imputed_data_test, target_train, target_test, classif, measures=10):
    score = 0
    metric = accuracy_score if classif else nrms

    scaler = preprocessing.StandardScaler()
    imputed_data_train = scaler.fit_transform(imputed_data_train)
    imputed_data_test = scaler.transform(imputed_data_test)

    for seed in range(measures):
        model = (RandomForestClassifier(n_estimators=100, max_leaf_nodes=1000, random_state=seed)
                 if classif else
                 RandomForestRegressor(n_estimators=100, max_leaf_nodes=1000, random_state=seed))
        model.fit(imputed_data_train, target_train[:, 0])
        pred = np.reshape(model.predict(imputed_data_test), (-1, 1))
        score += metric(target_test, pred)
    return score / measures
