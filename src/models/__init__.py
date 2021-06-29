""" Contains all the models that can be used to impute missing data. """

from models.daema import Daema
from models.holoclean import Holoclean
from models.mida import MIDA
from models.miss_forest import MissForestImpute
from models.baseline_imputations import MeanImputation, Identity

MODELS = {
    "DAEMA": Daema,
    "Holoclean": Holoclean,
    "MIDA": MIDA,
    "MissForest": MissForestImpute,
    "Mean": MeanImputation,
    "Real": Identity,  # Not a proper imputation algorithm, handled separately in the run.py file
}
