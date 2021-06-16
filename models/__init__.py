"""Contains all the models that can be used to impute missing datas."""

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
    "Real": Identity,  # Can't be used as is for a RealImputation (see run.py code).
}
