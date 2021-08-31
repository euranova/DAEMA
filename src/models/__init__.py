""" Contains all the models that can be used to impute missing data. """

from .daema import Daema
from .holoclean import Holoclean
from .mida import MIDA
from .miss_forest import MissForestImpute
from .baseline_imputations import MeanImputation, Identity

MODELS = {
    "DAEMA": Daema,
    "Holoclean": Holoclean,
    "MIDA": MIDA,
    "MissForest": MissForestImpute,
    "Mean": MeanImputation,
    "Real": Identity,  # Not a proper imputation algorithm, handled separately in the run.py file
}
