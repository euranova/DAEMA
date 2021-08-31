""" Handles the program arguments (default values, doc, ...). """

import argparse

from models import MODELS
from pipeline.datasets import DATASETS


def get_args(args):
    """ Creates the parser and returns it.

    :param args: List[String]; args to parse
    :return: ArgumentParser; arguments of the program
    """
    parser = argparse.ArgumentParser(description='Runs the experiments described in the DAEMA paper.')

    ds_settings = parser.add_argument_group('Dataset settings')
    ds_settings.add_argument('--datasets', choices=list(DATASETS), nargs='+', default=list(DATASETS),
                             help="Datasets to use")
    ds_settings.add_argument('--dataset_seeds', nargs='*', type=int, default=[],
                             help="Seeds to use to produce the datasets. Leave it empty to use the same as "
                                  "for the model. Else, runs as many experiments as entered seeds.")
    ds_settings.add_argument('--scaler', choices=["Standard", "MinMax"], default="Standard",
                             help="Scaler to use")
    ds_settings.add_argument('--ms_prop', type=float, default=0.2,
                             help="Proportion of missingness in the selected rows in [0, 1]")
    ds_settings.add_argument('--ms_setting', choices=["mcar", "mnar"], default="mcar",
                             help="Missingness setting to use")
    ds_settings.add_argument('--ms_method', choices=["uniform", "random"], default="uniform",
                             help="Missingness method to use")
    ds_settings.add_argument('--train_size', type=float, default=0.7,
                             help="Proportion of training samples in [0, 1]")

    common_model_settings = parser.add_argument_group("Common model initialisation and training settings")
    common_model_settings.add_argument('--seeds', nargs='+', type=int, default=list(range(10)),
                                       help="Seeds to use to initialise and train the model. "
                                            "Runs as many experiments as entered seeds.")
    common_model_settings.add_argument('--model', choices=list(MODELS),
                                       default="DAEMA", help="Model to use")
    common_model_settings.add_argument('--metric_steps', type=int, nargs='+',
                                       default=[39200, 39400, 39600, 39800, 40000],
                                       help="Steps after which to measure the metrics")
    common_model_settings.add_argument('--batch_size', type=int, default=64,
                                       help="Batchsize for the training")
    common_model_settings.add_argument('--lr', type=float, default=0.001,
                                       help="Learning rate for the training")

    daema_args = parser.add_argument_group("DAEMA-specific arguments")
    daema_args.add_argument('--daema_pre_drop', type=float, default=0.2,
                            help="Artificial missingness rate for the DAEMA algorithm in [0, 1]")
    daema_args.add_argument('--daema_loss_type', choices=["classic", "full", "dropout_only"], default="classic",
                            help="Loss to use")
    daema_args.add_argument('--daema_mask_input', choices=["FC", "ELEMENTWISE", None], default="FC",
                            help="Type of input to the feature encoder")
    daema_args.add_argument('--daema_ways', type=int, default=None,
                            help="Number of ways to compute each latent feature")
    daema_args.add_argument('--daema_feats', type=int, default=None,
                            help="Number of latent features to compute")
    daema_args.add_argument('--daema_attention_mode', choices=["classic", "full", "sep", "no"], default="full",
                            help="Attention type to use")
    daema_args.add_argument('--daema_activation', choices=["Tanh", "Sigmoid", None], default=None,
                            help="Activation to use for the final layer")

    holoclean_args = parser.add_argument_group("Holoclean-specific arguments")
    holoclean_args.add_argument('--holoclean_dropout', type=float, default=0,
                                help="Dropout rate for the Holoclean algorithm in [0, 1]")
    holoclean_args.add_argument('--holoclean_embedding_size', type=int, default=64,
                                help="Latent space dimension for the Holoclean algorithm")

    mide_args = parser.add_argument_group("MIDA-specific arguments")
    mide_args.add_argument('--mida_theta', type=int, default=7,
                           help="Theta to use for the MIDA architecture")
    mide_args.add_argument('--mida_depth', type=int, default=3,
                           help="Depth to use for the MIDA architecture")

    missforest_args = parser.add_argument_group("MissForest-specific arguments")
    missforest_args.add_argument('--mf_n_estimators', type=int, default=100,
                                 help="Number of estimators to use for the MissForest algorithm")
    missforest_args.add_argument('--mf_max_leaf_nodes', type=int, default=None,
                                 help="Maximum number of leaf nodes per estimator for the MissForest algorithm")
    missforest_args.add_argument('--mf_max_iter', type=int, default=10,
                                 help="Maximum number of iterations for the MissForest algorithm")

    metric_settings = parser.add_argument_group("Metric settings")
    metric_settings.add_argument('--metrics', nargs='+', choices=["NRMS", "RF"], default=["NRMS", "RF"],
                                 help="Metrics to use")
    return parser.parse_args(args)
