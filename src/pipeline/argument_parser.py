""" Contains the argument parser definition. """

import argparse
from models.daema import Generator


def get_args():
    parser = argparse.ArgumentParser(description='Runs the experiments described in the paper DAEMA.')

    # Dataset settings
    datasets = ["EEG", "Glass", "Breast", "Ionosphere", "Shuttle", "Boston", "CASP"]
    parser.add_argument('--datasets', choices=datasets, nargs='+', default=datasets, help="Datasets to use")
    parser.add_argument('--dataset_seeds', nargs='*', type=int, default=[],
                        help="Seeds to use to produce the datasets. Leave it empty to use the same as for the model. "
                             "Else, runs as many experiments as entered seeds.")
    parser.add_argument('--scaler', choices=["Standard", "MinMax"], default="Standard",
                        help="Scaler to use")
    parser.add_argument('--ms_prop', type=float, default=0.2,
                        help="Proportion of missingness in the selected rows in [0, 1]")
    parser.add_argument('--ms_setting', choices=["mcar", "mnar"], default="mcar",
                        help="Missingness setting to use")
    parser.add_argument('--ms_method', choices=["uniform", "random"], default="uniform",
                        help="Missingness method to use")
    parser.add_argument('--train_size', type=float, default=0.7,
                        help="Proportion of training samples in [0, 1]")


    # model initialisation and training settings
    parser.add_argument('--seeds', nargs='+', type=int, default=list(range(10)),
                        help="Seeds to use to initialise and train the model. "
                             "Runs as many experiments as entered seeds.")
    parser.add_argument('--model', choices=["DAEMA", "Holoclean", "MIDA", "MissForest", "Mean", "Real"],
                        default="DAEMA", help="Model to use")
    parser.add_argument('--metric_steps', type=int, nargs='+', default=[39200, 39400, 39600, 39800, 40000],
                        help="Steps after which to measure the metrics")
    parser.add_argument('--batch_size', type=int, default=64,
                        help="Batchsize for the training")
    parser.add_argument('--lr', type=float, default=0.001,
                        help="Learning rate for the training")
    # DAEMA
    parser.add_argument('--daema_pre_drop', type=float, default=0.2,
                        help="Artificial missingness rate for the DAEMA algorithm in [0, 1]")
    parser.add_argument('--daema_loss_type', choices=["classic", "full", "dropout_only"], default="classic",
                        help="Loss to use")
    parser.add_argument('--daema_mask_input', choices=["FC", "ELEMENTWISE", None], default="FC",
                        help="Type of input to the feature encoder")
    parser.add_argument('--daema_ways', type=int, default=None,
                        help="Number of ways to compute each latent feature")
    parser.add_argument('--daema_feats', type=int, default=None,
                        help="Number of latent features to compute")
    parser.add_argument('--daema_attention_mode', choices=["classic", "full", "sep", "no"], default="full",
                        help="Attention type to use")
    parser.add_argument('--daema_activation', choices=["Tanh", "Sigmoid", None], default=None,
                        help="Activation to use for the final layer")
    # Holoclean
    parser.add_argument('--holoclean_dropout', type=float, default=0,
                        help="Dropout rate for the Holoclean algorithm in [0, 1]")
    parser.add_argument('--holoclean_k', type=int, default=64,
                        help="Latent space dimension for the Holoclean algorithm")
    # MIDA
    parser.add_argument('--mida_theta', type=int, default=9,
                        help="Theta to use for the MIDA architecture")
    parser.add_argument('--mida_depth', type=int, default=3,
                        help="Depth to use for the MIDA architecture")
    # MissForest
    parser.add_argument('--mf_n_estimators', type=int, default=100,
                        help="Number of estimators to use for the MissForest algorithm")
    parser.add_argument('--mf_max_leaf_nodes', type=int, default=None,
                        help="Maximum number of leaf nodes per estimator for the MissForest algorithm")
    parser.add_argument('--mf_max_iter', type=int, default=10,
                        help="Maximum number of iterations for the MissForest algorithm")


    # metric settings
    parser.add_argument('--metrics', type=str, nargs='+', default=["NRMS", "RF"],
                        help="Metrics to use in {EEG, Glass, Breast, Ionosphere, Shuttle, Boston, CASP}")
    return parser.parse_args()
