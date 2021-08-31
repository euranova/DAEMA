""" Tests the code. """

from torch.utils.data import DataLoader

from models import MODELS
from pipeline import argument_parser
from pipeline.datasets import DATASETS, get_dataset
from run import main


def test_datasets():
    """ Tests all the datasets defined in pipeline.datasets.DATASETS. """
    for ds_name in DATASETS:
        train_set, test_set, _ = get_dataset(ds_name, seed=42)

        for set_ in (train_set, test_set):
            dl = DataLoader(list(zip(*set_)), batch_size=5)
            for data, missing_data, mask in dl:
                assert len(data) == 5, f"The {ds_name} dataset has less than 5 samples."
                assert data.shape[1] > 1, f"The {ds_name} dataset has none or one column only."
                print("data:", data, "missing_data:", missing_data, "mask:", mask, sep="\n")
                break


def test_general(capsys):
    """ Tests most of the code by checking it produces the expected result. """
    main(argument_parser.get_args(["--metric_steps", "50", "--datasets", "Boston", "--seeds", "0", "1"]))
    captured = capsys.readouterr()
    with open("tests/current_output.txt", "w") as f:
        assert f.write(captured.out)
    with open("tests/gold_output.txt", "r") as f:
        assert captured.out == f.read()


def test_models():
    """ Tests all the models (only checks if these run). """
    for model in MODELS:
        main(argument_parser.get_args(["--model", model, "--metric_steps", "0", "1", "5", "--datasets", "Boston",
                                       "--seeds", "0"]))
