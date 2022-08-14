"""Trains a model on a dataset."""

from chemprop.args import TrainArgs
from chemprop.train import cross_validate
from chemprop.train import run_training


if __name__ == '__main__':
    cross_validate(args=TrainArgs().parse_args(), train_func=run_training)
