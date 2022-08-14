"""Optimizes hyperparameters using Bayesian optimization."""

from chemprop.args import HyperoptArgs
from chemprop.hyperparameter_optimization import hyperopt

if __name__ == "__main__":
    hyperopt(args=HyperoptArgs().parse_args())
