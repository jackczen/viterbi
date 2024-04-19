import logging

import click
import numpy as np
import pytest

from src.cli.types import MATRIX_TYPE, INT_LIST_TYPE
from src.viterbi.hidden_markov_model import HiddenMarkovModel
from src.viterbi.viterbi_algorithm import viterbi

_logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO)


@click.group(help="Solves a Hidden Markov Model using Viterbi's algorithm.")
def cli():
    _configure_logging()


@cli.command(help="Solves HMM models using Viterbi's algorithm.")
@click.argument(
    "mat",
    type=MATRIX_TYPE,
)
@click.argument(
    "sampling",
    type=MATRIX_TYPE,
)
@click.argument(
    "sequence",
    type=INT_LIST_TYPE,
)
def solve(mat, sampling, sequence):
    hmm = HiddenMarkovModel(
        transition_matrix=np.array(mat, dtype=np.longdouble),
        sampling_probabilities=np.array(sampling, dtype=np.longdouble),
    )
    res = viterbi(hmm, sequence)

    _logger.info(f"Most likely path: {res}")


@cli.command(help="Runs unit tests for Viterbi's algorithm and CLI.")
def unit():
    pytest.main(['-x', 'tests'])


if __name__ == "__main__":
    cli()
