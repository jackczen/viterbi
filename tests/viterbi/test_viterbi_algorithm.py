from typing import Sequence

import numpy as np
import pytest

from src.viterbi.hidden_markov_model import HiddenMarkovModel
from src.viterbi.viterbi_algorithm import viterbi


@pytest.mark.parametrize(
    "hmm,sequence,expected_result",
    [
        pytest.param(
            HiddenMarkovModel(
                transition_matrix=np.array(
                    [[0.8, 0.2], [0.5, 0.5]], dtype=np.longdouble
                ),
                # Encoding Red -> 0, Black -> 1
                sampling_probabilities=np.array(
                    [[0.4, 0.6], [0.7, 0.3]], dtype=np.longdouble
                ),
            ),
            [0, 0, 1, 0],  # Red, Red, Black, Red
            [0, 0, 0, 0],  # Bag 0, Bag 0, Bag 0, Bag 0
        ),
        pytest.param(
            HiddenMarkovModel(
                transition_matrix=np.array(
                    [[0.8, 0.2], [0.5, 0.5]], dtype=np.longdouble
                ),
                # Encoding Red -> 0, Black -> 1
                sampling_probabilities=np.array(
                    [[0.4, 0.6], [0.7, 0.3]], dtype=np.longdouble
                ),
            ),
            [0, 0, 2, 0],  # Red, Red, ???, Red
            None,
            marks=pytest.mark.xfail(raises=IndexError),
        ),
    ],
)
def test_viterbi(
    hmm: HiddenMarkovModel,
    sequence: Sequence[int],
    expected_result: Sequence[int],
) -> None:
    actual = viterbi(hmm, sequence)

    assert expected_result == actual
