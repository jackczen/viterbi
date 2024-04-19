from typing import Any

import pytest

import numpy as np

from src.viterbi.hidden_markov_model import HiddenMarkovModel


@pytest.mark.parametrize(
    "transition_matrix,sampling_probabilities",
    [
        pytest.param(
            np.array([[0.8, 0.2], [0.5, 0.5]], dtype=np.longdouble),
            np.array([[0.4, 0.6], [0.7, 0.3]], dtype=np.longdouble),
        ),
        pytest.param(
            np.array([[0.8, 0.1], [0.5, 0.5]], dtype=np.longdouble),
            np.array([[0.4, 0.6], [0.7, 0.3]], dtype=np.longdouble),
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        pytest.param(
            np.array([[0.8, 0.2], [0.5, 0.5]], dtype=np.longdouble),
            np.array([[0.4, 0.6], [0.7, 0.2]], dtype=np.longdouble),
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        pytest.param(
            np.array([[0.8, 0.2], [0.5, 0.5], [0.2, 0.8]], dtype=np.longdouble),
            np.array([[0.4, 0.6], [0.7, 0.2]], dtype=np.longdouble),
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        pytest.param(
            np.array([[0.8, 0.1, 0.1], [0.5, 0.4, 0.1]], dtype=np.longdouble),
            np.array([[0.4, 0.6], [0.7, 0.2]], dtype=np.longdouble),
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        pytest.param(
            np.array([[0.8, 0.1, 0.1], [0.5, 0.4, 0.1]], dtype=np.longdouble),
            np.array([[0.4, 0.6]], dtype=np.longdouble),
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        pytest.param(
            np.array([[0.8, -0.4, 0.6], [0.5, 0.4, 0.1]], dtype=np.longdouble),
            np.array([[0.4, 0.6], [0.3, 0.7]], dtype=np.longdouble),
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        pytest.param(
            np.array([[0.8, 0.1, 0.1], [0.5, 0.4, 0.1]], dtype=np.longdouble),
            np.array([[0.4, 0.6], [1.3, -0.3]], dtype=np.longdouble),
            marks=pytest.mark.xfail(raises=ValueError),
        ),
    ],
)
def test_hmm_constructor(
    transition_matrix: np.ndarray[Any, np.dtype[np.longdouble]],
    sampling_probabilities: np.ndarray[Any, np.dtype[np.longdouble]],
) -> None:
    HiddenMarkovModel(
        transition_matrix=transition_matrix,
        sampling_probabilities=sampling_probabilities,
    )


@pytest.mark.parametrize(
    "hmm,expected_result",
    [
        pytest.param(
            HiddenMarkovModel(
                transition_matrix=np.array(
                    [[0.8, 0.2], [0.5, 0.5]], dtype=np.longdouble
                ),
                sampling_probabilities=np.array(
                    [[0.4, 0.6], [0.7, 0.3]], dtype=np.longdouble
                ),
            ),
            np.array([0.7142857143, 0.2857142857], dtype=np.longdouble),
        ),
        pytest.param(
            HiddenMarkovModel(
                transition_matrix=np.array(
                    [[1.0, 0.0], [0.0, 1.0]], dtype=np.longdouble
                ),
                sampling_probabilities=np.array(
                    [[0.4, 0.6], [0.7, 0.3]], dtype=np.longdouble
                ),
            ),
            None,
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        pytest.param(
            HiddenMarkovModel(
                transition_matrix=np.array(
                    [[0.4, 0.4, 0.2], [0.2, 0.4, 0.4], [0.1, 0.3, 0.6]],
                    dtype=np.longdouble,
                ),
                sampling_probabilities=np.array(
                    [[1.0], [1.0], [1.0]], dtype=np.longdouble
                ),
            ),
            np.array([0.1935483871, 0.3548387097, 0.4516129032]),
        ),
    ],
)
def test_stationary_probabilities(
    hmm: HiddenMarkovModel, expected_result: np.ndarray[Any, np.dtype[np.longdouble]]
) -> None:
    actual = hmm.steady_state_probabilities()

    assert expected_result.all() == actual.all()
