from __future__ import annotations

from typing import Any

from attrs import frozen, field, Attribute
import numpy as np
from scipy import linalg


def valid_tranisition_matrix(
    hmm: HiddenMarkovModel,
    attribute: Attribute,
    mat: np.ndarray[Any, np.dtype[np.longdouble]],
) -> None:
    """Validates that the provided matrix is both square and stochastic.

    Args:
        hmm: the parent Hidden Markov Model.
        attribute: the attrs attribute for mat.
        mat: the transition matrix field of hmm.

    Raises:
         ValueError: if mat is either
            (i) not two-dimensional;
            (ii) not square;
            (iii) not non-negative;
            (iv) not stochastic.
    """
    shape = mat.shape
    if len(shape) != 2:
        raise ValueError("Matrix is not two-dimensions!")
    if shape[0] != shape[1]:
        raise ValueError("Matrix is not square!")

    for entry in mat.flatten():
        if entry < 0.0:
            raise ValueError("Rows include negative probabilities!")

    for rowSum in mat.sum(axis=1):
        if rowSum - 1.0 > 1e-5:
            raise ValueError("Matrix is not stochastic!")


def valid_sampling_probabilities(
    hmm: HiddenMarkovModel,
    attribute: Attribute,
    sampling_probabilities: np.ndarray[Any, np.dtype[np.longdouble]],
) -> None:
    """Validates that each conditional distribution is a valid probability
    distribution and that there is a conditional distribution for each state
    in the transition matrix.

    Args:
        hmm: the parent Hidden Markov Model.
        attribute: the attrs attribute for mat.
        sampling_probabilities: the sampling_probabilities field of hmm.

    Raises:
         ValueError: if sampling_probabilities
            (i) does not have size (n,m), where n is the dimension of
                hmm.transition_matrix;
            (ii) does not have non-negative rows (distributions on marbles);
            (iii) does not have "rows" (distributions on marbles) that all
                sum to 1.0.
    """
    mat_shape = hmm.transition_matrix.shape
    sampling_shape = sampling_probabilities.shape

    if len(mat_shape) != len(sampling_shape):
        raise ValueError("Does not provide a distribution for each state!")

    for entry in sampling_probabilities.flatten():
        if entry < 0.0:
            raise ValueError("Distributions on marbles include negative probabilities!")

    for rowSum in sampling_probabilities.sum(axis=1):
        if rowSum - 1.0 > 1e-5:
            raise ValueError("Not a valid distribution!")


@frozen()
class HiddenMarkovModel:
    """Represents a hidden Markov model (HMM).

    Attributes:
        transition_matrix: The transition matrix describing the probabilities
            of transitioning between different "bags."

            This matrix must be square and stochastic (all rows rum to 1.0),
            and T[i][j] will be interpreted as the probability of
            transitioning from bag i to bag j.

        sampling_probabilities: The probabilities of drawing each kind of
            marble from each bag.

            This array must have shape (n, m), where n is the number of "bags"
            in the HMM and m is the number of "marbles" in the HMM. S[i] will
            be interpreted as the distribution of sampling marbles, given we
            are in bag i, and S[i][j] will be interpreted as the probability
            of drawing marble j, given we are in bag i. The entries of S[i]
            must sum to 1.0 for every bag i.
    """

    transition_matrix: np.ndarray[Any, np.dtype[np.longdouble]] = field(
        validator=valid_tranisition_matrix
    )
    sampling_probabilities: np.ndarray[Any, np.dtype[np.longdouble]] = field(
        validator=valid_sampling_probabilities
    )

    def steady_state_probabilities(self) -> np.ndarray[Any, np.dtype[np.longdouble]]:
        """Computes the steady-state probabilities of this Hidden Markov Model.

        By definition, the steady-state probability vector w satisfies:

        wT = w

        This is equivalent to being a left eigenvector with eigenvalue 1. So,
        we can find this vector with linear algebra via:

        ker((T - I)^T) = { a*w }

        for some scalar a. We then can normalize a*w to find w.

        Returns:
            The steady-state probability vector for this HMM's transition
            matrix.
        """
        n = len(self.transition_matrix)

        kernel_matrix = (self.transition_matrix - np.identity(n)).transpose()
        eigen_basis = linalg.null_space(kernel_matrix)

        # Because this HMM's transition matrix is stochastic, 1 is an
        # eigenvalue. However, it still is not guaranteed to have a unique
        # steady-state distribution, so we must check. Otherwise, we can't
        # return a steady-state distribution that makes sense.
        geometric_multiplicity = eigen_basis.shape[1]

        if geometric_multiplicity > 1:
            raise ValueError(
                "HMM does not have a unique stationary distribution!\n"
                f"Eigen-basis for lambda=1: {eigen_basis}"
            )

        steady_state_vector = eigen_basis.flatten()

        # Before returning, we need to normalize the distribution:
        return steady_state_vector / steady_state_vector.sum()
