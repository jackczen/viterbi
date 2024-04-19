import logging
from typing import Sequence

import numpy as np

from src.viterbi.hidden_markov_model import HiddenMarkovModel

_logger = logging.getLogger(__name__)


def viterbi(hmm: HiddenMarkovModel, sequence: Sequence[int]) -> Sequence[int]:
    """Runs Viterbi's algorithm for determining the most likely sequences of
    "bags" for a given sequence of "marbles" for some Hidden Markov Model.

    Args:
        hmm: A Hidden Markov Model.
        sequence: A sequence of marbles to draw. Each marble in the sequence
            must be a valid sample in the HMM. For instance, if our HMM only
            contains sampling probabilities for marbles 0, 1, and 2, this
            sequence cannot contain marble 3.

    Returns:
         The most likely sequence of bags visited, given the observed sequence
         of marbles.

    Raises:
        IndexError: if the provided sequences contains some marble that is not
            represented in the HMM's sampling probabilities.
    """
    if not sequence:
        raise ValueError("Cannot provide empty sequence of events!")

    # Compute the initial probability vector of Pr(A) * Pr(E | A) for each
    # "bag" A in the HMM and for the first "marble" E in the provided sequence.
    # This new vector is computed efficiently (and maintaining the integrity of
    # 128-bit floats) via the pair-wise vector computation Pr(.) * Pr(first marble | .):
    first_marble = sequence[0]
    probabilities = (
        hmm.steady_state_probabilities()
        * hmm.sampling_probabilities.transpose()[first_marble]
    )
    _logger.info(f"Vector after marble 1: {probabilities}")

    num_bags = len(hmm.transition_matrix)

    # Each time we observe a new marble and compute the most-likely parent
    # for each bag, we should keep track of these most-likely parents. This
    # will allow us to reconstruct the most likely path from the most likely
    # bag at the end.
    parents_by_marble = []

    for i, marble in enumerate(sequence[1:]):
        new_probabilities = np.zeros(num_bags, dtype=np.longdouble)
        new_parents = []

        # We must find the most likely way to add each bag.
        for bag in range(num_bags):
            # We need to keep track of the most likely parent, and the
            # respective probability of being in that parent state,
            # transitioning to our current bag.
            max_parent_probability = np.longdouble(0.0)
            most_likely_parent = 0  # Placeholder

            for parent_candidate, parent_probability in enumerate(probabilities):
                # Find Pr(parent so far) * T(parent -> bag)
                p = parent_probability * hmm.transition_matrix[parent_candidate][bag]

                if p > max_parent_probability:
                    max_parent_probability = p
                    most_likely_parent = parent_candidate

            # Now that we know that most likely parent, we multiply the
            # probability of observing the current marble in our current bag,
            # i.e., Pr(parent so far) * T(parent -> bag) * Pr(marble | bag).
            p = max_parent_probability * hmm.sampling_probabilities[bag][marble]
            new_probabilities[bag] = p
            new_parents.append(most_likely_parent)
            _logger.info(f"Selected {most_likely_parent} -> {bag} with p={p}")

        probabilities = new_probabilities
        parents_by_marble.append(new_parents)
        _logger.info(f"Vector after marble {i + 2}: {new_probabilities}")

    # Now, we find the most likely end bag and reconstruct the path that was
    # taken to reach it.
    most_likely_end_bag = int(np.argmax(probabilities))

    rev_path = [most_likely_end_bag]
    last_visited_bag = most_likely_end_bag
    for parent_array in reversed(parents_by_marble):
        parent = parent_array[last_visited_bag]

        rev_path.append(parent)
        last_visited_bag = parent

    # We re-constructed the path beginning with the final bag and ending with
    # the first bag, so we must finally reverse our path before returning.
    rev_path.reverse()
    return rev_path
