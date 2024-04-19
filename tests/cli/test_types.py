from typing import Any, Sequence

import numpy as np
import pytest
from click import BadParameter

from src.cli.types import MATRIX_TYPE, INT_LIST_TYPE


@pytest.mark.parametrize(
    "value,expected_result",
    [
        pytest.param(
            "[ [0., 0],  [1.000000, 2.200]  ]",
            np.array([[0.0, 0.0], [1.0, 2.2]], dtype=np.longdouble),
        ),
        pytest.param(
            np.array([[0.0, 1.2, 3.4], [5.8, 20.0, 21.2]], dtype=np.float32),
            np.array([[0.0, 1.2, 3.4], [5.8, 20.0, 21.2]], dtype=np.longdouble),
        ),
        pytest.param(
            np.array([[0.0, 1.2, 3.4], [5.8, 20.0, 21.2]], dtype=np.longdouble),
            np.array([[0.0, 1.2, 3.4], [5.8, 20.0, 21.2]], dtype=np.longdouble),
        ),
        pytest.param(
            "[ [0. , - 0.3]   ,  [1.4 , 2.200] , [21 , 45]  ]",
            np.array([[0.0, -0.3], [1.4, 2.2], [21.0, 45.0]], dtype=np.longdouble),
        ),
        pytest.param("[]", None, marks=pytest.mark.xfail(raises=BadParameter)),
        pytest.param("[[0.0],]", None, marks=pytest.mark.xfail(raises=BadParameter)),
        pytest.param("[[0.0], []]", None, marks=pytest.mark.xfail(raises=BadParameter)),
        pytest.param(
            "[[0.0], [1.0, 3.2]]", None, marks=pytest.mark.xfail(raises=BadParameter)
        ),
        pytest.param(
            "[[0.0], [1.0]", None, marks=pytest.mark.xfail(raises=BadParameter)
        ),
        pytest.param(
            "[[[0.0]], [1.0]]", None, marks=pytest.mark.xfail(raises=BadParameter)
        ),
        pytest.param(
            "[[.1], [1.0]]", None, marks=pytest.mark.xfail(raises=BadParameter)
        ),
        pytest.param(
            "[[False], [2.13]]", None, marks=pytest.mark.xfail(raises=BadParameter)
        ),
    ],
)
def test_matrix_convert(
    value: Any, expected_result: np.ndarray[Any, np.dtype[np.longdouble]]
) -> None:
    actual = MATRIX_TYPE.convert(value, None, None)

    assert np.allclose(expected_result, actual)


@pytest.mark.parametrize(
    "value,expected_result",
    [
        pytest.param([], []),
        pytest.param([1], [1]),
        pytest.param([1, 2, -41, 213], [1, 2, -41, 213]),
        pytest.param([1.2], [1]),
        pytest.param([1, False], [1, 0]),
        pytest.param([1, [1, 23]], None, marks=pytest.mark.xfail(raises=BadParameter)),
        pytest.param("[]", []),
        pytest.param("[  1, 2,   33, -67 ]", [1, 2, 33, -67]),
        pytest.param(
            "[  1, 2,   33, -67", None, marks=pytest.mark.xfail(raises=BadParameter)
        ),
        pytest.param(
            "[  1, 2,   33, -67", None, marks=pytest.mark.xfail(raises=BadParameter)
        ),
        pytest.param(
            "[  1.2131, 2,   33, -67]",
            None,
            marks=pytest.mark.xfail(raises=BadParameter),
        ),
        pytest.param(
            "[  1, 2,   ---33, -67]", None, marks=pytest.mark.xfail(raises=BadParameter)
        ),
    ],
)
def test_list_convert(value: Any, expected_result: Sequence[int]) -> None:
    actual = INT_LIST_TYPE.convert(value, None, None)

    assert expected_result == actual
