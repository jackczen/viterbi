from typing import Any, Optional, Sequence

import click
import numpy as np
import re
from click import Parameter, Context

# Regular expressions for parsing command line arguments.
NUMBER_PATTERN = r"\-?\d+(?:\.\d*)?"
ONE_DIMENSIONAL_ARRAY_PATTERN = rf"\[(?:{NUMBER_PATTERN},)*{NUMBER_PATTERN}]"
TWO_DIMENSIONAL_ARRAY_PATTERN = (
    rf"\[(?:{ONE_DIMENSIONAL_ARRAY_PATTERN},)*{ONE_DIMENSIONAL_ARRAY_PATTERN}]"
)

INTEGER_PATTERN = r"\-?\d+"
INT_LIST_PATTERN = rf"(?:\[(?:{INTEGER_PATTERN},)*{NUMBER_PATTERN}])|(?:\[])"


class MatrixType(click.ParamType):
    """Represents a matrix type parameter. Validates and converts values
    from the command line into a two-dimensional NDArray (matrix)."""

    def _parse_string(
        self, value_string: str
    ) -> np.ndarray[Any, np.dtype[np.longdouble]]:
        # Delete all spaces and check if the string has a two-dimensional
        # array form using regular expressions.
        clean_string = value_string.replace(" ", "")
        match = re.fullmatch(TWO_DIMENSIONAL_ARRAY_PATTERN, clean_string)

        if match is None:
            self.fail("Provided string does not represent a 2D-array!")

        # Now that we know our array is a two-dimensional array, we can parse
        # it.
        row_strings = re.findall(ONE_DIMENSIONAL_ARRAY_PATTERN, clean_string)

        matrix = [
            [float(n) for n in re.findall(NUMBER_PATTERN, row_string)]
            for row_string in row_strings
        ]

        # If the rows of our new matrix do not all have the same size, then
        # numpy will throw an exception when we attempt to construct an array.
        try:
            return np.array(matrix, dtype=np.longdouble)
        except ValueError as e:
            self.fail(str(e))

    def convert(
        self, value: Any, param: Optional[Parameter], ctx: Optional[Context]
    ) -> np.ndarray[Any, np.dtype[np.longdouble]]:
        """Converts a value to a matrix. This value must already be a
        matrix or be a string in 2D-list format.

        We require that all strings provided follow the Python syntax for
        explicitly constructing non-empty two-dimensional lists of
        ints/floats. Moreover, the lengths of nested lists must be equal.

        Valid examples include:
            [ [1.2, 0.] , [3.66666, 2]   ]

            [ [ 1.2] , [ 3.66666], [4 ]  ]

            [[-0.3,0.],[4.2,-0.003]]

        Invalid examples include:
            [ [1.2, 0.] , [3.66666]   ]   # Not all rows have the same dim!

            [ [1.2, 0.] , [3.66666, 4.5]  # Missing opening '[' or closing ']'

            [ [1.2 23] [12, 72.]]         # Missing commas

            []                            # Matrices must be non-empty!

            [[1.2, 3.5], []]              # All rows must be non-empty!

            [[1.2, 3.5], [1.0]]           # All rows must have same lengths!

            [[.2, 0.5]]                   # Require '0.2' instead of '.2'

        Args:
            value: The value to convert.
            param: The parameter that is using this type to convert its value.
                May be ``None``.
            ctx: The current context that arrived at this value. May be
                ``None``.

        Returns:
             The provided value as a two-dimensional NDArray with data type
             float64.

        Raises:
            BadParameter: if the provided value is neither a 2D-NDArray nor a
                well-formatted string for a 2D-NDArray.
        """
        if isinstance(value, np.ndarray):
            if len(value.shape) != 2:
                self.fail("Provided array is not two-dimensional!")

            return np.array(value, dtype=np.longdouble)
        elif isinstance(value, str):
            return self._parse_string(value)
        else:
            self.fail(f"Unsupported data type for value: {value}")


class IntListType(click.ParamType):
    """Represents a list of integers type parameter. Validates and converts
    values from the command line into a one-dimensional list."""

    def _parse_list(self, value_list: list) -> Sequence[int]:
        try:
            return [int(e) for e in value_list]
        except Exception as e:
            self.fail(str(e))

    def _parse_string(self, value_string: str) -> Sequence[int]:
        # Delete all spaces and check if the string has a one-dimensional
        # array form using regular expressions.
        clean_string = value_string.replace(" ", "")
        match = re.fullmatch(INT_LIST_PATTERN, clean_string)

        if match is None:
            self.fail("Provided string does not represent a 2D-array!")

        # Now that we know our array is a one-dimensional array, we can parse
        # it.
        element_strings = re.findall(INTEGER_PATTERN, clean_string)
        return [int(n) for n in element_strings]

    def convert(
        self, value: Any, param: Optional[Parameter], ctx: Optional[Context]
    ) -> Sequence[int]:
        """Converts a value to a list of integers. This value must already be
        a list of integers or be a string representing a list of integers.

        We require that all strings provided follow the Python syntax for
        explicitly constructing one-dimensional lists of ints.

        Args:
            value: The value to convert.
            param: The parameter that is using this type to convert its value.
                May be ``None``.
            ctx: The current context that arrived at this value. May be
                ``None``.

        Returns:
             The provided value as a list of integers.

        Raises:
            BadParameter: if the provided value is neither a list of int-like
                objects nor a well-formatted string for a list of integers.
        """
        if isinstance(value, list):
            return self._parse_list(value)
        elif isinstance(value, str):
            return self._parse_string(value)
        else:
            self.fail(f"Unsupported data type for value: {value}")


MATRIX_TYPE = MatrixType()
INT_LIST_TYPE = IntListType()
