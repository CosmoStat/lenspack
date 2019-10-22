# -*- coding: utf-8 -*-


def add_int(x, y):
    """Add Integers

    Add two integer values.

    Parameters
    ----------
    x : int
        First value
    y : int
        Second value

    Returns
    -------
    int
        Result of addition

    Raises
    ------
    TypeError
        For invalid input types.

    """

    if not isinstance(x, int) or not isinstance(y, int):
        raise TypeError('Inputs must be integers.')

    return x + y


def add_float(x, y):
    """Add Floats

    Add two float values.

    Parameters
    ----------
    x : float
        First value
    y : float
        Second value

    Returns
    -------
    float
        Result of addition

    Raises
    ------
    TypeError
        For invalid input types.

    """

    if not isinstance(x, float) or not isinstance(y, float):
        raise TypeError('Inputs must be floats.')

    return x + y
