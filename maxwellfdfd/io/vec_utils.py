"""
Vector (cell) utility functions for MaxwellFDFD.

This module provides element-wise operations on "cell vectors" —
dictionaries mapping :class:`~maxwellfdfd.base.axis.Axis` keys to NumPy
arrays. These are used for per-component field operations (e.g.,
flipping, multiplying, dividing, negating E/H/J/M field cells).
"""

from typing import Dict, Union

import numpy as np

from maxwellfdfd.base import Axis, chkarg


def flip_array(array: np.ndarray) -> np.ndarray:
    """
    Flip an array along all of its dimensions.

    This is equivalent to applying ``flipdim`` along every axis in MATLAB.

    Parameters
    ----------
    array : np.ndarray
        The array to flip.

    Returns
    -------
    np.ndarray
        The array flipped along every dimension.

    Examples
    --------
    >>> import numpy as np
    >>> from maxwellfdfd.io import flip_array
    >>> a = np.array([[1, 2], [3, 4]])
    >>> flip_array(a)
    array([[4, 3],
           [2, 1]])
    """
    result = array
    for i in range(result.ndim):
        result = np.flip(result, axis=i)
    return result


def flip_vec(F_cell: Dict[Axis, np.ndarray]) -> Dict[Axis, np.ndarray]:
    """
    Flip each component of a cell vector along all dimensions.

    Parameters
    ----------
    F_cell : dict
        ``{Axis: np.ndarray}`` cell vector.

    Returns
    -------
    dict
        ``{Axis: np.ndarray}`` with each component flipped via
        :func:`flip_array`.

    Raises
    ------
    ValueError
        If ``F_cell`` is not a length-``Axis.count()`` dict.
    """
    chkarg(
        isinstance(F_cell, dict) and len(F_cell) == Axis.count(),
        f'"F_cell" should be length-{Axis.count()} dict.',
    )
    return {w: flip_array(F_cell[w]) for w in Axis.elems()}


def div_vec(
    F_cell: Dict[Axis, np.ndarray],
    G_cell: Dict[Axis, np.ndarray],
) -> Dict[Axis, np.ndarray]:
    """
    Element-wise division of two cell vectors.

    Parameters
    ----------
    F_cell : dict
        Numerator cell vector ``{Axis: np.ndarray}``.
    G_cell : dict
        Denominator cell vector ``{Axis: np.ndarray}``.

    Returns
    -------
    dict
        ``{Axis: np.ndarray}`` where each component is ``F_cell[w] /
        G_cell[w]``.

    Raises
    ------
    ValueError
        If the cells are not length-``Axis.count()`` or if corresponding
        components have mismatched shapes.
    """
    chkarg(
        isinstance(F_cell, dict) and len(F_cell) == Axis.count(),
        f'"F_cell" should be length-{Axis.count()} dict.',
    )
    chkarg(
        isinstance(G_cell, dict) and len(G_cell) == Axis.count(),
        f'"G_cell" should be length-{Axis.count()} dict.',
    )
    result: Dict[Axis, np.ndarray] = {}
    for w in Axis.elems():
        chkarg(
            F_cell[w].shape == G_cell[w].shape,
            f'{w}-component of "F_cell" and "G_cell" should have the same size.',
        )
        result[w] = F_cell[w] / G_cell[w]
    return result


def mult_vec(
    F_cell: Dict[Axis, np.ndarray],
    G_cell: Dict[Axis, np.ndarray],
) -> Dict[Axis, np.ndarray]:
    """
    Element-wise multiplication of two cell vectors.

    Parameters
    ----------
    F_cell : dict
        First cell vector ``{Axis: np.ndarray}``.
    G_cell : dict
        Second cell vector ``{Axis: np.ndarray}``.

    Returns
    -------
    dict
        ``{Axis: np.ndarray}`` where each component is ``F_cell[w] *
        G_cell[w]``.

    Raises
    ------
    ValueError
        If the cells have different lengths or mismatched component shapes.
    """
    chkarg(
        isinstance(F_cell, dict) and isinstance(G_cell, dict),
        '"F_cell" and "G_cell" should be dicts.',
    )
    chkarg(
        len(F_cell) == len(G_cell),
        '"F_cell" and "G_cell" should be the same length.',
    )
    result: Dict[Axis, np.ndarray] = {}
    for w in F_cell:
        chkarg(
            F_cell[w].shape == G_cell[w].shape,
            f'Components should have the same size.',
        )
        result[w] = F_cell[w] * G_cell[w]
    return result


def neg_vec(F_cell: Dict[Axis, np.ndarray]) -> Dict[Axis, np.ndarray]:
    """
    Negate each component of a cell vector.

    Parameters
    ----------
    F_cell : dict
        Cell vector ``{Axis: np.ndarray}``.

    Returns
    -------
    dict
        ``{Axis: np.ndarray}`` with each component negated.

    Raises
    ------
    ValueError
        If ``F_cell`` is not a length-``Axis.count()`` dict.
    """
    chkarg(
        isinstance(F_cell, dict) and len(F_cell) == Axis.count(),
        f'"F_cell" should be length-{Axis.count()} dict.',
    )
    return {w: -F_cell[w] for w in Axis.elems()}
