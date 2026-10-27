"""Tier 1 unit tests: base utility functions (expand2row, expand2mat,
istypeof, isphysQcell).

These exercise the small helper functions that are used pervasively for
argument validation and MATLAB-style scalar/vector expansion.
"""
import numpy as np

from maxwellfdfd.base import Axis, PhysQ
from maxwellfdfd.base.utils import (
    expand2row,
    expand2mat,
    isexpandable2row,
    istypeof,
    isphysQcell,
    create_spdiag,
)


# ----------------------------------------------------------- expand2row
def test_expand2row_scalar():
    # A scalar expands to a length-n constant ROW of shape (1, n)
    # (MATLAB row-vector convention used throughout the library).
    out = expand2row(3.0, 4)
    assert out.shape == (1, 4)
    assert np.array_equal(out, np.array([[3.0, 3.0, 3.0, 3.0]]))


def test_expand2row_zero_dim_array():
    # 0-d arrays must be treated as scalars (regression for np.isscalar bug).
    out = expand2row(np.asarray(5.0), 3)
    assert out.shape == (1, 3)
    assert np.all(out == 5.0)


def test_expand2row_passthrough():
    v = np.array([1.0, 2.0, 3.0])
    out = expand2row(v, 3)
    assert out.shape == (1, 3)
    assert np.array_equal(out, np.array([[1.0, 2.0, 3.0]]))


def test_expand2row_2d_row_vector():
    # A (1, n) row vector should be flattened and passed through.
    v = np.array([[1.0, 2.0, 3.0]])
    out = expand2row(v, 3)
    assert out.shape == (1, 3)
    assert np.array_equal(out, np.array([[1.0, 2.0, 3.0]]))


def test_isexpandable2row_wrong_length():
    assert not isexpandable2row(np.array([1.0, 2.0]), 3)


# ----------------------------------------------------------- expand2mat
def test_expand2mat_scalar():
    out = expand2mat(2.0, 2, 2)
    assert out.shape == (2, 2)
    assert np.all(out == 2.0)


# ------------------------------------------------------------- istypeof
def test_istypeof_numpy_array():
    # istypeof must work by class-name walk (regression for eval() bug).
    assert istypeof(np.array([1.0]), 'ndarray')


def test_istypeof_axis():
    assert istypeof(Axis.X, 'Axis')


def test_istypeof_negative():
    assert not istypeof(42, 'Axis')


# ---------------------------------------------------------- isphysQcell
def test_isphysQcell_single_physq():
    # A single PhysQ is NOT a physQcell: the cell form is strictly a 2D
    # (n, 2) array of [PhysQ, power] rows; bare PhysQ instances are
    # accepted separately via isinstance (see Scalar2d/Scalar3d and
    # PhysUnit.value).
    assert not isphysQcell(PhysQ.L)


def test_isphysQcell_2d_cell():
    assert isphysQcell([[PhysQ.L, 1], [PhysQ.OMEGA, -1]])


def test_isphysQcell_negative():
    assert not isphysQcell([[1, 2], [3, 4]])


# --------------------------------------------------------- create_spdiag
def test_create_spdiag():
    v = np.array([1.0, 2.0, 3.0])
    d = create_spdiag(v)
    assert d.shape == (3, 3)
    assert np.allclose(d.diagonal(), v)
