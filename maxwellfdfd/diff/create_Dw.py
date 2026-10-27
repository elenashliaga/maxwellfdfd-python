"""
Create forward difference matrix for FDFD operators.

This module provides functions to create forward difference matrices used in
finite-difference frequency-domain (FDFD) simulations. The matrices implement
forward differences without division by grid spacing, which is applied separately.
"""

from typing import Union

import numpy as np
import scipy.sparse as sp

from maxwellfdfd.base import Axis, Dir


def create_Dw(
    w: Union[Axis, Dir],
    N: np.ndarray,
    f1: complex,
    fg: complex
) -> sp.csr_matrix:
    """
    Create forward difference matrix without dl division.

    Creates the forward difference matrix for a specified direction. For the
    backward difference matrix, take the transpose or conjugate transpose
    appropriately, as shown in create_Ds().

    Parameters
    ----------
    w : Axis or Dir
        Direction for the difference operator (Axis.X, Axis.Y, Axis.Z for 3D,
        or Dir.H, Dir.V for 2D).
    N : ndarray
        Grid dimensions as a 1D array. For 3D: (Nx, Ny, Nz). For 2D: (Nx, Ny).
    f1 : complex
        Factor multiplied to the first element in the w-direction.
        Used for implementing symmetry boundary conditions.
    fg : complex
        Factor multiplied to the ghost element in the w-direction.
        Used for implementing Bloch boundary conditions.

    Returns
    -------
    Dw : scipy.sparse.csr_matrix
        Sparse forward difference matrix of shape (M, M) where M = prod(N).

    Raises
    ------
    TypeError
        If w is not an instance of Axis or Dir.
    ValueError
        If N is not a valid array of integers or if dimensions are inconsistent.

    Examples
    --------
    >>> import numpy as np
    >>> from maxwellfdfd.base import Axis
    >>> N = np.array([4, 5, 6])
    >>> Dw = create_Dw(Axis.X, N, 1.0, 0.0)
    >>> Dw.shape
    (120, 120)

    Notes
    -----
    The forward difference matrix implements:
        (Dw * f)[i] = f[i+1] - f[i]

    For periodic boundaries with Bloch wave vector k:
        fg = exp(-1j * k * L)

    For symmetry boundaries:
        f1 = 2 (for certain boundary condition combinations)

    The matrix uses circshift to handle the periodic wrapping, with
    appropriate factors applied at the boundaries.
    """
    # Validate inputs
    if not isinstance(w, (Axis, Dir)):
        raise TypeError('"w" should be instance of Axis or Dir')
    
    if not isinstance(N, np.ndarray) or N.ndim != 1:
        raise ValueError('"N" should be a 1D numpy array with integer elements')
    
    if not np.all(N > 0):
        raise ValueError('All elements of N must be positive')
    
    expected_length = w.count()
    if len(N) != expected_length:
        raise ValueError(f'"N" should be length-{expected_length} row vector with integer elements')
    
    # Translate spatial indices (i,j,k) into matrix indices
    M = int(np.prod(N))
    row_ind = np.arange(1, M + 1, dtype=int)
    col_ind_curr = np.arange(1, M + 1, dtype=int)
    
    # Create column indices for next point using circshift.
    # MATLAB: circshift(col_ind_next, shift) where shift is a per-dimension
    # vector.  In numpy we MUST pass axis= explicitly; without it,
    # np.roll flattens the array and applies a single scalar shift,
    # which corrupts the derivative operator.
    col_ind_next = np.arange(1, M + 1, dtype=int).reshape(tuple(N))
    shift = np.zeros(len(N), dtype=int)
    # NOTE: use int(w), not w.value. The enum's .value is a string label
    # (e.g. 'x'); int(w) returns the 0-based index used for array indexing.
    shift[int(w)] = -1
    col_ind_next = np.roll(col_ind_next, tuple(shift), axis=tuple(range(len(N))))
    
    # Create coefficient arrays
    a_curr = np.ones(N, dtype=complex)
    a_next = np.ones(N, dtype=complex)
    
    # Set f1 factor at first element in w-direction
    # Create indexing tuple for the first element
    ind_curr = [slice(None)] * len(N)
    ind_curr[int(w)] = slice(0, 1)
    a_curr[tuple(ind_curr)] = f1
    
    # Set fg factor at last element in w-direction
    ind_next = [slice(None)] * len(N)
    ind_next[int(w)] = slice(N[int(w)] - 1, N[int(w)])
    a_next[tuple(ind_next)] = fg
    
    # Create the sparse matrix
    # Row indices: [row_ind(:); row_ind(:)]
    # Column indices: [col_ind_curr(:); col_ind_next(:)]
    # Values: [-a_curr(:); a_next(:)]
    rows = np.concatenate([row_ind.ravel() - 1, row_ind.ravel() - 1])
    cols = np.concatenate([col_ind_curr.ravel() - 1, col_ind_next.ravel() - 1])
    data = np.concatenate([-a_curr.ravel(), a_next.ravel()])
    
    # Construct directly as csr_matrix so the return type matches the
    # annotation regardless of scipy version. (In newer scipy versions,
    # coo_matrix.tocsr() returns csr_array, not csr_matrix.)
    Dw = sp.csr_matrix((data, (rows, cols)), shape=(M, M))
    
    return Dw