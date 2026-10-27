"""
Utility functions for the diff module.

This module provides helper functions for vector operations and index reordering
used in the finite-difference frequency-domain (FDFD) solver.
"""

from typing import Dict, List, Union

import numpy as np


def mult_vec(
    F_cell: Union[List[np.ndarray], Dict[object, np.ndarray]],
    G_cell: Union[List[np.ndarray], Dict[object, np.ndarray]]
) -> Union[List[np.ndarray], Dict[object, np.ndarray]]:
    """
    Element-wise multiplication of cell arrays.

    Performs element-wise multiplication of corresponding arrays in two cell arrays.
    The cell arrays can be either lists or dictionaries with enum keys.

    Parameters
    ----------
    F_cell : list of ndarray or dict
        First cell array containing arrays to multiply.
    G_cell : list of ndarray or dict
        Second cell array containing arrays to multiply.

    Returns
    -------
    Prod_cell : list of ndarray or dict
        Cell array with element-wise products of corresponding arrays.

    Raises
    ------
    ValueError
        If F_cell and G_cell have different lengths or if corresponding
        arrays have different shapes.

    Examples
    --------
    >>> import numpy as np
    >>> F = [np.array([1, 2, 3]), np.array([4, 5, 6])]
    >>> G = [np.array([2, 3, 4]), np.array([5, 6, 7])]
    >>> mult_vec(F, G)
    [array([ 2,  6, 12]), array([20, 30, 42])]
    """
    # Handle dictionary input
    if isinstance(F_cell, dict):
        if not isinstance(G_cell, dict):
            raise ValueError("F_cell and G_cell must both be dictionaries or both be lists")
        
        if len(F_cell) != len(G_cell):
            raise ValueError("F_cell and G_cell must have the same length")
        
        Prod_cell = {}
        for key in F_cell:
            if key not in G_cell:
                raise ValueError(f"Key {key} not found in G_cell")
            if F_cell[key].shape != G_cell[key].shape:
                raise ValueError(f"Arrays for key {key} have different shapes")
            Prod_cell[key] = F_cell[key] * G_cell[key]
        return Prod_cell
    
    # Handle list input
    if len(F_cell) != len(G_cell):
        raise ValueError("F_cell and G_cell must have the same length")
    
    n = len(F_cell)
    Prod_cell = []
    for w in range(n):
        if F_cell[w].shape != G_cell[w].shape:
            raise ValueError(f"{w}-component of F_cell and G_cell have different shapes")
        Prod_cell.append(F_cell[w] * G_cell[w])
    
    return Prod_cell


def div_vec(
    F_cell: Union[List[np.ndarray], Dict[object, np.ndarray]],
    G_cell: Union[List[np.ndarray], Dict[object, np.ndarray]]
) -> Union[List[np.ndarray], Dict[object, np.ndarray]]:
    """
    Element-wise division of cell arrays.

    Performs element-wise division of corresponding arrays in two cell arrays.
    The cell arrays can be either lists or dictionaries with enum keys.

    Parameters
    ----------
    F_cell : list of ndarray or dict
        Numerator cell array containing arrays.
    G_cell : list of ndarray or dict
        Denominator cell array containing arrays.

    Returns
    -------
    Div_cell : list of ndarray or dict
        Cell array with element-wise quotients of corresponding arrays.

    Raises
    ------
    ValueError
        If F_cell and G_cell have different lengths or if corresponding
        arrays have different shapes.

    Examples
    --------
    >>> import numpy as np
    >>> F = [np.array([2, 6, 12]), np.array([20, 30, 42])]
    >>> G = [np.array([1, 2, 3]), np.array([4, 5, 6])]
    >>> div_vec(F, G)
    [array([2., 3., 4.]), array([5., 6., 7.])]
    """
    # Handle dictionary input
    if isinstance(F_cell, dict):
        if not isinstance(G_cell, dict):
            raise ValueError("F_cell and G_cell must both be dictionaries or both be lists")
        
        if len(F_cell) != len(G_cell):
            raise ValueError("F_cell and G_cell must have the same length")
        
        Div_cell = {}
        for key in F_cell:
            if key not in G_cell:
                raise ValueError(f"Key {key} not found in G_cell")
            if F_cell[key].shape != G_cell[key].shape:
                raise ValueError(f"Arrays for key {key} have different shapes")
            Div_cell[key] = F_cell[key] / G_cell[key]
        return Div_cell
    
    # Handle list input
    if len(F_cell) != len(G_cell):
        raise ValueError("F_cell and G_cell must have the same length")
    
    n = len(F_cell)
    Div_cell = []
    for w in range(n):
        if F_cell[w].shape != G_cell[w].shape:
            raise ValueError(f"{w}-component of F_cell and G_cell have different shapes")
        Div_cell.append(F_cell[w] / G_cell[w])
    
    return Div_cell


def reordering_indices(dof: int, N: np.ndarray) -> np.ndarray:
    """
    Generate reordering indices to reduce matrix bandwidth.

    This function generates indices to reorder the elements of matrices and vectors
    to reduce the bandwidth of the Maxwell operator matrix. The reordering groups
    field components at each spatial location together, which improves the efficiency
    of sparse matrix operations.

    Parameters
    ----------
    dof : int
        Number of degrees of freedom per grid point (e.g., 3 for 3D vector fields).
    N : ndarray
        Array of grid dimensions (Nx, Ny, Nz) or (Nx, Ny) for 2D.

    Returns
    -------
    r : ndarray
        Reordering indices as a 1D array.

    Raises
    ------
    ValueError
        If dof is not a positive integer or if N is not a valid array of integers.

    Examples
    --------
    >>> import numpy as np
    >>> reordering_indices(3, np.array([4, 5, 6]))
    array([  0,  20,  40,   1,  21,  41,   2,  22,  42,   3,  23,  43,
             4,  24,  44,   5,  25,  45,   6,  26,  46,   7,  27,  47,
             8,  28,  48,   9,  29,  49,  10,  30,  50,  11,  31,  51,
            12,  32,  52,  13,  33,  53,  14,  34,  54,  15,  35,  55,
            16,  36,  56,  17,  37,  57,  18,  38,  58,  19,  39,  59,
            60,  80, 100,  61,  81, 101,  62,  82, 102,  63,  83, 103,
            64,  84, 104,  65,  85, 105,  66,  86, 106,  67,  87, 107,
            68,  88, 108,  69,  89, 109,  70,  90, 110,  71,  91, 111,
            72,  92, 112,  73,  93, 113,  74,  94, 114,  75,  95, 115,
            76,  96, 116,  77,  97, 117,  78,  98, 118,  79,  99, 119])
    """
    if not isinstance(dof, int) or dof <= 0:
        raise ValueError("dof must be a positive integer")
    
    if not isinstance(N, np.ndarray) or N.ndim != 1:
        raise ValueError("N must be a 1D numpy array")
    
    if not np.all(N > 0):
        raise ValueError("All elements of N must be positive")
    
    # Generate sequential indices
    r = np.arange(dof * np.prod(N), dtype=int)

    # Reshape to (prod(N), dof) using Fortran (column-major) order to match
    # MATLAB's reshape semantics.  MATLAB fills column-by-column, so we must
    # use order='F' here; the default C-order reshape would scramble the
    # field-component grouping.
    r = r.reshape(int(np.prod(N)), dof, order='F')

    # Transpose to group DOF at each spatial location together
    r = r.T

    # Flatten using Fortran (column-major) order, matching MATLAB's r(:)
    r = r.ravel(order='F')
    
    return r