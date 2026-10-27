"""
Direct solver for FDFD equations.

This module provides a direct solver for the finite-difference frequency-domain
equations using sparse LU decomposition.
"""

from typing import Dict, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from maxwellfdfd.base import Axis, FT
from maxwellfdfd.grid import Grid3d
from maxwellfdfd.diff.matrix_equation import MatrixEquation
from maxwellfdfd.diff.eh_from_eh import eh_from_eh


def solve_eq_direct(
    eqtype,
    pml,
    omega: complex,
    eps_cell: Dict[Axis, np.ndarray],
    mu_cell: Dict[Axis, np.ndarray],
    s_factor_cell: Dict[tuple, np.ndarray],
    J_cell: Dict[Axis, np.ndarray],
    M_cell: Dict[Axis, np.ndarray],
    grid3d: Grid3d
) -> Tuple[Dict[Axis, np.ndarray], Dict[Axis, np.ndarray]]:
    """
    Solve FDFD equations using direct solver.

    Solves the matrix equation A*f = b using sparse LU decomposition.
    This is suitable for small to medium-sized problems where the
    system matrix can be factorized efficiently.

    Parameters
    ----------
    eqtype : EquationType
        Equation type specifying which field to solve for and grid type.
    pml : PML
        PML type (PML.SC for stretched-coordinate, PML.U for uniaxial).
    omega : complex
        Angular frequency of the simulation.
    eps_cell : dict
        Permittivity cell arrays with Axis keys. Each array has shape (Nx, Ny, Nz).
    mu_cell : dict
        Permeability cell arrays with Axis keys. Each array has shape (Nx, Ny, Nz).
    s_factor_cell : dict
        PML stretching factor cell arrays with (Axis, GT) keys.
    J_cell : dict
        Electric current source cell arrays with Axis keys.
    M_cell : dict
        Magnetic current source cell arrays with Axis keys.
    grid3d : Grid3d
        3D grid object containing grid information.

    Returns
    -------
    E : dict
        Dictionary with Axis keys (Axis.X, Axis.Y, Axis.Z) containing
        the E-field components as 3D arrays of shape N.
    H : dict
        Dictionary with Axis keys (Axis.X, Axis.Y, Axis.Z) containing
        the H-field components as 3D arrays of shape N.

    Raises
    ------
    TypeError
        If grid3d is not Grid3d.
    ValueError
        If solver fails to converge.

    Examples
    --------
    >>> import numpy as np
    >>> from maxwellfdfd.base import Axis, BC, GT, FT, PML, EquationType, PhysUnit
    >>> from maxwellfdfd.grid import Grid3d
    >>> from maxwellfdfd.diff import solve_eq_direct
    >>> unit = PhysUnit()
    >>> lprim = [np.array([0.0, 0.5, 1.0]), np.array([0.0, 0.5, 1.0]), np.array([0.0, 0.5, 1.0])]
    >>> Npml = [[1, 1], [1, 1], [1, 1]]
    >>> bc = {Axis.X: BC.E, Axis.Y: BC.E, Axis.Z: BC.E}
    >>> grid3d = Grid3d(unit, lprim, Npml, bc)
    >>> eqtype = EquationType(FT.E, GT.Prim)
    >>> eps_cell = {Axis.X: np.ones((2, 2, 2)), Axis.Y: np.ones((2, 2, 2)), Axis.Z: np.ones((2, 2, 2))}
    >>> mu_cell = {Axis.X: np.ones((2, 2, 2)), Axis.Y: np.ones((2, 2, 2)), Axis.Z: np.ones((2, 2, 2))}
    >>> s_factor_cell = {}
    >>> J_cell = {Axis.X: np.zeros((2, 2, 2)), Axis.Y: np.zeros((2, 2, 2)), Axis.Z: np.zeros((2, 2, 2))}
    >>> M_cell = {Axis.X: np.zeros((2, 2, 2)), Axis.Y: np.zeros((2, 2, 2)), Axis.Z: np.zeros((2, 2, 2))}
    >>> omega = 2 * np.pi * 3e8 / 1.55e-6
    >>> E, H = solve_eq_direct(eqtype, PML.SC, omega, eps_cell, mu_cell, s_factor_cell, J_cell, M_cell, grid3d)
    >>> E[Axis.X].shape
    (2, 2, 2)

    Notes
    -----
    The direct solver uses scipy.sparse.linalg.spsolve which performs
    sparse LU decomposition. This is efficient for small to medium-sized
    problems but may be memory-intensive for large problems.

    For large problems, consider using the iterative solver solve_eq_iterative
    instead.

    The solver handles the following equation types:
    - FT.E: Solve for E-field, compute H-field from E
    - FT.H: Solve for H-field, compute E-field from H
    - FT.ELEMS: Solve for combined E and H fields

    The solution is converted to structured arrays using eh_from_eh.
    """
    # Validate inputs
    if not isinstance(grid3d, Grid3d):
        raise TypeError('"grid3d" should be instance of Grid3d')
    
    # Create matrix equation
    eq = MatrixEquation(eqtype, pml, omega, eps_cell, mu_cell,
                       s_factor_cell, J_cell, M_cell, grid3d)
    
    # Get matrix operation
    A, b = eq.matrix_op()
    
    # Get matrix-free operator for computing secondary field
    _, _, GfromF = eq.matrixfree_op()
    
    # Solve using sparse LU decomposition
    try:
        f = spla.spsolve(A, b)
    except Exception as e:
        raise ValueError(f"Direct solver failed: {e}")
    
    # Compute secondary field
    g = GfromF(f)
    
    # Determine which field is primary and which is secondary
    if eqtype.f == FT.E:
        e = f
        h = g
    elif eqtype.f == FT.H:
        e = g
        h = f
    else:  # FT.ELEMS
        # Split combined solution
        n = len(f) // 2
        e = f[:n]
        h = f[n:]
    
    # Convert to structured arrays
    E, H = eh_from_eh(e, h, grid3d)
    
    return E, H