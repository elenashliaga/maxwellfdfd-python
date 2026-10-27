"""
Field conversion utilities for FDFD.

This module provides functions to convert between flattened field vectors
and structured field arrays for the finite-difference frequency-domain solver.
"""

from typing import Dict, Tuple

import numpy as np

from maxwellfdfd.base import Axis
from maxwellfdfd.grid import Grid3d


def eh_from_eh(
    e: np.ndarray,
    h: np.ndarray,
    grid3d: Grid3d
) -> Tuple[Dict[Axis, np.ndarray], Dict[Axis, np.ndarray]]:
    """
    Convert E and H fields from flattened vectors to structured arrays.

    Reshapes flattened field vectors into 3D arrays organized by field components.
    The output is a dictionary with Axis enum keys containing the field components.

    Parameters
    ----------
    e : ndarray
        Flattened E-field vector. Length should be 3 * prod(N).
    h : ndarray
        Flattened H-field vector. Length should be 3 * prod(N).
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
        If e or h have incorrect dimensions.

    Examples
    --------
    >>> import numpy as np
    >>> from maxwellfdfd.base import Axis, BC
    >>> from maxwellfdfd.grid import Grid3d
    >>> from maxwellfdfd.base import PhysUnit
    >>> unit = PhysUnit()
    >>> lprim = [np.array([0.0, 0.5, 1.0]), np.array([0.0, 0.5, 1.0]), np.array([0.0, 0.5, 1.0])]
    >>> Npml = [[1, 1], [1, 1], [1, 1]]
    >>> bc = {Axis.X: BC.E, Axis.Y: BC.E, Axis.Z: BC.E}
    >>> grid3d = Grid3d(unit, lprim, Npml, bc)
    >>> e = np.random.randn(24) + 1j * np.random.randn(24)
    >>> h = np.random.randn(24) + 1j * np.random.randn(24)
    >>> E, H = eh_from_eh(e, h, grid3d)
    >>> E[Axis.X].shape
    (2, 2, 2)

    Notes
    -----
    The flattened field vectors have the structure:
        [Ex_1, Ex_2, ..., Ex_M, Ey_1, Ey_2, ..., Ey_M, Ez_1, Ez_2, ..., Ez_M]

    where M = prod(N) is the total number of grid points.

    The function reshapes these into 3D arrays:
        Ex: shape (Nx, Ny, Nz)
        Ey: shape (Nx, Ny, Nz)
        Ez: shape (Nx, Ny, Nz)

    This conversion is useful for visualization and post-processing of
    simulation results.
    """
    # Validate inputs
    if not isinstance(grid3d, Grid3d):
        raise TypeError('"grid3d" should be instance of Grid3d')
    
    N = grid3d.N
    M = int(np.prod(N))
    
    # Validate e and h dimensions
    if len(e) != 3 * M:
        raise ValueError(f'"e" should have length {3 * M}, got {len(e)}')
    
    if len(h) != 3 * M:
        raise ValueError(f'"h" should have length {3 * M}, got {len(h)}')
    
    # Reshape e and h to (3, M) using Fortran (column-major) order.
    # The solution vector from the solver is in *interleaved* order after
    # reordering: [Ex(0), Ey(0), Ez(0), Ex(1), Ey(1), Ez(1), ...].
    # Fortran-order reshape correctly groups the three field components at
    # each spatial location into rows of the (3, M) matrix.
    e_reshaped = e.reshape(3, M, order='F')
    h_reshaped = h.reshape(3, M, order='F')

    # Extract each component and reshape back to 3D arrays.
    # The second reshape uses default C-order, matching the original
    # .ravel() flattening used when building the system vectors.
    Ex = e_reshaped[0, :].reshape(tuple(N))
    Ey = e_reshaped[1, :].reshape(tuple(N))
    Ez = e_reshaped[2, :].reshape(tuple(N))

    Hx = h_reshaped[0, :].reshape(tuple(N))
    Hy = h_reshaped[1, :].reshape(tuple(N))
    Hz = h_reshaped[2, :].reshape(tuple(N))
    
    # Create dictionaries with Axis keys
    E = {Axis.X: Ex, Axis.Y: Ey, Axis.Z: Ez}
    H = {Axis.X: Hx, Axis.Y: Hy, Axis.Z: Hz}
    
    return E, H