"""
Create curl operators for Maxwell's equations in FDFD.

This module provides functions to create curl operators for the finite-difference
frequency-domain (FDFD) solver. The curl operators are used to discretize the
curl operations in Maxwell's equations.
"""

from typing import Dict, Tuple, Union

import numpy as np
import scipy.sparse as sp

from maxwellfdfd.base import Axis, GT, Sign
from maxwellfdfd.grid import Grid3d
from maxwellfdfd.diff.create_Ds import create_Ds
from maxwellfdfd.diff.create_masks import create_masks


def create_curls(
    ge: GT,
    dl_factor_cell: Union[Dict[tuple, np.ndarray], None],
    grid3d: Grid3d
) -> Tuple[sp.csr_matrix, sp.csr_matrix]:
    """
    Create curl operators for Maxwell's equations.

    Creates the curl operators Ce and Cm for the E-field and H-field
    respectively. The curl operators are constructed from forward and
    backward derivative matrices with appropriate boundary condition masking.

    Parameters
    ----------
    ge : GT
        Grid type for the E-field. GT.Prim for primary grid, GT.Dual for dual grid.
    dl_factor_cell : dict or None
        PML stretching factors. If None, no PML stretching is applied.
        Dictionary with (Axis, GT) keys containing stretching factor arrays.
    grid3d : Grid3d
        3D grid object containing grid information.

    Returns
    -------
    Ce : scipy.sparse.csr_matrix
        Curl operator for the E-field. Shape is (3*M, 3*M) where M = prod(N).
    Cm : scipy.sparse.csr_matrix
        Curl operator for the H-field. Shape is (3*M, 3*M) where M = prod(N).

    Raises
    ------
    TypeError
        If ge is not GT or grid3d is not Grid3d.

    Examples
    --------
    >>> import numpy as np
    >>> from maxwellfdfd.base import GT, Axis, BC
    >>> from maxwellfdfd.grid import Grid3d
    >>> from maxwellfdfd.base import PhysUnit
    >>> unit = PhysUnit()
    >>> lprim = [np.array([0.0, 0.5, 1.0]), np.array([0.0, 0.5, 1.0]), np.array([0.0, 0.5, 1.0])]
    >>> Npml = [[1, 1], [1, 1], [1, 1]]
    >>> bc = {Axis.X: BC.E, Axis.Y: BC.E, Axis.Z: BC.E}
    >>> grid3d = Grid3d(unit, lprim, Npml, bc)
    >>> Ce, Cm = create_curls(GT.Prim, None, grid3d)
    >>> Ce.shape
    (24, 24)

    Notes
    -----
    The curl operator is defined as:
        curl(F) = (dFz/dy - dFy/dz, dFx/dz - dFz/dx, dFy/dx - dFx/dy)

    In matrix form:
        Cp = [  0, -Df_z,  Df_y;
               Df_z,   0, -Df_x;
              -Df_y, Df_x,   0 ]

    where Df_x, Df_y, Df_z are forward derivative matrices.

    For the backward derivative operator:
        Cd = [  0, -Db_z,  Db_y;
               Db_z,   0, -Db_x;
              -Db_y, Db_x,   0 ]

    The assignment of Ce and Cm depends on the grid type:
    - If ge == GT.Prim: Ce = Cp, Cm = Cd
    - If ge == GT.Dual: Ce = Cd, Cm = Cp

    Boundary conditions are implemented by zeroing out appropriate
    rows and columns of the curl matrices.
    """
    # Validate inputs
    if not isinstance(ge, GT):
        raise TypeError('"ge" should be instance of GT')
    
    if not isinstance(grid3d, Grid3d):
        raise TypeError('"grid3d" should be instance of Grid3d')
    
    # Create forward and backward derivative matrices
    # create_Ds expects a Sign as its first argument: Sign.P (forward), Sign.N (backward)
    Df = create_Ds(Sign.P, ge, dl_factor_cell, grid3d)
    Db = create_Ds(Sign.N, ge, dl_factor_cell, grid3d)
    
    # Create mask indices
    ind_Mp, ind_Md = create_masks(ge, grid3d)
    
    # Get grid size
    M = int(np.prod(grid3d.N))
    Z = sp.csr_matrix((M, M))
    
    # Form primary curl matrix (using forward derivatives)
    Cp = sp.bmat([
        [Z, -Df[Axis.Z], Df[Axis.Y]],
        [Df[Axis.Z], Z, -Df[Axis.X]],
        [-Df[Axis.Y], Df[Axis.X], Z]
    ], format='csr')
    
    # Apply masking to primary curl matrix
    # Zero columns corresponding to primary grid mask
    Cp[:, ind_Mp] = 0
    # Zero rows corresponding to dual grid mask
    Cp[ind_Md, :] = 0
    
    # Form dual curl matrix (using backward derivatives)
    Cd = sp.bmat([
        [Z, -Db[Axis.Z], Db[Axis.Y]],
        [Db[Axis.Z], Z, -Db[Axis.X]],
        [-Db[Axis.Y], Db[Axis.X], Z]
    ], format='csr')
    
    # Apply masking to dual curl matrix
    # Zero columns corresponding to dual grid mask
    Cd[:, ind_Md] = 0
    # Zero rows corresponding to primary grid mask
    Cd[ind_Mp, :] = 0
    
    # Assign Ce and Cm based on grid type
    if ge == GT.PRIM:
        Ce = Cp
        Cm = Cd
    else:  # ge == GT.DUAL
        Ce = Cd
        Cm = Cp
    
    return Ce, Cm