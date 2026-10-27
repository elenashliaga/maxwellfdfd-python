"""
Create divergence operators for Maxwell's equations in FDFD.

This module provides functions to create divergence operators for the
finite-difference frequency-domain (FDFD) solver. The divergence operators
are used to enforce divergence-free conditions in electromagnetic simulations.
"""

from typing import Dict, Tuple, Union

import numpy as np
import scipy.sparse as sp

from maxwellfdfd.base import Axis, GT, Sign
from maxwellfdfd.grid import Grid3d
from maxwellfdfd.diff.create_Ds import create_Ds


def create_divs(
    ge: GT,
    dl_factor_cell: Union[Dict[tuple, np.ndarray], None],
    grid3d: Grid3d
) -> Tuple[sp.csr_matrix, sp.csr_matrix]:
    """
    Create divergence operators for Maxwell's equations.

    Creates the divergence operators Dive and Divm for the E-field and H-field
    respectively. The divergence operators are constructed from forward and
    backward derivative matrices.

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
    Dive : scipy.sparse.csr_matrix
        Divergence operator for the E-field. Shape is (M, 3*M) where M = prod(N).
    Divm : scipy.sparse.csr_matrix
        Divergence operator for the H-field. Shape is (M, 3*M) where M = prod(N).

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
    >>> Dive, Divm = create_divs(GT.Prim, None, grid3d)
    >>> Dive.shape
    (8, 24)

    Notes
    -----
    The divergence operator is defined as:
        div(F) = dFx/dx + dFy/dy + dFz/dz

    In matrix form:
        Divp = [Db_x, Db_y, Db_z]  (using backward derivatives)
        Divd = [Df_x, Df_y, Df_z]  (using forward derivatives)

    The assignment of Dive and Divm depends on the grid type:
    - If ge == GT.Prim: Dive = Divp, Divm = Divd
    - If ge == GT.Dual: Dive = Divd, Divm = Divp

    The divergence operators satisfy the vector identity:
        div(curl(F)) = 0

    This can be verified by computing:
        Dive * Cm ≈ 0
        Divm * Ce ≈ 0

    where Ce and Cm are the curl operators.
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
    
    # Form primary divergence matrix (using backward derivatives)
    Divp = sp.hstack([Db[Axis.X], Db[Axis.Y], Db[Axis.Z]], format='csr')
    
    # Form dual divergence matrix (using forward derivatives)
    Divd = sp.hstack([Df[Axis.X], Df[Axis.Y], Df[Axis.Z]], format='csr')
    
    # Assign Dive and Divm based on grid type
    if ge == GT.PRIM:
        Dive = Divp
        Divm = Divd
    else:  # ge == GT.DUAL
        Dive = Divd
        Divm = Divp
    
    return Dive, Divm