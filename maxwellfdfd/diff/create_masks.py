"""
Create mask operators for boundary conditions in FDFD.

This module provides functions to create mask indices for implementing
boundary conditions (PEC, PMC, periodic) in finite-difference frequency-domain
simulations.
"""

from typing import Tuple, Union

import numpy as np

from maxwellfdfd.base import Axis, BC, Dir, GT
from maxwellfdfd.grid import Grid2d, Grid3d


def create_masks(
    ge: GT,
    gridnd: Union[Grid2d, Grid3d]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create mask indices for boundary conditions.

    Generating the indices to mask is better than generating a diagonal mask
    matrix, because multiplying the mask matrix cannot make NaN or Inf elements
    zeros.

    Parameters
    ----------
    ge : GT
        Grid type for the E-field. GT.Prim for primary grid, GT.Dual for dual grid.
    gridnd : Grid2d or Grid3d
        Grid object containing grid information.

    Returns
    -------
    ind_Mp : ndarray
        Indices of elements to mask on the primary grid.
    ind_Md : ndarray
        Indices of elements to mask on the dual grid.

    Raises
    ------
    TypeError
        If ge is not GT or gridnd is not Grid2d or Grid3d.

    Examples
    --------
    >>> import numpy as np
    >>> from maxwellfdfd.base import GT, BC, Axis
    >>> from maxwellfdfd.grid import Grid3d
    >>> from maxwellfdfd.base import PhysUnit
    >>> unit = PhysUnit()
    >>> lprim = [np.array([0.0, 0.5, 1.0]), np.array([0.0, 0.5, 1.0]), np.array([0.0, 0.5, 1.0])]
    >>> Npml = [[1, 1], [1, 1], [1, 1]]
    >>> bc = {Axis.X: BC.E, Axis.Y: BC.E, Axis.Z: BC.E}
    >>> grid3d = Grid3d(unit, lprim, Npml, bc)
    >>> ind_Mp, ind_Md = create_masks(GT.Prim, grid3d)
    >>> ind_Mp.shape
    (24,)

    Notes
    -----
    The mask indices identify grid points that need to be zeroed out to
    implement boundary conditions:
    
    - PEC (BC.E): Electric field tangential components are zero
    - PMC (BC.M): Magnetic field tangential components are zero
    - Periodic (BC.P): No masking needed (handled by Bloch factors)

    For the primary grid (E-field), we mask:
    - E-field components tangential to PEC boundaries
    - H-field components tangential to PMC boundaries (on dual grid)

    The function returns indices rather than diagonal matrices for efficiency,
    as direct indexing is faster than matrix multiplication for masking.
    """
    # Validate inputs
    if not isinstance(ge, GT):
        raise TypeError('"ge" should be instance of GT')
    
    if not isinstance(gridnd, (Grid2d, Grid3d)):
        raise TypeError('"gridnd" should be instance of Grid2d or Grid3d')
    
    # Determine direction type (Axis for 3D, Dir for 2D)
    if isinstance(gridnd, Grid2d):
        v = Dir
    else:  # Grid3d
        v = Axis
    
    # Get grid properties
    N = gridnd.N
    bc = gridnd.bc
    
    # Create mask for primary grid
    mask_p = {}
    for w in v.elems():
        # Create mask array for w-component of field on primary grid
        mask = np.zeros(N, dtype=bool)

        # Get directions orthogonal to w
        us = [u for u in v.elems() if u != w]

        # Mask elements at boundaries
        for u in us:
            ui = int(u)  # 0-based index for list/ndarray indexing
            # Check if we need to mask at the boundary in direction u
            if (ge == GT.PRIM and bc[ui] == BC.E) or (ge == GT.DUAL and bc[ui] == BC.M):
                # Create indexing tuple
                ind = [slice(None)] * len(N)
                ind[ui] = 0  # First element in u-direction
                mask[tuple(ind)] = True

        mask_p[w] = mask.ravel()
    
    # Concatenate all primary grid masks
    ind_Mp = np.concatenate([mask_p[w] for w in v.elems()])
    ind_Mp = np.where(ind_Mp)[0]  # Get indices where mask is True
    
    # Create mask for dual grid
    mask_d = {}
    for w in v.elems():
        wi = int(w)  # 0-based index for list/ndarray indexing
        # Create mask array for w-component of field on dual grid
        mask = np.zeros(N, dtype=bool)

        # Check if we need to mask at the boundary in direction w
        if (ge == GT.PRIM and bc[wi] == BC.E) or (ge == GT.DUAL and bc[wi] == BC.M):
            # Create indexing tuple
            ind = [slice(None)] * len(N)
            ind[wi] = 0  # First element in w-direction
            mask[tuple(ind)] = True

        mask_d[w] = mask.ravel()
    
    # Concatenate all dual grid masks
    ind_Md = np.concatenate([mask_d[w] for w in v.elems()])
    ind_Md = np.where(ind_Md)[0]  # Get indices where mask is True
    
    return ind_Mp, ind_Md