"""
Create stretched coordinate PML operators for FDFD.

This module provides functions to create forward and backward derivative matrices
with stretched coordinate PML (Perfectly Matched Layer) support.
"""

from typing import Dict, List, Union

import numpy as np
import scipy.sparse as sp

from maxwellfdfd.base import Axis, BC, Dir, GT, Sign
from maxwellfdfd.base import create_spdiag
from maxwellfdfd.grid import Grid2d, Grid3d
from maxwellfdfd.diff.create_Dw import create_Dw
from maxwellfdfd.diff._utils import mult_vec


def create_Ds(
    s: Sign,
    ge: GT,
    dl_factor_cell: Union[List[np.ndarray], Dict[tuple, np.ndarray], None],
    gridnd: Union[Grid2d, Grid3d]
) -> Dict[Union[Axis, Dir], sp.csr_matrix]:
    """
    Create forward or backward derivative matrix with PML stretching.

    Creates derivative matrices for the finite-difference frequency-domain (FDFD)
    solver. The matrices can include stretched coordinate PML effects through
    the dl_factor_cell parameter.

    Parameters
    ----------
    s : Sign
        Sign of the derivative. Sign.P for forward derivative (Df),
        Sign.N for backward derivative (Db).
    ge : GT
        Grid type for the E-field. GT.Prim for primary grid, GT.Dual for dual grid.
    dl_factor_cell : list of ndarray, dict, or None
        PML stretching factors. If None, no PML stretching is applied.
        For 3D: dictionary with (Axis, GT) keys or 3x2 list.
        For 2D: dictionary with (Dir, GT) keys or 2x2 list.
    gridnd : Grid2d or Grid3d
        Grid object containing grid information.

    Returns
    -------
    Ds_cell : dict
        Dictionary of derivative matrices with Axis (for 3D) or Dir (for 2D) keys.
        Each matrix is a sparse CSR matrix.

    Raises
    ------
    TypeError
        If s is not Sign, ge is not GT, or gridnd is not Grid2d or Grid3d.
    ValueError
        If dl_factor_cell has invalid dimensions or types.

    Examples
    --------
    >>> import numpy as np
    >>> from maxwellfdfd.base import Sign, GT, Axis, BC
    >>> from maxwellfdfd.grid import Grid3d
    >>> from maxwellfdfd.base import PhysUnit
    >>> unit = PhysUnit()
    >>> lprim = [np.array([0.0, 0.5, 1.0]), np.array([0.0, 0.5, 1.0]), np.array([0.0, 0.5, 1.0])]
    >>> Npml = [[1, 1], [1, 1], [1, 1]]
    >>> bc = {Axis.X: BC.E, Axis.Y: BC.E, Axis.Z: BC.E}
    >>> grid3d = Grid3d(unit, lprim, Npml, bc)
    >>> Df = create_Ds(Sign.P, GT.Prim, None, grid3d)
    >>> Df[Axis.X].shape
    (8, 8)

    Notes
    -----
    The forward derivative matrix Df (s=Sign.P) implements:
        (Df * f)[i] = (f[i+1] - f[i]) / dl[i]

    The backward derivative matrix Db (s=Sign.N) implements:
        (Db * f)[i] = (f[i] - f[i-1]) / dl[i-1]

    For periodic boundaries with Bloch wave vector k:
        fg = exp(-1j * k * L)

    For PEC/PMC boundaries, appropriate factors are applied to implement
    the boundary conditions correctly.

    The PML stretching factors modify the effective grid spacing:
        dl_eff = dl * dl_factor
    """
    # Validate inputs
    if not isinstance(s, Sign):
        raise TypeError('"s" should be instance of Sign')
    
    if not isinstance(ge, GT):
        raise TypeError('"ge" should be instance of GT')
    
    if not isinstance(gridnd, (Grid2d, Grid3d)):
        raise TypeError('"gridnd" should be instance of Grid2d or Grid3d')
    
    # Determine direction type (Axis for 3D, Dir for 2D)
    if isinstance(gridnd, Grid2d):
        v = Dir
    else:  # Grid3d
        v = Axis
    
    # Validate dl_factor_cell
    if dl_factor_cell is not None:
        if isinstance(dl_factor_cell, dict):
            # Dictionary with (Axis/Dir, GT) keys
            expected_keys = [(d, g) for d in v.elems() for g in GT.elems()]
            for key in expected_keys:
                if key not in dl_factor_cell:
                    raise ValueError(f"Missing key {key} in dl_factor_cell")
        elif isinstance(dl_factor_cell, list):
            # List of lists: [v.count()][GT.count()]
            if len(dl_factor_cell) != v.count():
                raise ValueError(f"dl_factor_cell should have {v.count()} rows")
            for row in dl_factor_cell:
                if len(row) != GT.count():
                    raise ValueError(f"Each row of dl_factor_cell should have {GT.count()} elements")
        else:
            raise TypeError("dl_factor_cell should be None, dict, or list")
    
    # Get grid properties
    N = gridnd.N
    bc = gridnd.bc
    kBloch = gridnd.kBloch
    L = gridnd.L
    
    # Get the relevant grid type for the grid spacing.
    # This mirrors the MATLAB `GT.elems(s)` mapping:
    #   forward difference (Sign.P) uses dual grid spacing  -> GT.Dual
    #   backward difference (Sign.N) uses primary grid spacing -> GT.Prim
    # (Note: Sign.alter() returns a Sign, not a GT, so it must not be used here.)
    g = GT.DUAL if s == Sign.P else GT.PRIM
    
    # Create derivative matrices
    Ds_cell = {}
    
    if s == Sign.P:  # Ds == Df (forward derivative)
        for w in v.elems():
            wi = int(w)  # 0-based index for list/ndarray indexing
            f1 = 1.0

            if bc[wi] == BC.P:
                # Periodic boundary with Bloch phase factor
                fg = np.exp(-1j * kBloch[wi] * L[wi])
            else:
                # PEC or PMC boundary - ghost point is zero
                fg = 0.0

            Ds_cell[w] = create_Dw(w, N, f1, fg)
    else:  # Ds == Db (backward derivative)
        for w in v.elems():
            wi = int(w)  # 0-based index for list/ndarray indexing
            # Determine f1 factor for symmetry
            if (ge == GT.PRIM and bc[wi] == BC.M) or (ge == GT.DUAL and bc[wi] == BC.E):
                f1 = 2.0  # Symmetry of operator for this case
            else:
                f1 = 1.0

            if bc[wi] == BC.P:
                # Periodic boundary with Bloch phase factor
                fg = np.exp(-1j * kBloch[wi] * L[wi])
            else:
                # PEC or PMC boundary
                fg = 0.0  # f1 = 1 or 2 takes care of the ghost point

            Ds_cell[w] = create_Dw(w, N, f1, fg)
            # Conjugate transpose rather than transpose (hence non-symmetry for kBloch != 0)
            Ds_cell[w] = -Ds_cell[w].conj().T
    
    # Apply dl division
    # Get grid spacing
    dl = gridnd.dl
    
    # gridnd.dl is keyed by (Axis/Dir, GT) tuples, so build a per-axis list.
    dl_for_g = [dl[(w, g)] for w in v.elems()]

    if dl_factor_cell is None:
        # No PML stretching - use original dl
        dl_cell = dl_for_g
    else:
        # Apply PML stretching factors
        if isinstance(dl_factor_cell, dict):
            dl_factor = [dl_factor_cell[(w, g)] for w in v.elems()]
        else:
            # int(w)/int(g) give the 0-based indices (.value is a string label)
            dl_factor = [dl_factor_cell[int(w)][int(g)] for w in v.elems()]

        # Multiply dl by stretching factors
        dl_cell = mult_vec(dl_factor, dl_for_g)
    
    # Create meshgrid of dl values
    if len(N) == 2:
        # 2D case
        dl_mesh = np.meshgrid(dl_cell[0], dl_cell[1], indexing='ij')
    else:
        # 3D case
        dl_mesh = np.meshgrid(dl_cell[0], dl_cell[1], dl_cell[2], indexing='ij')
    
    # Apply inverse dl to derivative matrices
    for i, w in enumerate(v.elems()):
        dl_inv = 1.0 / dl_mesh[i]
        dl_inv_diag = create_spdiag(dl_inv.ravel())
        Ds_cell[w] = dl_inv_diag @ Ds_cell[w]
    
    return Ds_cell