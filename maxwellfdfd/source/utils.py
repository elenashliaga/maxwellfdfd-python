"""
Utility functions for MaxwellFDFD source module.

This module provides utility functions for source assignment and index
calculation.
"""

from typing import TYPE_CHECKING
import numpy as np
import warnings

from maxwellfdfd.base import Axis, GT, chkarg

if TYPE_CHECKING:
    from maxwellfdfd.grid import Grid3d


def ind_for_loc(loc: float, axis: Axis, gt: GT, grid3d: 'Grid3d') -> int:
    """
    Find the index of a grid location closest to a given location.
    
    This function searches for a grid location that matches the given
    location. If an exact match is not found, it returns the index of
    the closest grid location and issues a warning.
    
    Parameters
    ----------
    loc : float
        The location to search for.
    axis : Axis
        The axis along which to search.
    gt : GT
        The grid type (primary or dual).
    grid3d : Grid3d
        3D grid instance.
    
    Returns
    -------
    int
        The index of the grid location closest to `loc`.
    
    Notes
    -----
    This function uses np.isclose to find exact matches within a small
    tolerance. If no exact match is found, it uses np.argmin to find
    the closest grid point.
    
    Examples
    --------
    >>> from maxwellfdfd.base import Axis, GT
    >>> from maxwellfdfd.source.utils import ind_for_loc
    >>> 
    >>> # Find index for location 0.5 on x-axis, primary grid
    >>> ind = ind_for_loc(0.5, Axis.X, GT.PRIM, grid3d)
    """
    l_array = grid3d.l[(axis, gt)]
    
    # Try to find exact match
    ind = np.where(np.isclose(l_array, loc))[0]
    
    if len(ind) == 0:
        # No exact match, find closest
        ind = np.argmin(np.abs(l_array - loc))
        warnings.warn(
            f'{gt} grid in {axis}-axis of "grid3d" does not have location {loc}; '
            f'closest grid vertex at {l_array[ind]} will be taken instead.',
            RuntimeWarning,
            stacklevel=2
        )
    else:
        ind = ind[0]
    
    return ind


def assign_source(grid3d: 'Grid3d', src_array: list) -> dict:
    """
    Assign sources to the grid.
    
    This function assigns multiple sources to the grid by superposing
    their contributions. Each source generates current density values
    at specific grid points, and these are added together.
    
    Parameters
    ----------
    grid3d : Grid3d
        3D grid instance.
    src_array : list of Source
        List of Source instances to assign.
    
    Returns
    -------
    JM_cell : list of ndarray
        List of current density arrays for each axis [Jx, Jy, Jz].
    
    Notes
    -----
    - Multiple sources are superposed by adding their contributions.
    - The returned JM_cell contains the total current density from all
      sources.
    - Each element of JM_cell corresponds to a different axis (x, y, z).
    
    Examples
    --------
    >>> from maxwellfdfd.source import PointSrc
    >>> from maxwellfdfd.source.utils import assign_source
    >>> 
    >>> # Create multiple sources
    >>> src1 = PointSrc(Axis.Z, np.array([0, 0, 0]))
    >>> src2 = PointSrc(Axis.Z, np.array([1, 1, 1]))
    >>> 
    >>> # Assign sources to grid
    >>> JM_cell = assign_source(grid3d, [src1, src2])
    """
    # Validate inputs
    # Note: Grid3d validation will be done when grid module is available
    # chkarg(isinstance(grid3d, Grid3d), '"grid3d" should be instance of Grid3d.')
    # Lazy import to avoid circular import (source.py <-> utils.py).
    from .source import Source
    chkarg(
        all(isinstance(src, Source) for src in src_array),
        '"src_array" should be list of Source instances.'
    )
    
    # Initialize JM_cell with zeros, keyed by Axis (consistent with
    # eps_cell/mu_cell and what MatrixEquation expects). Use complex dtype
    # so the (complex) source amplitudes keep their imaginary part.
    JM_cell: dict = {}
    for w in Axis.elems():
        JM_cell[w] = np.zeros(tuple(int(n) for n in grid3d.N), dtype=complex)
    
    # Superpose all sources
    for src in src_array:
        for w in Axis.elems():
            ind, JMw_patch = src.generate(w, grid3d)
            # np.size() handles both ndarray and scalar JMw_patch (point
            # sources return a scalar for the polarization axis).
            if np.size(JMw_patch) > 0:
                # Convert indices to proper format for numpy indexing
                indices = []
                for i, idx in enumerate(ind):
                    if isinstance(idx, (int, np.integer)):
                        indices.append(idx)
                    elif isinstance(idx, np.ndarray):
                        indices.append(idx)
                    elif isinstance(idx, str) and idx == ':':
                        indices.append(slice(None))
                    else:
                        indices.append(idx)
                
                # Add source contribution
                JM_cell[w][tuple(indices)] += JMw_patch
    
    return JM_cell