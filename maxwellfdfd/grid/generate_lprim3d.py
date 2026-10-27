"""
Generate 3D grid from domain and shapes.

This module provides the generate_lprim3d function for generating
3D grids that respect domain boundaries, PML regions, shapes,
and sources.
"""

from typing import Dict, List, Tuple, Union
import numpy as np
import warnings

from maxwellfdfd.base import Axis, GT, Sign, expand2mat
from maxwellfdfd.shape import Domain, Shape, Interval


def generate_lprim3d(
    domain: Domain,
    Lpml: Union[float, np.ndarray],
    shape_array: List[Shape],
    src_array: List = None,
    withuniform: bool = False
) -> Tuple[Dict[Axis, np.ndarray], np.ndarray]:
    """
    Generate 3D grid from domain and shapes.
    
    This function generates a 3D grid that respects:
    - Domain boundaries
    - PML regions
    - Material shapes
    - Source locations
    
    The grid can be generated either uniformly or with adaptive
    cell sizes that respect material boundaries and source locations.
    
    Parameters
    ----------
    domain : Domain
        Domain shape defining the simulation region.
    Lpml : float or np.ndarray
        PML thicknesses. Can be:
        - Scalar: same thickness for all axes and both ends
        - Length-3 vector: different thicknesses per axis, same for both ends
        - 3x2 matrix: different thicknesses per axis and per end
    shape_array : list of Shape
        Shapes in the domain (e.g., material regions).
    src_array : list, optional
        Sources in the domain. Default is empty list.
    withuniform : bool, optional
        Whether to use uniform grid. Default is False.
    
    Returns
    -------
    tuple
        (lprim_cell, Npml) where:
        - lprim_cell is dict mapping Axis to primary grid arrays
        - Npml is 3x2 array of PML cell counts
    
    Raises
    ------
    ValueError
        If grid generation fails or parameters are invalid.
    
    Notes
    -----
    The grid generation algorithm:
    1. Collects unique intervals from shapes (to avoid redundancy)
    2. Collects primary and dual grid points from sources
    3. For each axis:
       a. Generates partial grid using generate_lprim1d_part
       b. Completes grid using complete_lprim1d
       c. Calculates PML cell counts
    4. If dual grid points are not preserved, retries without them
    
    The algorithm preserves provided subgrids to maintain material and
    source alignment, even if there are roundoff errors in the generated
    filler regions.
    
    Warnings
    --------
    If the generated uniform grid cell size significantly differs from
    the intended cell size (error > 1e-9), a warning is issued.
    
    Examples
    --------
    >>> from maxwellfdfd.base import Axis
    >>> from maxwellfdfd.shape import Domain, Box
    >>> domain = Domain(Box(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.1))
    >>> Lpml = 0.1
    >>> shape_array = []
    >>> src_array = []
    >>> lprim_cell, Npml = generate_lprim3d(
    ...     domain, Lpml, shape_array, src_array, withuniform=True
    ... )
    >>> lprim_cell[Axis.X][0], lprim_cell[Axis.X][-1]
    (0.0, 1.0)
    """
    # Validate domain
    if not isinstance(domain, Domain):
        raise ValueError('"domain" should be instance of Domain.')
    
    # Validate and expand Lpml
    Lpml = np.asarray(Lpml)
    if not np.all(np.isreal(Lpml)):
        raise ValueError('Element of "Lpml" should be real.')
    Lpml = expand2mat(Lpml, Axis.count(), Sign.count())
    
    # Validate shape_array
    if not isinstance(shape_array, list):
        raise ValueError('"shape_array" should be a list.')
    for i, shape in enumerate(shape_array):
        if not isinstance(shape, Shape):
            raise ValueError(f'Element {i} of "shape_array" should be instance of Shape.')
    
    # Validate src_array
    if src_array is None:
        src_array = []
    if not isinstance(src_array, list):
        raise ValueError('"src_array" should be a list.')
    # Note: We don't validate Source type here to avoid circular imports
    
    # Validate withuniform
    if not isinstance(withuniform, bool):
        raise ValueError('"withuniform" should be logical.')
    
    lprim_cell = {}
    Npml = np.full((Axis.count(), Sign.count()), np.nan)
    
    if withuniform:
        # Generate uniform grid
        for axis in Axis.elems():
            # NOTE: dl_max, L, bound are properties (ndarray), not methods.
            dl_intended = domain.dl_max[int(axis)]
            Nw = int(np.round(domain.L[int(axis)] / dl_intended))
            lprim = np.linspace(
                domain.bound[int(axis), int(Sign.N)],
                domain.bound[int(axis), int(Sign.P)],
                Nw + 1
            )
            
            if len(lprim) < 2:
                raise ValueError(
                    f'Generated grid for axis {axis} has fewer than 2 points.'
                )
            
            dl_generated = lprim[1] - lprim[0]
            error_dl = (dl_generated - dl_intended) / dl_intended
            
            if abs(error_dl) > 1e-9:
                warnings.warn(
                    f'Grid vertex separation {dl_generated} in generated uniform grid '
                    f'significantly differs from intended separation {dl_intended}: '
                    f'error is {error_dl * 100} percent.'
                )
            
            Npml[int(axis), 0] = len(lprim[lprim < lprim[0] + Lpml[int(axis), 0]])
            Npml[int(axis), 1] = len(lprim[lprim > lprim[-1] - Lpml[int(axis), 1]])
            lprim_cell[axis] = lprim
    else:
        # Generate adaptive grid
        
        # Collect unique intervals from shapes
        # Many intervals can be the same (e.g., in a photonic crystal)
        intervals = {axis: [] for axis in Axis.elems()}
        
        for shape in shape_array:
            for axis in Axis.elems():
                # NOTE: interval is a list attribute, not a method.
                inter_curr = shape.interval[int(axis)]
                
                # Check if this interval is already in the list
                is_new = True
                for existing in intervals[axis]:
                    if existing == inter_curr:
                        is_new = False
                        break
                
                if is_new:
                    intervals[axis].append(inter_curr)
        
        # Collect primary and dual grid points from sources
        lprim0 = {axis: np.array([]) for axis in Axis.elems()}
        ldual0 = {axis: np.array([]) for axis in Axis.elems()}
        
        for src in src_array:
            # Source.l returns {GT: [x_arr, y_arr, z_arr]}: a dict keyed by GT
            # whose values are length-3 lists indexed by axis.
            if hasattr(src, 'l'):
                src_l = src.l
                for axis in Axis.elems():
                    lprim0[axis] = np.concatenate([
                        lprim0[axis],
                        src_l[GT.PRIM][int(axis)]
                    ])
                    ldual0[axis] = np.concatenate([
                        ldual0[axis],
                        src_l[GT.DUAL][int(axis)]
                    ])
        
        # Remove dual grid points that coincide with primary grid points.
        # This happens e.g. for TFSF sources whose box boundaries produce
        # identical primary and dual locations, which would cause
        # generate_lprim1d_part to reject the input.
        for axis in Axis.elems():
            if len(lprim0[axis]) > 0 and len(ldual0[axis]) > 0:
                tol = 1e-10
                keep = np.ones(len(ldual0[axis]), dtype=bool)
                for dp in ldual0[axis]:
                    if np.any(np.abs(lprim0[axis] - dp) < tol):
                        keep &= np.abs(ldual0[axis] - dp) >= tol
                ldual0[axis] = ldual0[axis][keep]
        
        # Generate grid for each axis
        for axis in Axis.elems():
            try:
                # First try: include dual grid points from sources
                from .generate_lprim1d_part import generate_lprim1d_part
                from .complete_lprim1d import complete_lprim1d
                
                lprim_part = generate_lprim1d_part(
                    domain.interval[int(axis)],
                    Lpml[int(axis), :],
                    intervals[axis],
                    lprim0[axis],
                    ldual0[axis]
                )
                lprim = complete_lprim1d(lprim_part)
            except Exception as err1:
                # Second try: exclude dual grid points
                try:
                    lprim_part = generate_lprim1d_part(
                        domain.interval[int(axis)],
                        Lpml[int(axis), :],
                        intervals[axis],
                        lprim0[axis],
                        np.array([])
                    )
                    lprim = complete_lprim1d(lprim_part)
                    
                    # Calculate dual grid and check if all points are preserved
                    ldual = (lprim[:-1] + lprim[1:]) / 2
                    
                    if len(ldual0[axis]) > 0:
                        missing = np.setdiff1d(ldual0[axis], ldual)
                        if len(missing) > 0:
                            # Dual grid points were not preserved
                            raise ValueError(
                                f'{axis}-axis grid generation failed: '
                                f'dual grid points {missing} were not preserved.'
                            ) from err1
                except Exception as err2:
                    raise ValueError(
                        f'{axis}-axis grid generation failed.'
                    ) from err2
            
            # Calculate PML cell counts
            Npml[int(axis), 0] = len(lprim[lprim < lprim[0] + Lpml[int(axis), 0]])
            Npml[int(axis), 1] = len(lprim[lprim > lprim[-1] - Lpml[int(axis), 1]])
            lprim_cell[axis] = lprim
    
    return lprim_cell, Npml