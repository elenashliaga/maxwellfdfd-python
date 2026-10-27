"""
Generate 1D primary grid part from domain and intervals.

This module provides the generate_lprim1d_part function for generating
partial 1D grids that respect domain boundaries, PML regions, and
material intervals.
"""

from typing import List, Union
import numpy as np
import warnings

from maxwellfdfd.base import Sign, expand2row
from maxwellfdfd.shape import Interval


def generate_lprim1d_part(
    domain: Interval,
    Lpml: np.ndarray,
    interval_array: List[Interval],
    lprim0_array: np.ndarray,
    ldual0_array: np.ndarray
) -> List[Union[np.ndarray, float]]:
    """
    Generate 1D primary grid part from domain and intervals.
    
    This function generates a partial 1D grid that respects:
    - Domain boundaries
    - PML regions
    - Material intervals
    - Initial primary and dual grid points
    
    The output is a list in the format:
    [subgrid1, dl_target1, subgrid2, dl_target2, ..., subgridN]
    which can be passed to complete_lprim1d to generate the full grid.
    
    Parameters
    ----------
    domain : Interval
        Domain interval defining the simulation region.
    Lpml : np.ndarray
        PML thicknesses [negative, positive].
    interval_array : list of Interval
        Intervals defining regions with specific grid requirements.
    lprim0_array : np.ndarray
        Initial primary grid points.
    ldual0_array : np.ndarray
        Initial dual grid points.
    
    Returns
    -------
    list
        List of subgrids and target dl values in the format:
        [subgrid1, dl_target1, subgrid2, dl_target2, ..., subgridN]
    
    Raises
    ------
    ValueError
        If primary and dual grids share points, or if parameters are invalid.
    
    Notes
    -----
    The algorithm:
    1. Collects boundary points from domain, PML interfaces, and intervals
    2. Generates primary grid points around dual grid points
    3. Determines minimum dl for each interval between primary points
    4. Creates subgrids with consistent cell sizes on both sides of primary nodes
    
    The same cell size on both sides of primary nodes is important for
    correct averaging of material properties (e.g., eps = 2/(1/eps1 + 1/eps2)).
    
    Examples
    --------
    >>> from maxwellfdfd.base import Sign
    >>> from maxwellfdfd.shape import Interval
    >>> domain = Interval(0.0, 10.0, 1.0)
    >>> Lpml = np.array([1.0, 1.0])
    >>> interval_array = [Interval(2.0, 4.0, 0.1), Interval(6.0, 8.0, 0.1)]
    >>> lprim0_array = np.array([])
    >>> ldual0_array = np.array([])
    >>> lprim_part = generate_lprim1d_part(
    ...     domain, Lpml, interval_array, lprim0_array, ldual0_array
    ... )
    """
    # Validate domain
    if not isinstance(domain, Interval):
        raise ValueError('"domain" should be instance of Interval.')
    dl_max = domain.dl_max
    
    # Validate and expand Lpml
    Lpml = np.asarray(Lpml)
    if not np.all(np.isreal(Lpml)):
        raise ValueError('Element of "Lpml" should be real.')
    if not np.all(Lpml >= 0):
        raise ValueError('Element of "Lpml" should be positive.')
    # NOTE: expand2row returns a (1, n) array; flatten to 1-D so it can be
    # indexed as Lpml[0], Lpml[1]. Also, Interval.L is a method, not a property.
    Lpml = expand2row(Lpml, Sign.count()).reshape(-1)
    
    if domain.L() < Lpml[0] + Lpml[1]:
        raise ValueError(
            'Size of "domain" should be at least the total depth of PML.'
        )
    
    # Validate interval_array
    if not isinstance(interval_array, list):
        raise ValueError('"interval_array" should be a list.')
    for i, interval in enumerate(interval_array):
        if not isinstance(interval, Interval):
            raise ValueError(f'Element {i} of "interval_array" should be instance of Interval.')
    
    # Validate lprim0_array and ldual0_array
    lprim0_array = np.asarray(lprim0_array)
    if not np.all(np.isreal(lprim0_array)):
        raise ValueError('"lprim0_array" should have real elements.')
    
    ldual0_array = np.asarray(ldual0_array)
    if not np.all(np.isreal(ldual0_array)):
        raise ValueError('"ldual0_array" should have real elements.')
    
    # Collect boundary points
    # NOTE: Interval.bound is an ndarray attribute, not a method.
    b_min = domain.bound[int(Sign.N)]
    b_max = domain.bound[int(Sign.P)]
    b_pml_min = b_min + Lpml[0]
    b_pml_max = b_max - Lpml[1]
    
    lprim0_array = np.concatenate([
        lprim0_array,
        np.array([b_min, b_pml_min, b_pml_max, b_max])
    ])
    
    # Add primary grid points from intervals
    lprim_inter = np.array([])
    for interval in interval_array:
        lprim_inter = np.concatenate([lprim_inter, interval.lprim])
    
    is_in = (lprim_inter > b_min) & (lprim_inter < b_max)
    lprim0_array = np.concatenate([lprim0_array, lprim_inter[is_in]])
    
    # Remove duplicates and sort
    lprim0_array = np.unique(lprim0_array)
    
    # Remove near-duplicates (within dl_max * 1e-8)
    def isequal_approx(a, b):
        return abs(a - b) < dl_max * 1e-8
    
    dlprim0_array = np.diff(lprim0_array)
    ind_unique = ~np.array([isequal_approx(dlprim0_array[i], 0) for i in range(len(dlprim0_array))])
    lprim0_array = np.concatenate([lprim0_array[:-1][ind_unique], [lprim0_array[-1]]])
    
    # Check if last two points are duplicates
    if len(lprim0_array) >= 2 and isequal_approx(lprim0_array[-1], lprim0_array[-2]):
        lprim0_array = lprim0_array[:-1]
    
    # Verify boundaries
    if not (np.isclose(lprim0_array[0], b_min) and np.isclose(lprim0_array[-1], b_max)):
        raise ValueError('Primary grid does not match domain boundaries.')
    
    # Process ldual0_array
    ldual0_array = np.unique(ldual0_array)
    
    # Check for common points between primary and dual grids
    common = np.intersect1d(lprim0_array, ldual0_array)
    if len(common) > 0:
        raise ValueError(f'Primary and dual grid share points: {common}')
    
    # Generate primary grid points around dual grid points
    dl_prim0_array = np.diff(lprim0_array)
    dl_dual0_array = np.diff(ldual0_array)
    if len(dl_dual0_array) > 0:
        dl_dual0_array = np.minimum(
            np.concatenate([[dl_dual0_array[0]], dl_dual0_array]),
            np.concatenate([dl_dual0_array, [dl_dual0_array[-1]]])
        )
    else:
        dl_dual0_array = np.array([np.inf])
    
    n_dual0 = len(ldual0_array)
    lprim_by_ldual0 = np.array([])
    
    for j in range(n_dual0):
        val = ldual0_array[j]
        ind = np.where(lprim0_array < val)[0]
        if len(ind) == 0:
            continue
        ind = ind[-1]
        
        dl_min = min([dl_max, dl_dual0_array[j], dl_prim0_array[ind]])
        
        # Check if any interval contains this dual point
        for interval in interval_array:
            # NOTE: Interval.contains() returns a (truth, distance) tuple.
            if interval.contains(val)[0]:
                dl = interval.dl_max
                if dl < dl_min:
                    dl_min = dl
        
        # Add primary points on both sides of dual point
        lprim_new = val - dl_min / 2
        if lprim_new >= b_min and lprim_new <= b_max:
            lprim_by_ldual0 = np.concatenate([lprim_by_ldual0, [lprim_new]])
        
        lprim_new = val + dl_min / 2
        if lprim_new >= b_min and lprim_new <= b_max:
            lprim_by_ldual0 = np.concatenate([lprim_by_ldual0, [lprim_new]])
    
    lprim0_array = np.concatenate([lprim0_array, lprim_by_ldual0])
    lprim0_array = np.unique(lprim0_array)
    
    # For each interval between primary grid points, find the smallest dl
    lprim0_mid_array = (lprim0_array[:-1] + lprim0_array[1:]) / 2
    dl_prim0_array = np.diff(lprim0_array)
    
    n_prim0 = len(lprim0_array)
    dl_mid_array = np.full(n_prim0 - 1, np.nan)
    
    for j in range(n_prim0 - 1):
        val = lprim0_mid_array[j]
        dl_min = min([dl_max, dl_prim0_array[j]])
        
        # Check if any interval contains this midpoint
        for interval in interval_array:
            # NOTE: Interval.contains() returns a (truth, distance) tuple.
            if interval.contains(val)[0]:
                dl = interval.dl_max
                if dl < dl_min:
                    dl_min = dl
        
        dl_mid_array[j] = dl_min
    
    # Get minimum dl for each boundary (use min with adjacent intervals)
    dl_boundary_array = np.minimum(
        np.concatenate([[np.inf], dl_mid_array]),
        np.concatenate([dl_mid_array, [np.inf]])
    )
    
    # Create subgrids
    val = lprim0_array[0]
    dl = dl_boundary_array[0]
    prev = np.array([val, val + dl])
    lprim_part_cell = [prev]
    
    for j in range(1, n_prim0):
        val = lprim0_array[j]
        dl = dl_boundary_array[j]
        
        if np.isclose(val, b_max):
            curr = np.array([val - dl, val])
        else:  # val != b_min or b_max
            # Same cell size on both sides of primary node
            curr = np.array([val - dl, val, val + dl])
        
        # Check if we can merge with previous subgrid
        if (isequal_approx(curr[0], prev[-2]) and isequal_approx(curr[1], prev[-1])):
            # curr[0] == prev[-2] and curr[1] == prev[-1]
            curr = np.concatenate([prev, curr[2:]])
            lprim_part_cell[-1] = curr
        elif isequal_approx(curr[0], prev[-1]):
            # curr[0] == prev[-1]
            curr = np.concatenate([prev, curr[1:]])
            lprim_part_cell[-1] = curr
        else:
            # Need to add target dl between subgrids
            lprim_part_cell.append(dl_mid_array[j - 1])
            lprim_part_cell.append(curr)
        
        prev = curr
    
    return lprim_part_cell