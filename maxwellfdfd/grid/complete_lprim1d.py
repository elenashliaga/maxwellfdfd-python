"""
Complete 1D primary grid by filling gaps between subgrids.

This module provides the complete_lprim1d function and its helper functions
for generating smoothly varying 1D grids from partially specified grids.
"""

from typing import List, Tuple, Union
import numpy as np
from scipy.optimize import brentq


def complete_lprim1d(lprim_part_cell: List[Union[np.ndarray, float]]) -> np.ndarray:
    """
    Complete a 1D primary grid by filling gaps between subgrids.
    
    This function generates a full 1D primary grid from a partially generated
    grid. The input is a list where each element is either:
    - A subgrid (np.ndarray): Complete grid between two points
    - A target dl (float): Target grid cell size between subgrids
    
    The function fills the gaps between subgrids using geometric progression
    to ensure smooth transitions while respecting the target cell sizes.
    
    Parameters
    ----------
    lprim_part_cell : list
        List where each element is either a subgrid (np.ndarray with at least
        2 elements in ascending order) or a target dl (float). The list must
        have an odd number of elements in the format:
        [subgrid1, dl_target1, subgrid2, dl_target2, ..., subgridN]
    
    Returns
    -------
    np.ndarray
        Completed primary grid locations.
    
    Raises
    ------
    ValueError
        If grid generation fails due to incompatible parameters.
    
    Notes
    -----
    The grid generation algorithm uses geometric progression with:
    - Target ratio rt = 1.9 (can be made stricter with rt = 1.2)
    - Maximum ratio rmax = 2.0 (can be made stricter with rmax = 1.3)
    
    The algorithm preserves the provided subgrids to maintain material and
    source alignment, even if there are roundoff errors in the generated
    filler regions.
    
    Examples
    --------
    >>> import numpy as np
    >>> lprim_part_cell = [
    ...     np.array([0.0, 0.5]),
    ...     1.0,
    ...     np.array([10.0, 11.0, 12.0]),
    ...     1.5,
    ...     np.array([20.5, 21.0]),
    ...     2.0,
    ...     np.array([30.0, 31.0])
    ... ]
    >>> lprim = complete_lprim1d(lprim_part_cell)
    >>> lprim[0], lprim[-1]
    (0.0, 31.0)
    """
    # Validate input
    if not isinstance(lprim_part_cell, list):
        raise ValueError('"lprim_part_cell" should be a list.')
    
    numelems = len(lprim_part_cell)
    if numelems % 2 == 0:
        raise ValueError('"lprim_part_cell" should have odd number of elements.')
    
    # Parse input into list of {subgrid, dl_target} pairs
    ds = []
    for i in range((numelems + 1) // 2):
        if i == (numelems + 1) // 2 - 1:
            # Last element: only subgrid
            curr = [lprim_part_cell[2 * i]]
        else:
            # Middle elements: subgrid and dl_target
            curr = [lprim_part_cell[2 * i], lprim_part_cell[2 * i + 1]]
            
            # Validate dl_target
            dl_target = curr[1]
            if not np.isscalar(dl_target) or not np.isreal(dl_target):
                raise ValueError(
                    f'Element #{2 * i + 1} of "lprim_part_cell" should be real.'
                )
        
        # Validate subgrid
        subgrid = np.asarray(curr[0])
        if len(subgrid) < 2:
            raise ValueError(
                f'Element #{2 * i} of "lprim_part_cell" should have at least 2 elements.'
            )
        if not np.all(np.isreal(subgrid)):
            raise ValueError(
                f'Element #{2 * i} of "lprim_part_cell" should have real elements.'
            )
        if not np.all(np.diff(subgrid) > 0):
            raise ValueError(
                f'Element #{2 * i} of "lprim_part_cell" should be in ascending order.'
            )
        
        # Check ordering between consecutive subgrids
        if i >= 1:
            prev_subgrid = np.asarray(ds[-1][0])
            if prev_subgrid[-1] >= subgrid[0]:
                raise ValueError(
                    f'Subgrids {prev_subgrid} and {subgrid} in "lprim_part_cell" '
                    'should be sorted in ascending order.'
                )
        
        ds.append(curr)
    
    # Handle single subgrid case
    if numelems == 1:
        return np.asarray(ds[0][0])
    
    # Fill gaps between neighboring subgrids
    rt = 1.9  # target ratio of geometric sequence
    rmax = 2.0  # maximum ratio of geometric sequence
    numgrids = len(ds)
    
    curr = ds[0]  # [subgrid, dl_target] or [subgrid]
    lprim_array = np.asarray(curr[0])
    
    for i in range(1, numgrids):
        dl_n = lprim_array[-1] - lprim_array[-2]  # dl of current subgrid
        next_part = ds[i]  # [subgrid, dl_target] or [subgrid]
        dl_p = np.asarray(next_part[0])[1] - np.asarray(next_part[0])[0]  # dl of next subgrid
        gap = np.array([lprim_array[-1], np.asarray(next_part[0])[0]])
        dl_target = curr[1] if len(curr) > 1 else None
        
        try:
            filler = _fill_targeted_geometric(dl_n, gap, dl_target, dl_p, rt, rmax)
        except Exception as err:
            raise ValueError(
                f'Grid generation failed between subgrids {curr[0]} and {next_part[0]} '
                f'with target dl = {dl_target}: {err}'
            ) from err
        
        # Preserve provided subgrids despite roundoff errors in filler
        # This is important for assigning materials and sources at intended locations
        lprim_array = np.concatenate([
            lprim_array,
            filler[1:-1],  # Exclude endpoints to preserve subgrids
            np.asarray(next_part[0])
        ])
        curr = next_part
    
    # Check for stiff second derivatives
    dlprim_array = np.diff(lprim_array)
    ind = _find_stiff_ddl(dlprim_array, rmax)
    if len(ind) > 0:
        raise ValueError(
            f'Grid generation failed: grid vertex locations '
            f'[..., {lprim_array[ind[0]]}, {lprim_array[ind[0]+1]}, {lprim_array[ind[0]+2]}, ...] '
            f'are separated by [..., {dlprim_array[ind[0]]}, {dlprim_array[ind[0]+1]}, ...] '
            'that do not vary smoothly.'
        )
    
    return lprim_array


def _find_stiff_ddl(dl_array: np.ndarray, rt: float) -> np.ndarray:
    """
    Find indices where second derivative is too stiff.
    
    Parameters
    ----------
    dl_array : np.ndarray
        Array of cell sizes.
    rt : float
        Target ratio threshold.
    
    Returns
    -------
    np.ndarray
        Indices where the ratio of consecutive cell sizes exceeds the threshold.
    """
    if len(dl_array) < 2:
        return np.array([])
    
    dl1 = dl_array[:-1]
    dl2 = dl_array[1:]
    
    if rt >= 1:
        ind = np.concatenate([
            np.where(dl1 / dl2 > rt)[0],
            np.where(dl1 / dl2 < 1/rt)[0]
        ])
    else:
        ind = np.concatenate([
            np.where(dl1 / dl2 > 1/rt)[0],
            np.where(dl1 / dl2 < rt)[0]
        ])
    
    return np.unique(ind)


def _isequal_approx(a: float, b: float) -> bool:
    """
    Check if two values are approximately equal.
    
    Parameters
    ----------
    a, b : float
        Values to compare.
    
    Returns
    -------
    bool
        True if values are approximately equal.
    """
    d = abs(a - b)
    return d < min([a, b]) * 1e-8


def _is_smooth(dl_array: np.ndarray, rt: float) -> bool:
    """
    Check if cell sizes vary smoothly.
    
    Parameters
    ----------
    dl_array : np.ndarray
        Array of cell sizes.
    rt : float
        Target ratio threshold.
    
    Returns
    -------
    bool
        True if cell sizes vary smoothly.
    """
    return len(_find_stiff_ddl(dl_array, rt)) == 0


def _fill_constant(
    dl_min: float,
    dl_max: float,
    gap: np.ndarray,
    rt: float,
    rmax: float
) -> np.ndarray:
    """
    Fill gap with constant cell size.
    
    Parameters
    ----------
    dl_min : float
        Minimum cell size.
    dl_max : float
        Maximum cell size.
    gap : np.ndarray
        Length-2 array [gap_start, gap_end].
    rt : float
        Target ratio (unused in constant fill).
    rmax : float
        Maximum ratio threshold.
    
    Returns
    -------
    np.ndarray
        Grid locations filling the gap.
    
    Raises
    ------
    ValueError
        If dl_min > dl_max or if gap cannot be filled smoothly.
    """
    if dl_min > dl_max and not _isequal_approx(dl_min, dl_max):
        raise ValueError(
            f'"dl_min = {dl_min}" should not be greater than "dl_max = {dl_max}".'
        )
    
    if not _is_smooth(np.array([dl_min, dl_max]), rt):
        raise ValueError('"dl_min" and "dl_max" should be similar.')
    
    L = gap[1] - gap[0]
    if L <= 0:
        raise ValueError('Second element of "gap" should be greater than the first element.')
    
    # Candidate numbers of grid cells
    numcells = [int(np.floor(L / dl_max)), int(np.ceil(L / dl_max))]
    dl = L / np.array(numcells)
    
    # Check if either candidate works
    if not (_is_smooth(np.array([dl_min, dl[0]]), rmax) or
            _is_smooth(np.array([dl_min, dl[1]]), rmax)):
        raise ValueError(
            f'dl = {dl[0]} is too small or dl = {dl[1]} is too large for gap size = {L}.'
        )
    
    # Prefer coarser grid unless it fails
    if not _is_smooth(np.array([dl_min, dl[0]]), rmax):
        numcells = numcells[1]  # finer grid
    else:
        numcells = numcells[0]  # coarser grid
    
    return np.linspace(gap[0], gap[1], numcells + 1)


def _fill_targeted_geometric_sym(
    dl_sym: float,
    gap: np.ndarray,
    dl_t: float,
    rt: float,
    rmax: float
) -> np.ndarray:
    """
    Fill gap with symmetric geometric progression.
    
    Generate a filler subgrid whose grid vertex separations increase
    geometrically from both ends. The target dl should be equal to or greater
    than dl_n before the gap and dl_p after the gap, with dl_n == dl_p.
    
    Parameters
    ----------
    dl_sym : float
        dl of the subgrids before and after the gap (dl_n == dl_p).
    gap : np.ndarray
        Length-2 array [gap_start, gap_end].
    dl_t : float
        Target dl in the gap.
    rt : float
        Target ratio of geometric sequence.
    rmax : float
        Maximum ratio of geometric sequence.
    
    Returns
    -------
    np.ndarray
        Grid locations filling the gap.
    
    Raises
    ------
    ValueError
        If dl_t < dl_sym or if gap cannot be filled.
    """
    if dl_t < dl_sym and not _isequal_approx(dl_t, dl_sym):
        raise ValueError(
            f'"dl_t = {dl_t}" should be equal to or greater than "dl_sym = {dl_sym}".'
        )
    
    L = gap[1] - gap[0]
    if L <= 0:
        raise ValueError('Second element of "gap" should be greater than the first element.')
    
    # If smooth, use constant fill
    if _is_smooth(np.array([dl_sym, dl_t]), rt):
        return _fill_constant(dl_sym, dl_t, gap, rt, rmax)
    
    dl_max = dl_t
    dl_min = dl_sym
    
    # Guess graded dl's
    n = int(np.ceil(np.log(dl_max / dl_min) / np.log(rt)))
    r = (dl_max / dl_min) ** (1.0 / n)
    dl_array = dl_min * (r ** np.arange(1, n + 1))
    
    L_graded = np.sum(dl_array)
    
    if 2 * L_graded > L:
        # Try best to geometrically increase dl close to dl_t
        def func1(n_val):
            return dl_min * rt * (rt ** (n_val - 1) + 2 * (rt ** (n_val - 1) - 1) / (rt - 1)) - L
        
        def func2(n_val):
            return dl_min * rt * 2 * (rt ** n_val - 1) / (rt - 1) - L
        
        try:
            n1 = brentq(func1, 1, 100)
            n2 = brentq(func2, 1, 100)
        except ValueError:
            raise ValueError(
                f'dl = {dl_min} is too small or dl = {dl_max} is too large for gap size = {L}.'
            )
        
        # Choose the one closer to its ceiling
        if abs(np.ceil(n1) - n1) < abs(np.ceil(n2) - n2):
            n = int(np.ceil(n1))
            i = 0
        else:
            n = int(np.ceil(n2))
            i = 1
        
        if n < 1:
            raise ValueError(
                f'dl = {dl_min} is too small or dl = {dl_max} is too large for gap size = {L}.'
            )
        
        if i == 0:
            def func_r(s):
                return dl_min * s * (s ** (n - 1) + 2 * (s ** (n - 1) - 1) / (s - 1)) - L
            
            try:
                r = brentq(func_r, 1/rt, rt)
            except ValueError:
                raise ValueError(
                    f'dl = {dl_min} is too small or dl = {dl_max} is too large for gap size = {L}.'
                )
            
            if r > rt or r < 1/rt:
                raise ValueError(
                    f'dl = {dl_min} is too small or dl = {dl_max} is too large for gap size = {L}.'
                )
            
            dl_filler = dl_min * (r ** np.concatenate([np.arange(1, n + 1), np.arange(n - 1, 0, -1)]))
        else:
            def func_r(s):
                return dl_min * s * (s ** n - 1) / (s - 1) - L / 2
            
            try:
                r = brentq(func_r, 1/rt, rt)
            except ValueError:
                raise ValueError(
                    f'dl = {dl_min} is too small or dl = {dl_max} is too large for gap size = {L}.'
                )
            
            if r > rt or r < 1/rt:
                raise ValueError(
                    f'dl = {dl_min} is too small or dl = {dl_max} is too large for gap size = {L}.'
                )
            
            dl_filler = dl_min * (r ** np.concatenate([np.arange(1, n + 1), np.arange(n, 0, -1)]))
    else:
        # Slightly under-fill the gap with dl_max and graded dl's
        n_dl_max = int(np.floor((L - 2 * L_graded) / dl_max))
        dl_max_array = np.full(n_dl_max, dl_max)
        L_dl_max = np.sum(dl_max_array)
        
        # Slightly over-fill the gap with dl_min, graded dl's, and L_dl_max
        if 2 * L_graded + L_dl_max > L:
            raise ValueError(
                f'dl = {dl_min} is too small or dl = {dl_max} is too large for gap size = {L}.'
            )
        
        n_dl_min = int(np.ceil((L - 2 * L_graded - L_dl_max) / 2 / dl_min))
        dl_min_array = np.full(n_dl_min, dl_min)
        L_dl_min = np.sum(dl_min_array)
        
        # Update the graded dl's
        L_graded = (L - L_dl_max - 2 * L_dl_min) / 2
        
        def func_r(s):
            return dl_min * s * (s ** n - 1) / (s - 1) - L_graded
        
        try:
            r = brentq(func_r, 1/rt, rt)
        except ValueError:
            raise ValueError(
                f'dl = {dl_min} is too small or dl = {dl_max} is too large for gap size = {L}.'
            )
        
        dl_array = dl_min * (r ** np.arange(1, n + 1))
        dl_filler = np.concatenate([
            dl_min_array,
            dl_array,
            dl_max_array,
            dl_array[::-1],
            dl_min_array
        ])
    
    # Construct filler
    return np.concatenate([[gap[0]], gap[0] + np.cumsum(dl_filler)])


def _fill_targeted_geometric(
    dl_n: float,
    gap: np.ndarray,
    dl_t: float,
    dl_p: float,
    rt: float,
    rmax: float
) -> np.ndarray:
    """
    Fill gap with asymmetric geometric progression.
    
    Generate a filler subgrid whose grid vertex separations increase
    geometrically from both ends. Handles the case where dl_n != dl_p.
    
    Parameters
    ----------
    dl_n : float
        dl of the subgrid before the gap.
    gap : np.ndarray
        Length-2 array [gap_start, gap_end].
    dl_t : float
        Target dl in the gap.
    dl_p : float
        dl of the subgrid after the gap.
    rt : float
        Target ratio of geometric sequence.
    rmax : float
        Maximum ratio of geometric sequence.
    
    Returns
    -------
    np.ndarray
        Grid locations filling the gap.
    
    Raises
    ------
    ValueError
        If dl_t < dl_n or dl_t < dl_p, or if gap cannot be filled.
    """
    if (dl_t < dl_n and not _isequal_approx(dl_t, dl_n)) or \
       (dl_t < dl_p and not _isequal_approx(dl_t, dl_p)):
        raise ValueError(
            f'"dl_t = {dl_t}" should be equal to or greater than '
            f'"dl_n = {dl_n}" and "dl_p = {dl_p}".'
        )
    
    L = gap[1] - gap[0]
    if L <= 0:
        raise ValueError('Second element of "gap" should be greater than the first element.')
    
    if _isequal_approx(dl_n, dl_p):
        return _fill_targeted_geometric_sym(dl_n, gap, dl_t, rt, rmax)
    
    # dl_n < dl_p or dl_n > dl_p
    dl_max = max(dl_n, dl_p)
    dl_min = min(dl_n, dl_p)
    
    # Guess graded dl's
    n = int(np.ceil(np.log(dl_max / dl_min) / np.log(rt)))
    r = (dl_max / dl_min) ** (1.0 / n)
    dl_array = dl_min * (r ** np.arange(1, n + 1))
    
    # Slightly under-fill the gap with dl_max and graded dl's
    L_graded = np.sum(dl_array)
    if L_graded > L:
        raise ValueError(
            f'dl = {dl_min} is too small or dl = {dl_max} is too large for gap size = {L}.'
        )
    
    if dl_n < dl_p:
        filler_n = np.concatenate([[gap[0]], gap[0] + np.cumsum(dl_array)])
        gap_sym = np.array([filler_n[-1], gap[1]])
        filler_sym = _fill_targeted_geometric_sym(dl_p, gap_sym, dl_t, rt, rmax)
        filler = np.concatenate([filler_n[:-1], filler_sym])
    else:
        # dl_n > dl_p
        filler_p = np.concatenate([[gap[1]], gap[1] - np.cumsum(dl_array)])
        filler_p = filler_p[::-1]
        gap_sym = np.array([gap[0], filler_p[0]])
        filler_sym = _fill_targeted_geometric_sym(dl_n, gap_sym, dl_t, rt, rmax)
        filler = np.concatenate([filler_sym, filler_p[1:]])
    
    return filler