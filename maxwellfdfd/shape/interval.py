"""
Interval class for 1D intervals in MaxwellFDFD.

This module defines the Interval class, which represents a 1D interval
and is used as a building block for higher-dimensional shapes.
"""

import numpy as np
from typing import Tuple, Union
from maxwellfdfd.base import Sign, chkarg


class Interval:
    """
    1D interval class used as a building block for shapes.
    
    An Interval represents a 1D range [min, max] with associated
    primary grid locations and maximum grid cell size.
    
    Parameters
    ----------
    lprim_array : array_like
        Row vector of primary grid locations. Must be nonempty with real elements.
    dl_max : float
        Maximum grid cell size allowed. Must be positive.
    
    Attributes
    ----------
    lprim : ndarray
        Sorted unique primary grid locations.
    bound : ndarray
        Bounds of the interval as [min, max].
    dl_max : float
        Maximum grid cell size.
    
    Examples
    --------
    >>> import numpy as np
    >>> from maxwellfdfd.shape import Interval
    >>> interval = Interval(np.array([0, 1, 2, 3]), 0.5)
    >>> interval.lprim
    array([0, 1, 2, 3])
    >>> interval.bound
    array([0, 3])
    >>> interval.L()
    3
    >>> truth, distance = interval.contains(1.5)
    >>> truth
    True
    """
    
    def __init__(self, lprim_array: np.ndarray, dl_max: float):
        """
        Initialize an Interval.
        
        Parameters
        ----------
        lprim_array : array_like
            Row vector of primary grid locations.
        dl_max : float
            Maximum grid cell size allowed.
        """
        # Validate lprim_array
        lprim_arr = np.asarray(lprim_array)
        chkarg(
            lprim_arr.ndim == 1 and lprim_arr.size > 0 and np.isrealobj(lprim_arr),
            '"lprim_array" should be nonempty 1D array with real elements.'
        )
        
        # Store sorted unique primary locations
        self.lprim = np.unique(lprim_arr)
        
        # Set bounds
        self.bound = np.array([np.min(lprim_arr), np.max(lprim_arr)])
        
        # Validate and store dl_max
        # Accept numpy scalar types as well as Python int/float.
        chkarg(
            isinstance(dl_max, (int, float, np.number)) and dl_max > 0,
            '"dl_max" should be positive.'
        )
        self.dl_max = float(dl_max)
    
    def L(self) -> float:
        """
        Get the length of the interval.
        
        Returns
        -------
        float
            Length of the interval (bound[1] - bound[0]).
        """
        return self.bound[int(Sign.P)] - self.bound[int(Sign.N)]
    
    def contains(self, val: Union[float, np.ndarray]) -> Union[bool, Tuple[bool, np.ndarray]]:
        """
        Check if value(s) are contained in the interval.
        
        Parameters
        ----------
        val : float or array_like
            Value(s) to check.
        
        Returns
        -------
        truth : bool or ndarray
            Boolean indicating if each value is contained in the interval.
        distance : ndarray, optional
            Distance from the value to the nearest boundary.
            Only returned if requested.
        
        Examples
        --------
        >>> import numpy as np
        >>> from maxwellfdfd.shape import Interval
        >>> interval = Interval(np.array([0, 1, 2, 3]), 0.5)
        >>> interval.contains(1.5)
        True
        >>> interval.contains(5.0)
        False
        >>> truth, distance = interval.contains(1.5)
        >>> distance
        1.5
        """
        # Validate input
        val_arr = np.asarray(val)
        chkarg(
            np.isrealobj(val_arr),
            '"val" should be array with real elements.'
        )
        
        bn = self.bound[int(Sign.N)]
        bp = self.bound[int(Sign.P)]
        
        truth = (val_arr >= bn) & (val_arr <= bp)
        
        # Return distance if requested (check if caller expects multiple returns)
        # In Python, we can't detect nargout like MATLAB, so we always return both
        # but the caller can ignore the distance if not needed
        distance = np.minimum(np.abs(val_arr - bn), np.abs(val_arr - bp))
        
        return truth, distance