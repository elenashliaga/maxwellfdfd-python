"""
Point class for point shapes.

This module defines the Point class, which represents a point in 3D space.
"""

from typing import Dict
import numpy as np
from maxwellfdfd.base import Axis, chkarg
from .zero_vol_shape import ZeroVolShape


class Point(ZeroVolShape):
    """
    Point shape in 3D space.
    
    Point represents a single point in 3D space. It does not have a volume,
    but it is used to force a user-defined primary grid point in simulations.
    
    Parameters
    ----------
    location : array_like
        Location of the point as [x, y, z].
    
    Attributes
    ----------
    location : ndarray
        Point coordinates as [x, y, z].
    
    Examples
    --------
    >>> import numpy as np
    >>> from maxwellfdfd.shape import Point
    >>> point = Point(np.array([0, 0, 0]))
    >>> point.location
    array([0, 0, 0])
    """
    
    def __init__(self, location: np.ndarray):
        """
        Initialize a Point.
        
        Parameters
        ----------
        location : array_like
            Location of the point.
        """
        # Validate location
        location_arr = np.asarray(location)
        chkarg(
            location_arr.shape == (Axis.count(),) and np.isrealobj(location_arr),
            f'"location" should be length-{Axis.count()} array with real elements.'
        )
        self.location = location_arr
        
        # Define level set function
        # LSF is -max(|x-x0|, |y-y0|, |z-z0|)
        def lsf(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
            """
            Level set function for the point.
            
            Parameters
            ----------
            x, y, z : ndarray
                Coordinate arrays.
            
            Returns
            -------
            ndarray
                Level set function values.
            """
            chkarg(np.isrealobj(x), '"x" should be array with real elements.')
            chkarg(np.isrealobj(y), '"y" should be array with real elements.')
            chkarg(np.isrealobj(z), '"z" should be array with real elements.')
            chkarg(
                x.shape == y.shape == z.shape,
                '"x", "y", "z" should have same size.'
            )
            
            loc = [x, y, z]
            level = np.full_like(x, -np.inf)
            
            for axis in Axis.elems():
                level = np.maximum(level, np.abs(loc[int(axis)] - self.location[int(axis)]))
            
            level = -level
            return level
        
        # Create lprim_cell
        lprim_cell: Dict[Axis, np.ndarray] = {}
        for axis in Axis.elems():
            lprim_cell[axis] = np.array([self.location[int(axis)]])
        
        # Call parent constructor (dl_max is meaningless for a point)
        super().__init__(lprim_cell, lsf)