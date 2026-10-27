"""
ZeroVolShape class for shapes without volume.

This module defines the ZeroVolShape class, which is a base class
for shapes that have zero volume (points, lines, planes, etc.).
"""

from typing import Callable, Dict, Union
import numpy as np
from maxwellfdfd.base import Axis, Sign, chkarg
from .shape import Shape


class ZeroVolShape(Shape):
    """
    Base class for shapes without volume.
    
    ZeroVolShape is the superclass of shapes without volume such as
    planes, points, and lines. It modifies the level set function to
    help with visualization, as zero-volume shapes are hard to visualize.
    
    Parameters
    ----------
    lprim_cell : dict
        Dictionary mapping Axis enums to arrays of primary grid locations.
    lsf : callable
        Level set function with signature lsf(x, y, z) -> ndarray.
    dl_max : float or array_like, optional
        Maximum grid cell size allowed.
    
    Notes
    -----
    The level set function is modified to shift it upward when all values
    are negative, which helps with visualization of zero-volume shapes.
    """
    
    def __init__(
        self,
        lprim_cell: Dict[Axis, np.ndarray],
        lsf: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
        dl_max: Union[float, np.ndarray] = np.inf
    ):
        """
        Initialize a ZeroVolShape.
        
        Parameters
        ----------
        lprim_cell : dict
            Dictionary mapping Axis enums to arrays of primary grid locations.
        lsf : callable
            Level set function.
        dl_max : float or array_like, optional
            Maximum grid cell size.
        """
        # Create wrapper level set function that handles visualization
        def lsf_zv(
            x: Union[float, np.ndarray],
            y: Union[float, np.ndarray],
            z: Union[float, np.ndarray],
            force_draw: bool = False
        ) -> np.ndarray:
            """
            Modified level set function for zero-volume shapes.
            
            Parameters
            ----------
            x, y, z : float or array_like
                Coordinates.
            force_draw : bool, optional
                If True, shift the function upward when all values are negative.
            
            Returns
            -------
            ndarray
                Level set function values.
            """
            level = lsf(x, y, z)
            
            if force_draw:
                level_copy = level.flatten()
                if np.all(level_copy <= 0):
                    # If the level set function is negative, nothing is drawn,
                    # so shift the function upward.
                    max_level = np.max(level_copy)
                    # Get second largest value
                    below_max = level_copy[level_copy < max_level]
                    if below_max.size > 0:
                        max_level = np.max(below_max)
                    level = level - max_level
            
            return level
        
        # Call parent constructor with modified level set function
        super().__init__(lprim_cell, lsf_zv, dl_max)