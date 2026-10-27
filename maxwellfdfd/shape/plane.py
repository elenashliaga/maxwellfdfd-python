"""
Plane class for plane shapes.

This module defines the Plane class, which represents a plane in 3D space.
"""

from typing import Dict, Union
import numpy as np
from maxwellfdfd.base import Axis, chkarg, expand2row
from .zero_vol_shape import ZeroVolShape


class Plane(ZeroVolShape):
    """
    Plane shape in 3D space.
    
    Plane represents a plane normal to one of the Cartesian axes.
    It does not have a volume, but it is used to force a user-defined
    primary grid plane in simulations.
    
    Parameters
    ----------
    normal_axis : Axis
        Axis normal to the plane (Axis.X, Axis.Y, or Axis.Z).
    intercept : float
        Location of the plane in the normal_axis direction.
    dl_max : float or array_like, optional
        Maximum grid cell size allowed. Can be a scalar or length-3 vector.
        If not provided, defaults to Inf. In the normal_axis direction,
        dl_max is meaningless.
    
    Attributes
    ----------
    normal_axis : Axis
        Axis normal to the plane.
    intercept : float
        Location of the plane.
    
    Examples
    --------
    >>> from maxwellfdfd.base import Axis
    >>> from maxwellfdfd.shape import Plane
    >>> plane = Plane(Axis.Y, 100)
    >>> plane.normal_axis
    <Axis.Y: 'y'>
    >>> plane.intercept
    100.0
    """
    
    def __init__(
        self,
        normal_axis: Axis,
        intercept: float,
        dl_max: Union[float, np.ndarray] = np.inf
    ):
        """
        Initialize a Plane.
        
        Parameters
        ----------
        normal_axis : Axis
            Axis normal to the plane.
        intercept : float
            Location of the plane.
        dl_max : float or array_like, optional
            Maximum grid cell size.
        """
        # Validate normal_axis
        chkarg(isinstance(normal_axis, Axis), '"normal_axis" should be instance of Axis.')
        self.normal_axis = normal_axis
        
        # Validate intercept
        chkarg(np.isreal(intercept), '"intercept" should be real.')
        self.intercept = float(intercept)
        
        # Define level set function
        # LSF is -|n - n0|
        def lsf(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
            """
            Level set function for the plane.
            
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
            level = -np.abs(loc[int(normal_axis)] - self.intercept)
            return level
        
        # Create lprim_cell
        lprim_cell: Dict[Axis, np.ndarray] = {}
        for axis in Axis.elems():
            if axis == normal_axis:
                lprim_cell[axis] = np.array([self.intercept])
            else:
                lprim_cell[axis] = np.array([-np.inf, np.inf])
        
        # Process dl_max
        if isinstance(dl_max, (int, float)):
            dl_max_arr = np.full((1, Axis.count()), float(dl_max))
        else:
            dl_max_arr = expand2row(dl_max, Axis.count())
        
        # Set dl_max to Inf in normal direction
        dl_max_arr[0, int(normal_axis)] = np.inf
        
        # Call parent constructor
        super().__init__(lprim_cell, lsf, dl_max_arr)