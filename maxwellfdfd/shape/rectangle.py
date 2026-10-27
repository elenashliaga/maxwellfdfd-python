"""
Rectangle class for rectangular patch shapes.

This module defines the Rectangle class, which represents a rectangular
patch on a plane in 3D space.
"""

from typing import Dict, Union
import numpy as np
from maxwellfdfd.base import Axis, Dir, Sign, chkarg, expand2row
from .zero_vol_shape import ZeroVolShape
from .box import Box
from .plane import Plane


class Rectangle(ZeroVolShape):
    """
    Rectangular patch on a plane in 3D space.
    
    Rectangle represents a rectangle on a plane. It does not have a volume,
    but it is used to force a user-defined primary grid plane and four
    primary grid lines (corresponding to the four sides) in simulations.
    
    Parameters
    ----------
    normal_axis : Axis
        Axis normal to the plane of the rectangle (Axis.X, Axis.Y, or Axis.Z).
    intercept : float
        Location of the rectangle in the normal_axis direction.
    rect : ndarray
        Bounds of the rectangle in the plane as shape (2, 2) array.
        If normal_axis == Axis.Y, it is [[zmin, zmax], [xmin, xmax]].
    dl_max : float or array_like, optional
        Maximum grid cell size allowed. Can be a scalar or length-3 vector.
        If not provided, defaults to Inf. In the normal_axis direction,
        dl_max is meaningless.
    
    Attributes
    ----------
    normal_axis : Axis
        Axis normal to the plane.
    intercept : float
        Location of the rectangle.
    rect : ndarray
        Bounds of the rectangle in the plane.
    
    Examples
    --------
    >>> import numpy as np
    >>> from maxwellfdfd.base import Axis
    >>> from maxwellfdfd.shape import Rectangle
    >>> rect = Rectangle(Axis.Z, 100, np.array([[0, 100], [0, 100]]))
    >>> rect.normal_axis
    <Axis.Z: 'z'>
    >>> rect.intercept
    100.0
    >>> rect.rect
    array([[  0, 100],
           [  0, 100]])
    """
    
    def __init__(
        self,
        normal_axis: Axis,
        intercept: float,
        rect: np.ndarray,
        dl_max: Union[float, np.ndarray] = np.inf
    ):
        """
        Initialize a Rectangle.
        
        Parameters
        ----------
        normal_axis : Axis
            Axis normal to the plane.
        intercept : float
            Location of the rectangle.
        rect : ndarray
            Bounds of the rectangle in the plane.
        dl_max : float or array_like, optional
            Maximum grid cell size.
        """
        # Validate normal_axis
        chkarg(isinstance(normal_axis, Axis), '"normal_axis" should be instance of Axis.')
        self.normal_axis = normal_axis
        
        # Validate intercept
        chkarg(np.isreal(intercept), '"intercept" should be real.')
        self.intercept = float(intercept)
        
        # Validate rect
        chkarg(
            rect.shape == (Dir.count(), Sign.count()) and np.isrealobj(rect),
            f'"rect" should be {Dir.count()}-by-{Sign.count()} array with real elements.'
        )
        self.rect = rect
        
        # Get cyclic axes
        h, v, n = normal_axis.cycle()
        
        # Create bound array
        bound = np.full((Axis.count(), Sign.count()), np.nan)
        bound[int(normal_axis), :] = [self.intercept, self.intercept]
        bound[int(h), :] = rect[int(Dir.H), :]
        bound[int(v), :] = rect[int(Dir.V), :]
        
        # Create box and plane for level set function
        box = Box(bound)
        plane = Plane(normal_axis, self.intercept)
        
        # Define level set function
        # LSF is min(box.lsf, plane.lsf) - intersection of box and plane
        def lsf(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
            """
            Level set function for the rectangle.
            
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
            
            # Create copies and set normal axis coordinate to intercept
            loc_copy = [loc[0].copy(), loc[1].copy(), loc[2].copy()]
            loc_copy[int(normal_axis)] = np.full_like(loc[int(normal_axis)], self.intercept)
            
            # Get level set values from box and plane
            level_box = box.lsf(loc_copy[0], loc_copy[1], loc_copy[2])
            level_plane = plane.lsf(x, y, z)
            
            level = np.minimum(level_box, level_plane)
            return level
        
        # Create lprim_cell
        lprim_cell: Dict[Axis, np.ndarray] = {}
        for axis in Axis.elems():
            lprim_cell[axis] = box.bound[int(axis), :]
        
        # Process dl_max
        if isinstance(dl_max, (int, float)):
            dl_max_arr = np.full((1, Axis.count()), float(dl_max))
        else:
            dl_max_arr = expand2row(dl_max, Axis.count())
        
        # Set dl_max to Inf in normal direction
        dl_max_arr[0, int(normal_axis)] = np.inf
        
        # Call parent constructor
        super().__init__(lprim_cell, lsf, dl_max_arr)