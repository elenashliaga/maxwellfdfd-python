"""
Box class for 3D rectangular boxes.

This module defines the Box class, which represents a 3D rectangular
box aligned with the Cartesian coordinate axes.
"""

from typing import Dict, Union
import numpy as np
from maxwellfdfd.base import Axis, Sign, chkarg
from .shape import Shape


class Box(Shape):
    """
    3D box shape aligned with Cartesian coordinate axes.
    
    Box is the most basic shape in MaxwellFDFD. Its sides are aligned
    with the axes of the Cartesian coordinate system.
    
    Parameters
    ----------
    bound : ndarray
        Bounds of the box as shape (3, 2) array:
        [[xmin, xmax], [ymin, ymax], [zmin, zmax]].
    dl_max : float or array_like, optional
        Maximum grid cell size allowed. Can be a scalar or length-3 vector.
        If not provided, defaults to Inf.
    
    Attributes
    ----------
    bound : ndarray
        Box bounds as shape (3, 2) array.
    
    Examples
    --------
    >>> import numpy as np
    >>> from maxwellfdfd.shape import Box
    >>> box = Box(np.array([[-100, 100], [-50, 50], [0, 20]]))
    >>> box.bound
    array([[-100.,  100.],
           [ -50.,   50.],
           [   0.,   20.]])
    >>> box.contains(0, 0, 10)
    True
    >>> box.contains(200, 0, 10)
    False
    """
    
    def __init__(
        self,
        bound: np.ndarray,
        dl_max: Union[float, np.ndarray] = np.inf
    ):
        """
        Initialize a Box.
        
        Parameters
        ----------
        bound : ndarray
            Box bounds as shape (3, 2) array.
        dl_max : float or array_like, optional
            Maximum grid cell size.
        """
        # Validate bound
        chkarg(
            bound.shape == (Axis.count(), Sign.count()) and np.isrealobj(bound),
            f'"bound" should be {Axis.count()}-by-{Sign.count()} array with real elements.'
        )
        
        # Calculate semisides and center
        s = np.diff(bound, axis=1) / 2  # semisides
        s = s.flatten()  # row vector
        chkarg(
            np.all(s >= 0),
            '"bound" should have smaller lower bound than upper bound in all axes.'
        )
        
        c = np.mean(bound, axis=1)  # center
        
        # Define level set function
        # LSF is 1 - max(|r-c| ./ s) where r = [x, y, z]
        def lsf(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
            """
            Level set function for the box.
            
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
                level = np.maximum(level, np.abs(loc[int(axis)] - c[int(axis)]) / s[int(axis)])
            
            level = 1 - level
            return level
        
        # Create lprim_cell
        lprim_cell: Dict[Axis, np.ndarray] = {}
        for axis in Axis.elems():
            lprim_cell[axis] = bound[int(axis), :]
        
        # Call parent constructor
        super().__init__(lprim_cell, lsf, dl_max)