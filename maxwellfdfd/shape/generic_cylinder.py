"""
GenericCylinder class for cylinders with arbitrary 2D cross sections.

This module defines the GenericCylinder class, which is the base class
for all cylinder shapes with arbitrary 2D cross sections.
"""

from abc import ABC
from typing import Callable, Dict, Union
import numpy as np
from maxwellfdfd.base import Axis, chkarg
from .shape import Shape


class GenericCylinder(Shape, ABC):
    """
    Abstract base class for cylinders with arbitrary 2D cross sections.
    
    GenericCylinder represents a cylinder whose cross section is any 2D shape.
    The axis of the cylinder should be aligned with one of the axes of the
    Cartesian coordinate system.
    
    Parameters
    ----------
    normal_axis : Axis
        Axis of the cylinder (Axis.X, Axis.Y, or Axis.Z).
    lsf2d : callable
        2D level set function for the cross section with signature
        lsf2d(p, q) -> ndarray, where (p, q) are coordinates in the
        lateral domain.
    lprim_cell : dict
        Dictionary mapping Axis enums to arrays of primary grid locations.
    dl_max : float or array_like, optional
        Maximum grid cell size allowed. Can be a scalar or length-3 vector.
        If not provided, defaults to Inf.
    
    Attributes
    ----------
    normal_axis : Axis
        Axis of the cylinder.
    lsf2d : callable
        2D level set function for the cross section.
    
    Notes
    -----
    The level set function is the minimum of the 2D cross section level set
    function and the level set function for the height constraint.
    """
    
    def __init__(
        self,
        normal_axis: Axis,
        lsf2d: Callable[[np.ndarray, np.ndarray], np.ndarray],
        lprim_cell: Dict[Axis, np.ndarray],
        dl_max: Union[float, np.ndarray] = np.inf
    ):
        """
        Initialize a GenericCylinder.
        
        Parameters
        ----------
        normal_axis : Axis
            Axis of the cylinder.
        lsf2d : callable
            2D level set function for the cross section.
        lprim_cell : dict
            Dictionary mapping Axis enums to arrays of primary grid locations.
        dl_max : float or array_like, optional
            Maximum grid cell size.
        """
        # Validate normal_axis
        chkarg(isinstance(normal_axis, Axis), '"normal_axis" should be instance of Axis.')
        self.normal_axis = normal_axis
        
        # Validate lsf2d
        chkarg(callable(lsf2d), '"lsf2d" should be callable.')
        self.lsf2d = lsf2d
        
        # Validate lprim_cell
        chkarg(
            isinstance(lprim_cell, dict) and len(lprim_cell) == Axis.count(),
            f'"lprim_cell" should be dictionary with {Axis.count()} entries.'
        )
        
        # Get cyclic axes
        h, v, n = normal_axis.cycle()
        
        # Calculate bounds and parameters in normal direction
        bound_n = [np.min(lprim_cell[n]), np.max(lprim_cell[n])]
        sn = (bound_n[1] - bound_n[0]) / 2  # semiside in normal direction
        cn = np.mean(bound_n)  # center in normal direction
        
        # Define level set function
        # LSF is min(lsf2d(rhv), 1 - |rn-cn|/sn)
        def lsf(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
            """
            Level set function for the cylinder.
            
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
            
            # Get 2D level set value
            level_2d = lsf2d(loc[int(h)], loc[int(v)])
            
            # Get height constraint level set value
            level_height = 1 - np.abs(loc[int(n)] - cn) / sn
            
            # Intersection of regions
            level = np.minimum(level_2d, level_height)
            return level
        
        # Call parent constructor
        super().__init__(lprim_cell, lsf, dl_max)