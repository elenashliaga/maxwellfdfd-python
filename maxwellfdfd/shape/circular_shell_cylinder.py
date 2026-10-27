"""
CircularShellCylinder class for hollow cylindrical pipes.

This module defines the CircularShellCylinder class, which represents a
hollow cylindrical pipe with circular cross sections.
"""

from typing import Dict, Union
import numpy as np
from maxwellfdfd.base import Axis, Dir, chkarg
from .generic_cylinder import GenericCylinder


class CircularShellCylinder(GenericCylinder):
    """
    Hollow cylindrical pipe with circular cross sections.
    
    The axis of the cylinder should be aligned with an axis of the
    Cartesian coordinate system.
    
    Parameters
    ----------
    normal_axis : Axis
        Axis of the cylinder (Axis.X, Axis.Y, or Axis.Z).
    height : float
        Size of the cylinder along its axis.
    center : array_like
        Center of the cylinder as [x, y, z].
    r1 : float
        First radius (inner or outer).
    r2 : float
        Second radius (outer or inner).
    dl_max : float or array_like, optional
        Maximum grid cell size allowed. Can be a scalar or length-3 vector.
        If not provided, defaults to Inf.
    
    Attributes
    ----------
    normal_axis : Axis
        Axis of the cylinder.
    height : float
        Height of the cylinder.
    center : ndarray
        Center coordinates as [x, y, z].
    r_inner : float
        Inner radius.
    r_outer : float
        Outer radius.
    
    Examples
    --------
    >>> import numpy as np
    >>> from maxwellfdfd.base import Axis
    >>> from maxwellfdfd.shape import CircularShellCylinder
    >>> cylinder = CircularShellCylinder(Axis.Z, 100, np.array([0, 0, 50]), 25, 50)
    >>> cylinder.r_inner
    25.0
    >>> cylinder.r_outer
    50.0
    """
    
    def __init__(
        self,
        normal_axis: Axis,
        height: float,
        center: np.ndarray,
        r1: float,
        r2: float,
        dl_max: Union[float, np.ndarray] = np.inf
    ):
        """
        Initialize a CircularShellCylinder.
        
        Parameters
        ----------
        normal_axis : Axis
            Axis of the cylinder.
        height : float
            Height of the cylinder.
        center : array_like
            Center of the cylinder.
        r1 : float
            First radius.
        r2 : float
            Second radius.
        dl_max : float or array_like, optional
            Maximum grid cell size.
        """
        # Validate normal_axis
        chkarg(isinstance(normal_axis, Axis), '"normal_axis" should be instance of Axis.')
        
        # Validate height
        chkarg(
            isinstance(height, (int, float)) and height > 0,
            '"height" should be positive.'
        )
        self.height = float(height)
        
        # Validate center
        center_arr = np.asarray(center)
        chkarg(
            center_arr.shape == (Axis.count(),) and np.isrealobj(center_arr),
            f'"center" should be length-{Axis.count()} array with real elements.'
        )
        self.center = center_arr
        
        # Validate radii
        chkarg(
            isinstance(r1, (int, float)) and r1 > 0,
            '"r1" should be positive.'
        )
        chkarg(
            isinstance(r2, (int, float)) and r2 > 0,
            '"r2" should be positive.'
        )
        
        # Sort radii
        rs = sorted([float(r1), float(r2)])
        self.r_inner = rs[0]
        self.r_outer = rs[1]
        
        # Get cyclic axes
        h, v, n = normal_axis.cycle()
        
        # Calculate semisides and bounds
        s = np.full(Axis.count(), np.nan)
        s[int(h)] = self.r_outer
        s[int(v)] = self.r_outer
        s[int(n)] = self.height / 2
        bound = np.vstack([self.center - s, self.center + s]).T
        
        # Define 2D level set function for the circular shell
        # LSF2D is min(1 - sqrt((p-cx)^2/R^2 + (q-cy)^2/R^2), sqrt((p-cx)^2/r^2 + (q-cy)^2/r^2) - 1)
        def lsf2d(p: np.ndarray, q: np.ndarray) -> np.ndarray:
            """
            2D level set function for the circular shell cross section.
            
            Parameters
            ----------
            p, q : ndarray
                Coordinate arrays in the lateral domain.
            
            Returns
            -------
            ndarray
                Level set function values.
            """
            chkarg(np.isrealobj(p), '"p" should be array with real elements.')
            chkarg(np.isrealobj(q), '"q" should be array with real elements.')
            chkarg(
                p.shape == q.shape,
                '"p" and "q" should have same size.'
            )
            
            c = self.center[[int(h), int(v)]]
            loc = [p, q]
            
            # Level set function for inside of outer circle
            level1 = np.zeros_like(p)
            for d in Dir.elems():
                level1 += ((loc[int(d)] - c[int(d)]) / self.r_outer) ** 2
            level1 = 1 - np.sqrt(level1)
            
            # Level set function for outside of inner circle
            level2 = np.zeros_like(p)
            for d in Dir.elems():
                level2 += ((loc[int(d)] - c[int(d)]) / self.r_inner) ** 2
            level2 = np.sqrt(level2) - 1
            
            # Intersection of two regions
            level = np.minimum(level1, level2)
            return level
        
        # Create lprim_cell
        lprim_cell: Dict[Axis, np.ndarray] = {}
        for axis in Axis.elems():
            lprim_cell[axis] = bound[int(axis), :]
        
        # Call parent constructor
        super().__init__(normal_axis, lsf2d, lprim_cell, dl_max)