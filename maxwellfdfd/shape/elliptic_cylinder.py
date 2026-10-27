"""
EllipticCylinder class for cylinders with elliptical cross sections.

This module defines the EllipticCylinder class, which represents a cylinder
with an elliptical cross section.
"""

from typing import Dict, Union
import numpy as np
from maxwellfdfd.base import Axis, Dir, chkarg
from .generic_cylinder import GenericCylinder


class EllipticCylinder(GenericCylinder):
    """
    Cylinder with elliptical cross section.
    
    The major and minor axes of the ellipse as well as the axis of the cylinder
    are aligned with the axes of the Cartesian coordinate system.
    
    Parameters
    ----------
    normal_axis : Axis
        Axis of the cylinder (Axis.X, Axis.Y, or Axis.Z).
    height : float
        Size of the cylinder along its axis.
    center : array_like
        Center of the cylinder as [x, y, z].
    semiaxes : array_like
        Semiaxes of the ellipse as [a, b]. If normal_axis == Axis.Y,
        a is the semiaxis in the z-axis and b is the semiaxis in the x-axis.
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
    semiaxes : ndarray
        Semiaxes of the ellipse as [a, b].
    
    Examples
    --------
    >>> import numpy as np
    >>> from maxwellfdfd.base import Axis
    >>> from maxwellfdfd.shape import EllipticCylinder
    >>> cylinder = EllipticCylinder(Axis.Z, 100, np.array([0, 0, 50]), np.array([100, 50]))
    >>> cylinder.normal_axis
    <Axis.Z: 'z'>
    >>> cylinder.height
    100.0
    >>> cylinder.center
    array([ 0,  0, 50])
    >>> cylinder.semiaxes
    array([100,  50])
    """
    
    def __init__(
        self,
        normal_axis: Axis,
        height: float,
        center: np.ndarray,
        semiaxes: np.ndarray,
        dl_max: Union[float, np.ndarray] = np.inf
    ):
        """
        Initialize an EllipticCylinder.
        
        Parameters
        ----------
        normal_axis : Axis
            Axis of the cylinder.
        height : float
            Height of the cylinder.
        center : array_like
            Center of the cylinder.
        semiaxes : array_like
            Semiaxes of the ellipse.
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
        
        # Validate semiaxes
        semiaxes_arr = np.asarray(semiaxes)
        chkarg(
            semiaxes_arr.shape == (Dir.count(),) and np.isrealobj(semiaxes_arr) and np.all(semiaxes_arr > 0),
            f'"semiaxes" should be length-{Dir.count()} array with positive elements.'
        )
        self.semiaxes = semiaxes_arr
        
        # Get cyclic axes
        h, v, n = normal_axis.cycle()
        
        # Calculate semisides and bounds
        s = np.full(Axis.count(), np.nan)
        s[int(h)] = self.semiaxes[int(Dir.H)]
        s[int(v)] = self.semiaxes[int(Dir.V)]
        s[int(n)] = self.height / 2
        bound = np.vstack([self.center - s, self.center + s]).T
        
        # Define 2D level set function for the ellipse
        # LSF2D is 1 - sqrt(((p-ch)/a)^2 + ((q-cv)/b)^2)
        def lsf2d(p: np.ndarray, q: np.ndarray) -> np.ndarray:
            """
            2D level set function for the elliptical cross section.
            
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
            
            level = np.zeros_like(p)
            for d in Dir.elems():
                level += ((loc[int(d)] - c[int(d)]) / self.semiaxes[int(d)]) ** 2
            
            level = 1 - np.sqrt(level)
            return level
        
        # Create lprim_cell
        lprim_cell: Dict[Axis, np.ndarray] = {}
        for axis in Axis.elems():
            lprim_cell[axis] = bound[int(axis), :]
        
        # Call parent constructor
        super().__init__(normal_axis, lsf2d, lprim_cell, dl_max)