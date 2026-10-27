"""
Ellipsoid class for 3D ellipsoids.

This module defines the Ellipsoid class, which represents a 3D ellipsoid
aligned with the Cartesian coordinate axes.
"""

from typing import Dict, Union
import numpy as np
from maxwellfdfd.base import Axis, chkarg
from .shape import Shape


class Ellipsoid(Shape):
    """
    3D ellipsoid shape aligned with Cartesian coordinate axes.
    
    The three axes of the ellipsoid are aligned with the axes of the
    Cartesian coordinate system.
    
    Parameters
    ----------
    center : array_like
        Center of the ellipsoid as [x, y, z].
    semiaxes : array_like
        Semiaxes of the ellipsoid as [a, b, c].
    dl_max : float or array_like, optional
        Maximum grid cell size allowed. Can be a scalar or length-3 vector.
        If not provided, defaults to Inf.
    
    Attributes
    ----------
    center : ndarray
        Center coordinates as [x, y, z].
    semiaxes : ndarray
        Semiaxes as [a, b, c].
    
    Examples
    --------
    >>> import numpy as np
    >>> from maxwellfdfd.shape import Ellipsoid
    >>> ellipsoid = Ellipsoid(np.array([0, 0, 0]), np.array([100, 50, 50]))
    >>> ellipsoid.center
    array([0, 0, 0])
    >>> ellipsoid.semiaxes
    array([100,  50,  50])
    >>> ellipsoid.contains(0, 0, 0)
    True
    >>> ellipsoid.contains(100, 0, 0)
    False
    """
    
    def __init__(
        self,
        center: np.ndarray,
        semiaxes: np.ndarray,
        dl_max: Union[float, np.ndarray] = np.inf
    ):
        """
        Initialize an Ellipsoid.
        
        Parameters
        ----------
        center : array_like
            Center of the ellipsoid.
        semiaxes : array_like
            Semiaxes of the ellipsoid.
        dl_max : float or array_like, optional
            Maximum grid cell size.
        """
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
            semiaxes_arr.shape == (Axis.count(),) and np.isrealobj(semiaxes_arr) and np.all(semiaxes_arr > 0),
            f'"semiaxes" should be length-{Axis.count()} array with positive elements.'
        )
        self.semiaxes = semiaxes_arr
        
        # Calculate bounds
        bound = np.vstack([self.center - self.semiaxes, self.center + self.semiaxes]).T
        
        # Define level set function
        # LSF is 1 - sqrt(sum(((r-c) ./ semiaxes).^2))
        def lsf(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
            """
            Level set function for the ellipsoid.
            
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
            level = np.zeros_like(x)
            
            for axis in Axis.elems():
                level += ((loc[int(axis)] - self.center[int(axis)]) / self.semiaxes[int(axis)]) ** 2
            
            level = 1 - np.sqrt(level)
            return level
        
        # Create lprim_cell
        lprim_cell: Dict[Axis, np.ndarray] = {}
        for axis in Axis.elems():
            lprim_cell[axis] = bound[int(axis), :]
        
        # Call parent constructor
        super().__init__(lprim_cell, lsf, dl_max)