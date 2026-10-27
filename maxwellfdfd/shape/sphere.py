"""
Sphere class for 3D spheres.

This module defines the Sphere class, which represents a 3D sphere.
"""

from typing import Union
import numpy as np
from maxwellfdfd.base import chkarg
from .ellipsoid import Ellipsoid


class Sphere(Ellipsoid):
    """
    3D sphere shape.
    
    Sphere is a special case of Ellipsoid where all semiaxes are equal.
    
    Parameters
    ----------
    center : array_like
        Center of the sphere as [x, y, z].
    radius : float
        Radius of the sphere.
    dl_max : float or array_like, optional
        Maximum grid cell size allowed. Can be a scalar or length-3 vector.
        If not provided, defaults to Inf.
    
    Attributes
    ----------
    center : ndarray
        Center coordinates as [x, y, z].
    radius : float
        Radius of the sphere.
    semiaxes : ndarray
        Semiaxes as [radius, radius, radius].
    
    Examples
    --------
    >>> import numpy as np
    >>> from maxwellfdfd.shape import Sphere
    >>> sphere = Sphere(np.array([0, 0, 0]), 100)
    >>> sphere.center
    array([0, 0, 0])
    >>> sphere.radius
    100.0
    >>> sphere.contains(0, 0, 0)
    True
    >>> sphere.contains(100, 0, 0)
    False
    """
    
    def __init__(
        self,
        center: np.ndarray,
        radius: float,
        dl_max: Union[float, np.ndarray] = np.inf
    ):
        """
        Initialize a Sphere.
        
        Parameters
        ----------
        center : array_like
            Center of the sphere.
        radius : float
            Radius of the sphere.
        dl_max : float or array_like, optional
            Maximum grid cell size.
        """
        # Validate radius
        chkarg(
            isinstance(radius, (int, float)) and radius > 0,
            '"radius" should be positive.'
        )
        self.radius = float(radius)
        
        # Create semiaxes array (all equal to radius)
        semiaxes = np.full(3, self.radius)
        
        # Call parent constructor
        super().__init__(center, semiaxes, dl_max)