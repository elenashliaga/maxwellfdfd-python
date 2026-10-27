"""
CircularCylinder class for cylinders with circular cross sections.

This module defines the CircularCylinder class, which represents a cylinder
with a circular cross section.
"""

from typing import Union
import numpy as np
from maxwellfdfd.base import chkarg
from .elliptic_cylinder import EllipticCylinder


class CircularCylinder(EllipticCylinder):
    """
    Cylinder with circular cross section.
    
    CircularCylinder is a special case of EllipticCylinder where both
    semiaxes are equal (the radius).
    
    Parameters
    ----------
    normal_axis : Axis
        Axis of the cylinder (Axis.X, Axis.Y, or Axis.Z).
    height : float
        Size of the cylinder along its axis.
    center : array_like
        Center of the cylinder as [x, y, z].
    radius : float
        Radius of the circle.
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
    radius : float
        Radius of the circle.
    semiaxes : ndarray
        Semiaxes as [radius, radius].
    
    Examples
    --------
    >>> import numpy as np
    >>> from maxwellfdfd.base import Axis
    >>> from maxwellfdfd.shape import CircularCylinder
    >>> cylinder = CircularCylinder(Axis.Z, 100, np.array([0, 0, 50]), 50)
    >>> cylinder.radius
    50.0
    >>> cylinder.semiaxes
    array([50., 50.])
    """
    
    def __init__(
        self,
        normal_axis,
        height: float,
        center: np.ndarray,
        radius: float,
        dl_max: Union[float, np.ndarray] = np.inf
    ):
        """
        Initialize a CircularCylinder.
        
        Parameters
        ----------
        normal_axis : Axis
            Axis of the cylinder.
        height : float
            Height of the cylinder.
        center : array_like
            Center of the cylinder.
        radius : float
            Radius of the circle.
        dl_max : float or array_like, optional
            Maximum grid cell size.
        """
        # Validate radius
        chkarg(
            isinstance(radius, (int, float)) and radius > 0,
            '"radius" should be positive.'
        )
        self.radius = float(radius)
        
        # Create semiaxes array (both equal to radius)
        semiaxes = np.full(2, self.radius)
        
        # Call parent constructor
        super().__init__(normal_axis, height, center, semiaxes, dl_max)