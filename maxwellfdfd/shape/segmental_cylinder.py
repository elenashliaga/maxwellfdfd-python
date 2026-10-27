"""
SegmentalCylinder class for cylinders with segmental cross sections.

This module defines the SegmentalCylinder class, which represents a cylinder
with a circular segment (region surrounded by an arc and chord) cross section.
"""

from typing import Dict, Union
import numpy as np
from maxwellfdfd.base import Axis, Dir, chkarg
from .generic_cylinder import GenericCylinder
from .sectoral_cylinder import SectoralCylinder


class SegmentalCylinder(GenericCylinder):
    """
    Cylinder with circular segment cross section.
    
    The cross section is a circular segment (region surrounded by an arc
    and the chord connecting its end points). The axis of the cylinder
    should be aligned with one of the axes of the Cartesian coordinate system.
    
    Parameters
    ----------
    normal_axis : Axis
        Axis of the cylinder (Axis.X, Axis.Y, or Axis.Z).
    height : float
        Size of the cylinder along its axis.
    center : array_like
        Center of the cylinder as [x, y, z]. For normal_axis = Axis.Z,
        (x, y) is the coordinate of the center of the circle.
    radius : float
        Radius of the circle.
    theta : float
        Beginning angle of the segment in radians.
    d_theta : float
        Angular width of the segment in radians between -pi and pi
        (the angle of a segment should not be reflex).
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
    theta : float
        Beginning angle of the segment.
    d_theta : float
        Angular width of the segment.
    
    Examples
    --------
    >>> import numpy as np
    >>> from maxwellfdfd.base import Axis
    >>> from maxwellfdfd.shape import SegmentalCylinder
    >>> cylinder = SegmentalCylinder(Axis.Z, 100, np.array([0, 0, 50]), 50, np.pi/6, np.pi/3)
    >>> cylinder.radius
    50.0
    >>> cylinder.theta
    0.5235987755982988
    >>> cylinder.d_theta
    1.0471975511965976
    """
    
    def __init__(
        self,
        normal_axis: Axis,
        height: float,
        center: np.ndarray,
        radius: float,
        theta: float,
        d_theta: float,
        dl_max: Union[float, np.ndarray] = np.inf
    ):
        """
        Initialize a SegmentalCylinder.
        
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
        theta : float
            Beginning angle of the segment.
        d_theta : float
            Angular width of the segment.
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
        
        # Validate radius
        chkarg(
            isinstance(radius, (int, float)) and radius > 0,
            '"radius" should be positive.'
        )
        self.radius = float(radius)
        
        # Validate theta
        chkarg(np.isreal(theta), '"theta" should be real.')
        self.theta = float(theta)
        
        # Validate d_theta
        chkarg(
            np.isreal(d_theta) and d_theta < np.pi and d_theta > -np.pi,
            '"d_theta" should be real between -pi and pi.'
        )
        self.d_theta = float(d_theta)
        
        # Create sectoral cylinder for reference
        sectoral = SectoralCylinder(normal_axis, height, center, radius, theta, d_theta, dl_max)
        
        # Calculate chord parameters
        thetas = [self.theta, self.theta + self.d_theta]
        
        # Direction bisecting the segment
        mc_dir = np.mean([np.cos(thetas), np.sin(thetas)], axis=1)
        mc_dir = mc_dir / np.linalg.norm(mc_dir)
        
        # Get cyclic axes
        h, v, n = normal_axis.cycle()
        
        # Midpoint of chord
        midpt = self.center[[int(h), int(v)]] + np.mean(radius * np.array([np.cos(thetas), np.sin(thetas)]), axis=1)
        
        # Define 2D level set function for the segment
        def lsf2d(p: np.ndarray, q: np.ndarray) -> np.ndarray:
            """
            2D level set function for the segmental cross section.
            
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
            
            # Level set function of sectoral cylinder
            level1 = sectoral.lsf2d(p, q)
            
            # Level set function for chord constraint
            lm = [p - midpt[int(Dir.H)], q - midpt[int(Dir.V)]]
            level2 = lm[int(Dir.H)] * mc_dir[int(Dir.H)] + lm[int(Dir.V)] * mc_dir[int(Dir.V)]
            
            # Intersection of regions
            level = np.minimum(level1, level2)
            return level
        
        # Create lprim_cell
        lprim_cell: Dict[Axis, np.ndarray] = {}
        lprim_cell[h] = self.radius * np.cos(thetas) + self.center[int(h)]
        lprim_cell[v] = self.radius * np.sin(thetas) + self.center[int(v)]
        lprim_cell[n] = np.array([-self.height, self.height]) / 2 + self.center[int(n)]
        
        # Add cardinal directions if they are in the sector
        if sectoral.lsf_th(np.array([0]))[0] > 0:
            lprim_cell[h] = np.append(lprim_cell[h], self.center[int(h)] + self.radius)
        if sectoral.lsf_th(np.array([np.pi / 2]))[0] > 0:
            lprim_cell[v] = np.append(lprim_cell[v], self.center[int(v)] + self.radius)
        if sectoral.lsf_th(np.array([np.pi]))[0] > 0:
            lprim_cell[h] = np.append(lprim_cell[h], self.center[int(h)] - self.radius)
        if sectoral.lsf_th(np.array([3 * np.pi / 2]))[0] > 0:
            lprim_cell[v] = np.append(lprim_cell[v], self.center[int(v)] - self.radius)
        
        # Call parent constructor
        super().__init__(normal_axis, lsf2d, lprim_cell, dl_max)