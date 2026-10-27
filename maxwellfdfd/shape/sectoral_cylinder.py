"""
SectoralCylinder class for cylinders with sectoral cross sections.

This module defines the SectoralCylinder class, which represents a cylinder
with a circular sector (angular portion of a disk) cross section.
"""

from typing import Callable, Dict, Union
import numpy as np
from maxwellfdfd.base import Axis, Dir, chkarg
from .generic_cylinder import GenericCylinder


class SectoralCylinder(GenericCylinder):
    """
    Cylinder with circular sector cross section.
    
    The cross section is a circular sector (angular portion of a disk).
    The axis of the cylinder should be aligned with one of the axes of the
    Cartesian coordinate system.
    
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
        Beginning angle of the sector in radians.
    d_theta : float
        Angular width of the sector in radians between -2*pi and 2*pi.
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
        Beginning angle of the sector.
    d_theta : float
        Angular width of the sector.
    lsf_th : callable
        Level set function for angles.
    
    Examples
    --------
    >>> import numpy as np
    >>> from maxwellfdfd.base import Axis
    >>> from maxwellfdfd.shape import SectoralCylinder
    >>> cylinder = SectoralCylinder(Axis.Z, 100, np.array([0, 0, 50]), 50, np.pi/6, np.pi/3)
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
        Initialize a SectoralCylinder.
        
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
            Beginning angle of the sector.
        d_theta : float
            Angular width of the sector.
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
            np.isreal(d_theta) and d_theta <= 2*np.pi and d_theta >= -2*np.pi,
            '"d_theta" should be real between -2*pi and 2*pi.'
        )
        self.d_theta = float(d_theta)
        
        # Calculate theta parameters
        thetas = np.sort([self.theta, self.theta + self.d_theta])
        cth = np.mean(thetas)  # center of theta range
        cth = ((cth + np.pi) % (2 * np.pi)) - np.pi  # -pi <= cth < pi
        sth = np.diff(thetas) / 2  # semiwidth of theta range
        
        # Calculate radius parameters
        cr = self.radius / 2  # center of radius range
        sr = self.radius / 2  # semiwidth of radius range
        
        # Define level set function for angles
        def lsf_th(th: np.ndarray) -> np.ndarray:
            """
            Level set function for angles.
            
            Parameters
            ----------
            th : ndarray
                Angle values.
            
            Returns
            -------
            ndarray
                Level set function values.
            """
            dth = th - cth
            dth = ((dth + np.pi) % (2 * np.pi)) - np.pi
            level = 1 - np.abs(dth / sth)
            return level
        
        self.lsf_th = lsf_th
        
        # Get cyclic axes
        h, v, n = normal_axis.cycle()
        
        # Define 2D level set function for the sector
        def lsf2d(p: np.ndarray, q: np.ndarray) -> np.ndarray:
            """
            2D level set function for the sectoral cross section.
            
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
            
            lc = [p - self.center[int(h)], q - self.center[int(v)]]
            
            theta_pt = np.arctan2(lc[int(Dir.V)], lc[int(Dir.H)])  # -pi <= theta_pt < pi
            r_pt = np.sqrt(lc[int(Dir.H)]**2 + lc[int(Dir.V)]**2)
            zero_r_pt = (r_pt == 0)  # atan2(0,0) is not well-defined
            
            dr = r_pt - cr
            level = np.minimum(lsf_th(theta_pt), 1 - np.abs(dr / sr))
            level[zero_r_pt] = 0
            return level
        
        # Create lprim_cell
        lprim_cell: Dict[Axis, np.ndarray] = {}
        lprim_cell[h] = np.concatenate([
            self.radius * np.cos(thetas) + self.center[int(h)],
            [self.center[int(h)]]
        ])
        lprim_cell[v] = np.concatenate([
            self.radius * np.sin(thetas) + self.center[int(v)],
            [self.center[int(v)]]
        ])
        lprim_cell[n] = np.array([-self.height, self.height]) / 2 + self.center[int(n)]
        
        # Add cardinal directions if they are in the sector
        if lsf_th(0) > 0:  # +x direction
            lprim_cell[h] = np.append(lprim_cell[h], self.center[int(h)] + self.radius)
        if lsf_th(np.pi / 2) > 0:  # +y direction
            lprim_cell[v] = np.append(lprim_cell[v], self.center[int(v)] + self.radius)
        if lsf_th(np.pi) > 0:  # -x direction
            lprim_cell[h] = np.append(lprim_cell[h], self.center[int(h)] - self.radius)
        if lsf_th(3 * np.pi / 2) > 0:  # -y direction
            lprim_cell[v] = np.append(lprim_cell[v], self.center[int(v)] - self.radius)
        
        # Call parent constructor
        super().__init__(normal_axis, lsf2d, lprim_cell, dl_max)