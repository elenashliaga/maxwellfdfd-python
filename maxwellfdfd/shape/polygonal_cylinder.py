"""
PolygonalCylinder class for cylinders with polygonal cross sections.

This module defines the PolygonalCylinder class, which represents a cylinder
with a polygonal cross section.
"""

from typing import Dict, Union
import numpy as np
from maxwellfdfd.base import Axis, Dir, chkarg
from .generic_cylinder import GenericCylinder


class PolygonalCylinder(GenericCylinder):
    """
    Cylinder with polygonal cross section.
    
    The cross section is a polygon. It is assumed that the polygon is convex
    and the mean of its vertices is inside the polygon. The axis of the cylinder
    should be aligned with one of the axes of the Cartesian coordinate system.
    
    Parameters
    ----------
    normal_axis : Axis
        Axis of the cylinder (Axis.X, Axis.Y, or Axis.Z).
    height : float
        Size of the cylinder along its axis.
    normal_center : float
        Center of the axis of the cylinder.
    vertex_array : ndarray
        n-by-2 array of the vertices of the polygon with n vertices.
        For normal_axis == Axis.Y, it is in the format of [[z1, x1], ..., [zn, xn]].
    dl_max : float or array_like, optional
        Maximum grid cell size allowed. Can be a scalar or length-3 vector.
        If not provided, defaults to Inf.
    
    Attributes
    ----------
    normal_axis : Axis
        Axis of the cylinder.
    height : float
        Height of the cylinder.
    normal_center : float
        Center of the axis.
    vertex_array : ndarray
        Array of polygon vertices.
    
    Examples
    --------
    >>> import numpy as np
    >>> from maxwellfdfd.base import Axis
    >>> from maxwellfdfd.shape import PolygonalCylinder
    >>> # Regular triangle
    >>> vertices = np.array([[0, 0], [50, 0], [25, 25*np.sqrt(3)]])
    >>> cylinder = PolygonalCylinder(Axis.Z, 100, 50, vertices)
    >>> cylinder.vertex_array.shape
    (3, 2)
    """
    
    def __init__(
        self,
        normal_axis: Axis,
        height: float,
        normal_center: float,
        vertex_array: np.ndarray,
        dl_max: Union[float, np.ndarray] = np.inf
    ):
        """
        Initialize a PolygonalCylinder.
        
        Parameters
        ----------
        normal_axis : Axis
            Axis of the cylinder.
        height : float
            Height of the cylinder.
        normal_center : float
            Center of the axis.
        vertex_array : ndarray
            Array of polygon vertices.
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
        
        # Validate normal_center
        chkarg(np.isreal(normal_center), '"normal_center" should be real.')
        self.normal_center = float(normal_center)
        
        # Validate vertex_array
        v_arr = np.asarray(vertex_array)
        chkarg(
            v_arr.ndim == 2 and v_arr.shape[1] == Dir.count() and np.isrealobj(v_arr),
            f'"vertex_array" should be {Dir.count()}-column matrix with real elements.'
        )
        self.vertex_array = v_arr
        
        # Get number of vertices and center of mass
        Nv = v_arr.shape[0]
        c = np.mean(v_arr, axis=0)
        
        # Add first vertex at the end to close the polygon
        vc_array = np.vstack([v_arr, v_arr[0, :]])
        vc_array = vc_array - c  # vertices in center-of-mass coordinates
        
        # Check that vertices are in counter-clockwise direction
        z_array = vc_array[:, 0] + 1j * vc_array[:, 1]
        theta_array = np.angle(z_array[1:] / z_array[:-1])
        chkarg(
            np.all(theta_array > 0),
            '"vertex_array" should represent points arranged in counter-clockwise direction.'
        )
        
        # Calculate side vectors and outward normal directions
        s_array = np.diff(vc_array, axis=0)
        
        # Unnormalized outward normal directions
        n_array = np.column_stack([s_array[:, 1], -s_array[:, 0]])
        ln_array = np.sqrt(np.sum(n_array**2, axis=1))
        n_array = n_array / ln_array[:, np.newaxis]
        
        # Distances from the center
        d_array = np.sum(vc_array[:-1, :] * n_array, axis=1)
        chkarg(
            np.all(d_array > 0),
            'mean point of vertices should be in polygon.'
        )
        
        # Define 2D level set function for the polygon
        def lsf2d(p: np.ndarray, q: np.ndarray) -> np.ndarray:
            """
            2D level set function for the polygonal cross section.
            
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
            
            lc = [p - c[int(Dir.H)], q - c[int(Dir.V)]]
            
            level = np.full_like(p, -np.inf)
            for i in range(Nv):
                leveli = lc[int(Dir.H)] * n_array[i, int(Dir.H)] + lc[int(Dir.V)] * n_array[i, int(Dir.V)]
                leveli = leveli / d_array[i]
                level = np.maximum(level, leveli)
            
            level = 1 - level
            return level
        
        # Get cyclic axes
        h, v, n = normal_axis.cycle()
        
        # Create lprim_cell
        lprim_cell: Dict[Axis, np.ndarray] = {}
        lprim_cell[h] = v_arr[:, int(Dir.H)]
        lprim_cell[v] = v_arr[:, int(Dir.V)]
        lprim_cell[n] = np.array([-self.height, self.height]) / 2 + self.normal_center
        
        # Call parent constructor
        super().__init__(normal_axis, lsf2d, lprim_cell, dl_max)