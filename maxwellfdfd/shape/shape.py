"""
Base Shape class for MaxwellFDFD geometric shapes.

This module defines the abstract Shape class, which is the base class
for all geometric shapes in MaxwellFDFD. Shapes are defined using
level set functions.
"""

from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Tuple, Union
import numpy as np
from maxwellfdfd.base import Axis, Sign, chkarg, expand2row
from .interval import Interval


class Shape(ABC):
    """
    Abstract base class for all geometric shapes in MaxwellFDFD.
    
    Shapes are defined using level set functions (LSF). A level set function
    takes a position (x, y, z) and returns a real number. For points in the
    interior, boundary, and exterior of the shape, LSF > 0, LSF == 0, and
    LSF < 0 respectively.
    
    Parameters
    ----------
    lprim_cell : dict
        Dictionary mapping Axis enums to arrays of primary grid locations.
        Must have entries for all three axes (x, y, z).
    lsf : callable
        Level set function with signature lsf(x, y, z) -> ndarray.
        Must be vectorized to handle array inputs.
    dl_max : float or array_like, optional
        Maximum grid cell size allowed. Can be a scalar or length-3 vector.
        If not provided, defaults to Inf.
    
    Attributes
    ----------
    interval : list of Interval
        List of Interval objects for each axis.
    lsf : callable
        Level set function defining the shape.
    n_subpxls : int
        Number of uniform sampling points in each Cartesian direction
        for subpixel smoothing (default: 10).
    
    Properties
    ----------
    lprim : dict
        Dictionary mapping Axis enums to primary grid location arrays.
    bound : ndarray
        Circumbox bounds as shape (3, 2) array: [[xmin, xmax], [ymin, ymax], [zmin, zmax]].
    cb_center : ndarray
        Center of the circumbox as [x, y, z].
    L : ndarray
        Circumbox dimensions as [Lx, Ly, Lz].
    dl_max : ndarray
        Maximum grid cell size in each direction as [dx_max, dy_max, dz_max].
    
    Examples
    --------
    >>> import numpy as np
    >>> from maxwellfdfd.base import Axis
    >>> from maxwellfdfd.shape import Shape
    >>> 
    >>> def lsf_box(x, y, z):
    ...     return 1 - np.maximum.reduce([np.abs(x), np.abs(y), np.abs(z)])
    >>> 
    >>> lprim_cell = {
    ...     Axis.X: np.array([-1, 1]),
    ...     Axis.Y: np.array([-1, 1]),
    ...     Axis.Z: np.array([-1, 1])
    ... }
    >>> shape = Shape(lprim_cell, lsf_box)
    """
    
    def __init__(
        self,
        lprim_cell: Dict[Axis, np.ndarray],
        lsf: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
        dl_max: Union[float, np.ndarray] = np.inf
    ):
        """
        Initialize a Shape.
        
        Parameters
        ----------
        lprim_cell : dict
            Dictionary mapping Axis enums to arrays of primary grid locations.
        lsf : callable
            Level set function.
        dl_max : float or array_like, optional
            Maximum grid cell size.
        """
        # Validate lprim_cell
        chkarg(
            isinstance(lprim_cell, dict) and len(lprim_cell) == Axis.count(),
            f'"lprim_cell" should be dictionary with {Axis.count()} entries.'
        )
        for axis in Axis.elems():
            chkarg(
                axis in lprim_cell and np.isrealobj(lprim_cell[axis]) and lprim_cell[axis].size > 0,
                f'"lprim_cell" should contain nonempty array for {axis}.'
            )
        
        # Validate and store level set function
        chkarg(callable(lsf), '"lsf" should be callable.')
        self.lsf = lsf
        
        # Validate and expand dl_max
        if isinstance(dl_max, (int, float)):
            dl_max_scalar = float(dl_max)
            if not np.isinf(dl_max_scalar):
                chkarg(dl_max_scalar > 0, '"dl_max" should be positive.')
            dl_max_arr = np.full((1, Axis.count()), dl_max_scalar)
        else:
            dl_max_arr = np.asarray(dl_max)
            chkarg(
                np.isrealobj(dl_max_arr) and np.all(dl_max_arr > 0),
                'elements of "dl_max" should be positive.'
            )
            dl_max_arr = expand2row(dl_max_arr, Axis.count())
        
        self._dl_max = dl_max_arr
        
        # Create intervals for each axis
        self.interval: List[Interval] = []
        for axis in Axis.elems():
            interval = Interval(lprim_cell[axis], dl_max_arr[0, int(axis)])
            self.interval.append(interval)
        
        # Set default subpixel sampling
        self.n_subpxls = 10
    
    @property
    def lprim(self) -> Dict[Axis, np.ndarray]:
        """
        Get primary grid locations for each axis.
        
        Returns
        -------
        dict
            Dictionary mapping Axis enums to primary grid location arrays.
        """
        result = {}
        for axis in Axis.elems():
            result[axis] = self.interval[int(axis)].lprim
        return result
    
    @property
    def bound(self) -> np.ndarray:
        """
        Get circumbox bounds.
        
        Returns
        -------
        ndarray
            Shape (3, 2) array: [[xmin, xmax], [ymin, ymax], [zmin, zmax]].
        """
        bound = np.full((Axis.count(), Sign.count()), np.nan)
        for axis in Axis.elems():
            bound[int(axis), :] = self.interval[int(axis)].bound
        return bound
    
    @property
    def cb_center(self) -> np.ndarray:
        """
        Get center of circumbox.
        
        Returns
        -------
        ndarray
            Center coordinates as [x, y, z].
        """
        center = np.mean(self.bound, axis=1)
        return center
    
    @property
    def L(self) -> np.ndarray:
        """
        Get circumbox dimensions.
        
        Returns
        -------
        ndarray
            Dimensions as [Lx, Ly, Lz].
        """
        L = np.full(Axis.count(), np.nan)
        for axis in Axis.elems():
            L[int(axis)] = self.interval[int(axis)].L()
        return L
    
    @property
    def dl_max(self) -> np.ndarray:
        """
        Get maximum grid cell size in each direction.
        
        Returns
        -------
        ndarray
            Maximum grid cell sizes as [dx_max, dy_max, dz_max].
        """
        dl_max = np.full(Axis.count(), np.nan)
        for axis in Axis.elems():
            dl_max[int(axis)] = self.interval[int(axis)].dl_max
        return dl_max
    
    def circumbox_contains(
        self,
        x: Union[float, np.ndarray],
        y: Union[float, np.ndarray],
        z: Union[float, np.ndarray]
    ) -> Union[bool, np.ndarray]:
        """
        Check if point(s) are contained in the circumbox.
        
        Parameters
        ----------
        x, y, z : float or array_like
            Coordinates of point(s) to check.
        
        Returns
        -------
        bool or ndarray
            Boolean array indicating if each point is in the circumbox.
        
        Examples
        --------
        >>> import numpy as np
        >>> from maxwellfdfd.base import Axis
        >>> from maxwellfdfd.shape import Box
        >>> box = Box(np.array([[-1, 1], [-1, 1], [-1, 1]]))
        >>> box.circumbox_contains(0, 0, 0)
        True
        >>> box.circumbox_contains(2, 0, 0)
        False
        """
        # Validate inputs
        x_arr = np.asarray(x)
        y_arr = np.asarray(y)
        z_arr = np.asarray(z)
        
        chkarg(np.isrealobj(x_arr), '"x" should be array with real elements.')
        chkarg(np.isrealobj(y_arr), '"y" should be array with real elements.')
        chkarg(np.isrealobj(z_arr), '"z" should be array with real elements.')
        chkarg(
            x_arr.shape == y_arr.shape == z_arr.shape,
            '"x", "y", "z" should have same size.'
        )
        
        truth = np.ones_like(x_arr, dtype=bool)
        loc = [x_arr, y_arr, z_arr]
        
        for axis in Axis.elems():
            contains_result = self.interval[int(axis)].contains(loc[int(axis)])
            if isinstance(contains_result, tuple):
                truth &= contains_result[0]
            else:
                truth &= contains_result
        
        return truth
    
    def contains(
        self,
        x: Union[float, np.ndarray],
        y: Union[float, np.ndarray],
        z: Union[float, np.ndarray]
    ) -> Union[bool, np.ndarray]:
        """
        Check if point(s) are contained in the shape.
        
        Parameters
        ----------
        x, y, z : float or array_like
            Coordinates of point(s) to check.
        
        Returns
        -------
        bool or ndarray
            Boolean array indicating if each point is in the shape.
        
        Examples
        --------
        >>> import numpy as np
        >>> from maxwellfdfd.shape import Box
        >>> box = Box(np.array([[-1, 1], [-1, 1], [-1, 1]]))
        >>> box.contains(0, 0, 0)
        True
        >>> box.contains(2, 0, 0)
        False
        """
        # Validate inputs
        x_arr = np.asarray(x)
        y_arr = np.asarray(y)
        z_arr = np.asarray(z)
        
        chkarg(np.isrealobj(x_arr), '"x" should be array with real elements.')
        chkarg(np.isrealobj(y_arr), '"y" should be array with real elements.')
        chkarg(np.isrealobj(z_arr), '"z" should be array with real elements.')
        chkarg(
            x_arr.shape == y_arr.shape == z_arr.shape,
            '"x", "y", "z" should have same size.'
        )
        
        truth = self.lsf(x_arr, y_arr, z_arr) >= 0
        return truth
    
    def smoothing_params(self, box: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Calculate smoothing parameters for subpixel smoothing.
        
        This method computes the relative volume and normal direction
        for a given box (pixel) using subpixel sampling.
        
        Parameters
        ----------
        box : ndarray
            Box bounds as shape (3, 2) array: [[xmin, xmax], [ymin, ymax], [zmin, zmax]].
        
        Returns
        -------
        rvol : float
            Relative volume of the shape within the box (0 to 1).
        ndir : ndarray
            Normal direction vector [nx, ny, nz] (normalized).
        
        Notes
        -----
        This method uses uniform sampling within the box to estimate
        the volume fraction and surface normal at the interface.
        """
        # Validate box
        chkarg(
            box.shape == (Axis.count(), Sign.count()) and np.isrealobj(box),
            f'"box" should be {Axis.count()}-by-{Sign.count()} array with real elements.'
        )
        
        # Create probing points
        l_probe: Dict[Axis, np.ndarray] = {}
        dl_sub = np.diff(box.T, axis=0).flatten() / self.n_subpxls
        N_subpxls = self.n_subpxls ** 3
        
        for axis in Axis.elems():
            bnd = box[int(axis), :]
            # Probing points at centers of subpixels
            l_probe[axis] = np.linspace(
                bnd[int(Sign.N)] - dl_sub[int(axis)] / 2,
                bnd[int(Sign.P)] + dl_sub[int(axis)] / 2,
                self.n_subpxls + 2
            )
        
        # Create meshgrid and evaluate level set function
        X, Y, Z = np.meshgrid(l_probe[Axis.X], l_probe[Axis.Y], l_probe[Axis.Z], indexing='ij')
        F = self.lsf(X, Y, Z)
        
        # Values at interior points of box (excluding boundary points)
        Fint = F[1:-1, 1:-1, 1:-1]
        is_contained = (Fint > 0)
        at_interface = (Fint == 0)
        
        # Calculate relative volume
        rvol = (np.sum(is_contained) + 0.5 * np.sum(at_interface)) / N_subpxls
        
        # Calculate gradient for normal direction
        # Note: np.gradient uses different ordering than MATLAB's gradient
        Fx, Fy, Fz = np.gradient(F, dl_sub[int(Axis.Y)], dl_sub[int(Axis.X)], dl_sub[int(Axis.Z)])
        Fx = Fx[1:-1, 1:-1, 1:-1]
        Fy = Fy[1:-1, 1:-1, 1:-1]
        Fz = Fz[1:-1, 1:-1, 1:-1]
        
        gradF = np.column_stack([Fx.flatten(), Fy.flatten(), Fz.flatten()])
        normF = np.sqrt(np.sum(gradF**2, axis=1))
        
        # Avoid division by zero
        valid = normF > 0
        gradF[valid] = gradF[valid] / normF[valid, np.newaxis]
        
        # Average normal direction
        ndir = np.sum(gradF, axis=0) / N_subpxls
        ndir_norm = np.linalg.norm(ndir)
        if ndir_norm > 0:
            ndir = -ndir / ndir_norm
        
        return rvol, ndir