"""
Line class for line shapes.

This module defines the Line class, which represents a line in 3D space.
"""

from typing import Dict, Union
import numpy as np
from maxwellfdfd.base import Axis, Dir, chkarg, expand2row
from .zero_vol_shape import ZeroVolShape


class Line(ZeroVolShape):
    """
    Line shape in 3D space.
    
    Line represents a line aligned with one of the Cartesian axes.
    It does not have a volume, but it is used to force a user-defined
    primary grid line in simulations.
    
    Parameters
    ----------
    axis : Axis
        Direction of the line (Axis.X, Axis.Y, or Axis.Z).
    intercept : array_like
        Location of the line in the plane normal to the line.
        For axis=Axis.Y, it is [z, x].
    dl_max : float or array_like, optional
        Maximum grid cell size allowed. Can be a scalar or length-3 vector.
        If not provided, defaults to Inf. In directions normal to the
        line direction, dl_max is meaningless.
    
    Attributes
    ----------
    axis : Axis
        Direction of the line.
    intercept : ndarray
        Location of the line in the transverse plane.
    
    Examples
    --------
    >>> import numpy as np
    >>> from maxwellfdfd.base import Axis
    >>> from maxwellfdfd.shape import Line
    >>> line = Line(Axis.Z, np.array([0, 100]))
    >>> line.axis
    <Axis.Z: 'z'>
    >>> line.intercept
    array([  0, 100])
    """
    
    def __init__(
        self,
        axis: Axis,
        intercept: np.ndarray,
        dl_max: Union[float, np.ndarray] = np.inf
    ):
        """
        Initialize a Line.
        
        Parameters
        ----------
        axis : Axis
            Direction of the line.
        intercept : array_like
            Location of the line in the transverse plane.
        dl_max : float or array_like, optional
            Maximum grid cell size.
        """
        # Validate axis
        chkarg(isinstance(axis, Axis), '"axis" should be instance of Axis.')
        self.axis = axis
        
        # Validate intercept
        intercept_arr = np.asarray(intercept)
        chkarg(
            intercept_arr.shape == (Dir.count(),) and np.isrealobj(intercept_arr),
            f'"intercept" should be length-{Dir.count()} array with real elements.'
        )
        self.intercept = intercept_arr
        
        # Get cyclic axes
        h, v, n = axis.cycle()
        
        # Define level set function
        # LSF is -max(|h-h0|, |v-v0|)
        def lsf(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
            """
            Level set function for the line.
            
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
            level = -np.maximum(
                np.abs(loc[int(h)] - self.intercept[int(Dir.H)]),
                np.abs(loc[int(v)] - self.intercept[int(Dir.V)])
            )
            return level
        
        # Create lprim_cell
        lprim_cell: Dict[Axis, np.ndarray] = {}
        lprim_cell[axis] = np.array([-np.inf, np.inf])
        lprim_cell[h] = np.array([self.intercept[int(Dir.H)]])
        lprim_cell[v] = np.array([self.intercept[int(Dir.V)]])
        
        # Process dl_max
        if isinstance(dl_max, (int, float)):
            dl_max_arr = np.full((1, Axis.count()), float(dl_max))
        else:
            dl_max_arr = expand2row(dl_max, Axis.count())
        
        # Set dl_max to Inf in directions normal to the line
        dl_max_arr[0, int(h)] = np.inf
        dl_max_arr[0, int(v)] = np.inf
        
        # Call parent constructor
        super().__init__(lprim_cell, lsf, dl_max_arr)