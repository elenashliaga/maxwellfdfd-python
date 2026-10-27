"""
PointSrc class for MaxwellFDFD.

This module defines the PointSrc class, which represents an electric point
dipole source.
"""

from typing import List, Tuple, TYPE_CHECKING
import numpy as np
import warnings

from maxwellfdfd.base import Axis, GT, chkarg
from maxwellfdfd.shape import Point
from .source import Source
from .utils import ind_for_loc

if TYPE_CHECKING:
    from maxwellfdfd.grid import Grid3d


class PointSrc(Source):
    """
    Point source representing an electric point dipole.
    
    PointSrc is the most basic source type. It places an oscillating electric
    point dipole at the location specified in the constructor.
    
    Parameters
    ----------
    polarization_axis : Axis
        Direction of the dipole (Axis.X, Axis.Y, or Axis.Z).
    location : array_like
        Location of the dipole as [x, y, z].
    I : complex, optional
        Amplitude of the current that the dipole drives. Default is 1.0.
    
    Attributes
    ----------
    polarization : Axis
        Direction of the dipole.
    location : ndarray
        Point coordinates as [x, y, z].
    I : complex
        Current amplitude.
    
    Notes
    -----
    In the finite-difference grid, PointSrc is located at one of the E-field
    points. This poses a condition on the location argument in the constructor:
    the location in the directions normal to the dipole polarization should be
    at dual grid points, whereas the location in the direction along the dipole
    polarization should be at a primary grid point. Therefore, make sure that
    the location of the dipole does not overlap with the locations of the
    vertices of Shape in the directions normal to the dipole polarization;
    otherwise dynamic grid generation in maxwell_run() will fail.
    
    Examples
    --------
    >>> import numpy as np
    >>> from maxwellfdfd.base import Axis
    >>> from maxwellfdfd.source import PointSrc
    >>> 
    >>> # Create a z-polarized point source at the origin
    >>> src = PointSrc(Axis.Z, np.array([0, 0, 0]))
    >>> 
    >>> # Use the constructed src in maxwell_run()
    >>> # [E, H] = maxwell_run({...}, 'SRCJ', src)
    """
    
    def __init__(
        self,
        polarization_axis: Axis,
        location: np.ndarray,
        I: complex = 1.0
    ):
        """
        Initialize a PointSrc.
        
        Parameters
        ----------
        polarization_axis : Axis
            Direction of the dipole.
        location : array_like
            Location of the dipole as [x, y, z].
        I : complex, optional
            Current amplitude. Default is 1.0.
        """
        # Validate inputs
        chkarg(isinstance(polarization_axis, Axis),
               '"polarization_axis" should be instance of Axis.')
        
        location_arr = np.asarray(location)
        chkarg(
            location_arr.shape == (Axis.count(),) and np.isrealobj(location_arr),
            f'"location" should be length-{Axis.count()} array with real elements.'
        )
        chkarg(np.iscomplex(I) or np.isreal(I), '"I" should be complex.')
        
        # Create lgrid and laltgrid
        lgrid: List[np.ndarray] = [np.array([]) for _ in range(Axis.count())]
        laltgrid: List[np.ndarray] = [np.array([]) for _ in range(Axis.count())]
        
        for w in Axis.elems():
            if w == polarization_axis:
                laltgrid[int(w)] = np.array([location_arr[int(w)]])
            else:
                lgrid[int(w)] = np.array([location_arr[int(w)]])
        
        # Create point shape
        point = Point(location_arr)
        
        # Initialize parent class
        super().__init__(lgrid, laltgrid, point)
        
        self._polarization = polarization_axis
        self._location = location_arr
        self._I = complex(I)
    
    @property
    def polarization(self) -> Axis:
        """Direction of the dipole."""
        return self._polarization
    
    @property
    def location(self) -> np.ndarray:
        """Point coordinates as [x, y, z]."""
        return self._location
    
    @property
    def I(self) -> complex:
        """Current amplitude."""
        return self._I
    
    def generate_kernel(
        self,
        w_axis: Axis,
        grid3d: 'Grid3d'
    ) -> Tuple[List[np.ndarray | slice | str], np.ndarray]:
        """
        Generate the point source kernel for a specific axis.
        
        Parameters
        ----------
        w_axis : Axis
            Axis for which to generate the source.
        grid3d : Grid3d
            3D grid instance.
        
        Returns
        -------
        index_cell : list
            List of indices for each axis.
        JMw_patch : ndarray
            Current density values for the source.
        """
        index_cell: List[np.ndarray | slice | str] = []
        q, r, p = self._polarization.cycle()  # q, r: axes normal to polarization axis p
        
        if w_axis == p:
            # Generate indices for all axes
            for v in Axis.elems():
                l = self._location[int(v)]
                if v == p:
                    g = self._gt.alter()
                else:  # v == q or r
                    g = self._gt
                
                iv = ind_for_loc(l, v, g, grid3d)
                index_cell.append(iv)
            
            # Calculate current density
            # NOTE: Grid3d.dl is keyed by (Axis, GT) tuples.
            dq = grid3d.dl[(q, self._gt)][index_cell[int(q)]]
            dr = grid3d.dl[(r, self._gt)][index_cell[int(r)]]
            JMw_patch = self._I / (dq * dr)  # I = J * (area)
        else:  # w_axis == q or r
            JMw_patch = np.array([])
            index_cell = [np.array([]) for _ in range(Axis.count())]
        
        return index_cell, JMw_patch