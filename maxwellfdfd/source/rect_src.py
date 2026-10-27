"""
RectSrc class for MaxwellFDFD.

This module defines the RectSrc class, which represents a constant electric
dipole distribution over a rectangular patch.
"""

from typing import List, Tuple, TYPE_CHECKING
import numpy as np

from maxwellfdfd.base import Axis, Dir, GT, Sign, chkarg
from maxwellfdfd.shape import Rectangle
from .source import Source
from .utils import ind_for_loc

if TYPE_CHECKING:
    from maxwellfdfd.grid import Grid3d


class RectSrc(Source):
    """
    Rectangular source representing constant electric dipole distribution.
    
    RectSrc is similar to PlaneSrc, but it places constant electric dipoles on
    a rectangular patch rather than an entire plane.
    
    Parameters
    ----------
    normal_axis : Axis
        Direction normal to the rectangular patch (Axis.X, Axis.Y, or Axis.Z).
    intercept : float
        Location of the rectangular patch in the normal_axis direction.
    rect : array_like
        Description of the rectangular patch in the format of [[hmin, hmax],
        [vmin, vmax]]. If normal_axis == Axis.Y, then it is [[zmin, zmax],
        [xmin, xmax]].
    polarization : Axis
        Direction of the dipoles distributed on the rectangular patch. Must be
        orthogonal to normal_axis.
    K : complex, optional
        Amplitude of the surface current density. Default is 1.0.
    
    Attributes
    ----------
    normal_axis : Axis
        Plane normal axis.
    intercept : float
        Intercept between plane and normal axis.
    rect : ndarray
        Rectangular patch boundaries.
    polarization : Axis
        Direction of the dipoles.
    K : complex
        Sheet current density.
    
    Notes
    -----
    In the finite-difference grid, RectSrc excites dipoles at the E-field
    points. This poses a condition on the intercept argument in the constructor:
    intercept should be at a dual grid point in the normal_axis direction.
    Therefore, make sure that intercept does not overlap with the locations of
    the vertices of Shape in the normal_axis direction; otherwise dynamic grid
    generation in maxwell_run() will fail.
    
    Examples
    --------
    >>> import numpy as np
    >>> from maxwellfdfd.base import Axis
    >>> from maxwellfdfd.source import RectSrc
    >>> 
    >>> # Create a rectangular source on y = -2000 plane
    >>> rect = np.array([[0, 10], [20, 180]])
    >>> src = RectSrc(Axis.Y, -2000, rect, Axis.X)
    """
    
    def __init__(
        self,
        normal_axis: Axis,
        intercept: float,
        rect: np.ndarray,
        polarization: Axis,
        K: complex = 1.0
    ):
        """
        Initialize a RectSrc.
        
        Parameters
        ----------
        normal_axis : Axis
            Direction normal to the rectangular patch.
        intercept : float
            Location of the rectangular patch.
        rect : array_like
            Rectangular patch boundaries [[hmin, hmax], [vmin, vmax]].
        polarization : Axis
            Direction of the dipoles.
        K : complex, optional
            Surface current density amplitude.
        """
        # Validate inputs
        chkarg(isinstance(normal_axis, Axis),
               '"normal_axis" should be instance of Axis.')
        chkarg(np.isreal(intercept), '"intercept" should be real.')
        
        rect_arr = np.asarray(rect)
        chkarg(
            rect_arr.shape == (Dir.count(), Sign.count()) and np.isrealobj(rect_arr),
            f'"rect" should be {Dir.count()}-by-{Sign.count()} array with real elements.'
        )
        chkarg(
            isinstance(polarization, Axis) and polarization != normal_axis,
            '"polarization" should be instance of Axis and different from "normal_axis".'
        )
        chkarg(np.iscomplex(K) or np.isreal(K), '"K" should be complex.')
        
        # Create lgrid and laltgrid
        lgrid: List[np.ndarray] = [np.array([]) for _ in range(Axis.count())]
        laltgrid: List[np.ndarray] = [np.array([]) for _ in range(Axis.count())]
        lgrid[int(normal_axis)] = np.array([intercept])
        
        # Create rectangle shape
        rectangle = Rectangle(normal_axis, intercept, rect_arr)
        
        # Initialize parent class
        super().__init__(lgrid, laltgrid, rectangle)
        
        self._normal_axis = normal_axis
        self._intercept = intercept
        self._rect = rect_arr
        self._polarization = polarization
        self._K = complex(K)
    
    @property
    def normal_axis(self) -> Axis:
        """Plane normal axis."""
        return self._normal_axis
    
    @property
    def intercept(self) -> float:
        """Intercept between plane and normal axis."""
        return self._intercept
    
    @property
    def rect(self) -> np.ndarray:
        """Rectangular patch boundaries."""
        return self._rect
    
    @property
    def polarization(self) -> Axis:
        """Direction of the dipoles."""
        return self._polarization
    
    @property
    def K(self) -> complex:
        """Sheet current density."""
        return self._K
    
    def generate_kernel(
        self,
        w_axis: Axis,
        grid3d: 'Grid3d'
    ) -> Tuple[List[np.ndarray | slice | str], np.ndarray]:
        """
        Generate the rectangular source kernel for a specific axis.
        
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
        
        if w_axis != self._polarization:
            JMw_patch = np.array([])
            index_cell = [np.array([]) for _ in range(Axis.count())]
        else:  # w_axis == self._polarization
            n = self._normal_axis
            p = w_axis  # polarization axis
            
            g = self._gt
            ind_n = ind_for_loc(self._intercept, n, g, grid3d)
            dn = grid3d.dl[(n, g)][ind_n]
            JM = self._K / dn  # K*dt = (current through dn*dt) = J * (dn*dt)
            
            index_cell = [slice(None), slice(None), slice(None)]
            index_cell[int(n)] = ind_n
            
            # Calculate overlap ratios
            r_overlap: List[np.ndarray] = []
            
            # Note: Grid2d will be available when grid module is implemented
            # For now, we'll use a simplified approach
            for d in Dir.elems():
                # Get axis for this direction
                # This would normally use Grid2d to get the correct axis
                axis_idx = int(d) if int(n) != int(d) else (int(d) + 1) % 3
                
                # Get interval for this direction
                interval = self._rect[int(d), :]
                
                # Find overlapping indices (simplified)
                l = grid3d.l[axis_idx, g]
                ind_h = np.where(l >= interval[int(Sign.N)])[0]
                ind_p = np.where(l <= interval[int(Sign.P)])[0]
                
                if len(ind_h) > 0 and len(ind_p) > 0:
                    indices = np.intersect1d(ind_h, ind_p)
                    if len(indices) > 0:
                        index_cell[axis_idx] = indices
                        
                        # Calculate overlap ratios
                        r = np.ones(len(indices))
                        if len(indices) == 1:
                            r[0] = (interval[int(Sign.P)] - interval[int(Sign.N)]) / (
                                l[indices[0] + 1] - l[indices[0]] if indices[0] + 1 < len(l) else
                                l[indices[0]] - l[indices[0] - 1]
                            )
                        else:
                            r[0] = (l[indices[0] + 1] - interval[int(Sign.N)]) / (
                                l[indices[0] + 1] - l[indices[0]]
                            )
                            r[-1] = (interval[int(Sign.P)] - l[indices[-1]]) / (
                                l[indices[-1]] - l[indices[-1] - 1]
                            )
                        r_overlap.append(r)
                    else:
                        r_overlap.append(np.array([]))
                else:
                    r_overlap.append(np.array([]))
            
            # Create JMw_patch using outer product of overlap ratios
            if len(r_overlap) == 2 and r_overlap[0].size > 0 and r_overlap[1].size > 0:
                JMw_patch = JM * np.outer(r_overlap[0], r_overlap[1])
            else:
                JMw_patch = np.array([])
        
        return index_cell, JMw_patch