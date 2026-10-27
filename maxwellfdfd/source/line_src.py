"""
LineSrc class for MaxwellFDFD.

This module defines the LineSrc class, which represents a constant electric
dipole distribution along a line.
"""

from typing import List, Tuple, TYPE_CHECKING, Union
import numpy as np

from maxwellfdfd.base import Axis, Dir, GT, chkarg
from maxwellfdfd.shape import Line
from .source import Source
from .with_bloch import WithBloch
from .utils import ind_for_loc

if TYPE_CHECKING:
    from maxwellfdfd.grid import Grid3d


class LineSrc(Source, WithBloch):
    """
    Line source representing a constant electric dipole distribution.
    
    LineSrc is to generate a cylindrical wave in a homogeneous medium. It can
    create a cylindrical wave propagating not only in the radial direction, but
    also in the direction slanted toward the line axis. It also supports the
    polarization normal to the line axis.
    
    Parameters
    ----------
    axis : Axis
        Direction of the line (Axis.X, Axis.Y, or Axis.Z).
    intercept : array_like
        Location of the line in the plane normal to the line. For axis == Axis.Y,
        it is in the format of [z, x].
    polarization : Axis, optional
        Direction of the dipoles. If unassigned, polarization = axis is used.
    IorK : complex, optional
        For polarization == axis: complex amplitude of the current.
        For polarization ~= axis: complex amplitude of the sheet current density.
        Default is 1.0.
    theta : float, optional
        Oblique incidence angle in radians. |theta| should not exceed pi/2.
        Default is 0.0.
    wvlen : float, optional
        Wavelength of the cylindrical wave in the background medium. Required
        to set up the Bloch boundary condition for oblique incidence. Default
        is None.
    
    Attributes
    ----------
    axis : Axis
        Axis of the line.
    intercept : ndarray
        Location of intercept in transverse plane [h, v].
    polarization : Axis
        Direction of the dipoles.
    kBloch : ndarray
        Bloch k-vector [kx, ky, kz].
    IorK : complex
        Current I if polarization == axis; sheet current density K otherwise.
    
    Notes
    -----
    In the finite-difference grid, the dipoles in LineSrc are located at the
    E-field points. This poses a condition on the intercept argument in the
    constructor:
    - If polarization == axis: both elements of intercept should be at dual
      grid points.
    - If polarization ~= axis: the element of intercept corresponding to the
      polarization component should be at a primary grid point, and the other
      element of intercept should be at a dual grid point.
    
    Therefore, make sure that the elements of intercept at primary grid points
    do not overlap with the locations of the vertices of Shape; otherwise
    dynamic grid generation in maxwell_run() will fail.
    
    Examples
    --------
    >>> import numpy as np
    >>> from maxwellfdfd.base import Axis
    >>> from maxwellfdfd.source import LineSrc
    >>> 
    >>> # Create a z-axis line source at origin
    >>> src = LineSrc(Axis.Z, np.array([0, 0]))
    >>> 
    >>> # Create with oblique incidence
    >>> src = LineSrc(Axis.Z, np.array([0, 0]), theta=np.pi/6, wvlen=1.0)
    """
    
    def __init__(
        self,
        axis: Axis,
        intercept: np.ndarray,
        polarization: Axis | None = None,
        IorK: complex = 1.0,
        theta: float | np.ndarray = 0.0,
        wvlen: float | None = None
    ):
        """
        Initialize a LineSrc.
        
        Parameters
        ----------
        axis : Axis
            Direction of the line.
        intercept : array_like
            Location of the line in the plane normal to the line.
        polarization : Axis, optional
            Direction of the dipoles.
        IorK : complex, optional
            Current or sheet current density amplitude.
        theta : float or array_like, optional
            Oblique incidence angle or Bloch wave vector.
        wvlen : float, optional
            Wavelength for oblique incidence.
        """
        # Validate inputs
        chkarg(isinstance(axis, Axis), '"axis" should be instance of Axis.')
        
        intercept_arr = np.asarray(intercept)
        chkarg(
            intercept_arr.shape == (Dir.count(),) and np.isrealobj(intercept_arr),
            f'"intercept" should be length-{Dir.count()} array with real elements.'
        )
        
        if polarization is None:
            polarization = axis
        chkarg(isinstance(polarization, Axis),
               '"polarization" should be instance of Axis.')
        
        chkarg(np.iscomplex(IorK) or np.isreal(IorK), '"IorK" should be complex.')
        
        # Handle theta and wvlen
        ka_Bloch: float
        if wvlen is None:
            # theta is actually ka_Bloch
            if isinstance(theta, (int, float)):
                ka_Bloch = float(theta)
            else:
                theta_arr = np.asarray(theta)
                chkarg(
                    theta_arr.shape == (Dir.count(),) and np.isrealobj(theta_arr),
                    f'"ka_Bloch" should be length-{Dir.count()} array with real elements.'
                )
                ka_Bloch = theta_arr[int(Dir.H)]
        else:
            # theta is incidence angle, wvlen is wavelength
            chkarg(
                isinstance(theta, (int, float)) and -np.pi/2 <= theta <= np.pi/2,
                '"theta" should be polar angle in radian between -pi/2 and pi/2.'
            )
            chkarg(isinstance(wvlen, (int, float)) and wvlen > 0,
                   '"wvlen" should be positive.')
            ka_Bloch = (2 * np.pi / wvlen) * np.sin(theta)
        
        # Create lgrid and laltgrid
        lgrid: List[np.ndarray] = [np.array([]) for _ in range(Axis.count())]
        laltgrid: List[np.ndarray] = [np.array([]) for _ in range(Axis.count())]
        
        h, v, _ = axis.cycle()  # h, v ~= axis
        
        if polarization == h:
            laltgrid[int(h)] = np.array([intercept_arr[int(Dir.H)]])
            lgrid[int(v)] = np.array([intercept_arr[int(Dir.V)]])
        elif polarization == v:
            lgrid[int(h)] = np.array([intercept_arr[int(Dir.H)]])
            laltgrid[int(v)] = np.array([intercept_arr[int(Dir.V)]])
        else:  # polarization == axis
            lgrid[int(h)] = np.array([intercept_arr[int(Dir.H)]])
            lgrid[int(v)] = np.array([intercept_arr[int(Dir.V)]])
        
        # Create line shape
        line = Line(axis, intercept_arr)
        
        # Initialize parent class
        super().__init__(lgrid, laltgrid, line)
        
        self._axis = axis
        self._intercept = intercept_arr
        self._polarization = polarization
        self._IorK = complex(IorK)
        
        # Set kBloch
        self._kBloch = np.zeros(Axis.count())
        self._kBloch[int(axis)] = ka_Bloch
    
    @property
    def axis(self) -> Axis:
        """Axis of the line."""
        return self._axis
    
    @property
    def intercept(self) -> np.ndarray:
        """Location of intercept in transverse plane [h, v]."""
        return self._intercept
    
    @property
    def polarization(self) -> Axis:
        """Direction of the dipoles."""
        return self._polarization
    
    @property
    def kBloch(self) -> np.ndarray:
        """Bloch k-vector [kx, ky, kz]."""
        return self._kBloch
    
    @property
    def IorK(self) -> complex:
        """Current I if polarization == axis; sheet current density K otherwise."""
        return self._IorK
    
    def generate_kernel(
        self,
        w_axis: Axis,
        grid3d: 'Grid3d'
    ) -> Tuple[List[np.ndarray | slice | str], np.ndarray]:
        """
        Generate the line source kernel for a specific axis.
        
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
        # Set Bloch boundary condition
        grid3d.set_kBloch(self)
        
        h, v, a = self._axis.cycle()  # h, v: axes in plane normal to this line
        p = self._polarization
        
        index_cell: List[np.ndarray | slice | str] = []
        
        if w_axis != p:
            JMw_patch = np.array([])
            index_cell = [np.array([]) for _ in range(Axis.count())]
        else:  # w_axis == p
            index_cell = [slice(None), slice(None), slice(None)]
            
            if p == a:
                # Set index_cell{h} and index_cell{v}
                g = self._gt
                axes = [h, v]
                for d in Dir.elems():
                    ind = ind_for_loc(self._intercept[int(d)], axes[int(d)], g, grid3d)
                    index_cell[int(axes[int(d)])] = ind
                
                # Set Jw
                dh = grid3d.dl[(h, self._gt)][index_cell[int(h)]]
                dv = grid3d.dl[(v, self._gt)][index_cell[int(v)]]
                I = self._IorK
                JMw = I / (dh * dv)  # I = J * (area)
                ga = self._gt.alter()
            else:  # p ~= a
                # Set index_cell{h}
                if h == p:
                    g = self._gt.alter()
                else:
                    g = self._gt
                ind = ind_for_loc(self._intercept[int(Dir.H)], h, g, grid3d)
                index_cell[int(h)] = ind
                
                # Set index_cell{v}
                g = g.alter()
                ind = ind_for_loc(self._intercept[int(Dir.V)], v, g, grid3d)
                index_cell[int(v)] = ind
                
                # Set Jw
                q = h if v == p else v
                dq = grid3d.dl[(q, self._gt)][index_cell[int(q)]]
                K = self._IorK
                JMw = K / dq
                ga = self._gt
            
            # Set Jw_patch with Bloch phase
            la = grid3d.l[(a, ga)]
            ka = self._kBloch[int(a)]
            
            # The patch is 1D along the line direction `a`. The index_cell
            # already places it at the correct (h, v) transverse indices with a
            # slice along `a`, so no transpose is needed (the previous 3-axis
            # transpose assumed a 3D array and raised "axes don't match array").
            JMw_patch = JMw * np.exp(-1j * ka * la)
        
        return index_cell, JMw_patch