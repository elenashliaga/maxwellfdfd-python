"""
PlaneSrc class for MaxwellFDFD.

This module defines the PlaneSrc class, which represents a constant electric
dipole distribution over an entire plane.
"""

from typing import List, Tuple, TYPE_CHECKING, Union
import numpy as np

from maxwellfdfd.base import Axis, Dir, GT, chkarg
from maxwellfdfd.shape import Plane
from .source import Source
from .with_bloch import WithBloch
from .utils import ind_for_loc

if TYPE_CHECKING:
    from maxwellfdfd.grid import Grid3d


class PlaneSrc(Source, WithBloch):
    """
    Plane wave source representing constant electric dipole distribution.
    
    PlaneSrc is to generate a plane wave in a homogeneous medium. It supports
    oblique incidence, i.e., emission in the direction different from the
    normal direction of the plane.
    
    Parameters
    ----------
    normal_axis : Axis
        Direction normal to the plane (Axis.X, Axis.Y, or Axis.Z).
    intercept : float
        Location of the plane in the normal_axis direction.
    polarization : Axis or float
        Direction of the dipoles distributed on the plane. Can be an Axis
        instance orthogonal to normal_axis, or an angle in radians measured
        in the 2D Cartesian coordinate system normal to normal_axis.
    K : complex, optional
        Amplitude of the surface current density. Default is 1.0.
    theta : float or array_like, optional
        Oblique incidence angle measured from normal_axis, or in-plane Bloch
        wavevector. |theta| should not exceed pi/2.
    wvlen : float, optional
        Wavelength of the plane wave in the background medium. Required to set
        up the Bloch boundary condition for oblique incidence.
    
    Attributes
    ----------
    normal_axis : Axis
        Plane normal axis.
    intercept : float
        Intercept between plane and normal axis.
    phi : float
        Angle of polarization with respect to first Cartesian direction in
        this plane.
    kBloch : ndarray
        Bloch k-vector [kx, ky, kz].
    K : complex
        Surface current density.
    
    Notes
    -----
    In the finite-difference grid, PlaneSrc excites dipoles at the E-field
    points. This poses a condition on the intercept argument in the constructor:
    intercept should be at a dual grid point in the normal_axis direction.
    Therefore, make sure that intercept does not overlap with the locations of
    the vertices of Shape in the normal_axis direction; otherwise dynamic grid
    generation in maxwell_run() will fail.
    
    Examples
    --------
    >>> import numpy as np
    >>> from maxwellfdfd.base import Axis
    >>> from maxwellfdfd.source import PlaneSrc
    >>> 
    >>> # Create a z-polarized plane wave on y = 0 plane
    >>> src = PlaneSrc(Axis.Y, 0, Axis.Z)
    >>> 
    >>> # Create with oblique incidence
    >>> src = PlaneSrc(Axis.Y, 0, Axis.Z, theta=np.pi/6, wvlen=1.0)
    """
    
    def __init__(
        self,
        normal_axis: Axis,
        intercept: float,
        polarization: Axis | float,
        K: complex = 1.0,
        theta: float | np.ndarray = 0.0,
        wvlen: float | None = None
    ):
        """
        Initialize a PlaneSrc.
        
        Parameters
        ----------
        normal_axis : Axis
            Direction normal to the plane.
        intercept : float
            Location of the plane.
        polarization : Axis or float
            Direction of the dipoles.
        K : complex, optional
            Surface current density amplitude.
        theta : float or array_like, optional
            Oblique incidence angle or Bloch wave vector.
        wvlen : float, optional
            Wavelength for oblique incidence.
        """
        # Validate inputs
        chkarg(isinstance(normal_axis, Axis),
               '"normal_axis" should be instance of Axis.')
        p, q, _ = normal_axis.cycle()
        
        chkarg(np.isreal(intercept), '"intercept" should be real.')
        
        # Handle polarization
        phi_: float
        if isinstance(polarization, Axis):
            chkarg(polarization != normal_axis,
                   '"polarization" should be orthogonal to "normal_axis".')
            if polarization == p:
                phi_ = 0.0
            else:
                phi_ = np.pi / 2
        else:
            phi_ = float(polarization)
        
        chkarg(np.iscomplex(K) or np.isreal(K), '"K" should be complex.')
        
        # Handle theta and wvlen
        kt_Bloch: np.ndarray
        if wvlen is None:
            # theta is actually kt_Bloch
            if isinstance(theta, (int, float)):
                kt_Bloch = np.array([0.0, 0.0])
            else:
                theta_arr = np.asarray(theta)
                chkarg(
                    theta_arr.shape == (Dir.count(),) and np.isrealobj(theta_arr),
                    f'"kt_Bloch" should be length-{Dir.count()} array with real elements.'
                )
                kt_Bloch = theta_arr.copy()
        else:
            # theta is incidence angle, wvlen is wavelength
            chkarg(
                isinstance(theta, (int, float)) and -np.pi/2 <= theta <= np.pi/2,
                '"theta" should be polar angle in radian between -pi/2 and pi/2.'
            )
            chkarg(isinstance(wvlen, (int, float)) and wvlen > 0,
                   '"wvlen" should be positive.')
            
            kt = (2 * np.pi / wvlen) * np.sin(theta)  # tangential component of k
            kp = kt * np.cos(phi_ + np.pi / 2)
            kq = kt * np.sin(phi_ + np.pi / 2)
            kt_Bloch = np.array([kp, kq])
        
        # Create lgrid and laltgrid
        lgrid: List[np.ndarray] = [np.array([]) for _ in range(Axis.count())]
        laltgrid: List[np.ndarray] = [np.array([]) for _ in range(Axis.count())]
        lgrid[int(normal_axis)] = np.array([intercept])
        
        # Create plane shape
        plane = Plane(normal_axis, intercept)
        
        # Initialize parent class
        super().__init__(lgrid, laltgrid, plane)
        
        self._normal_axis = normal_axis
        self._intercept = intercept
        self._phi = phi_
        self._K = complex(K)
        
        # Set kBloch
        self._kBloch = np.zeros(Axis.count())
        self._kBloch[int(p)] = kt_Bloch[int(Dir.H)]
        self._kBloch[int(q)] = kt_Bloch[int(Dir.V)]
    
    @property
    def normal_axis(self) -> Axis:
        """Plane normal axis."""
        return self._normal_axis
    
    @property
    def intercept(self) -> float:
        """Intercept between plane and normal axis."""
        return self._intercept
    
    @property
    def phi(self) -> float:
        """Angle of polarization with respect to first Cartesian direction."""
        return self._phi
    
    @property
    def kBloch(self) -> np.ndarray:
        """Bloch k-vector [kx, ky, kz]."""
        return self._kBloch
    
    @property
    def K(self) -> complex:
        """Surface current density."""
        return self._K
    
    def generate_kernel(
        self,
        w_axis: Axis,
        grid3d: 'Grid3d'
    ) -> Tuple[List[np.ndarray | slice | str], np.ndarray]:
        """
        Generate the plane wave source kernel for a specific axis.
        
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
        
        index_cell: List[np.ndarray | slice | str] = []
        
        if w_axis == self._normal_axis:
            JMw_patch = np.array([])
            index_cell = [np.array([]) for _ in range(Axis.count())]
        else:
            n = self._normal_axis
            w = w_axis
            v = [axis for axis in Axis.elems() if axis != n and axis != w][0]
            
            g = self._gt
            ind_n = ind_for_loc(self._intercept, n, g, grid3d)
            dn = grid3d.dl[(n, g)][ind_n]
            J = self._K / dn  # K*dt = (current through dn*dt) = J * (dn*dt)
            
            # Calculate Jw based on polarization
            p, q, _ = n.cycle()
            if w == p:
                Jw = J * np.cos(self._phi)
            else:
                Jw = J * np.sin(self._phi)
            
            if np.abs(Jw) < 1e-15:
                JMw_patch = np.array([])
                index_cell = [np.array([]) for _ in range(Axis.count())]
            else:
                # Set Jw_patch with Bloch phase
                lw = grid3d.l[(w, g.alter())]
                lv = grid3d.l[(v, g)]
                kw = self._kBloch[int(w)]
                kv = self._kBloch[int(v)]
                
                # Create phase factors
                phase_w = np.exp(-1j * kw * lw)
                phase_v = np.exp(-1j * kv * lv)
                
                # Outer product to create 2D phase pattern (axes in w, v order)
                JMw_patch = Jw * np.outer(phase_w, phase_v)

                # The patch is assigned at a single index along the normal
                # axis n, yielding a 2D slice whose axes are the in-plane
                # axes {w, v} in ascending axis-index order. Reorder if the
                # outer product's (w, v) order does not match.
                if int(v) < int(w):
                    JMw_patch = JMw_patch.T
                
                # Set index_cell
                index_cell = [slice(None), slice(None), slice(None)]
                index_cell[int(n)] = ind_n
        
        return index_cell, JMw_patch