"""
TFSFPlaneSrc class for MaxwellFDFD.

This module defines the TFSFPlaneSrc class, which represents a plane wave
propagating in a total field box of the total-field/scattered-field (TF/SF)
method.
"""

from typing import List, Tuple, TYPE_CHECKING, Dict
import numpy as np

from maxwellfdfd.base import Axis, Sign, chkarg
from maxwellfdfd.shape import Box
from .source import Source

if TYPE_CHECKING:
    from maxwellfdfd.grid import Grid3d
    from maxwellfdfd.base import Oscillation


class TFSFPlaneSrc(Source):
    """
    Total-field/scattered-field plane wave source.
    
    TFSFPlaneSrc is a source type for the TF/SF method. It generates a plane
    wave inside a total field box. The box should be empty except for
    scatterers. It supports an arbitrary propagation direction and
    polarization direction.
    
    Parameters
    ----------
    bound : array_like
        Description of the box in the format of [[xmin, xmax], [ymin, ymax],
        [zmin, zmax]].
    propagation_axis : Axis or array_like
        Direction of the wavevector of the plane wave. Can be an Axis instance
        or [kx, ky, kz]. The latter does not need to be normalized.
    polarization_axis : Axis or array_like
        Axis of the polarization of the F-field (E or H depending on whether
        the source is used as 'SRCJ' or 'SRCM'). Can be an Axis instance or
        [Fx, Fy, Fz]. The latter does not need to be normalized.
    F0 : complex, optional
        Complex amplitude of the plane wave measured at the origin.
        Default is 1.0.
    
    Attributes
    ----------
    khat : ndarray
        Unit vector along k-vector of plane wave [nx, ny, nz].
    Fhat : ndarray
        Unit vector along polarization direction [px, py, pz].
    F0 : complex
        Complex amplitude of plane wave.
    
    Methods
    -------
    set_bg_material(material)
        Sets the background material that determines the wavelength of the
        plane wave.
    create_incidentF(osc, grid3d)
        Returns {Fx_array, Fy_array, Fz_array} that has the F-field of the
        plane wave inside the total field box region.
    setJM(JM_cell, grid3d)
        Sets the J or M that creates the F-field inside the total field box.
    
    Notes
    -----
    TFSFPlaneSrc is similar to PlaneSrc, but has a few differences:
    - PlaneSrc requires a periodic boundary condition along the transverse
      direction, so it can be used only for a periodic structure.
    - PlaneSrc supports an arbitrary propagation direction by utilizing the
      Bloch boundary condition internally, but it does not support an arbitrary
      polarization direction: the polarization direction is restricted to
      Axis.X, Axis.Y, Axis.Z.
    - TFSFPlaneSrc supports a fully arbitrary polarization direction.
    
    When passed as a SRC parameter of maxwell_run(), the scatterer objects are
    described in the SOBJ parameter group and the background objects are
    described in the OBJ parameter group. No background objects should be
    inside the total field box, because TFSFPlaneSrc assumes that the total
    field box is completely filled with the default material of the simulation
    domain.
    
    Examples
    --------
    >>> import numpy as np
    >>> from maxwellfdfd.base import Axis
    >>> from maxwellfdfd.source import TFSFPlaneSrc
    >>> 
    >>> # Create a TF/SF plane wave source
    >>> bound = np.array([[-1, 1], [-1, 1], [-1, 1]])
    >>> src = TFSFPlaneSrc(bound, Axis.Y, Axis.Z)
    """
    
    def __init__(
        self,
        bound: np.ndarray,
        propagation_axis: Axis | np.ndarray,
        polarization_axis: Axis | np.ndarray,
        F0: complex = 1.0
    ):
        """
        Initialize a TFSFPlaneSrc.
        
        Parameters
        ----------
        bound : array_like
            Box boundaries [[xmin, xmax], [ymin, ymax], [zmin, zmax]].
        propagation_axis : Axis or array_like
            Wavevector direction.
        polarization_axis : Axis or array_like
            Polarization direction.
        F0 : complex, optional
            Plane wave amplitude.
        """
        # Validate inputs
        bound_arr = np.asarray(bound)
        chkarg(
            bound_arr.shape == (Axis.count(), Sign.count()) and np.isrealobj(bound_arr),
            f'"bound" should be {Axis.count()}-by-{Sign.count()} array with real elements.'
        )
        
        # Handle propagation_axis
        khat: np.ndarray
        if isinstance(propagation_axis, Axis):
            khat = np.zeros(Axis.count())
            khat[int(propagation_axis)] = 1.0
        else:
            khat = np.asarray(propagation_axis)
            chkarg(
                khat.shape == (Axis.count(),) and np.isrealobj(khat),
                f'"propagation_axis" should be length-{Axis.count()} array with real elements.'
            )
        
        # Normalize khat
        khat_norm = np.linalg.norm(khat)
        chkarg(khat_norm > 0, '"propagation_axis" should not be zero vector.')
        khat = khat / khat_norm
        
        # Handle polarization_axis
        Fhat: np.ndarray
        if isinstance(polarization_axis, Axis):
            Fhat = np.zeros(Axis.count())
            Fhat[int(polarization_axis)] = 1.0
        else:
            Fhat = np.asarray(polarization_axis)
            chkarg(
                Fhat.shape == (Axis.count(),) and np.isrealobj(Fhat),
                f'"polarization_axis" should be length-{Axis.count()} array with real elements.'
            )
        
        # Normalize Fhat
        Fhat_norm = np.linalg.norm(Fhat)
        chkarg(Fhat_norm > 0, '"polarization_axis" should not be zero vector.')
        Fhat = Fhat / Fhat_norm
        
        # Check orthogonality
        chkarg(
            np.abs(np.dot(khat, Fhat)) < 1e-12,
            '"propagation_axis" and "polarization_axis" should be orthogonal.'
        )
        
        chkarg(np.iscomplex(F0) or np.isreal(F0), '"F0" should be complex.')
        
        # Create lgrid and laltgrid
        lgrid: List[np.ndarray] = []
        laltgrid: List[np.ndarray] = []
        for w in Axis.elems():
            lgrid.append(np.array([bound_arr[int(w), int(Sign.N)], bound_arr[int(w), int(Sign.P)]]))
            laltgrid.append(np.array([bound_arr[int(w), int(Sign.N)], bound_arr[int(w), int(Sign.P)]]))
        
        # Create box shape
        box = Box(bound_arr)
        
        # Initialize parent class with forceprim=True
        super().__init__(lgrid, laltgrid, box, forceprim=True)
        
        self._khat = khat
        self._Fhat = Fhat
        self._F0 = complex(F0)
        self._N_bg: complex = np.nan
        self._grid3d: 'Grid3d | None' = None
        self._JM: List[np.ndarray] = [np.array([]) for _ in range(Axis.count())]
    
    @property
    def khat(self) -> np.ndarray:
        """Unit vector along k-vector of plane wave."""
        return self._khat
    
    @property
    def Fhat(self) -> np.ndarray:
        """Unit vector along polarization direction."""
        return self._Fhat
    
    @property
    def F0(self) -> complex:
        """Complex amplitude of plane wave."""
        return self._F0
    
    def set_bg_material(self, material: 'Material') -> None:
        """
        Set the background material.
        
        Parameters
        ----------
        material : Material
            Background material instance.
        
        Raises
        ------
        ValueError
            If material is anisotropic.
        """
        # Note: Material validation will be done when material module is available
        # chkarg(isinstance(material, Material), '"mat_bg" should be instance of Material.')
        # chkarg(material.isiso, 'anisotropic "mat_bg" is currently not supported.')
        
        # For now, assume isotropic material
        eps = material.eps if hasattr(material, 'eps') else 1.0
        mu = material.mu if hasattr(material, 'mu') else 1.0
        
        # Extract diagonal elements for isotropic case
        if isinstance(eps, np.ndarray) and eps.ndim == 2:
            eps = np.diag(eps)[0]
        if isinstance(mu, np.ndarray) and mu.ndim == 2:
            mu = np.diag(mu)[0]
        
        self._N_bg = np.sqrt(complex(eps) * complex(mu))
    
    def create_incidentF(
        self,
        osc: 'Oscillation',
        grid3d: 'Grid3d'
    ) -> List[np.ndarray]:
        """
        Create the incident F-field inside the total field box.
        
        Parameters
        ----------
        osc : Oscillation
            Oscillation parameters.
        grid3d : Grid3d
            3D grid instance.
        
        Returns
        -------
        F_cell : list of ndarray
            List of F-field components [Fx, Fy, Fz].
        """
        # Note: Oscillation validation will be done when module is available
        # chkarg(isinstance(osc, Oscillation), '"osc" should be instance of Oscillation.')
        # chkarg(isinstance(grid3d, Grid3d), '"grid3d" should be instance of Grid3d.')
        
        self._grid3d = grid3d
        
        F_cell: List[np.ndarray] = []
        bound = self.shape.bound
        k = self._khat * 2 * np.pi * self._N_bg / osc.in_L0()  # k = (2*pi*n/lambda)
        
        for w in Axis.elems():
            # Initialize with zeros (complex dtype for field values)
            F_w = np.zeros(grid3d.N, dtype=complex)
            
            # Get grid locations
            p, q, _ = w.cycle()
            l_p = grid3d.l[(p, self._gt)]
            l_q = grid3d.l[(q, self._gt)]
            l_w = grid3d.l[(w, self._gt.alter())]
            
            # Find indices within the box
            ind_p = np.where((l_p >= bound[int(p), int(Sign.N)]) & (l_p <= bound[int(p), int(Sign.P)]))[0]
            ind_q = np.where((l_q >= bound[int(q), int(Sign.N)]) & (l_q <= bound[int(q), int(Sign.P)]))[0]
            ind_w = np.where((l_w >= bound[int(w), int(Sign.N)]) & (l_w <= bound[int(w), int(Sign.P)]))[0]
            
            if len(ind_p) > 0 and len(ind_q) > 0 and len(ind_w) > 0:
                # Create meshgrid in (p, q, w) order
                Lp, Lq, Lw = np.meshgrid(l_p[ind_p], l_q[ind_q], l_w[ind_w], indexing='ij')
                
                # Calculate phase: k · r with correct axis-component mapping
                phase = k[int(p)] * Lp + k[int(q)] * Lq + k[int(w)] * Lw
                
                # Compute field values in (p, q, w) order
                field_vals = (self._F0 * self._Fhat[int(w)]) * np.exp(-1j * phase)
                
                # Transpose from (p, q, w) to (x, y, z) order
                perm = [0, 0, 0]
                perm[int(p)] = 0
                perm[int(q)] = 1
                perm[int(w)] = 2
                field_vals = np.transpose(field_vals, perm)
                
                # Map indices to (x, y, z) order for F_w assignment
                idx = [None, None, None]
                idx[int(p)] = ind_p
                idx[int(q)] = ind_q
                idx[int(w)] = ind_w
                F_w[np.ix_(idx[0], idx[1], idx[2])] = field_vals
            
            F_cell.append(F_w)
        
        return F_cell
    
    def setJM(self, JM_cell: List[np.ndarray], grid3d: 'Grid3d') -> None:
        """
        Set the J or M that creates the F-field inside the total field box.
        
        Parameters
        ----------
        JM_cell : list of ndarray
            List of J or M arrays [JMx, JMy, JMz].
        grid3d : Grid3d
            3D grid instance (must be the same as used in create_incidentF).
        
        Raises
        ------
        ValueError
            If grid3d is not the same as used in create_incidentF.
        """
        chkarg(grid3d is self._grid3d,
               'same "grid3d" should be used as in create_incidentF().')
        
        # Validate JM_cell
        for w in Axis.elems():
            chkarg(
                tuple(JM_cell[int(w)].shape) == tuple(int(n) for n in grid3d.N),
                f'"JM_cell" element {w} should have shape {grid3d.N}.'
            )
        
        self._JM = [arr.copy() for arr in JM_cell]
    
    def generate_kernel(
        self,
        w_axis: Axis,
        grid3d: 'Grid3d'
    ) -> Tuple[List[np.ndarray | slice | str], np.ndarray]:
        """
        Generate the TF/SF plane wave source kernel for a specific axis.
        
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
        
        Raises
        ------
        RuntimeError
            If background material is not set.
        """
        if np.isnan(self._N_bg):
            raise RuntimeError('"mat_bg" is not set in this TFSFPlaneSrc.')
        
        index_cell = [slice(None), slice(None), slice(None)]
        JMw_patch = self._JM[int(w_axis)]
        
        return index_cell, JMw_patch