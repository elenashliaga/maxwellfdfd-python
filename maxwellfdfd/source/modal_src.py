"""
ModalSrc class for MaxwellFDFD.

This module defines the ModalSrc class, which represents an electric dipole
distribution that generates a mode of a waveguide.
"""

from typing import List, Tuple, TYPE_CHECKING, Dict, Any
import numpy as np

from maxwellfdfd.base import Axis, Dir, GT, FT, chkarg
from maxwellfdfd.shape import Plane
from .source import Source
from .utils import ind_for_loc

if TYPE_CHECKING:
    from maxwellfdfd.grid import Grid3d, Grid2d
    from maxwellfdfd.base import Oscillation


class ModalSrc(Source):
    """
    Modal source for generating waveguide modes.
    
    ModalSrc is used to store the transverse J-field distribution on a plane
    to generate a mode of a waveguide. The plane should be orthogonal to the
    axis of the waveguide. The mode is calculated by an external mode solver.
    
    Parameters
    ----------
    normal_axis : Axis
        Direction normal to the plane (Axis.X, Axis.Y, or Axis.Z). This is also
        the axis of the waveguide.
    intercept : float
        Location of the plane in the normal_axis direction.
    opts : dict
        Options dictionary that controls the behavior of the mode solver.
        Must contain 'clue' key with value 'order' or 'guess'.
    KA : float, optional
        Integral of the norm of the surface current density. Indicates the
        strength of the mode amplitude. Default is 1.0.
    
    Attributes
    ----------
    normal_axis : Axis
        Plane normal axis (waveguide axis).
    intercept : float
        Intercept between plane and normal axis.
    KA : float
        Surface integral of surface current density K.
    opts : dict
        Options to control method to calculate mode.
    ispreped : bool
        True if this ModalSrc is prepared; False otherwise.
    beta_r : float
        Real part of complex wavevector.
    Lp : float
        Propagation length.
    
    Notes
    -----
    In the finite-difference grid, ModalSrc excites dipoles at the E-field
    points. This poses a condition on the intercept argument in the constructor:
    intercept should be at a dual grid point in the normal_axis direction.
    Therefore, make sure that intercept does not overlap with the locations of
    the vertices of Shape in the normal_axis direction; otherwise dynamic grid
    generation in maxwell_run() will fail.
    
    The opts dictionary should have:
    - 'clue': 'order' or 'guess'
    - If 'clue' == 'order': 'order' key with positive integer value
    - If 'clue' == 'guess': 'neff' key with complex value
    
    Examples
    --------
    >>> import numpy as np
    >>> from maxwellfdfd.base import Axis
    >>> from maxwellfdfd.source import ModalSrc
    >>> 
    >>> # Create a modal source for fundamental mode
    >>> opts = {'clue': 'order', 'order': 1}
    >>> src = ModalSrc(Axis.Y, -1000, opts)
    """
    
    def __init__(
        self,
        normal_axis: Axis,
        intercept: float,
        opts: Dict[str, Any],
        KA: float = 1.0
    ):
        """
        Initialize a ModalSrc.
        
        Parameters
        ----------
        normal_axis : Axis
            Direction normal to the plane.
        intercept : float
            Location of the plane.
        opts : dict
            Mode solver options.
        KA : float, optional
            Mode amplitude integral.
        """
        # Validate inputs
        chkarg(isinstance(normal_axis, Axis),
               '"normal_axis" should be instance of Axis.')
        chkarg(np.isreal(intercept), '"intercept" should be real.')
        chkarg(isinstance(opts, dict), '"opts" should be dictionary.')
        
        # Validate opts
        clue = opts.get('clue')
        chkarg(clue in ['order', 'guess'],
               '"opts.clue" should be "order" or "guess".')
        
        if clue == 'guess':
            neff = opts.get('neff')
            chkarg(
                isinstance(neff, complex) or (isinstance(neff, (int, float))),
                '"opts.neff" should be complex.'
            )
        else:  # clue == 'order'
            order = opts.get('order', 1)
            chkarg(
                isinstance(order, int) and order > 0,
                '"opts.order" should be positive integer.'
            )
        
        chkarg(np.isreal(KA), '"KA" should be real.')
        
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
        self._opts = opts.copy()
        self._KA = float(KA)
        
        # Private properties
        self._grid2d: 'Grid2d | None' = None
        self._osc: 'Oscillation | None' = None
        self._E2d: List[np.ndarray] = [np.array([]) for _ in range(Axis.count())]
        self._H2d: List[np.ndarray] = [np.array([]) for _ in range(Axis.count())]
        self._JMh: np.ndarray = np.array([])
        self._JMv: np.ndarray = np.array([])
        self._neff: complex = 0.0 + 0.0j
    
    @property
    def normal_axis(self) -> Axis:
        """Plane normal axis."""
        return self._normal_axis
    
    @property
    def intercept(self) -> float:
        """Intercept between plane and normal axis."""
        return self._intercept
    
    @property
    def KA(self) -> float:
        """Surface integral of surface current density K."""
        return self._KA
    
    @property
    def opts(self) -> Dict[str, Any]:
        """Options to control method to calculate mode."""
        return self._opts.copy()
    
    @property
    def ispreped(self) -> bool:
        """True if this ModalSrc is prepared; False otherwise."""
        return self._JMh.size > 0 and self._JMv.size > 0
    
    @property
    def beta_r(self) -> float:
        """Real part of complex wavevector."""
        beta = 2 * np.pi * self._neff / self._osc.in_L0()
        return np.real(beta)
    
    @property
    def Lp(self) -> float:
        """Propagation length."""
        beta = 2 * np.pi * self._neff / self._osc.in_L0()
        return -1.0 / np.imag(beta)

    @property
    def neff(self) -> complex:
        """Effective refractive index of the solved mode.

        Zero until the mode is solved (by `prep` or a manual `setEH` call).
        """
        return self._neff

    def prep(
        self,
        ge: GT,
        pml,
        osc,
        grid3d: 'Grid3d',
        eps_cell,
        mu_cell,
        s_factor_cell,
    ) -> None:
        """Solve for the waveguide mode and populate this ModalSrc.

        Python port of MATLAB `modesolver/prep_modalsrc.m`. Builds the 2D
        eigenoperator on the cross-section grid (see `maxwellfdfd.modesolver`),
        solves for the mode selected by `opts` ('guess' -> shift-invert around
        the neff guess; 'order' -> spectrum-shifted solve of the real
        operator), reconstructs all six field components and stores them via
        `setEH`.

        Notes
        -----
        - The intercept plane must lie OUTSIDE the PML regions along the
          normal axis (checked inside `wgmode_matrix`).
        - The MATLAB option `opts.H2d` (initial eigenvector guess) is not
          ported; 'guess' uses shift-invert directly, which is robust for a
          reasonable neff estimate.

        Parameters
        ----------
        ge : GT
            Grid type of the E field.
        pml : PML
            PML type (PML.SC or PML.U).
        osc : Oscillation
            Oscillation parameters.
        grid3d : Grid3d
            3D grid of the simulation domain.
        eps_cell, mu_cell : dict of Axis to ndarray
            Complex material arrays (from `mean_material_node`).
        s_factor_cell : dict of (Axis, GT) to ndarray
            PML s-factors (from `generate_s_factor`).
        """
        from maxwellfdfd.base import PML as PMLType
        from maxwellfdfd.modesolver import (
            wgmode_matrix, solve_mode, solve_mode_order,
        )

        A, En_Htr, gEh_HvEn, gEv_HhEn, Hn_Etr = wgmode_matrix(
            ge, pml, osc.in_omega0(), eps_cell, mu_cell, s_factor_cell,
            grid3d, self._normal_axis, self._intercept,
        )

        if self._opts.get('clue') == 'order':
            # Real (lossless) operator for mode-order selection.
            Ar, *_ = wgmode_matrix(
                ge, pml, np.real(osc.in_omega0()),
                {w: np.real(eps_cell[w]) for w in Axis.elems()},
                {w: np.real(mu_cell[w]) for w in Axis.elems()},
                {key: np.real(val) for key, val in s_factor_cell.items()},
                grid3d, self._normal_axis, self._intercept,
            )
            v, beta = solve_mode_order(Ar, int(self._opts.get('order', 1)))
        else:  # clue == 'guess'
            beta_guess = 2 * np.pi * complex(self._opts.get('neff', 1.0)) \
                / osc.in_L0()
            v, beta = solve_mode(A, beta_guess)

        neff = beta * osc.in_L0() / (2 * np.pi)

        # Reconstruct the remaining field components from Htr = [Hh; Hv].
        N = v.size // 2
        Hh = v[:N]
        Hv = v[N:]
        En = En_Htr @ v
        Eh = (gEh_HvEn @ np.concatenate([Hv, En])) / (1j * beta)
        Ev = (gEv_HhEn @ np.concatenate([Hh, En])) / (1j * beta)
        Hn = Hn_Etr @ np.concatenate([Eh, Ev])

        h, vv, n = self._normal_axis.cycle()  # (p, q, r): h, v, normal
        Nh = int(grid3d.N[int(h)])
        Nv = int(grid3d.N[int(vv)])

        E_cell = [None] * Axis.count()
        H_cell = [None] * Axis.count()
        E_cell[int(h)] = np.asarray(Eh).reshape(Nh, Nv)
        E_cell[int(vv)] = np.asarray(Ev).reshape(Nh, Nv)
        E_cell[int(n)] = np.asarray(En).reshape(Nh, Nv)
        H_cell[int(h)] = np.asarray(Hh).reshape(Nh, Nv)
        H_cell[int(vv)] = np.asarray(Hv).reshape(Nh, Nv)
        H_cell[int(n)] = np.asarray(Hn).reshape(Nh, Nv)

        self.setEH(neff, osc, E_cell, H_cell, ge, grid3d)
        print(f'\tmode solved: beta = {beta}, n_eff = {neff}')
    
    def setEH(
        self,
        neff: complex,
        osc: 'Oscillation',
        E_cell: List[np.ndarray],
        H_cell: List[np.ndarray],
        ge: GT,
        grid3d: 'Grid3d'
    ) -> None:
        """
        Set the electromagnetic field distribution for the mode.
        
        Parameters
        ----------
        neff : complex
            Effective refractive index of the mode.
        osc : Oscillation
            Oscillation parameters.
        E_cell : list of ndarray
            E-field components on the plane.
        H_cell : list of ndarray
            H-field components on the plane.
        ge : GT
            Grid type for the fields.
        grid3d : Grid3d
            3D grid instance.
        """
        # Note: Grid2d and Oscillation validation will be done when modules are available
        chkarg(np.iscomplex(neff) or np.isreal(neff), '"neff" should be complex.')
        self._neff = complex(neff)
        
        # chkarg(isinstance(osc, Oscillation), '"osc" should be instance of Oscillation.')
        self._osc = osc
        
        # chkarg(isinstance(grid3d, Grid3d), '"grid3d" should be instance of Grid3d.')
        # self._grid2d = Grid2d(grid3d, self._normal_axis)
        
        # chkarg(isinstance(ge, GT), '"ge" should be instance of GT.')
        
        # Note: The following would use Grid2d when available
        # Nh = self._grid2d.N(Dir.H)
        # Nv = self._grid2d.N(Dir.V)
        # h = self._grid2d.axis(Dir.H)
        # v = self._grid2d.axis(Dir.V)
        
        # For now, use the arrays directly
        Nh = E_cell[0].shape[0]
        Nv = E_cell[0].shape[1]
        
        # Validate E_cell and H_cell
        for w in Axis.elems():
            chkarg(
                E_cell[int(w)].shape == (Nh, Nv),
                f'"E_cell" element {w} should be {Nh}-by-{Nv} array.'
            )
            chkarg(
                H_cell[int(w)].shape == (Nh, Nv),
                f'"H_cell" element {w} should be {Nh}-by-{Nv} array.'
            )
        
        # Set JMh and JMv based on grid type
        # Note: This would use proper axis mapping with Grid2d
        h_axis = (int(self._normal_axis) + 1) % 3
        v_axis = (int(self._normal_axis) + 2) % 3
        
        if self._gt == ge:  # source is J
            self._JMh = H_cell[v_axis]
            self._JMv = -H_cell[h_axis]
        else:  # source is M
            self._JMh = E_cell[v_axis]
            self._JMv = -E_cell[h_axis]
        
        # Store E2d and H2d
        # Note: This would use array2scalar when available
        self._E2d = [arr.copy() for arr in E_cell]
        self._H2d = [arr.copy() for arr in H_cell]
    
    def generate_kernel(
        self,
        w_axis: Axis,
        grid3d: 'Grid3d'
    ) -> Tuple[List[np.ndarray | slice | str], np.ndarray]:
        """
        Generate the modal source kernel for a specific axis.
        
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
            If JMh and JMv are not set.
        """
        if self._JMh.size == 0 or self._JMv.size == 0:
            raise RuntimeError('"JMh" and "JMv" are not set in this ModalSrc.')
        
        index_cell: List[np.ndarray | slice | str] = []
        
        if w_axis == self._normal_axis:
            JMw_patch = np.array([])
            index_cell = [np.array([]) for _ in range(Axis.count())]
        else:
            # Global axis indices/objects of the cross-section grid.
            h_int = (int(self._normal_axis) + 1) % 3
            v_int = (int(self._normal_axis) + 2) % 3
            h_ax = Axis.elems()[h_int]
            v_ax = Axis.elems()[v_int]
            n = self._normal_axis
            
            g = self._gt
            ind_n = ind_for_loc(self._intercept, n, g, grid3d)
            
            # Set index_cell
            index_cell = [slice(None), slice(None), slice(None)]
            index_cell[int(n)] = ind_n
            
            # Calculate normalization factor
            dn = grid3d.dl[(n, g)][ind_n]
            dha = grid3d.dl[(h_ax, g.alter())]
            dhg = grid3d.dl[(h_ax, g)]
            dva = grid3d.dl[(v_ax, g.alter())]
            dvg = grid3d.dl[(v_ax, g)]
            
            # Calculate volume elements
            dVh = dn * np.outer(dha, dvg)
            dVv = dn * np.outer(dhg, dva)
            
            # Calculate current KA
            KA_curr = np.abs(self._JMh) * dVh + np.abs(self._JMv) * dVv
            KA_curr = np.sum(KA_curr)  # KA_curr is real
            
            # Get phase factor
            JMhv = np.concatenate([self._JMh.ravel(), self._JMv.ravel()])
            i_pf = np.argmax(np.abs(JMhv))
            phasefactor = JMhv[i_pf] / np.abs(JMhv[i_pf])
            KA_curr = KA_curr * phasefactor  # KA_curr is complex
            
            # Calculate normalization factor
            norm_factor = self._KA / KA_curr
            
            # Set JMw_patch
            if int(w_axis) == h_int:
                JMw_patch = norm_factor * self._JMh
            else:
                JMw_patch = norm_factor * self._JMv

            # Promote the 2D (Nh, Nv) patch to 3D with a singleton axis
            # along the waveguide normal, apply the INVERSE permutation of
            # [h, v, n] (MATLAB: ipermute(JMw_patch, int([h v n]))), then
            # drop the singleton normal axis so the patch matches the shape
            # of the target slice JM_cell[w][..., ind_n, ...] taken by
            # assign_source.
            JMw_patch = np.asarray(JMw_patch)
            if JMw_patch.ndim == 2:
                JMw_patch = JMw_patch[:, :, np.newaxis]
            inv_perm = np.argsort([h_int, v_int, int(n)])
            JMw_patch = np.transpose(JMw_patch, inv_perm)
            shape = list(JMw_patch.shape)
            shape.pop(int(n))
            JMw_patch = JMw_patch.reshape(shape)

        return index_cell, JMw_patch