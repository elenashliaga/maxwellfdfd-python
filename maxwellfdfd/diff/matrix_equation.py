"""
Matrix equation class for FDFD system.

This module provides the MatrixEquation class that constructs and manages
the matrix equation for finite-difference frequency-domain simulations.
"""

from typing import Callable, Dict, List, Tuple, Union

import numpy as np
import scipy.sparse as sp

from maxwellfdfd.base import Axis, BC, FT, GT, PML, create_spdiag
from maxwellfdfd.grid import Grid3d
from maxwellfdfd.diff.create_curls import create_curls
from maxwellfdfd.diff._utils import div_vec, mult_vec, reordering_indices


class MatrixEquation:
    """
    Matrix equation class for FDFD system.

    This class constructs the matrix equation A*f = b for Maxwell's equations
    in the frequency domain. It supports different equation types (E-field,
    H-field, or combined), PML implementations, and various boundary conditions.

    Parameters
    ----------
    eqtype : EquationType
        Equation type specifying which field to solve for and grid type.
    pml : PML
        PML type (PML.SC for stretched-coordinate, PML.U for uniaxial).
    omega : complex
        Angular frequency of the simulation.
    eps_cell : dict
        Permittivity cell arrays with Axis keys. Each array has shape (Nx, Ny, Nz).
    mu_cell : dict
        Permeability cell arrays with Axis keys. Each array has shape (Nx, Ny, Nz).
    s_factor_cell : dict
        PML stretching factor cell arrays with (Axis, GT) keys.
    J_cell : dict
        Electric current source cell arrays with Axis keys.
    M_cell : dict
        Magnetic current source cell arrays with Axis keys.
    grid3d : Grid3d
        3D grid object containing grid information.

    Attributes
    ----------
    Ce : scipy.sparse.csr_matrix
        Curl operator for E-field.
    Cm : scipy.sparse.csr_matrix
        Curl operator for H-field.
    eps : ndarray
        Flattened permittivity vector.
    mu : ndarray
        Flattened permeability vector.
    j : ndarray
        Flattened electric current source vector.
    m : ndarray
        Flattened magnetic current source vector.
    omega : complex
        Angular frequency.
    pm : ndarray
        PEC mask vector (1 for non-PEC, 0 for PEC).
    r : ndarray
        Reordering indices for bandwidth reduction.
    s : ndarray
        Inverse reordering indices.
    ft : FT
        Field type of equation (FT.E, FT.H, or FT.ELEMS).

    Methods
    -------
    matrix_op()
        Get matrix operation A and right-hand side b.
    matrixfree_op()
        Get matrix-free operators for iterative solving.

    Examples
    --------
    >>> import numpy as np
    >>> from maxwellfdfd.base import Axis, BC, GT, FT, PML, EquationType, PhysUnit
    >>> from maxwellfdfd.grid import Grid3d
    >>> unit = PhysUnit()
    >>> lprim = [np.array([0.0, 0.5, 1.0]), np.array([0.0, 0.5, 1.0]), np.array([0.0, 0.5, 1.0])]
    >>> Npml = [[1, 1], [1, 1], [1, 1]]
    >>> bc = {Axis.X: BC.E, Axis.Y: BC.E, Axis.Z: BC.E}
    >>> grid3d = Grid3d(unit, lprim, Npml, bc)
    >>> eqtype = EquationType(FT.E, GT.Prim)
    >>> eps_cell = {Axis.X: np.ones((2, 2, 2)), Axis.Y: np.ones((2, 2, 2)), Axis.Z: np.ones((2, 2, 2))}
    >>> mu_cell = {Axis.X: np.ones((2, 2, 2)), Axis.Y: np.ones((2, 2, 2)), Axis.Z: np.ones((2, 2, 2))}
    >>> s_factor_cell = {}
    >>> J_cell = {Axis.X: np.zeros((2, 2, 2)), Axis.Y: np.zeros((2, 2, 2)), Axis.Z: np.zeros((2, 2, 2))}
    >>> M_cell = {Axis.X: np.zeros((2, 2, 2)), Axis.Y: np.zeros((2, 2, 2)), Axis.Z: np.zeros((2, 2, 2))}
    >>> omega = 2 * np.pi * 3e8 / 1.55e-6
    >>> eq = MatrixEquation(eqtype, PML.SC, omega, eps_cell, mu_cell, s_factor_cell, J_cell, M_cell, grid3d)
    >>> A, b = eq.matrix_op()
    >>> A.shape
    (24, 24)
    """

    def __init__(
        self,
        eqtype,
        pml: PML,
        omega: complex,
        eps_cell: Dict[Axis, np.ndarray],
        mu_cell: Dict[Axis, np.ndarray],
        s_factor_cell: Dict[tuple, np.ndarray],
        J_cell: Dict[Axis, np.ndarray],
        M_cell: Dict[Axis, np.ndarray],
        grid3d: Grid3d
    ):
        """Initialize the MatrixEquation."""
        # Validate inputs
        if not isinstance(grid3d, Grid3d):
            raise TypeError('"grid3d" should be instance of Grid3d')
        
        N = grid3d.N
        
        # Validate eps_cell, mu_cell, J_cell, M_cell
        for axis in Axis.elems():
            if axis not in eps_cell:
                raise ValueError(f"Missing {axis} in eps_cell")
            if eps_cell[axis].shape != tuple(N):
                raise ValueError(f"eps_cell[{axis}] should have shape {tuple(N)}")
            
            if axis not in mu_cell:
                raise ValueError(f"Missing {axis} in mu_cell")
            if mu_cell[axis].shape != tuple(N):
                raise ValueError(f"mu_cell[{axis}] should have shape {tuple(N)}")
            
            if axis not in J_cell:
                raise ValueError(f"Missing {axis} in J_cell")
            if J_cell[axis].shape != tuple(N):
                raise ValueError(f"J_cell[{axis}] should have shape {tuple(N)}")
            
            if axis not in M_cell:
                raise ValueError(f"Missing {axis} in M_cell")
            if M_cell[axis].shape != tuple(N):
                raise ValueError(f"M_cell[{axis}] should have shape {tuple(N)}")
        
        # Reorder the indices to reduce bandwidth
        self.r = reordering_indices(Axis.count(), N)
        self.s = np.argsort(self.r)  # Inverse reordering
        
        # Construct curls
        dl_factor_cell = None
        if pml == PML.SC:
            dl_factor_cell = s_factor_cell
        
        ge = eqtype.ge
        self.Ce, self.Cm = create_curls(ge, dl_factor_cell, grid3d)
        
        # Construct material parameters
        if pml == PML.U:
            # Uniaxial PML - modify material parameters
            # Create s-factor grids
            sm = [s_factor_cell[(Axis.X, ge.alter())],
                  s_factor_cell[(Axis.Y, ge.alter())],
                  s_factor_cell[(Axis.Z, ge.alter())]]
            se = [s_factor_cell[(Axis.X, ge)],
                  s_factor_cell[(Axis.Y, ge)],
                  s_factor_cell[(Axis.Z, ge)]]
            
            # Create meshgrids
            smx, smy, smz = np.meshgrid(sm[0], sm[1], sm[2], indexing='ij')
            sex, sey, sez = np.meshgrid(se[0], se[1], se[2], indexing='ij')
            
            sm_dict = {Axis.X: smx, Axis.Y: smy, Axis.Z: smz}
            se_dict = {Axis.X: sex, Axis.Y: sey, Axis.Z: sez}
            
            # Modify mu_cell
            mu_cell = mult_vec(mu_cell, [sm_dict[Axis.Y], sm_dict[Axis.Z], sm_dict[Axis.X]])
            mu_cell = mult_vec(mu_cell, [sm_dict[Axis.Z], sm_dict[Axis.X], sm_dict[Axis.Y]])
            mu_cell = div_vec(mu_cell, [se_dict[Axis.X], se_dict[Axis.Y], se_dict[Axis.Z]])
            
            # Modify eps_cell
            eps_cell = mult_vec(eps_cell, [se_dict[Axis.Y], se_dict[Axis.Z], se_dict[Axis.X]])
            eps_cell = mult_vec(eps_cell, [se_dict[Axis.Z], se_dict[Axis.X], se_dict[Axis.Y]])
            eps_cell = div_vec(eps_cell, [sm_dict[Axis.X], sm_dict[Axis.Y], sm_dict[Axis.Z]])
        
        # Flatten material parameters
        self.mu = np.concatenate([mu_cell[Axis.X].ravel(),
                                  mu_cell[Axis.Y].ravel(),
                                  mu_cell[Axis.Z].ravel()])
        self.eps = np.concatenate([eps_cell[Axis.X].ravel(),
                                   eps_cell[Axis.Y].ravel(),
                                   eps_cell[Axis.Z].ravel()])
        
        # Flatten source terms
        self.j = np.concatenate([J_cell[Axis.X].ravel(),
                                 J_cell[Axis.Y].ravel(),
                                 J_cell[Axis.Z].ravel()])
        self.m = np.concatenate([M_cell[Axis.X].ravel(),
                                 M_cell[Axis.Y].ravel(),
                                 M_cell[Axis.Z].ravel()])
        
        self.omega = omega
        self.ft = eqtype.f
        
        # Mask elements corresponding to PEC
        if self.ft == FT.E:
            ind_pec = np.isinf(np.abs(self.eps))
            self.eps = self.eps.copy()
            self.eps[ind_pec] = 1.0
            self.pm = np.ones_like(self.eps, dtype=float)  # PEC mask
            self.pm[ind_pec] = 0.0
    
    def matrix_op(self) -> Tuple[sp.csr_matrix, np.ndarray]:
        """
        Get matrix operation A and right-hand side b.

        Constructs the matrix A and right-hand side b for the equation A*f = b.
        The form depends on the field type:
        - FT.E: (PM * Cm * inv(mu) * Ce * PM - omega^2 * eps) * e = b
        - FT.H: (Ce * inv(eps) * Cm - omega^2 * mu) * h = b
        - FT.ELEMS: [[-1j*omega*eps, Cm], [Ce, 1j*omega*mu]] * [e; h] = [j; -m]

        Returns
        -------
        A : scipy.sparse.csr_matrix
            System matrix with reordered indices.
        b : ndarray
            Right-hand side vector with reordered indices.

        Examples
        --------
        >>> A, b = eq.matrix_op()
        >>> f = sp.linalg.spsolve(A, b)
        """
        if self.ft == FT.E:
            # E-field equation
            INV_MU = create_spdiag(1.0 / self.mu)
            EPS = create_spdiag(self.eps)
            PM = create_spdiag(self.pm)
            
            A = PM @ (self.Cm @ INV_MU @ self.Ce) @ PM - (self.omega ** 2) * EPS
            b = -1j * self.omega * self.j - self.Cm @ (self.m / self.mu)
            
            # Apply reordering
            A = A[self.r, :][:, self.r]
            b = b[self.r]
            
        elif self.ft == FT.H:
            # H-field equation
            INV_EPS = create_spdiag(1.0 / self.eps)
            MU = create_spdiag(self.mu)
            
            A = (self.Ce @ INV_EPS @ self.Cm) - (self.omega ** 2) * MU
            b = -1j * self.omega * self.m + self.Ce @ (self.j / self.eps)
            
            # Apply reordering
            A = A[self.r, :][:, self.r]
            b = b[self.r]
            
        else:  # FT.ELEMS - combined E and H
            EPS = create_spdiag(self.eps)
            MU = create_spdiag(self.mu)
            
            # Block matrix
            A_top = sp.hstack([-1j * self.omega * EPS, self.Cm], format='csr')
            A_bottom = sp.hstack([self.Ce, 1j * self.omega * MU], format='csr')
            A = sp.vstack([A_top, A_bottom], format='csr')
            
            b = np.concatenate([self.j, -self.m])
            
            # Apply reordering
            N = len(self.r)
            rr = np.concatenate([self.r, self.r + N])
            A = A[rr, :][:, rr]
            b = b[rr]
        
        return A, b
    
    def matrixfree_op(self) -> Tuple[Callable[[np.ndarray, str], np.ndarray], np.ndarray, Callable[[np.ndarray], np.ndarray]]:
        """
        Get matrix-free operators for iterative solving.

        Returns matrix-free operators that can be used with iterative solvers
        like scipy.sparse.linalg.bicg or scipy.sparse.linalg.gmres.

        Returns
        -------
        Op : callable
            Matrix-free operator function with signature Op(x, transp_flag).
            transp_flag can be 'transp' or 'notransp'.
        b : ndarray
            Right-hand side vector with reordered indices.
        GfromF : callable
            Function to compute secondary field from primary field.
            GfromF(f) returns the other field (H from E or E from H).

        Examples
        --------
        >>> Op, b, GfromF = eq.matrixfree_op()
        >>> from scipy.sparse.linalg import bicg
        >>> f, info = bicg(Op, b)
        >>> g = GfromF(f)
        """
        if self.ft == FT.E:
            # E-field equation
            def h_A(e: np.ndarray) -> np.ndarray:
                return (self.pm * (self.Cm @ ((self.Ce @ (self.pm * e)) / self.mu)) -
                        (self.omega ** 2) * (self.eps * e))
            
            def h_Atr(e: np.ndarray) -> np.ndarray:
                return (self.pm * (self.Ce.conj().T @ ((self.Cm.conj().T @ (self.pm * e)) / self.mu)) -
                        (self.omega ** 2) * (self.eps * e))
            
            b = -1j * self.omega * self.j - self.Cm @ (self.m / self.mu)
            
            def h_GfromF_temp(e: np.ndarray) -> np.ndarray:
                return (self.Ce @ e + self.m) / (-1j * self.omega * self.mu)
            
        else:  # FT.H
            # H-field equation
            def h_A(h: np.ndarray) -> np.ndarray:
                return (self.Ce @ ((self.Cm @ h) / self.eps) -
                        (self.omega ** 2) * (self.mu * h))
            
            def h_Atr(h: np.ndarray) -> np.ndarray:
                return (self.Cm.conj().T @ ((self.Ce.conj().T @ h) / self.eps) -
                        (self.omega ** 2) * (self.mu * h))
            
            b = -1j * self.omega * self.m + self.Ce @ (self.j / self.eps)
            
            def h_GfromF_temp(h: np.ndarray) -> np.ndarray:
                return (self.Cm @ h - self.j) / (1j * self.omega * self.eps)
        
        # Create operator function with reordering
        def Op(x: np.ndarray, transp_flag: str) -> np.ndarray:
            if transp_flag == 'transp':
                y = h_Atr(x[self.s])
            elif transp_flag == 'notransp':
                y = h_A(x[self.s])
            else:
                raise ValueError("transp_flag must be 'transp' or 'notransp'")
            
            y = y[self.r]
            return y
        
        # Create GfromF function with reordering
        def GfromF(x: np.ndarray) -> np.ndarray:
            y = h_GfromF_temp(x[self.s])
            y = y[self.r]
            return y
        
        return Op, b, GfromF