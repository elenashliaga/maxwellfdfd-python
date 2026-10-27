"""
Iterative solver for FDFD equations.

This module provides an iterative solver for the finite-difference frequency-domain
equations using Krylov subspace methods.
"""

from typing import Dict, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from maxwellfdfd.base import Axis, FT
from maxwellfdfd.grid import Grid3d
from maxwellfdfd.diff.matrix_equation import MatrixEquation
from maxwellfdfd.diff.eh_from_eh import eh_from_eh


def solve_eq_iterative(
    maxit: int,
    tol: float,
    F0type: str,
    eqtype,
    pml,
    omega: complex,
    eps_cell: Dict[Axis, np.ndarray],
    mu_cell: Dict[Axis, np.ndarray],
    s_factor_cell: Dict[tuple, np.ndarray],
    J_cell: Dict[Axis, np.ndarray],
    M_cell: Dict[Axis, np.ndarray],
    grid3d: Grid3d,
    precond: str = 'none',
    ilu_drop_tol: float = 1e-3,
    ilu_fill_factor: float = 5.0,
    algorithm: str = 'bicgstab',
    gmres_restart: int = 100,
) -> Tuple[Dict[Axis, np.ndarray], Dict[Axis, np.ndarray], float, int, np.ndarray]:
    """
    Solve FDFD equations using iterative solver.

    Solves the matrix equation A*f = b using the biconjugate gradient
    stabilized (BiCGSTAB) method. This is suitable for large problems
    where direct factorization would be too memory-intensive.

    Parameters
    ----------
    maxit : int
        Maximum number of iterations.
    tol : float
        Tolerance for convergence.
    F0type : str
        Initial guess type. Either 'zero' for zero initial guess or
        'rand' for random initial guess.
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

    Returns
    -------
    E : dict
        Dictionary with Axis keys (Axis.X, Axis.Y, Axis.Z) containing
        the E-field components as 3D arrays of shape N.
    H : dict
        Dictionary with Axis keys (Axis.X, Axis.Y, Axis.Z) containing
        the H-field components as 3D arrays of shape N.
    relres : float
        Relative residual of the solution.
    iter : int
        Number of iterations performed.
    resvec : ndarray
        Residual history array of length (iter + 1).

    Raises
    ------
    TypeError
        If grid3d is not Grid3d.
    ValueError
        If F0type is not 'zero' or 'rand', or if solver fails to converge.

    Examples
    --------
    >>> import numpy as np
    >>> from maxwellfdfd.base import Axis, BC, GT, FT, PML, EquationType, PhysUnit
    >>> from maxwellfdfd.grid import Grid3d
    >>> from maxwellfdfd.diff import solve_eq_iterative
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
    >>> E, H, relres, iter, resvec = solve_eq_iterative(100, 1e-6, 'zero', eqtype, PML.SC, omega, eps_cell, mu_cell, s_factor_cell, J_cell, M_cell, grid3d)
    >>> E[Axis.X].shape
    (2, 2, 2)

    Notes
    -----
    The iterative solver uses scipy.sparse.linalg.bicgstab which implements
    the BiCGSTAB algorithm. This is efficient for large, sparse, non-symmetric
    systems.

    For small to medium-sized problems, consider using the direct solver
    solve_eq_direct instead, which may be faster and more robust.

    The solver handles the following equation types:
    - FT.E: Solve for E-field, compute H-field from E
    - FT.H: Solve for H-field, compute E-field from H
    - FT.ELEMS: Solve for combined E and H fields

    The initial guess can be either zero or random. Zero is usually preferred
    for well-conditioned problems, while random may help with convergence
    for ill-conditioned problems.

    The residual history can be used to monitor convergence and diagnose
    convergence issues.
    """
    # Validate inputs
    if not isinstance(grid3d, Grid3d):
        raise TypeError('"grid3d" should be instance of Grid3d')
    
    if F0type not in ['zero', 'rand']:
        raise ValueError('"F0type" should be either "zero" or "rand"')
    
    # Create matrix equation
    eq = MatrixEquation(eqtype, pml, omega, eps_cell, mu_cell,
                       s_factor_cell, J_cell, M_cell, grid3d)
    
    # Get matrix operation
    A, b = eq.matrix_op()
    
    # Get matrix-free operator for computing secondary field
    _, _, GfromF = eq.matrixfree_op()
    
    # Set initial guess
    if F0type == 'zero':
        x0 = np.zeros_like(b)
    else:  # F0type == 'rand'
        x0 = np.random.randn(len(b)) + 1j * np.random.randn(len(b))
    
    # Solve using BiCGSTAB
    # scipy renamed bicgstab's `tol` to `rtol` in 1.12; support both.
    import inspect
    bicg_kwargs = inspect.signature(spla.bicgstab).parameters
    tol_key = 'rtol' if 'rtol' in bicg_kwargs else 'tol'

    # Row equilibration: FDFD matrices in normalized units have tiny entries
    # (||b|| ~ 1e-8), which cripples iterative solvers' stopping criteria and
    # amplifies roundoff. Solve (D^-1 A) x = D^-1 b with D = diag(row norms).
    row_norms = np.sqrt(np.asarray(
        A.tocsr().multiply(A.tocsr().conj()).sum(axis=1)).real).ravel()
    row_norms[row_norms == 0] = 1.0
    Dinv = sp.diags(1.0 / row_norms)
    A_eq = (Dinv @ A).tocsr()
    b_eq = (Dinv @ b)
    print(f'\trow equilibration: median row norm {np.median(row_norms):.3e} '
          f'-> 1', flush=True)

    # Optional ILU preconditioner (large ill-conditioned FDFD systems where
    # plain BiCGSTAB stagnates or breaks down).
    Minv = None
    if precond == 'ilu':
        import time as _time
        t0 = _time.time()
        print('\tbuilding ILU preconditioner '
              f'(drop_tol={ilu_drop_tol:g}, fill_factor={ilu_fill_factor:g})...',
              flush=True)
        try:
            shift_vec = 1e-10 * np.ones(A_eq.shape[0])
            A_p = (A_eq.tocsc() + sp.diags(shift_vec)).tocsc()
            ilu = None
            for permc in ('MMD_AT_PLUS_A', 'COLAMD'):
                for dp in (0.001, 0.01):
                    try:
                        ilu = spla.spilu(A_p, drop_tol=ilu_drop_tol,
                                         fill_factor=ilu_fill_factor,
                                         drop_rule='basic,area',
                                         permc_spec=permc,
                                         diag_pivot_thresh=dp)
                        print(f'\tILU({permc}, pivot={dp}) succeeded',
                              flush=True)
                        break
                    except Exception as e:
                        print(f'\tILU({permc}, pivot={dp}) failed: {e}',
                              flush=True)
                if ilu is not None:
                    break
            if ilu is None:
                raise RuntimeError('all ILU variants failed')
            nnz_ratio = ilu.L.nnz + ilu.U.nnz
            print(f'\tILU done in {_time.time() - t0:.1f} s, '
                  f'nnz(L+U) = {nnz_ratio}', flush=True)
            Minv = spla.LinearOperator(A_eq.shape, ilu.solve,
                                       dtype=A_eq.dtype)
        except Exception as e:
            print(f'\tILU build failed ({e}); falling back to no '
                  'preconditioner', flush=True)
            ilu = None

    _iter_count = [0]
    _bnorm = np.linalg.norm(b_eq)

    def _cb(xk):
        _iter_count[0] += 1
        if _iter_count[0] % 100 == 0:
            # Report the true relative residual on the EQUILIBRATED system.
            try:
                rr = np.linalg.norm(A_eq @ xk - b_eq) / max(_bnorm, 1e-300)
                print(f'\titer {_iter_count[0]}, relres = {rr:.3e}',
                      flush=True)
            except Exception:
                print(f'\titer {_iter_count[0]}', flush=True)

    try:
        if algorithm == 'gmres':
            # GMRES with restarts: more robust than BiCGSTAB on
            # ill-conditioned complex FDFD systems (no breakdown).
            f, info = spla.gmres(A_eq, b_eq, x0=x0, M=Minv,
                                 restart=gmres_restart,
                                 callback=_cb, callback_type='legacy',
                                 **{tol_key: tol}, maxiter=maxit)
        else:
            f, info = spla.bicgstab(A_eq, b_eq, x0=x0, M=Minv, callback=_cb,
                                    **{tol_key: tol}, maxiter=maxit)
        iter = _iter_count[0]

        if info != 0:
            if info > 0:
                raise ValueError(f"Iterative solver did not converge within "
                                 f"{maxiter} iterations")
            else:
                raise ValueError(f"Iterative solver failed with error code "
                                 f"{info}")

        # Compute relative residual on the equilibrated system (what the
        # solver actually minimized).
        residual = A_eq @ f - b_eq
        relres = np.linalg.norm(residual) / max(_bnorm, 1e-300)
        print(f'\t{algorithm} finished in {iter} iterations, '
              f'relres = {relres:.2e}', flush=True)
        resvec = np.array([1.0, relres])  # Simplified residual history

    except Exception as e:
        raise ValueError(f"Iterative solver failed: {e}")
    
    # Compute secondary field
    g = GfromF(f)
    
    # Determine which field is primary and which is secondary
    if eqtype.f == FT.E:
        e = f
        h = g
    elif eqtype.f == FT.H:
        e = g
        h = f
    else:  # FT.ELEMS
        # Split combined solution
        n = len(f) // 2
        e = f[:n]
        h = f[n:]
    
    # Convert to structured arrays
    E, H = eh_from_eh(e, h, grid3d)
    
    return E, H, relres, iter, resvec