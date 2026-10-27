"""
Solve the waveguide-mode eigenproblem built by `wgmode_matrix`.

Python port of MATLAB `maxwellfdfd/modesolver/solve_mode.m` (strategy
'beta' -> `solve_mode`) and `solve_mode_real.m` (strategy 'modeorder' ->
`solve_mode_order`).

Eigenvalue convention: ``A x = lambda x`` with ``lambda = gamma^2 = -beta^2``
(``gamma = 1j*beta``), so ``beta = sqrt(-lambda)`` and ``Re(beta) >= 0``.
"""

from typing import Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def solve_mode(A, beta_guess: complex) -> Tuple[np.ndarray, complex]:
    """Find the eigenpair of `A` closest to the guess propagation constant.

    Shift-invert around ``sigma = -beta_guess^2 + eps`` (the tiny ``eps``
    guards against the shift coinciding with an exact eigenvalue, which
    would make the shifted matrix singular -- same trick as MATLAB).

    Parameters
    ----------
    A : (2N, 2N) sparse matrix
        Eigenoperator from `wgmode_matrix`.
    beta_guess : complex
        Estimate of the propagation constant (from a neff guess:
        ``beta_guess = 2*pi*neff/lambda0``).

    Returns
    -------
    v : (2N,) complex ndarray
        Eigenvector [Hh(:); Hv(:)], unit norm.
    beta : complex
        Propagation constant with Re(beta) >= 0.
    """
    sigma = -(beta_guess ** 2) + np.finfo(float).eps
    vals, vecs = spla.eigs(A, k=1, sigma=sigma, which='LM')
    lam = vals[0]
    v = vecs[:, 0]
    v = v / np.linalg.norm(v)
    beta = np.sqrt(-lam)  # principal branch: Re(beta) >= 0
    if np.real(beta) < 0:
        beta = -beta
    return v, beta


def _power_iteration_largest_mag(A, tol=1e-10, maxiter=10000,
                                 rng=None) -> complex:
    """Largest-magnitude eigenvalue estimate by power iteration.

    Port of MATLAB `eig_largest_mag` (Rayleigh-quotient power iteration,
    complex-safe: the start vector is complex random).
    """
    m = A.shape[0]
    rng = rng or np.random.default_rng(0)
    vmax = rng.standard_normal(m) + 1j * rng.standard_normal(m)
    vmax = vmax / np.linalg.norm(vmax)
    lmax_prev = np.inf
    vmax_next = A @ vmax
    lmax = np.vdot(vmax, vmax_next)
    vmax_next = vmax_next / np.linalg.norm(vmax_next)
    i = 0
    while abs((lmax - lmax_prev) / lmax) > tol and i < maxiter:
        i += 1
        vmax = vmax_next
        lmax_prev = lmax
        vmax_next = A @ vmax
        lmax = np.vdot(vmax, vmax_next)
        vmax_next = vmax_next / np.linalg.norm(vmax_next)
    if abs((lmax - lmax_prev) / lmax) > tol:
        import warnings
        warnings.warn(
            'power iteration did not converge: '
            f'abs((lmax - lmax_prev)/lmax) = {abs((lmax - lmax_prev) / lmax)}',
            RuntimeWarning,
        )
    return lmax


def solve_mode_order(A, order: int, tol_power: float = 1e-10
                     ) -> Tuple[np.ndarray, complex]:
    """Find the `order`-th guided mode of a (nearly) real eigenoperator.

    Port of MATLAB `solve_mode_real`: assumes the eigenvalues are all real
    or close to real (small loss). The physical eigenvalues gamma^2 = -beta^2
    are negative; the largest-magnitude eigenvalue of A (Nyquist) is positive,
    so the spectrum is shifted such that the most negative eigenvalues of A
    become the smallest-magnitude eigenvalues of the shifted matrix, which
    are then found with shift-invert at 0.

    Parameters
    ----------
    A : (2N, 2N) sparse matrix
        REAL eigenoperator (build it with real omega / eps / mu / s-factors).
    order : int
        Mode order (1 = fundamental, largest beta).

    Returns
    -------
    v : (2N,) complex ndarray
        Eigenvector [Hh(:); Hv(:)] of the requested mode, unit norm.
    beta : complex
        Propagation constant, Re(beta) >= 0.
    """
    if order < 1:
        raise ValueError('"order" should be a positive integer.')

    A = A.tocsr()
    identity = sp.identity(A.shape[0], format='csr', dtype=A.dtype)

    lmax = np.real(_power_iteration_largest_mag(A, tol_power))
    lmax = abs(lmax)

    # Most negative eigenvalue (all eigenvalues of A - lmax*I are negative).
    B1 = (A - lmax * identity).tocsr()
    lneg = -abs(np.real(_power_iteration_largest_mag(B1, tol_power)))

    # Shift so that all eigenvalues become positive; the most negative
    # eigenvalues of A (largest beta) end up closest to zero.
    lshift = lmax + (1.0 + np.sqrt(tol_power)) * lneg

    B = (A - lshift * identity).tocsr()
    vals, vecs = spla.eigs(B, k=order + 2, sigma=0.0, which='LM')

    betas = np.sqrt(-(vals + lshift))
    ind = np.argsort(np.real(betas))[::-1]  # largest Re(beta) first
    k = ind[order - 1]
    v = vecs[:, k]
    v = v / np.linalg.norm(v)
    beta = betas[k]
    if np.real(beta) < 0:
        beta = -beta
    return v, beta
