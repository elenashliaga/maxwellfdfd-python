#!/usr/bin/env python3
"""
Slab-waveguide validation test for the modesolver port
(maxwellfdfd/modesolver + ModalSrc.prep).

A symmetric dielectric slab (core n1, thickness d, cladding n2 = air),
invariant along x (propagation) and y, is solved with the ported 2D
eigenoperator and compared against the ANALYTICAL slab dispersion:

    TE even(m=0) / odd(m=1):  tan(kd/2)  = +-(gamma/k)
    TM even(m=0) / odd(m=1):  tan(kd/2)  = +-(eps1/eps2)*(gamma/k)

with k = sqrt(k0^2 n1^2 - beta^2), gamma = sqrt(beta^2 - k0^2 n2^2).

The eigenoperator acts on [Hh; Hv] = [Hy; Hz] (normal_axis = X), so the
TM mode (H along y) is dominated by Hh and the TE mode (H in the x-z
plane) by Hv; the shift-invert guess selects between them.

Run (from maxwellfdfd-python/):
    python3 tests/test_slab.py
"""

import sys

import numpy as np
from scipy.optimize import brentq

from maxwellfdfd.base import Axis, BC, GT, PML, PhysUnit
from maxwellfdfd.grid import Grid3d
from maxwellfdfd.io.grid_utils import generate_s_factor
from maxwellfdfd.modesolver import wgmode_matrix, solve_mode

# ----------------------------------------------------------------------------
# Parameters (units: nm)
# ----------------------------------------------------------------------------
LAM = 982.0        # vacuum wavelength, nm
N1, N2 = 3.5, 1.0  # core / cladding refractive indices
D_SLAB = 200.0     # slab thickness, nm
DL = 5.0           # uniform grid step in z, nm
LZ_HALF = 600.0    # z-domain half-width, nm
LPML_Z = 100.0     # PML thickness in z, nm
NPML_Z = int(round(LPML_Z / DL))

K0 = 2.0 * np.pi / LAM


def analytic_neff(pol: str, m: int = 0) -> float:
    """Solve the analytic slab dispersion for the guided-mode neff.

    Scans F(neff) on a fine grid and brents the first sign change that is
    not a tan() pole (|F| small on both sides of the bracket).
    """

    def F(neff):
        beta = K0 * neff
        k = np.sqrt((K0 * N1) ** 2 - beta ** 2)
        gamma = np.sqrt(beta ** 2 - (K0 * N2) ** 2)
        ratio = gamma / k
        if pol == 'TM':
            ratio *= (N1 / N2) ** 2
        lhs = np.tan(k * D_SLAB / 2.0)
        if m == 0:      # even
            return lhs - ratio
        else:           # odd: cot(kd/2) + ratio = 0
            return 1.0 / lhs + ratio

    # Fine scan between the light lines; skip pole neighborhoods.
    ns = np.linspace(N2 + 1e-4, N1 - 1e-4, 200001)
    fs = np.array([F(n) for n in ns])
    POLE = 1e3
    for i in range(len(ns) - 1):
        if np.isfinite(fs[i]) and np.isfinite(fs[i + 1]) \
                and abs(fs[i]) < POLE and abs(fs[i + 1]) < POLE \
                and fs[i] * fs[i + 1] < 0:
            return brentq(F, ns[i], ns[i + 1], xtol=1e-13)
    raise RuntimeError(f'No {pol}{m} root found in [{ns[0]}, {ns[-1]}].')


def build_system():
    """Build grid3d, eps_cell, mu_cell, s_factor_cell for the slab."""
    # x: propagation axis, 2 cells, no PML; y: invariant, 1 cell;
    # z: cross-section with PML at both ends.
    lprim = {
        Axis.X: np.array([0.0, 50.0, 100.0]),
        Axis.Y: np.array([0.0, DL]),
        Axis.Z: np.arange(-LZ_HALF, LZ_HALF + DL / 2, DL),
    }
    Npml = np.zeros((3, 2), dtype=int)
    Npml[2, :] = NPML_Z
    bc = [BC.P, BC.P, BC.P]
    grid3d = Grid3d(PhysUnit(1e-9), lprim, Npml, bc)

    # Cell-centered slab profile along z.
    z_l = grid3d.l[(Axis.Z, GT.PRIM)]  # left vertices, length Nz
    z_c = z_l + DL / 2.0               # cell centers
    in_slab = np.abs(z_c) < D_SLAB / 2.0
    eps_z = np.where(in_slab, N1 ** 2, N2 ** 2).astype(complex)

    Nx, Ny, Nz = (int(n) for n in grid3d.N)
    eps3d = np.broadcast_to(eps_z[None, None, :], (Nx, Ny, Nz)).copy()
    eps_cell = {w: eps3d.copy() for w in Axis.elems()}
    mu_cell = {w: np.ones_like(eps3d) for w in Axis.elems()}

    omega = K0  # in 1/nm units (consistent with lengths in nm)
    s_factor_cell = generate_s_factor(omega, grid3d, 4, np.exp(-16))
    return grid3d, eps_cell, mu_cell, s_factor_cell, omega


def solve_slab_mode(pol: str, m: int, neff_guess: float):
    grid3d, eps_cell, mu_cell, s_factor_cell, omega = build_system()
    intercept = 50.0  # on the primary x-grid line, outside (zero) PML
    A, *_ = wgmode_matrix(
        GT.PRIM, PML.SC, omega, eps_cell, mu_cell, s_factor_cell,
        grid3d, Axis.X, intercept,
    )
    beta_guess = K0 * neff_guess
    v, beta = solve_mode(A, beta_guess)
    neff = beta / K0
    return neff, v, grid3d


def main():
    failures = []

    for pol, m in (('TM', 0), ('TE', 0)):
        n_exact = analytic_neff(pol, m)
        # Perturb the guess by 2% to make the test honest.
        guess = n_exact * (1.02 if pol == 'TM' else 0.98)
        neff, v, grid3d = solve_slab_mode(pol, m, guess)

        N = v.size // 2
        Hh = v[:N].reshape(int(grid3d.N[1]), int(grid3d.N[2]))  # Hy (y,z)
        Hv = v[N:].reshape(int(grid3d.N[1]), int(grid3d.N[2]))  # Hz

        # Polarization check: TM -> |Hy| dominates; TE -> |Hz| dominates.
        frac_h = np.linalg.norm(Hh) / np.linalg.norm(v)
        dom = 'Hy (TM-like)' if frac_h > 0.5 else 'Hz (TE-like)'

        err = abs(np.real(neff) - n_exact)
        im = abs(np.imag(neff))
        print(f'{pol}{m}:')
        print(f'  analytic neff = {n_exact:.10f}')
        print(f'  numeric  neff = {neff:.10f}   (|err| = {err:.2e})')
        print(f'  |Im(neff)|    = {im:.2e}   (PML back-reflection measure)')
        print(f'  polarization  : {dom} (|Hh|/|v| = {frac_h:.3f})')

        tol = 5e-3
        if err > tol:
            failures.append(f'{pol}{m}: neff error {err:.2e} > {tol}')
        if im > 1e-6:
            failures.append(f'{pol}{m}: |Im(neff)| = {im:.2e} > 1e-6')
        want_h = (pol == 'TM')
        if (frac_h > 0.5) != want_h:
            failures.append(f'{pol}{m}: wrong polarization (frac_h={frac_h:.3f})')

    if failures:
        print('\nFAILED:')
        for f in failures:
            print(f'  - {f}')
        sys.exit(1)
    print('\nAll slab-waveguide checks PASSED.')


if __name__ == '__main__':
    main()
