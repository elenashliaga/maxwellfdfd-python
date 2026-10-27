#!/usr/bin/env python3
"""
Integration smoke test: maxwell_run + ModalSrc on a dielectric slab.

A Si-like slab (eps = 11.82, full width a = 420 nm in y) guides a mode
propagating along x. The mode is launched with ModalSrc (guess clue) and the
driven solution is checked for:

  1. the neff printed by ModalSrc.prep matching a direct wgmode solve;
  2. power-flux balance left/right of the source (lossless, symmetric launch);
  3. the field being localized in the slab.

Run (from maxwellfdfd-python/):
    python3 tests/test_modalsrc_smoke.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from maxwellfdfd.base import Axis, BC, GT, PML
from maxwellfdfd.shape import Box
from maxwellfdfd.source import ModalSrc
from maxwellfdfd.io import maxwell_run

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'examples'))
from vis_utils import powerflux_patch

A = 420.0        # lattice constant / slab full width (nm)
WVLEN = 2500.0   # vacuum wavelength (nm)
EPS_SL = 11.82   # slab permittivity
AD = 25          # grid divider: dl = a/25 = 16.8 nm
MX = 20.5        # x half-extent in units of a
MY = 7.5         # y half-extent in units of a
SRC_LOC = 2 * A  # modal source plane (must be OUTSIDE the x-PML)


def direct_mode_neff():
    """Independent neff from the cross-section eigenoperator (y-z grid)."""
    from maxwellfdfd.base import PhysUnit
    from maxwellfdfd.grid import Grid3d
    from maxwellfdfd.io.grid_utils import generate_s_factor
    from maxwellfdfd.modesolver import wgmode_matrix, solve_mode

    dl = A / AD
    ly = np.arange(-MY * A, MY * A + dl / 2, dl)
    # Physical y-extent without PML is [-my*a, my*a]; trim to keep the span.
    n_int = int(round((2 * MY * A) / dl))
    ly = -MY * A + dl * np.arange(n_int + 1)
    lprim = {
        Axis.X: np.array([0.0, 2 * dl]),
        Axis.Y: ly,
        Axis.Z: np.array([0.0, dl]),
    }
    Npml = np.zeros((3, 2), dtype=int)
    Npml[1, :] = int(round(2 * A / dl))  # PML in y only
    grid3d = Grid3d(PhysUnit(1e-9), lprim, Npml, [BC.P, BC.P, BC.P])

    Nx, Ny, Nz = (int(n) for n in grid3d.N)
    y_l = grid3d.l[(Axis.Y, GT.PRIM)]
    y_c = y_l + dl / 2
    eps_y = np.where(np.abs(y_c) < A / 2, EPS_SL, 1.0).astype(complex)
    eps3d = np.broadcast_to(eps_y[None, :, None], (Nx, Ny, Nz)).copy()
    eps_cell = {w: eps3d.copy() for w in Axis.elems()}
    mu_cell = {w: np.ones_like(eps3d) for w in Axis.elems()}

    omega = 2 * np.pi / WVLEN
    s_factor_cell = generate_s_factor(omega, grid3d, 4, np.exp(-16))

    Aop, *_ = wgmode_matrix(
        GT.PRIM, PML.SC, omega, eps_cell, mu_cell, s_factor_cell,
        grid3d, Axis.X, dl,
    )
    # First TE-like guess (Ez-polarized here): start from a middle neff.
    for guess in (2.6, 2.4, 2.8, 2.2, 3.0):
        v, beta = solve_mode(Aop, 2 * np.pi * guess / WVLEN)
        neff = beta / (2 * np.pi / WVLEN)
        if 1.0 < np.real(neff) < np.sqrt(EPS_SL):
            return np.real(neff)
    raise RuntimeError('direct mode solve failed')


def main():
    n_direct = direct_mode_neff()
    print(f'direct cross-section solve: neff = {n_direct:.8f}')

    gray = [0.5, 0.5, 0.5]
    E, H, obj_array, src_array, extra = maxwell_run(
        'OSC', 1e-9, WVLEN,
        'DOM', ['vacuum', 'none', 1.0],
            np.array([[-MX * A, MX * A], [-MY * A, MY * A], [0, 1]]),
            np.array([A / AD, A / AD, 1]), BC.P, np.array([10 * A, 2 * A, 0]),
        'OBJ',
            ['Si', gray, EPS_SL],
            Box(np.array([[-MX * A, MX * A], [-A / 2, A / 2], [0, 1]])),
        'SRCJ', ModalSrc(Axis.X, SRC_LOC,
                         {'clue': 'guess', 'neff': complex(n_direct)}),
    )

    modal = src_array[0]
    n_modal = modal.neff
    grid3d = extra['grid3d']

    # Checks
    failures = []
    # NOTE: the direct solve uses a uniform grid, maxwell_run builds a
    # nonuniform (PML-graded) grid, so a small discrepancy is expected.
    err = abs(np.real(n_modal) - n_direct) / n_direct
    print(f'ModalSrc neff = {n_modal:.8f}   (rel.diff vs direct = {err:.2e})')
    if err > 2e-3:
        failures.append(f'ModalSrc neff mismatch: {err:.2e}')

    pr = powerflux_patch(E, H, Axis.X, SRC_LOC + 3 * A, grid3d)
    pl = -powerflux_patch(E, H, Axis.X, SRC_LOC - 3 * A, grid3d)
    print(f'power right = {pr:.6e}, power left = {pl:.6e}')
    if pr <= 0 or pl <= 0:
        failures.append('negative/zero power flux')
    else:
        bal = abs(pl - pr) / max(pr, pl)
        print(f'balance error = {bal * 100:.2f}%')
        if bal > 0.05:
            failures.append(f'power imbalance {bal * 100:.1f}% > 5%')

    # Field localization: |Ez| max inside slab vs outside at a far x-plane.
    Ez = E[Axis.Z]
    arr, l = Ez.data_original()
    y = l[Axis.Y]
    iy_slab = np.abs(y) <= A / 2
    iy_out = np.abs(y) >= 2 * A
    e_in = np.max(np.abs(arr[:, iy_slab, 0]), axis=1)
    e_out = np.max(np.abs(arr[:, iy_out, 0]), axis=1)
    ratio = np.max(e_in) / max(np.max(e_out), 1e-300)
    print(f'slab/outside |Ez| ratio at worst x = {ratio:.1f}')
    if ratio < 5:
        failures.append(f'field not localized in slab (ratio={ratio:.1f})')

    if failures:
        print('\nFAILED:')
        for f in failures:
            print(f'  - {f}')
        sys.exit(1)
    print('\nModalSrc integration smoke test PASSED.')


if __name__ == '__main__':
    main()
