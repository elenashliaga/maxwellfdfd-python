"""Tier 0 smoke test: core objects construct without raising.

Constructs the fundamental building blocks (enums, units, grids, materials,
sources) to catch constructor/argument-validation regressions. No numerical
result is checked here — only that construction succeeds.
"""
import numpy as np

from maxwellfdfd.base import Axis, BC, GT, PhysUnit, Oscillation, PhysQ
from maxwellfdfd.grid import Grid3d
from maxwellfdfd.shape import Box
from maxwellfdfd.material import Material
from maxwellfdfd.source import PointSrc


def _lprim(n, L):
    pts = np.linspace(0.0, L, n)
    return {w: pts.copy() for w in Axis.elems()}


def test_physunit_construct():
    u = PhysUnit(1e-9)
    assert u is not None


def test_oscillation_construct():
    u = PhysUnit(1e-9)
    osc = Oscillation(20.0, u)  # wavelength = 20 (in length units)
    assert osc is not None


def test_grid3d_construct_periodic():
    u = PhysUnit(1.0)
    # _lprim(8, 8.0) creates 8 points → 7 cells (N = points - 1)
    g = Grid3d(u, _lprim(8, 8.0), Npml=0, bc=BC.P)
    assert tuple(int(n) for n in g.N) == (7, 7, 7)


def test_grid3d_construct_pec():
    u = PhysUnit(1.0)
    # _lprim(6, 6.0) creates 6 points → 5 cells
    g = Grid3d(u, _lprim(6, 6.0), Npml=0, bc=[BC.E, BC.E, BC.E])
    assert g.Ntot == 5 * 5 * 5


def test_box_construct():
    b = Box(np.array([[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]]))
    assert b is not None


def test_material_vacuum():
    m = Material('vacuum', 'blue', 1.0)
    assert m.is_iso
    assert m.has_iso_eps and m.has_iso_mu


def test_pointsrc_construct():
    src = PointSrc(Axis.Z, np.array([0.0, 0.0, 0.5]))
    assert src.polarization == Axis.Z
