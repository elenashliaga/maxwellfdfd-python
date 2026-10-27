"""Tier 1 unit tests: Grid1d.

Checks the primary/dual grid vertex counts, total length, and the contains()
predicate on a simple uniform 1-D grid.
"""
import numpy as np

from maxwellfdfd.base import Axis, BC, GT, PhysUnit
from maxwellfdfd.grid import Grid1d


def _make_grid(n_pml=0, bc=BC.P):
    unit = PhysUnit(1.0)
    lprim = np.array([0.0, 0.5, 1.0, 1.5, 2.0])  # 5 nodes -> N = 4 cells
    Npml = np.array([n_pml, n_pml])
    return Grid1d(Axis.X, unit, lprim, Npml, bc)


def test_grid1d_N_and_L():
    g = _make_grid()
    assert g.N == 4
    assert np.isclose(g.L, 2.0)


def test_grid1d_prim_dual_vertex_counts():
    g = _make_grid()
    # Cell-center locations `l`: N points on BOTH primary and dual grids.
    assert len(g.l[GT.PRIM]) == g.N
    assert len(g.l[GT.DUAL]) == g.N
    # `lg` additionally includes the ghost vertex: N+1 points each.
    assert len(g.lg[GT.PRIM]) == g.N + 1
    assert len(g.lg[GT.DUAL]) == g.N + 1


def test_grid1d_prim_spans_domain():
    g = _make_grid()
    lp = g.lg[GT.PRIM]
    assert np.isclose(lp[0], 0.0)
    assert np.isclose(lp[-1], 2.0)


def test_grid1d_dual_midpoints():
    # With a uniform grid the dual cell centers sit at the prim midpoints;
    # `lg` additionally carries one ghost point beyond the negative end
    # (periodic BC): [-0.25, 0.25, ..., 1.75].
    g = _make_grid()
    ld = g.lg[GT.DUAL]
    expected = np.array([-0.25, 0.25, 0.75, 1.25, 1.75])
    assert np.allclose(ld, expected)
    assert np.allclose(g.l[GT.DUAL], expected[1:])


def test_grid1d_contains_inside():
    g = _make_grid()
    inside = g.contains(1.0)
    # contains() may return a bool or a (truth, distance) tuple.
    truth = inside[0] if isinstance(inside, tuple) else inside
    assert bool(truth)


def test_grid1d_contains_outside():
    g = _make_grid()
    inside = g.contains(10.0)
    truth = inside[0] if isinstance(inside, tuple) else inside
    assert not bool(truth)
