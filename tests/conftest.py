"""Shared pytest fixtures for the MaxwellFDFD test suite.

These fixtures build small, deterministic grids and material setups so that
Tier-1 unit tests and the analytical dipole validation (Test A) run fast
(sub-second) while still exercising the real numerical operators.
"""
import sys
import os

import numpy as np
import pytest

# Make sure the package (which lives in maxwellfdfd-python/) is importable
# when pytest is run from the project root without an editable install.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.dirname(_HERE)
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from maxwellfdfd.base import Axis, BC, GT, PhysUnit
from maxwellfdfd.grid import Grid3d


def _uniform_lprim(n, L):
    """Return ``{Axis: ndarray}`` of ``n`` uniformly spaced primary nodes on [0, L]."""
    pts = np.linspace(0.0, L, n)
    return {w: pts.copy() for w in Axis.elems()}


@pytest.fixture
def phys_unit():
    """A dimensionless :class:`PhysUnit` (length scale = 1)."""
    return PhysUnit(1.0)


@pytest.fixture
def small_grid(phys_unit):
    """A small 3-D uniform grid, N = (8, 8, 4), periodic BC, no PML.

    Small enough that every diff operator builds and applies in milliseconds.
    """
    lprim = {
        Axis.X: np.linspace(0.0, 8.0, 8),
        Axis.Y: np.linspace(0.0, 8.0, 8),
        Axis.Z: np.linspace(0.0, 4.0, 4),
    }
    return Grid3d(phys_unit, lprim, Npml=0, bc=BC.P)


@pytest.fixture
def vacuum_eps_mu(small_grid):
    """Vacuum ``eps``/``mu`` cells (all ones) keyed by :class:`Axis`."""
    N = tuple(int(n) for n in small_grid.N)
    ones = np.ones(N, dtype=complex)
    eps = {w: ones.copy() for w in Axis.elems()}
    mu = {w: ones.copy() for w in Axis.elems()}
    return eps, mu
