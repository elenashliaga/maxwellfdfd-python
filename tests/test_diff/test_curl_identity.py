"""Tier 1 unit tests: differential operators.

The central physics sanity check is the **vector-calculus identity**
``div(curl(F)) = 0`` for any smooth vector field F. Discretely this means
the composed operator ``Dive @ Ce`` (and ``Divm @ Cm``) must be numerically
zero. If this fails, the Yee-grid staggering or the derivative matrices are
wrong and *no* FDFD result can be trusted.

Also tests that ``create_Dw`` produces a correct forward-difference: it
annihilates constants and reproduces the slope of a linear ramp.
"""
import numpy as np
import scipy.sparse as sp

from maxwellfdfd.base import Axis, GT
from maxwellfdfd.diff import create_curls, create_divs, create_Dw


# ----------------------------------------------------- div(curl(F)) = 0
def test_div_curl_e_is_zero(small_grid):
    """Dive @ Cm must be (numerically) the zero operator.

    The discrete div-curl identity pairs a divergence with the curl living
    on the SAME grid: Divm @ Ce = 0 and Dive @ Cm = 0 (see the docstring of
    `create_divs`). The cross pairings Dive @ Ce / Divm @ Cm are NOT zero
    discretely -- mixed-axis forward/backward stencils do not cancel.
    """
    _, Cm = create_curls(GT.PRIM, None, small_grid)
    Dive, _ = create_divs(GT.PRIM, None, small_grid)
    product = (Dive @ Cm).toarray()
    # Relative to the operator norm, the composition should vanish.
    scale = max(1.0, np.abs(Cm.data).max() if Cm.nnz else 1.0,
                np.abs(Dive.data).max() if Dive.nnz else 1.0)
    assert np.max(np.abs(product)) < 1e-9 * scale, (
        f"||Dive @ Cm||_inf = {np.max(np.abs(product)):.3e}, "
        f"expected ~0 (div(curl)=0 identity violated)"
    )


def test_div_curl_m_is_zero(small_grid):
    """Divm @ Ce must be (numerically) the zero operator."""
    Ce, _ = create_curls(GT.PRIM, None, small_grid)
    _, Divm = create_divs(GT.PRIM, None, small_grid)
    product = (Divm @ Ce).toarray()
    scale = max(1.0, np.abs(Ce.data).max() if Ce.nnz else 1.0,
                np.abs(Divm.data).max() if Divm.nnz else 1.0)
    assert np.max(np.abs(product)) < 1e-9 * scale, (
        f"||Divm @ Ce||_inf = {np.max(np.abs(product)):.3e}, "
        f"expected ~0 (div(curl)=0 identity violated)"
    )


def test_div_curl_on_random_field(small_grid):
    """For a random field vector F, ||Divm(Ce(F))|| / ||F|| must be ~0."""
    rng = np.random.default_rng(42)
    Ce, _ = create_curls(GT.PRIM, None, small_grid)
    _, Divm = create_divs(GT.PRIM, None, small_grid)
    M = int(np.prod(small_grid.N))
    F = rng.standard_normal(3 * M) + 1j * rng.standard_normal(3 * M)
    residual = Divm @ (Ce @ F)
    rel = np.linalg.norm(residual) / np.linalg.norm(F)
    assert rel < 1e-9, f"||Div(Curl(F))||/||F|| = {rel:.3e}, expected < 1e-9"


# --------------------------------------------------------- create_Dw
def test_create_Dw_annihilates_constant():
    """Forward difference of a constant field is zero (interior)."""
    N = np.array([6, 6, 6])
    Dw = create_Dw(Axis.X, N, 1.0, 1.0)
    M = int(np.prod(N))
    Fx = np.ones(M)
    dF = Dw @ Fx
    # The forward difference of a constant vanishes everywhere.
    assert np.max(np.abs(dF)) < 1e-12


def test_create_Dw_linear_ramp():
    """Forward difference of x along the x-axis equals the cell size (1)."""
    N = np.array([8, 4, 4])
    Dw = create_Dw(Axis.X, N, 1.0, 1.0)
    # Build a field whose x-component is the x-index (ramp of slope 1).
    nx, ny, nz = (int(n) for n in N)
    X, Y, Z = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz),
                          indexing='ij')
    Fx = (X.astype(float)).ravel()
    dF = Dw @ Fx
    # The layout is component-major C-order, so the x index varies SLOWEST:
    # reshape back and check the forward difference block-wise. Interior
    # rows equal 1; the last x-plane wraps periodically (0 - 7 = -7).
    dF3 = dF.reshape(nx, ny, nz)
    assert np.allclose(dF3[:-1], 1.0)
    assert np.allclose(dF3[-1], -7.0)


def test_create_Dw_shape():
    N = np.array([5, 5, 5])
    Dw = create_Dw(Axis.Y, N, 1.0, 1.0)
    M = int(np.prod(N))
    assert Dw.shape == (M, M)
    assert sp.issparse(Dw)
