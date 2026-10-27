"""
Construct the waveguide-mode eigenoperator for FDFD.

Python port of MATLAB `maxwellfdfd/modesolver/wgmode_matrix.m`.

The operator ``A`` acts on H fields, not E fields: ``A x = lambda x`` solves
for the H fields of an eigenmode of a waveguide that is invariant along its
axis (``normal_axis``), with ``lambda = gamma^2 = -beta^2`` (``gamma = 1j*beta``).
For the eigenvalue equation see Sec. II of G. Veronis and S. Fan, Journal of
Lightwave Technology, vol. 25, no. 9, pp. 2511--2521 (2007).

Vector layout (component-major): the unknown is ``[Hh.ravel(); Hv.ravel()]``
where ``Hh``/``Hv`` are ``(Nh, Nv)`` arrays on the 2D staggered grid
(``h`` = horizontal, ``v`` = vertical direction of the cross-section) and
``ravel()`` uses C order.

Notes on the port:
- MATLAB reshapes ``(Nh, Nv)`` arrays column-major; this port ravels/rescales
  everything in C order, which is a uniform permutation of the MATLAB layout
  and therefore yields identical eigenvalues.
- MATLAB reorders ``A`` at the end to reduce bandwidth (``A(r, r)``); scipy's
  sparse LU applies its own ordering, so the reordering is omitted here and
  eigenvectors stay in the component-major layout.
"""

from typing import Dict, List, Tuple, Union

import numpy as np
import scipy.sparse as sp

from maxwellfdfd.base import Axis, Dir, GT, PML, Sign, create_spdiag
from maxwellfdfd.grid import Grid2d, Grid3d
from maxwellfdfd.diff.create_Ds import create_Ds
from maxwellfdfd.diff.create_masks import create_masks


def _zero_cols(mat: sp.csr_matrix, cols: np.ndarray) -> sp.csr_matrix:
    """Return a copy of sparse `mat` with the given columns zeroed."""
    if cols.size == 0:
        return mat
    m = mat.tolil()
    m[:, cols] = 0
    return m.tocsr()


def _zero_rows(mat: sp.csr_matrix, rows: np.ndarray) -> sp.csr_matrix:
    """Return a copy of sparse `mat` with the given rows zeroed."""
    if rows.size == 0:
        return mat
    m = mat.tolil()
    m[rows, :] = 0
    return m.tocsr()


def _slice_at_intercept(
    arr3d: np.ndarray,
    normal_axis: Axis,
    ind_n: int,
    h_axis: Axis,
    v_axis: Axis,
) -> np.ndarray:
    """Slice a 3D field array at index `ind_n` along `normal_axis` and
    transpose the remaining two axes into (h, v) order."""
    sl = [slice(None)] * 3
    sl[int(normal_axis)] = ind_n
    plane = arr3d[tuple(sl)]
    rem = [a for a in Axis.elems() if int(a) != int(normal_axis)]
    perm = [rem.index(h_axis), rem.index(v_axis)]
    return np.transpose(plane, perm)


def _real_cell(cell: Dict[Axis, np.ndarray]) -> Dict[Axis, np.ndarray]:
    """Take the real part of every array in an {Axis: ndarray} cell."""
    return {w: np.real(cell[w]) for w in Axis.elems()}


def _real_s_factor(
    s_factor_cell: Dict[Tuple[Axis, GT], np.ndarray],
) -> Dict[Tuple[Axis, GT], np.ndarray]:
    """Take the real part of every s-factor vector."""
    return {key: np.real(val) for key, val in s_factor_cell.items()}


def wgmode_matrix(
    ge: GT,
    pml: PML,
    omega: complex,
    eps_cell: Dict[Axis, np.ndarray],
    mu_cell: Dict[Axis, np.ndarray],
    s_factor_cell: Dict[Tuple[Axis, GT], np.ndarray],
    grid3d: Grid3d,
    normal_axis: Axis,
    intercept: float,
):
    """Build the waveguide-mode eigenoperator on the cross-section grid.

    Parameters
    ----------
    ge : GT
        Grid type of the E field (PRIM: E on primary / H on dual grid).
    pml : PML
        PML type (PML.SC stretched coordinates or PML.U uniaxial).
    omega : complex
        Angular frequency (in normalized units of `grid3d`).
    eps_cell, mu_cell : dict of Axis to ndarray
        Complex permittivity / permeability arrays of shape grid3d.N.
    s_factor_cell : dict of (Axis, GT) to ndarray
        PML s-factor vectors from `generate_s_factor`.
    grid3d : Grid3d
        3D grid of the (possibly 2D-like) simulation domain.
    normal_axis : Axis
        Waveguide axis (the mode propagates along it).
    intercept : float
        Location of the cross-section plane along `normal_axis`. It must
        coincide with a grid line of the primary or dual grid and lie
        OUTSIDE the PML regions (this is where the ModalSrc plane goes).

    Returns
    -------
    A : csr_matrix (2N x 2N, N = Nh*Nv)
        Eigenoperator acting on [Hh(:); Hv(:)]; eigenvalue = gamma^2 = -beta^2.
    En_Htr : csr_matrix (N x 2N)
        Reconstructs En from Htr:  En = En_Htr @ Htr.
    gEh_HvEn : csr_matrix (N x 2N)
        gamma*Eh from [Hv; En]:  Eh = (gEh_HvEn @ [Hv; En]) / (1j*beta).
    gEv_HhEn : csr_matrix (N x 2N)
        gamma*Ev from [Hh; En]:  Ev = (gEv_HhEn @ [Hh; En]) / (1j*beta).
    Hn_Etr : csr_matrix (N x 2N)
        Reconstructs Hn from Etr:  Hn = Hn_Etr @ [Eh; Ev].
    """
    if not isinstance(ge, GT):
        raise TypeError('"ge" should be instance of GT.')
    if not isinstance(pml, PML):
        raise TypeError('"pml" should be instance of PML.')
    if not isinstance(normal_axis, Axis):
        raise TypeError('"normal_axis" should be instance of Axis.')

    # ---- Locate the intercept on the grid --------------------------------
    l_ge = grid3d.l[(normal_axis, ge)]
    hits = np.where(np.isclose(l_ge, intercept))[0]
    gt_hit = ge
    if hits.size == 0:
        l_alt = grid3d.l[(normal_axis, ge.alter())]
        hits = np.where(np.isclose(l_alt, intercept))[0]
        gt_hit = ge.alter()
    if hits.size == 0:
        raise ValueError(
            f'{ge} grid along {normal_axis}-axis does not contain the '
            f'intercept {intercept}.'
        )
    ind_n = int(hits[0])

    # The modal-source plane must be outside the PML (both ends).
    lpml = grid3d.lpml  # (3, 2): [axis, Sign.N / Sign.P]
    if not (intercept > lpml[int(normal_axis), int(Sign.N)]
            and intercept < lpml[int(normal_axis), int(Sign.P)]):
        raise ValueError(
            '"intercept" should be outside the PML in the '
            f'{normal_axis}-direction (got intercept={intercept}, PML '
            f'interfaces at {lpml[int(normal_axis), :]}).'
        )

    # ---- Cross-section grid ------------------------------------------------
    grid2d = Grid2d(grid3d, normal_axis)
    h_axis, v_axis, n_axis = normal_axis.cycle()  # (p, q, r) with r == normal
    Nh = int(grid2d.N[0])
    Nv = int(grid2d.N[1])
    N = Nh * Nv

    # ---- eps and mu at the intercept plane --------------------------------
    eps_h = _slice_at_intercept(eps_cell[h_axis], normal_axis, ind_n, h_axis, v_axis)
    eps_v = _slice_at_intercept(eps_cell[v_axis], normal_axis, ind_n, h_axis, v_axis)
    eps_n = _slice_at_intercept(eps_cell[n_axis], normal_axis, ind_n, h_axis, v_axis)
    mu_h = _slice_at_intercept(mu_cell[h_axis], normal_axis, ind_n, h_axis, v_axis)
    mu_v = _slice_at_intercept(mu_cell[v_axis], normal_axis, ind_n, h_axis, v_axis)
    mu_n = _slice_at_intercept(mu_cell[n_axis], normal_axis, ind_n, h_axis, v_axis)

    # ---- PML ---------------------------------------------------------------
    dl_factor_cell = None
    if pml == PML.SC:
        dl_factor_cell = [
            [s_factor_cell[(h_axis, g)] for g in GT.elems()],
            [s_factor_cell[(v_axis, g)] for g in GT.elems()],
        ]
    elif pml == PML.U:
        smh, smv = np.meshgrid(
            s_factor_cell[(h_axis, ge.alter())],
            s_factor_cell[(v_axis, ge.alter())],
            indexing='ij',
        )
        seh, sev = np.meshgrid(
            s_factor_cell[(h_axis, ge)],
            s_factor_cell[(v_axis, ge)],
            indexing='ij',
        )
        eps_h = eps_h * sev / smh
        eps_v = eps_v * seh / smv
        eps_n = eps_n * seh * sev
        mu_h = mu_h * smv / seh
        mu_v = mu_v * smh / sev
        mu_n = mu_n * smh * smv
    else:
        raise ValueError(f'Unknown PML type: {pml}')

    # ---- Derivative operators ----------------------------------------------
    s = Sign.N if ge == GT.PRIM else Sign.P
    Ds = create_Ds(s, ge, dl_factor_cell, grid2d)
    Da = create_Ds(s.alter(), ge, dl_factor_cell, grid2d)

    # ---- Masks (indices of field samples forced to zero by BCs) ------------
    # create_masks returns (primary, dual) indices over the concatenated
    # [h-block; v-block] component-major layout.
    ind_Mp, ind_Md = create_masks(ge, grid2d)
    if ge == GT.PRIM:
        ind_ME, ind_MH = ind_Mp, ind_Md  # E on primary, H on dual
    else:
        ind_MH, ind_ME = ind_Mp, ind_Md  # H on primary, E on dual

    def split(ind):
        ind = np.asarray(ind, dtype=int)
        return ind[ind < N], ind[ind >= N] - N

    ind_MEh, ind_MEv = split(ind_ME)
    ind_MHh, ind_MHv = split(ind_MH)
    ind_MEn = np.array([], dtype=int)  # n-direction is periodic/1-cell: no mask
    ind_MHn = np.array([], dtype=int)

    # ---- Material diagonals -------------------------------------------------
    eps_vh = np.concatenate([eps_v.ravel(), eps_h.ravel()])
    mu_hv_vec = np.concatenate([mu_h.ravel(), mu_v.ravel()])

    EPS_vh = create_spdiag(eps_vh)
    INV_EPS_n = create_spdiag(1.0 / eps_n.ravel())
    INV_MU_n = create_spdiag(1.0 / mu_n.ravel())
    MU_hv = create_spdiag(mu_hv_vec)
    MU_h = create_spdiag(mu_h.ravel())
    MU_v = create_spdiag(mu_v.ravel())

    # ---- Derivative blocks with masked columns ------------------------------
    # create_Ds on a Grid2d returns matrices keyed by Dir.H / Dir.V, where
    # Dir.H acts along h_axis and Dir.V along v_axis.
    DhHh = _zero_cols(Da[Dir.H], ind_MHh)
    DvHv = _zero_cols(Da[Dir.V], ind_MHv)
    DhHv = _zero_cols(Ds[Dir.H], ind_MHv)
    DvHh = _zero_cols(Ds[Dir.V], ind_MHh)
    DhEn = _zero_cols(Da[Dir.H], ind_MEn)
    DvEn = _zero_cols(Da[Dir.V], ind_MEn)
    DhHn = _zero_cols(Ds[Dir.H], ind_MHn)
    DvHn = _zero_cols(Ds[Dir.V], ind_MHn)
    DhEv = _zero_cols(Da[Dir.H], ind_MEv)
    DvEh = _zero_cols(Da[Dir.V], ind_MEh)

    # ---- Eigenoperator A ------------------------------------------------------
    A = (
        -omega ** 2 * EPS_vh @ MU_hv
        + EPS_vh @ sp.vstack([DvEn, -DhEn]) @ INV_EPS_n @ sp.hstack([-DvHh, DhHv])
        - sp.vstack([DhHn, DvHn]) @ INV_MU_n @ sp.hstack([DhHh, DvHv]) @ MU_hv
    ).tocsr()

    # ---- Field-reconstruction operators --------------------------------------
    En_Htr = (1.0 / (omega * 1j)) * INV_EPS_n @ sp.hstack([-DvHh, DhHv])
    En_Htr = _zero_rows(En_Htr.tocsr(), ind_MEn)

    Hn_Etr = (-1.0 / (omega * 1j)) * INV_MU_n @ sp.hstack([-DvEh, DhEv])
    Hn_Etr = _zero_rows(Hn_Etr.tocsr(), ind_MHn)

    gEh_HvEn = sp.hstack([1j * omega * MU_v, -DhEn]).tocsr()
    gEh_HvEn = _zero_rows(gEh_HvEn, ind_MEh)

    gEv_HhEn = sp.hstack([-1j * omega * MU_h, -DvEn]).tocsr()
    gEv_HhEn = _zero_rows(gEv_HhEn, ind_MEv)

    # ---- PEC handling ---------------------------------------------------------
    # Make sure eigenvector entries corresponding to infinitely large
    # eps_vh (PEC) are zero: zero the corresponding rows/columns and put -1
    # on the diagonal (the eigenvalue gamma^2 = -beta^2 is never -1 ... wait,
    # it can be negative, so use the MATLAB reasoning: beta = 0 gives
    # eigenvalue (1/L)^2 > 0 which cannot equal -1).
    ind_pec = np.where(np.isinf(np.abs(eps_vh)))[0]
    if ind_pec.size > 0:
        diagvec = A.diagonal()
        diagvec[ind_pec] = -1.0
        A = _zero_cols(_zero_rows(A, ind_pec), ind_pec)
        A = (A + sp.diags(diagvec)).tocsr()

    return A, En_Htr.tocsr(), gEh_HvEn, gEv_HhEn, Hn_Etr.tocsr()
