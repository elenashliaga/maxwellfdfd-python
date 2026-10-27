"""
General utility functions for MaxwellFDFD io module.

This module provides array manipulation, complex-number handling,
node-array expansion, field-to-scalar conversion, slicing, grid-type
transformation, and amplitude computation utilities.
"""

from typing import Dict, List, Tuple, Union, Optional, TYPE_CHECKING

import numpy as np

from maxwellfdfd.base import (
    Axis, BC, Dir, FT, GT, PhysQ, Oscillation,
    chkarg, istypesizeof,
)

if TYPE_CHECKING:
    from maxwellfdfd.grid.grid2d import Grid2d
    from maxwellfdfd.grid.grid3d import Grid3d
    from .scalar2d import Scalar2d
    from .scalar3d import Scalar3d


# ======================================================================
# Complex number handling
# ======================================================================
def expand_complex(complex_array: np.ndarray) -> np.ndarray:
    """
    Expand a complex array into an interleaved real/imaginary array.

    The result has an extra leading dimension of size 2: index 0 holds
    the real part, index 1 holds the imaginary part.

    Parameters
    ----------
    complex_array : np.ndarray
        Array with complex elements.

    Returns
    -------
    np.ndarray
        Array of shape ``(2, *complex_array.shape)`` where ``[0]`` is the
        real part and ``[1]`` is the imaginary part.

    Examples
    --------
    >>> import numpy as np
    >>> from maxwellfdfd.io import expand_complex
    >>> a = np.array([1.0 + 2.0j, 3.0 + 4.0j])
    >>> expand_complex(a)
    array([[1., 3.],
           [2., 4.]])
    """
    chkarg(
        np.iscomplexobj(complex_array) or np.isrealobj(complex_array),
        '"complex_array" should be array with complex elements.',
    )
    return np.stack([np.real(complex_array), np.imag(complex_array)], axis=0)


def collapse_complex(ri_array: np.ndarray) -> np.ndarray:
    """
    Collapse an interleaved real/imaginary array into a complex array.

    The inverse of :func:`expand_complex`: index 0 of the leading
    dimension is the real part, index 1 is the imaginary part.

    Parameters
    ----------
    ri_array : np.ndarray
        Real array of shape ``(2, ...)``.

    Returns
    -------
    np.ndarray
        Complex array with the leading dimension removed.

    Examples
    --------
    >>> import numpy as np
    >>> from maxwellfdfd.io import collapse_complex
    >>> ri = np.array([[1.0, 3.0], [2.0, 4.0]])
    >>> collapse_complex(ri)
    array([1.+2.j, 3.+4.j])
    """
    chkarg(np.isrealobj(ri_array), '"ri_array" should be array with real elements.')
    return ri_array[0] + 1j * ri_array[1]


# ======================================================================
# Cell / dict array operations
# ======================================================================
def cell2array(
    cellarray: Union[List[np.ndarray], Dict[Axis, np.ndarray]],
    nD: int,
) -> np.ndarray:
    """
    Stack a list/dict of nD arrays into an (nD+1)-dimensional array.

    The leading dimension indexes the cell elements.

    Parameters
    ----------
    cellarray : list or dict of np.ndarray
        Collection of arrays, each of dimensionality ``nD``.
    nD : int
        Intended dimensionality of each element.

    Returns
    -------
    np.ndarray
        Stacked array of shape ``(ncell, *element_shape)``.

    Raises
    ------
    ValueError
        If ``nD`` is not a positive integer or elements have mismatched
        shapes.
    """
    chkarg(isinstance(nD, int) and nD > 0, '"nD" should be positive integer.')

    if isinstance(cellarray, dict):
        arrays = [cellarray[w] for w in sorted(cellarray.keys(), key=lambda a: int(a))]
    else:
        arrays = list(cellarray)

    ncell = len(arrays)
    if ncell == 0:
        return np.array([])

    dims = arrays[0].shape
    for arr in arrays:
        chkarg(
            arr.shape == dims,
            'elements of "cellarray" should have the same size.',
        )
    return np.stack(arrays, axis=0)


def real_cell(
    complex_cell: Dict[Axis, np.ndarray],
) -> Dict[Axis, np.ndarray]:
    """
    Take the real part of each element in a cell (dict) of arrays.

    Parameters
    ----------
    complex_cell : dict
        ``{Axis: np.ndarray}`` with complex elements.

    Returns
    -------
    dict
        ``{Axis: np.ndarray}`` with real elements.
    """
    chkarg(
        isinstance(complex_cell, dict),
        '"complex_cell" should be dict with complex elements.',
    )
    return {k: np.real(v) for k, v in complex_cell.items()}


# ======================================================================
# Node array expansion
# ======================================================================
def expand_node_array(
    grid3d: 'Grid3d',
    node_array: np.ndarray,
) -> np.ndarray:
    """
    Extend eps/mu node arrays to ghost points considering boundary conditions.

    For each axis, a ghost layer is prepended and appended. Under periodic
    boundary conditions, the ghost values wrap around; under PEC/PMC, the
    edge values are replicated.

    Parameters
    ----------
    grid3d : Grid3d
        The 3D grid providing boundary conditions.
    node_array : np.ndarray
        ``Nx × Ny × Nz`` array of node values.

    Returns
    -------
    np.ndarray
        ``(Nx+2) × (Ny+2) × (Nz+2)`` array with ghost layers.
    """
    result = node_array
    for w in Axis.elems():
        wi = int(w)
        Nw = int(grid3d.N[wi])
        if grid3d.bc[wi] == BC.P:
            ind_gn = Nw - 1  # last element (0-based)
            ind_gp = 0       # first element
        else:
            ind_gn = 0       # first element
            ind_gp = Nw - 1  # last element
        # Build slices for prepend and append
        prepend_slices = [slice(None)] * result.ndim
        prepend_slices[wi] = ind_gn
        append_slices = [slice(None)] * result.ndim
        append_slices[wi] = ind_gp
        # NOTE: the new singleton axis must be inserted at position `wi`, not
        # at the front (which is what bare np.newaxis does). Use expand_dims.
        result = np.concatenate(
            [np.expand_dims(result[tuple(prepend_slices)], wi),
             result,
             np.expand_dims(result[tuple(append_slices)], wi)],
            axis=wi,
        )
    return result


# ======================================================================
# Reordering indices
# ======================================================================
def reordering_indices(dof: int, N: Union[np.ndarray, List[int]]) -> np.ndarray:
    """
    Generate indices to reorder matrix/vector elements for reduced bandwidth.

    The Maxwell operator matrix has lower bandwidth when field components
    are interleaved (x, y, z, x, y, z, ...) rather than grouped
    (x, x, ..., y, y, ..., z, z, ...).

    Parameters
    ----------
    dof : int
        Degrees of freedom per grid cell (typically ``Axis.count()``).
    N : array-like
        Grid dimensions ``[Nx, Ny, Nz]``.

    Returns
    -------
    np.ndarray
        1D integer array of permutation indices.

    Examples
    --------
    >>> from maxwellfdfd.io import reordering_indices
    >>> reordering_indices(3, [2, 1, 1])
    array([0, 3, 1, 4, 2, 5])
    """
    chkarg(isinstance(dof, int) and dof > 0, '"dof" should be positive integer.')
    N = np.asarray(N, dtype=int)
    chkarg(
        np.all(N > 0) and N.ndim == 1,
        '"N" should be row vector with positive integer elements.',
    )
    total = dof * int(np.prod(N))
    r = np.arange(total, dtype=int)
    # Use Fortran (column-major) order to match diff/_utils.py and MATLAB.
    # The default C-order reshape/flatten produces a DIFFERENT permutation
    # that is inconsistent with MatrixEquation's reordering, breaking TFSF.
    r = r.reshape(int(np.prod(N)), dof, order='F')
    r = r.T
    return r.ravel(order='F')


# ======================================================================
# array2scalar — convert raw array to Scalar2d/Scalar3d
# ======================================================================
def _attach_extra_F(
    Fw_array: np.ndarray,
    ft: FT,
    gt: GT,
    w: Axis,
    bc_v: BC,
    v: Axis,
    kvLv: float,
) -> np.ndarray:
    """
    Augment a field array with ghost boundary elements along axis ``v``.

    This internal helper combines the E-field and H-field ghost-point
    attachment logic based on field type, grid type, and boundary
    condition.

    Parameters
    ----------
    Fw_array : np.ndarray
        The field component array.
    ft : FT
        Field type (E or H).
    gt : GT
        Grid type along axis ``v``.
    w : Axis
        The component axis of this field.
    bc_v : BC
        Boundary condition along axis ``v``.
    v : Axis
        The axis along which to attach ghost points.
    kvLv : float
        Bloch phase factor ``k_Bloch * L`` along axis ``v``.

    Returns
    -------
    np.ndarray
        The augmented array.
    """
    vi = int(v)
    Nv = Fw_array.shape[vi]

    # Scalar3d stores N points along PRIM axes and N+1 along DUAL axes
    # (N + int(gt)). The raw field already has N points on every axis, so:
    #   - PRIM axis: no ghost is needed (return unchanged).
    #   - DUAL axis: prepend exactly ONE ghost at index 0, so the original
    #     N values occupy indices 1..N (matches Scalar3d.data_original).
    if gt == GT.PRIM:
        return Fw_array

    # GT.DUAL: prepend one ghost.
    prepend_slices = [slice(None)] * Fw_array.ndim
    if bc_v == BC.P:
        # Periodic: ghost = last node value * Bloch phase.
        prepend_slices[vi] = Nv - 1
        ghost = Fw_array[tuple(prepend_slices)] * np.exp(1j * kvLv)
    else:
        # PEC/PMC: ghost = +/- first node value (image symmetry).
        prepend_slices[vi] = 0
        ft_match_bc = (ft == FT.E and bc_v == BC.E) or (ft == FT.H and bc_v == BC.M)
        if (v != w and ft_match_bc) or (v == w and not ft_match_bc):
            ghost = -Fw_array[tuple(prepend_slices)]
        else:
            ghost = Fw_array[tuple(prepend_slices)]
    Fw_array = np.concatenate(
        [np.expand_dims(ghost, axis=vi), Fw_array], axis=vi,
    )
    return Fw_array


def array2scalar(
    F_array: np.ndarray,
    grid: Union['Grid2d', 'Grid3d'],
    ge: GT,
    ft: FT,
    axis: Axis,
    osc: Oscillation,
    physQ: PhysQ,
    intercept: float = float('nan'),
) -> Union['Scalar2d', 'Scalar3d']:
    """
    Convert a raw field array into a Scalar2d or Scalar3d object.

    Attaches ghost boundary elements based on field type, grid type,
    and boundary conditions, then constructs the appropriate Scalar
    object.

    Parameters
    ----------
    F_array : np.ndarray
        Raw field array (without ghost points).
    grid : Grid2d or Grid3d
        The grid on which the field is defined.
    ge : GT
        Grid type of the E-field.
    ft : FT
        Field type (E or H).
    axis : Axis
        The component axis of this field.
    osc : Oscillation
        Oscillation parameters.
    physQ : PhysQ
        Physical quantity.
    intercept : float, optional
        Intercept for 2D slices. Only used when ``grid`` is a Grid2d.

    Returns
    -------
    Scalar2d or Scalar3d
        The constructed scalar field object.
    """
    # Lazy import to avoid circular dependency
    from .scalar2d import Scalar2d
    from .scalar3d import Scalar3d

    chkarg(
        istypesizeof(grid, 'Grid2d') or istypesizeof(grid, 'Grid3d'),
        '"grid" should be instance of Grid2d or Grid3d.',
    )

    is3D = istypesizeof(grid, 'Grid3d')

    if ft == FT.E:
        gt = ge  # grid type for E-field
    else:  # FT.H
        gt = ge.alter()  # grid type for H-field

    # Build gt_array: gt for all axes, except `axis` gets alter(gt)
    gt_array = [gt] * Axis.count()
    gt_array[int(axis)] = gt.alter()

    if is3D:
        grid3d = grid
        V = F_array
        for w in Axis.elems():
            wi = int(w)
            V = _attach_extra_F(
                V, ft, gt_array[wi], axis,
                grid3d.bc[wi], w,
                float(grid3d.kBloch[wi] * grid3d.L[wi]),
            )
        name = f'{physQ.symbol}_{axis}'
        return Scalar3d(V, grid3d, gt_array, osc, physQ, name)
    else:
        # 2D grid
        grid2d = grid
        # Map 2D directions to axes
        gt_array_2d = [gt_array[int(grid2d.axis[int(Dir.H)])],
                       gt_array[int(grid2d.axis[int(Dir.V)])]]

        if axis == grid2d.axis[int(Dir.H)]:
            axis_temp = Axis.X
        elif axis == grid2d.axis[int(Dir.V)]:
            axis_temp = Axis.Y
        else:
            axis_temp = Axis.Z

        C = F_array
        for d in Dir.elems():
            di = int(d)
            C = _attach_extra_F(
                C, ft, gt_array_2d[di], axis_temp,
                grid2d.bc[di], Axis.elems(di),
                float(grid2d.kBloch[di] * grid2d.L[di]),
            )
        name = f'{physQ.symbol}_{axis}'
        return Scalar2d(C, grid2d, gt_array_2d, osc, physQ, name, intercept)


# ======================================================================
# slice_scalar3d — slice a 3D scalar to 2D
# ======================================================================
def slice_scalar3d(
    scalar3d: 'Scalar3d',
    normal_axis: Axis,
    intercept: float,
) -> 'Scalar2d':
    """
    Slice a Scalar3d along a plane perpendicular to ``normal_axis``.

    If the intercept coincides with a grid plane, the data is extracted
    directly; otherwise, linear interpolation is used.

    Parameters
    ----------
    scalar3d : Scalar3d
        The 3D scalar field to slice.
    normal_axis : Axis
        The axis normal to the slicing plane.
    intercept : float
        The coordinate value at which to slice along ``normal_axis``.

    Returns
    -------
    Scalar2d
        The 2D slice.
    """
    from .scalar2d import Scalar2d
    from maxwellfdfd.grid.grid2d import Grid2d
    from scipy.interpolate import interpn

    chkarg(
        istypesizeof(scalar3d, 'Scalar3d'),
        '"scalar3d" should be instance of Scalar3d.',
    )
    chkarg(isinstance(normal_axis, Axis), '"normal_axis" should be instance of Axis.')
    chkarg(np.isreal(intercept), '"intercept" should be real.')

    grid3d = scalar3d.grid3d
    h, v, n = normal_axis.cycle()  # h, v are the in-plane axes; n == normal_axis

    chkarg(
        grid3d.comp[int(n)].contains(np.array([intercept])),
        f'"intercept" should be contained in {n}-axis range.',
    )

    lall = {
        w: grid3d.lall[(w, scalar3d.gt_array[int(w)])]
        for w in Axis.elems()
    }
    nall = lall[n]

    # Check if intercept is exactly on a grid plane
    matches = np.where(np.isclose(nall, intercept))[0]
    ni = int(n)

    if matches.size > 0:
        indn = int(matches[0])
        slices = [slice(None)] * 3
        slices[ni] = indn
        array = scalar3d.array[tuple(slices)]
    else:
        # Interpolate
        below = np.where(nall < intercept)[0]
        indn0 = int(below[-1]) if below.size > 0 else 0
        indn1 = indn0 + 1

        slices = [slice(None)] * 3
        slices[ni] = slice(indn0, indn1 + 1)
        V = scalar3d.array[tuple(slices)]

        l_interp = [lall[w] for w in Axis.elems()]
        l_interp[ni] = nall[indn0:indn1 + 1]

        xi_n = np.array([intercept])
        # Build query grid
        grids = []
        for w in Axis.elems():
            wi = int(w)
            if wi == ni:
                grids.append(xi_n)
            else:
                grids.append(lall[w])

        mesh = np.meshgrid(*[l_interp[wi] for wi in range(3)], indexing='ij')
        points = [l_interp[wi] for wi in range(3)]

        xi_mesh = np.meshgrid(*grids, indexing='ij')
        xi = np.stack(xi_mesh, axis=-1).reshape(-1, 3)

        val = interpn(points, V, xi)
        val = val.reshape([len(g) for g in grids])
        # Squeeze the normal axis (size 1)
        array = np.squeeze(val, axis=ni)

    # Permute to (h, v) order
    hi, vi = int(h), int(v)
    perm = [hi, vi, ni]
    # Remove the normal axis index and reorder
    order = [hi, vi]
    # Adjust indices for the removed dimension
    order = [x if x < ni else x - 1 for x in order]
    array = np.transpose(array, order)

    grid2d = Grid2d(grid3d, n)
    gt_array_2d = [scalar3d.gt_array[hi], scalar3d.gt_array[vi]]
    return Scalar2d(
        array, grid2d, gt_array_2d, scalar3d.osc,
        scalar3d.physQcell, scalar3d.name, intercept,
    )


# ======================================================================
# transform_gt — transform grid type of a scalar field
# ======================================================================
def transform_gt(
    scalarNd: Union['Scalar2d', 'Scalar3d'],
    axis: Axis,
    gt_array: List[GT],
) -> Union['Scalar2d', 'Scalar3d']:
    """
    Transform the grid type of a scalar field via interpolation.

    Re-interpolates the scalar field data onto a new set of grid types.

    Parameters
    ----------
    scalarNd : Scalar2d or Scalar3d
        The input scalar field.
    axis : Axis
        The component axis of the field.
    gt_array : list of GT
        Target grid types.

    Returns
    -------
    Scalar2d or Scalar3d
        The re-interpolated scalar field.
    """
    from scipy.interpolate import interpn

    chkarg(
        istypesizeof(scalarNd, 'Scalar2d') or istypesizeof(scalarNd, 'Scalar3d'),
        '"scalarNd" should be instance of Scalar2d or Scalar3d.',
    )
    chkarg(isinstance(axis, Axis), '"axis" should be instance of Axis.')

    physQcell = scalarNd.physQcell
    chkarg(
        len(physQcell) == 2 and physQcell[1] == 1,
        '"scalarNd" should represent quantity defined in PhysQ.',
    )

    physQ = physQcell[0]
    if physQ == PhysQ.E:
        ft = FT.E
    elif physQ == PhysQ.H:
        ft = FT.H
    else:
        raise ValueError('"physQ" should be either PhysQ.E or PhysQ.H.')

    if istypesizeof(scalarNd, 'Scalar2d'):
        Ndim = 2
        grid = scalarNd.grid2d
        axis_dummy = Dir.H
    else:
        Ndim = 3
        grid = scalarNd.grid3d
        axis_dummy = Axis.X

    chkarg(
        isinstance(gt_array, list) and len(gt_array) == Ndim
        and all(isinstance(g, GT) for g in gt_array),
        f'"gt_array" should be length-{Ndim} list of GT objects.',
    )

    data, l = scalarNd.data_expanded()

    # Target locations
    if Ndim == 3:
        li = {w: grid.l[(w, gt_array[int(w)])] for w in Axis.elems()}
        points = [l[w] for w in Axis.elems()]
        xi_grids = [li[w] for w in Axis.elems()]
    else:
        li = {d: grid.l[(d, gt_array[int(d)])] for d in Dir.elems()}
        points = [l[d] for d in Dir.elems()]
        xi_grids = [li[d] for d in Dir.elems()]

    xi_mesh = np.meshgrid(*xi_grids, indexing='ij')
    xi = np.stack(xi_mesh, axis=-1).reshape(-1, Ndim)

    V = interpn(points, data, xi)
    V = V.reshape([len(g) for g in xi_grids])

    osc = scalarNd.osc
    if Ndim == 3:
        ge = gt_array[int(axis)] if ft == FT.E else gt_array[int(axis)].alter()
        return array2scalar(V, grid, ge, ft, axis, osc, physQ)
    else:
        intercept = scalarNd.intercept
        return array2scalar(V, grid, gt_array[int(axis)], ft, axis, osc, physQ, intercept)


# ======================================================================
# vec2amp — compute field amplitude from vector components
# ======================================================================
def vec2amp(F_cell: Dict[Axis, 'Scalar3d']) -> 'Scalar3d':
    """
    Compute the amplitude (magnitude) of a vector field.

    Interpolates each component onto the dual grid, computes the
    Euclidean norm across components, and returns a Scalar3d.

    Parameters
    ----------
    F_cell : dict
        ``{Axis: Scalar3d}`` length-3 cell of field components.

    Returns
    -------
    Scalar3d
        The field amplitude ``|F|``.
    """
    from .scalar3d import Scalar3d
    from scipy.interpolate import interpn

    chkarg(
        isinstance(F_cell, dict) and len(F_cell) == Axis.count(),
        f'"F_cell" should be length-{Axis.count()} dict with Scalar3d as elements.',
    )

    Fx = F_cell[Axis.X]
    grid3d = Fx.grid3d
    osc = Fx.osc
    physQcell = Fx.physQcell
    name = Fx.name[:-2] if len(Fx.name) > 2 else Fx.name  # strip '_x' etc.

    # Target locations: dual grid
    li = {w: grid3d.l[(w, GT.DUAL)] for w in Axis.elems()}

    fi = np.full(tuple(int(n) for n in grid3d.N) + (Axis.count(),), np.nan)
    for w in Axis.elems():
        wi = int(w)
        Fw = F_cell[w]
        fw, l = Fw.data_ghost_expanded()
        points = [l[ax] for ax in Axis.elems()]
        xi_mesh = np.meshgrid(*[li[ax] for ax in Axis.elems()], indexing='ij')
        xi = np.stack(xi_mesh, axis=-1).reshape(-1, 3)
        val = interpn(points, fw, xi)
        fi[..., wi] = val.reshape(tuple(int(n) for n in grid3d.N))

    famp = np.sqrt(np.sum(np.abs(fi) ** 2, axis=-1))

    # Attach extra boundary points
    for w in Axis.elems():
        wi = int(w)
        bc_w = grid3d.bc[wi]
        if bc_w == BC.P:
            prepend = famp[np.index_exp[..., -1] if wi == 0 else
                           (slice(None),) * wi + (-1,) + (slice(None),) * (2 - wi)]
            append = famp[(slice(None),) * wi + (0,) + (slice(None),) * (2 - wi)]
        else:
            prepend = famp[(slice(None),) * wi + (0,) + (slice(None),) * (2 - wi)]
            append = famp[(slice(None),) * wi + (-1,) + (slice(None),) * (2 - wi)]
        famp = np.concatenate(
            [np.expand_dims(prepend, axis=wi), famp, np.expand_dims(append, axis=wi)],
            axis=wi,
        )

    gt_array = [GT.DUAL, GT.DUAL, GT.DUAL]
    return Scalar3d(famp, grid3d, gt_array, osc, physQcell, f'|{name}|')
