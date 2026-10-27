"""
System building for MaxwellFDFD.

This module provides the :func:`build_system` function that constructs
the FDFD simulation system (grid, materials, sources, PML factors) from
flexible parameter groups.
"""

from typing import Dict, List, Tuple, Union, Any, Optional, TYPE_CHECKING

import numpy as np

from maxwellfdfd.base import (
    Axis, BC, FT, GT, PML, Sign, EquationType,
    Oscillation, PhysUnit, PhysQ,
    chkarg, istypesizeof, expand2mat, expand2row, is2dlike,
)
from maxwellfdfd.shape import Shape, Domain
from maxwellfdfd.source import Source, TFSFPlaneSrc, ModalSrc, assign_source
from maxwellfdfd.grid import Grid3d, generate_lprim3d
from maxwellfdfd.material import Material, assign_material_node, mean_material_node
from maxwellfdfd.diff import MatrixEquation

from .em_object import EMObject
from .scalar3d import Scalar3d
from .utils import expand_node_array, reordering_indices
from .grid_utils import generate_s_factor
from .progmark import ProgMark

if TYPE_CHECKING:
    pass


def _create_material(osc: Oscillation, *args) -> Material:
    """
    Create a Material from flexible argument formats.

    Supported formats:
    - ``Material(name, color, permittivity)``: name without '/'
    - ``Material.fromtable(osc, datapath, color)``: datapath with '/'

    Parameters
    ----------
    osc : Oscillation
        Oscillation parameters (needed for table-based materials).
    *args
        Material description arguments.

    Returns
    -------
    Material
        The created material.
    """
    narg = len(args)
    # Drop trailing logical (is_scatterer flag) if present
    if isinstance(args[-1], bool):
        args = args[:-1]
        narg = len(args)

    chkarg(narg >= 2, '# of arguments should be at least 2.')
    matname = args[0]
    if isinstance(matname, str) and '/' not in matname:
        # Direct material specification
        return Material(*args)
    else:
        # Data table specification
        return Material.fromtable(osc, *args)


def _expand_node_to_ghost(grid3d, arr):
    """
    Extend a node array by ONE ghost layer per axis (N -> N+1 per axis).

    This matches the dual-grid ``lg`` vertex count expected by ``Scalar3d``
    when constructed with ``GT.DUAL`` (whose size is ``N + int(GT.DUAL) = N+1``).
    The ghost layer is *prepended*: under periodic BC it wraps from the
    opposite end; under PEC/PMC the edge value is replicated.

    Parameters
    ----------
    grid3d : Grid3d
        The 3D grid (provides per-axis cell counts and boundary conditions).
    arr : np.ndarray
        Node array of shape ``(Nx, Ny, Nz)``.

    Returns
    -------
    np.ndarray
        Array of shape ``(Nx+1, Ny+1, Nz+1)``.
    """
    result = arr
    for w in Axis.elems():
        wi = int(w)
        Nw = int(grid3d.N[wi])
        # Periodic -> wrap from the opposite end; otherwise replicate the edge.
        src_idx = (Nw - 1) if grid3d.bc[wi] == BC.P else 0
        sl = [slice(None)] * result.ndim
        sl[wi] = src_idx
        result = np.concatenate(
            [np.expand_dims(result[tuple(sl)], wi), result], axis=wi,
        )
    return result


def build_system(
    ge: GT,
    pml: PML,
    *args: Any,
    progmark: Optional[ProgMark] = None,
) -> Tuple[
    Oscillation, 'Grid3d',
    Dict[Tuple[Axis, GT], np.ndarray],
    Dict[Axis, np.ndarray], Dict[Axis, np.ndarray],
    Dict[Axis, np.ndarray], Dict[Axis, np.ndarray],
    List[EMObject], List[Source], List[Material],
    Dict[Axis, 'Scalar3d'], Dict[Axis, 'Scalar3d'],
    bool,
]:
    """
    Build the FDFD system from parameter groups.

    Constructs a simulation system from given objects and sources. The
    constructed system is typically used inside :func:`maxwell_run`.

    Parameters
    ----------
    ge : GT
        Grid type of the E-field.
    pml : PML
        PML kind.
    *args
        Parameter groups, each beginning with a string marker:
        ``'OSC'``, ``'DOM'``, ``'OBJ'``, ``'SOBJ'``, ``'SRCJ'``,
        ``'SRCM'``. See :func:`maxwell_run` for details.
    progmark : ProgMark, optional
        Progress marker for timing. Created internally if not given.

    Returns
    -------
    osc : Oscillation
        Oscillation parameters.
    grid3d : Grid3d
        The 3D grid.
    s_factor_cell : dict
        ``{(Axis, GT): np.ndarray}`` PML s-factors.
    eps_cell : dict
        ``{Axis: np.ndarray}`` permittivity at E-field positions.
    mu_cell : dict
        ``{Axis: np.ndarray}`` permeability at H-field positions.
    J_cell : dict
        ``{Axis: np.ndarray}`` electric current sources.
    M_cell : dict
        ``{Axis: np.ndarray}`` magnetic current sources.
    obj_array : list of EMObject
        Objects in the simulation domain.
    src_array : list of Source
        Sources in the simulation domain.
    mat_array : list of Material
        Materials used.
    eps_node : dict
        ``{Axis: Scalar3d}`` permittivity at grid nodes.
    mu_node : dict
        ``{Axis: Scalar3d}`` permeability at grid nodes.
    isiso : bool
        True if all materials are isotropic.

    Examples
    --------
    >>> from maxwellfdfd.base import GT, PML, BC
    >>> from maxwellfdfd.io import build_system
    >>> osc, grid3d, s_factor, eps, mu, J, M, objs, srcs, mats, eps_n, mu_n, isiso = \\
    ...     build_system(
    ...         GT.PRIM, PML.SC,
    ...         'OSC', 1e-9, 1550,
    ...         'DOM', ['Palik/SiO2', 'none'], [[-700,700],[-600,600],[-200,1700]], 20, BC.P, 200,
    ...         'OBJ', ['Palik/SiO2', 'none'], Box(...),
    ...         'SRCJ', PointSrc(Axis.X, [0, 0, 200]),
    ...     )
    """
    pm = progmark if progmark is not None else ProgMark()

    # Validate ge and pml
    chkarg(isinstance(ge, GT), 'first argument should be "ge" (GT).')
    chkarg(isinstance(pml, PML), 'second argument should be "pml" (PML).')

    # Initialize accumulators
    osc: Optional[Oscillation] = None
    obj_dom: Optional[EMObject] = None
    shape_array: List[Shape] = []
    sshape_array: List[Shape] = []
    mat_array: List[Material] = []
    obj_array: List[EMObject] = []
    sobj_array: List[EMObject] = []
    srcj_array: List[Source] = []
    srcm_array: List[Source] = []
    isepsgiven = False
    isTFSF = False

    # Domain parameters (set during DOM parsing)
    domain: Optional[Domain] = None
    bc: List[BC] = [BC.P] * Axis.count()
    Lpml: np.ndarray = np.zeros((Axis.count(), Sign.count()))
    deg_pml: float = 4
    R_pml: float = float(np.exp(-16))
    withuniformgrid: bool = False
    eps_node_cell: Dict[Axis, np.ndarray] = {}
    mu_node_cell: Dict[Axis, np.ndarray] = {}

    iarg = 0
    while iarg < len(args):
        arg = args[iarg]

        if isinstance(arg, str) and arg.upper() == 'OSC':
            # ---- OSC group ----
            iarg += 1
            arg = args[iarg]
            if isinstance(arg, (int, float)) and arg > 0:
                L0 = arg
                iarg += 1
                wvlen = args[iarg]
                chkarg(
                    isinstance(wvlen, (int, float, complex)),
                    f'argument #{iarg} should be "wvlen" (complex).',
                )
                unit = PhysUnit(L0)
                osc = Oscillation(complex(wvlen), unit)
            elif isinstance(arg, Oscillation):
                osc = arg
            else:
                raise ValueError(
                    f'argument #{iarg} should be either "L0" (positive) '
                    f'or "osc" (instance of Oscillation).'
                )

        elif isinstance(arg, str) and arg.upper() == 'DOM':
            # ---- DOM group ----
            iarg += 1
            arg = args[iarg]

            if isinstance(arg, EMObject):
                obj_dom = arg
                domain = obj_dom.shape
                chkarg(
                    isinstance(domain, Domain),
                    f'argument #{iarg} should be EMObject with Domain as shape.',
                )
            else:
                # Parse material
                if isinstance(arg, (list, tuple)):
                    mat_dom = _create_material(osc, *arg)
                elif isinstance(arg, Material):
                    mat_dom = arg
                else:
                    raise ValueError(
                        f'argument #{iarg} should be cell, Material, or EMObject.'
                    )

                iarg += 1
                arg = args[iarg]

                if isinstance(arg, Domain):
                    domain = arg
                elif isinstance(arg, np.ndarray) and arg.shape == (Axis.count(), Sign.count()):
                    # box_domain: [xmin xmax; ymin ymax; zmin zmax]
                    box_domain = arg
                    iarg += 1
                    dl_domain = args[iarg]
                    dl_domain = expand2row(dl_domain, Axis.count())
                    domain = Domain(box_domain, dl_domain)
                else:
                    raise ValueError(
                        f'argument #{iarg} should be "box_dom" or "domain".'
                    )
                obj_dom = EMObject(domain, mat_dom)

            mat_array.append(obj_dom.material)

            # Boundary conditions
            iarg += 1
            bc_arg = args[iarg]
            bc = list(expand2mat(bc_arg, Axis.count(), Sign.count())[:, int(Sign.N)])
            # Convert to BC list
            bc = [BC.elems(int(v)) for v in bc]

            # PML thicknesses
            iarg += 1
            Lpml_arg = args[iarg]
            Lpml = expand2mat(Lpml_arg, Axis.count(), Sign.count())

            # Optional: deg_pml
            # NOTE: must exclude bool, because isinstance(True, int) is True
            # in Python (bool is a subclass of int). Otherwise the
            # withuniformgrid flag would be consumed as deg_pml.
            iarg += 1
            if iarg < len(args) and isinstance(args[iarg], (int, float)) and not isinstance(args[iarg], bool):
                deg_pml = float(args[iarg])
            else:
                iarg -= 1

            # Optional: R_pml
            iarg += 1
            if iarg < len(args) and isinstance(args[iarg], (int, float)) and not isinstance(args[iarg], bool):
                R_pml = float(args[iarg])
            else:
                iarg -= 1

            # Optional: withuniformgrid
            iarg += 1
            if iarg < len(args) and isinstance(args[iarg], bool):
                withuniformgrid = args[iarg]
            else:
                iarg -= 1

        elif isinstance(arg, str) and arg.upper() in ('OBJ', 'SOBJ'):
            # ---- OBJ / SOBJ group ----
            is_scatterer = arg.upper() == 'SOBJ'
            iarg += 1
            arg = args[iarg]

            if isinstance(arg, np.ndarray) and np.iscomplexobj(arg) and arg.ndim == 3:
                # eps_node_array given directly
                isepsgiven = True
                eps_node_cell = {w: arg for w in Axis.elems()}
                mu_node_cell = {w: np.ones_like(arg) for w in Axis.elems()}
            else:
                # Parse objects
                obj_array_temp: List[EMObject] = []
                shape_array_temp: List[Shape] = []

                while (
                    isinstance(arg, (list, tuple))
                    or isinstance(arg, Material)
                    or (isinstance(arg, list) and all(isinstance(o, EMObject) for o in arg))
                    or isinstance(arg, EMObject)
                ):
                    if isinstance(arg, EMObject):
                        obj_array_temp.append(arg)
                        shape_array_temp.append(arg.shape)
                        mat_array.append(arg.material)
                        iarg += 1
                        if iarg >= len(args):
                            break
                        arg = args[iarg]
                    elif isinstance(arg, list) and all(isinstance(o, EMObject) for o in arg):
                        obj_array_temp.extend(arg)
                        for obj in arg:
                            shape_array_temp.append(obj.shape)
                            mat_array.append(obj.material)
                        iarg += 1
                        if iarg >= len(args):
                            break
                        arg = args[iarg]
                    else:
                        # Material + shapes
                        if isinstance(arg, (list, tuple)):
                            mat = _create_material(osc, *arg)
                        elif isinstance(arg, Material):
                            mat = arg
                        else:
                            break
                        mat_array.append(mat)

                        iarg += 1
                        if iarg >= len(args):
                            break
                        arg = args[iarg]

                        while isinstance(arg, Shape) or (
                            isinstance(arg, list) and all(isinstance(s, Shape) for s in arg)
                        ):
                            shapes = [arg] if isinstance(arg, Shape) else arg
                            shape_array_temp.extend(shapes)
                            for s in shapes:
                                obj_array_temp.append(EMObject(s, mat))
                            iarg += 1
                            if iarg >= len(args):
                                break
                            arg = args[iarg]

                iarg -= 1

                if is_scatterer:
                    sshape_array.extend(shape_array_temp)
                    sobj_array.extend(obj_array_temp)
                else:
                    shape_array.extend(shape_array_temp)
                    obj_array.extend(obj_array_temp)

        elif isinstance(arg, str) and arg.upper() == 'SRCJ':
            # ---- SRCJ group ----
            iarg += 1
            if iarg >= len(args) or not isinstance(args[iarg], (Source, list)):
                import warnings
                warnings.warn('no source is given.', UserWarning)
            else:
                arg = args[iarg]
                while isinstance(arg, Source) or (
                    isinstance(arg, list) and all(isinstance(s, Source) for s in arg)
                ):
                    srcs = [arg] if isinstance(arg, Source) else arg
                    for src in srcs:
                        if isinstance(src, TFSFPlaneSrc):
                            isTFSF = True
                        src.set_gridtype(ge)
                    srcj_array.extend(srcs)
                    iarg += 1
                    if iarg >= len(args):
                        break
                    arg = args[iarg]
                iarg -= 1

        elif isinstance(arg, str) and arg.upper() == 'SRCM':
            # ---- SRCM group ----
            iarg += 1
            if iarg >= len(args) or not isinstance(args[iarg], (Source, list)):
                import warnings
                warnings.warn('no source is given.', UserWarning)
            else:
                arg = args[iarg]
                while isinstance(arg, Source) or (
                    isinstance(arg, list) and all(isinstance(s, Source) for s in arg)
                ):
                    srcs = [arg] if isinstance(arg, Source) else arg
                    for src in srcs:
                        if isinstance(src, TFSFPlaneSrc):
                            isTFSF = True
                        src.set_gridtype(ge.alter())
                    srcm_array.extend(srcs)
                    iarg += 1
                    if iarg >= len(args):
                        break
                    arg = args[iarg]
                iarg -= 1

        iarg += 1

    # ---- Post-parse validation ----
    chkarg(osc is not None, 'OSC parameter groups should be set.')
    chkarg(obj_dom is not None, 'DOM parameter groups should be set.')
    obj_array = [obj_dom] + obj_array
    src_array = srcj_array + srcm_array

    if isTFSF and not sobj_array:
        import warnings
        warnings.warn(
            'TF/SF source is used, but scatterers are not defined in SOBJ group.',
            UserWarning,
        )

    pm.mark('initial setup')

    print(f'\tLength Unit: {osc.unit.value(PhysQ.L)} m')
    print(f'\twvlen = {osc.in_L0()}, freq = {osc.in_eV()} eV')

    # Deduplicate materials
    mat_array = list(dict.fromkeys(mat_array))
    isiso = True
    print('materials used:')
    for mat in mat_array:
        # NOTE: mat.eps / mat.mu are 3x3 ndarrays (unhashable, so set() fails).
        # Print the scalar value when isotropic, otherwise label as a tensor.
        eps_text = mat.eps[0, 0] if mat.has_iso_eps else 'tensor'
        mu_text = mat.mu[0, 0] if mat.has_iso_mu else 'tensor'
        print(f'\t{mat.name}: eps = {eps_text}, mu = {mu_text}')
        isiso = isiso and getattr(mat, 'isiso', True)

    # ---- Generate grid ----
    lprim, Npml = generate_lprim3d(
        domain, Lpml, shape_array + sshape_array, src_array, withuniformgrid,
    )
    grid3d = Grid3d(osc.unit, lprim, Npml, bc)

    if withuniformgrid:
        pm.mark('uniform grid generation')
    else:
        pm.mark('nonuniform grid generation')
    print(f'\t[Nx Ny Nz] = {grid3d.N}')

    # Warn about 2D-like simulations
    like2d, normal_axis = is2dlike(grid3d.N)
    if like2d and grid3d.N[int(normal_axis)] >= 2:
        import warnings
        warnings.warn(
            f'If this is a 2D structure, N{normal_axis} should be 1; '
            f'check d{normal_axis}\'s of objects and locations of sources',
            UserWarning,
        )

    # ---- Construct material parameters ----
    if not isepsgiven:
        eps_node_cell, mu_node_cell = assign_material_node(grid3d, obj_array)

    eps_cell = mean_material_node(grid3d, ge, eps_node_cell)
    mu_cell = mean_material_node(grid3d, ge.alter(), mu_node_cell)

    # ---- Construct PML s-factors ----
    s_factor_cell = generate_s_factor(osc.in_omega0(), grid3d, deg_pml, R_pml)
    pm.mark('eps and mu assignment')

    # ---- Handle modal sources and TF/SF ----
    if not isTFSF:
        for src in src_array:
            if isinstance(src, ModalSrc):
                if not getattr(src, 'ispreped', False):
                    src.prep(ge, pml, osc, grid3d, eps_cell, mu_cell, s_factor_cell)
                neff = src.neff
                beta = 2 * np.pi * neff / osc.in_L0()
                pm.mark('mode calculation')
                print(f'\tbeta = {beta}, n_eff = {neff}')
    else:
        # Set up J for TF/SF
        for src in src_array:
            if isinstance(src, TFSFPlaneSrc):
                tfsfsrc = src
                cb_center = tfsfsrc.shape.cb_center
                bgobj = None
                for bgobj_candidate in reversed(obj_array):
                    if bgobj_candidate.shape.contains(*cb_center):
                        bgobj = bgobj_candidate
                        break
                if bgobj is not None:
                    tfsfsrc.set_bg_material(bgobj.material)
                F0 = tfsfsrc.create_incidentF(osc, grid3d)

                JM = {w: np.zeros(tuple(int(n) for n in grid3d.N)) for w in Axis.elems()}

                if tfsfsrc.gt == ge:
                    eqtype_tfsf = EquationType(FT.E, ge)
                else:
                    eqtype_tfsf = EquationType(FT.H, ge)

                eq = MatrixEquation(
                    eqtype_tfsf, pml, osc.in_omega0(),
                    eps_cell, mu_cell, s_factor_cell, JM, JM, grid3d,
                )
                Op, _, _ = eq.matrixfree_op()

                # F0 is a list [Fx, Fy, Fz] from create_incidentF()
                x0 = np.concatenate([
                    F0[int(Axis.X)].flatten(),
                    F0[int(Axis.Y)].flatten(),
                    F0[int(Axis.Z)].flatten(),
                ])
                r = reordering_indices(Axis.count(), grid3d.N)
                x0 = x0[r]

                _op_raw = Op(x0, 'notransp')
                JM_flat = _op_raw / (-1j * osc.in_omega0())
                # JM_flat is in *interleaved* order (Op returns reordered
                # space).  De-interleave with Fortran-order reshape on
                # (dof, M), then spatially reshape each component with
                # C-order to match the derivative-operator convention.
                M_tfsf = int(np.prod(grid3d.N))
                JM_temp = JM_flat.reshape(Axis.count(), M_tfsf, order='F')
                JM = [
                    JM_temp[int(w)].reshape(tuple(int(n) for n in grid3d.N))
                    for w in Axis.elems()
                ]
                tfsfsrc.setJM(JM, grid3d)

        # Add scatterer objects to eps/mu
        eps_node_cell, mu_node_cell = assign_material_node(
            grid3d, sobj_array, eps_node_cell, mu_node_cell,
        )
        eps_cell = mean_material_node(grid3d, ge, eps_node_cell)
        mu_cell = mean_material_node(grid3d, ge.alter(), mu_node_cell)
        pm.mark('TF/SF source assignment')

    obj_array = obj_array + sobj_array

    # ---- Build Scalar3d for eps_node and mu_node ----
    eps_node: Dict[Axis, Scalar3d] = {}
    mu_node: Dict[Axis, Scalar3d] = {}
    for w in Axis.elems():
        # Scalar3d with GT.DUAL expects N+1 nodes per axis (matching the dual
        # grid vertex count lg[DUAL]). expand_node_array adds 2 ghost layers
        # (N+2) which is wrong here; _expand_node_to_ghost adds exactly one.
        eps_node_cell[w] = _expand_node_to_ghost(grid3d, eps_node_cell[w])
        mu_node_cell[w] = _expand_node_to_ghost(grid3d, mu_node_cell[w])

        eps_node[w] = Scalar3d(
            eps_node_cell[w], grid3d,
            [GT.DUAL, GT.DUAL, GT.DUAL], osc, PhysQ.EPS, '\\epsilon',
        )
        mu_node[w] = Scalar3d(
            mu_node_cell[w], grid3d,
            [GT.DUAL, GT.DUAL, GT.DUAL], osc, PhysQ.MU, '\\mu',
        )

    # ---- Construct sources ----
    J_cell = assign_source(grid3d, srcj_array)
    M_cell = assign_source(grid3d, srcm_array)
    pm.mark('J assignment')

    return (
        osc, grid3d, s_factor_cell,
        eps_cell, mu_cell, J_cell, M_cell,
        obj_array, src_array, mat_array,
        eps_node, mu_node, isiso,
    )
