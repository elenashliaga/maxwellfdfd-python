"""
Main solver interface for MaxwellFDFD.

This module provides the :func:`maxwell_run` function, the primary
entry point for running FDFD simulations. It constructs a simulation
domain from given objects and sources, and launches a finite-difference
frequency-domain (FDFD) solver to solve Maxwell's equations.

Parameter Groups
----------------
Each of OSC, DOM, OBJ, and SRC represents a group of parameters passed
as positional arguments (``*args``). Each group begins with a string
marker.

**OSC** — oscillation parameters:
    - ``'OSC', L0, wvlen`` — length unit and vacuum wavelength
    - ``'OSC', osc`` — an :class:`~maxwellfdfd.base.oscillation.Oscillation` instance

**DOM** — domain parameters:
    - ``'DOM', material, box, dl, bc, Lpml, [deg_pml], [R_pml], [withuniformgrid]``
    - ``'DOM', domain, dl, bc, Lpml, [deg_pml], [R_pml], [withuniformgrid]``

**OBJ / SOBJ** — object parameters:
    - ``'OBJ', material_1, shapes_1, ..., material_N, shapes_N``
    - ``'OBJ', obj_1, ..., obj_N``
    - ``'OBJ', eps_node_array``

**SRCJ / SRCM** — source parameters:
    - ``'SRCJ', src_1, ..., src_M``
"""

from typing import Dict, List, Tuple, Union, Any, Optional, TYPE_CHECKING

import numpy as np

from maxwellfdfd.base import (
    Axis, FT, GT, PML, PhysQ, EquationType,
    chkarg, istypesizeof,
)
from maxwellfdfd.diff import (
    MatrixEquation, create_divs, solve_eq_direct, solve_eq_iterative,
)

from .build_system import build_system
from .progmark import ProgMark
from .scalar3d import Scalar3d
from .utils import array2scalar
from .vec_utils import flip_vec, neg_vec, mult_vec
from .io_utils import write_input, fds

if TYPE_CHECKING:
    from maxwellfdfd.grid.grid3d import Grid3d
    from maxwellfdfd.base import Oscillation
    from .em_object import EMObject
    from maxwellfdfd.source.source import Source


def maxwell_run(
    *args: Any,
    inspect_only: bool = False,
    solveropts: Optional[Dict[str, Any]] = None,
) -> Tuple[
    Dict[Axis, Scalar3d],
    Dict[Axis, Scalar3d],
    List['EMObject'],
    List['Source'],
    Dict[str, Any],
]:
    """
    Run MaxwellFDFD.

    Constructs a simulation domain from given objects and sources, and
    launches a finite-difference frequency-domain (FDFD) solver to solve
    Maxwell's equations in the simulation domain.

    Parameters
    ----------
    *args
        Parameter groups (OSC, DOM, OBJ, SRCJ, SRCM), each beginning
        with a string marker. See module docstring for details.
    inspect_only : bool, optional
        If True, run without launching the solver — useful to inspect
        input arguments before expensive computation. Default is False.
    solveropts : dict, optional
        Solver options with keys:

        - ``'method'``: ``'direct'``, ``'iterative'``, ``'inputfile'``,
          ``'gpu'``, or ``'aws'``. Default ``'direct'``.
        - ``'maxit'``: max iterations for iterative solver. Default 1e6.
        - ``'tol'``: solver tolerance. Default 1e-6.
        - ``'eqtype'``: :class:`~maxwellfdfd.base.equation_type.EquationType`.
          Default ``EquationType(FT.E, GT.PRIM)``.
        - ``'pml'``: :class:`~maxwellfdfd.base.pml.PML` kind.
          Default ``PML.SC``.
        - ``'F0'``: initial guess — ``'zero'``, ``'rand'``, or field dict.
          Default ``'zero'``.
        - ``'returnAandb'``: return system matrix and RHS. Default False.
        - ``'returnDiv'``: return divergence operators. Default False.
        - ``'showstruct'``: visualize domain in inspect mode. Default True.
        - ``'filenamebase'``: base filename for ``'inputfile'`` method.

    Returns
    -------
    E_cell : dict
        ``{Axis: Scalar3d}`` E-field solution components.
    H_cell : dict
        ``{Axis: Scalar3d}`` H-field solution components.
    obj_array : list of EMObject
        Objects in the simulation domain.
    src_array : list of Source
        Sources in the simulation domain.
    extra : dict
        Additional results with keys:

        - ``'grid3d'``: the :class:`~maxwellfdfd.grid.grid3d.Grid3d`.
        - ``'J'``: ``{Axis: Scalar3d}`` electric current distribution.
        - ``'M'``: ``{Axis: Scalar3d}`` magnetic current distribution.
        - ``'eps'``: ``{Axis: Scalar3d}`` permittivity at grid nodes.
        - ``'mu'``: ``{Axis: Scalar3d}`` permeability at grid nodes.
        - ``'A'``, ``'b'``, ``'GfromF'``: (if ``returnAandb``).
        - ``'Dive'``, ``'Divm'``: (if ``returnDiv``).

    Examples
    --------
    >>> from maxwellfdfd.base import BC
    >>> from maxwellfdfd.shape import Box
    >>> from maxwellfdfd.source import PointSrc
    >>> from maxwellfdfd.io import maxwell_run
    >>> E, H, objs, srcs, extra = maxwell_run(
    ...     'OSC', 1e-9, 1550,
    ...     'DOM', ['Palik/SiO2', 'none'],
    ...         [[-700, 700], [-600, 600], [-200, 1700]], 20, BC.P, 200,
    ...     'OBJ',
    ...         ['Palik/SiO2', 'none'], Box([[-50, 50], [-50, 50], [-200, 1700]], [2, 2, 20]),
    ...     'SRCJ', PointSrc(Axis.X, [0, 0, 200]),
    ... )
    """
    DEFAULT_METHOD = 'direct'

    # ---- Set solver options with defaults ----
    opts: Dict[str, Any] = {}
    is_solveropts = solveropts is not None

    if not is_solveropts or 'showstruct' not in solveropts:
        opts['showstruct'] = True
    else:
        opts['showstruct'] = solveropts['showstruct']

    if not is_solveropts or 'method' not in solveropts:
        opts['method'] = DEFAULT_METHOD
    else:
        opts['method'] = solveropts['method']

    if is_solveropts and opts['method'] == 'inputfile':
        chkarg(
            'filenamebase' in solveropts,
            '"solveropts" should have "filenamebase" field.',
        )
        opts['filenamebase'] = solveropts['filenamebase']

    if not is_solveropts or 'maxit' not in solveropts:
        opts['maxit'] = int(1e6)
    else:
        chkarg(
            isinstance(solveropts['maxit'], (int, float)) and solveropts['maxit'] > 0,
            'solveropts.maxit should be positive.',
        )
        opts['maxit'] = int(solveropts['maxit'])

    if not is_solveropts or 'tol' not in solveropts:
        opts['tol'] = 1e-6
    else:
        chkarg(
            isinstance(solveropts['tol'], (int, float)) and solveropts['tol'] > 0,
            'solveropts.tol should be positive.',
        )
        opts['tol'] = float(solveropts['tol'])

    if not is_solveropts or 'eqtype' not in solveropts:
        opts['eqtype'] = EquationType(FT.E, GT.PRIM)
    else:
        chkarg(
            isinstance(solveropts['eqtype'], EquationType),
            'solveropts.eqtype should be instance of EquationType.',
        )
        opts['eqtype'] = solveropts['eqtype']

    if not is_solveropts or 'pml' not in solveropts:
        opts['pml'] = PML.SC
    else:
        chkarg(
            isinstance(solveropts['pml'], PML),
            'solveropts.pml should be instance of PML.',
        )
        opts['pml'] = solveropts['pml']

    if not is_solveropts or 'returnAandb' not in solveropts:
        opts['returnAandb'] = False
    else:
        chkarg(isinstance(solveropts['returnAandb'], bool),
               'solveropts.returnAandb should be bool.')
        opts['returnAandb'] = solveropts['returnAandb']

    if not is_solveropts or 'returnDiv' not in solveropts:
        opts['returnDiv'] = False
    else:
        chkarg(isinstance(solveropts['returnDiv'], bool),
               'solveropts.returnDiv should be bool.')
        opts['returnDiv'] = solveropts['returnDiv']

    if not is_solveropts or 'precond' not in solveropts:
        opts['precond'] = 'none'
    else:
        chkarg(solveropts['precond'] in ('none', 'ilu'),
               "solveropts.precond should be 'none' or 'ilu'.")
        opts['precond'] = solveropts['precond']

    if not is_solveropts or 'ilu_drop_tol' not in solveropts:
        opts['ilu_drop_tol'] = 1e-3
    else:
        opts['ilu_drop_tol'] = float(solveropts['ilu_drop_tol'])

    if not is_solveropts or 'ilu_fill_factor' not in solveropts:
        opts['ilu_fill_factor'] = 5.0
    else:
        opts['ilu_fill_factor'] = float(solveropts['ilu_fill_factor'])

    if not is_solveropts or 'algorithm' not in solveropts:
        opts['algorithm'] = 'bicgstab'
    else:
        chkarg(solveropts['algorithm'] in ('bicgstab', 'gmres'),
               "solveropts.algorithm should be 'bicgstab' or 'gmres'.")
        opts['algorithm'] = solveropts['algorithm']

    if not is_solveropts or 'gmres_restart' not in solveropts:
        opts['gmres_restart'] = 100
    else:
        opts['gmres_restart'] = int(solveropts['gmres_restart'])

    chkarg(len(args) > 0, 'first argument is not correct.')

    if inspect_only:
        print('maxwell_run begins (inspection only).')
    else:
        print('maxwell_run begins.')

    ge = opts['eqtype'].ge
    print(f'E-field grid type: {ge}')
    pm = ProgMark()

    # ---- Build the system ----
    (
        osc, grid3d, s_factor, eps, mu, J, M,
        obj_array, src_array, mat_array,
        eps_node, mu_node, isiso,
    ) = build_system(ge, opts['pml'], *args, progmark=pm)

    # ---- Handle F0 ----
    if not is_solveropts or 'F0' not in solveropts:
        opts['F0'] = 'zero'
    else:
        opts['F0'] = solveropts['F0']

    # ---- Initialize output containers ----
    E: Dict[Axis, np.ndarray] = {}
    H: Dict[Axis, np.ndarray] = {}
    extra: Dict[str, Any] = {}

    if inspect_only:
        # Inspect objects and sources
        if opts['showstruct']:
            try:
                from maxwellfdfd.vis import visobjsrc
                withpml = True
                visobjsrc(grid3d, obj_array, src_array, withpml)
            except (ImportError, AttributeError):
                print(
                    '\t(Visualization module not available; '
                    'skipping domain visualization.)'
                )
            pm.mark('domain visualization')

        # Visualize modal sources if present
        is_modalsrc = False
        for src in src_array:
            from maxwellfdfd.source import ModalSrc
            if isinstance(src, ModalSrc):
                is_modalsrc = True
                try:
                    from maxwellfdfd.vis import vis2d
                    E2d = src.E2d
                    cmax = max(
                        np.max(np.abs(E2d[w].array)) for w in Axis.elems()
                    )
                    for w in Axis.elems():
                        vis2d(E2d[w], obj_array, {'withabs': True, 'cmax': cmax})
                except (ImportError, AttributeError):
                    print('\t(Visualization module not available.)')
                    break

        if is_modalsrc:
            pm.mark('distributed source visualization')

        print('maxwell_run finishes (inspection only).\n')

    else:
        # ---- Solve ----
        if opts['method'] == 'inputfile':
            chkarg(
                isiso,
                "anisotropic materials are not supported for method == 'inputfile'.",
            )

            # data_original() returns (array, l_cell); extract just the array
            eps_node_array, _ = eps_node[Axis.X].data_original()

            write_input(
                opts.get('filenamebase', 'maxwell_input'),
                opts['eqtype'], osc, grid3d, s_factor,
                eps_node_array, eps, mu, J, M,
                opts['F0'], opts['tol'], opts['maxit'],
            )
            pm.mark('input file creation')
            print('maxwell_run finishes. (input file created)\n')

        elif opts['method'] == 'direct':
            E, H = solve_eq_direct(
                opts['eqtype'], opts['pml'], osc.in_omega0(),
                eps, mu, s_factor, J, M, grid3d,
            )
            pm.mark('solution calculation')
            print(f'solve for: {opts["eqtype"].f}-field')
            print('maxwell_run finishes.\n')

        elif opts['method'] == 'iterative':
            E, H, relres, iters, resvec = solve_eq_iterative(
                opts['maxit'], opts['tol'], opts['F0'],
                opts['eqtype'], opts['pml'], osc.in_omega0(),
                eps, mu, s_factor, J, M, grid3d,
                precond=opts.get('precond', 'none'),
                ilu_drop_tol=opts.get('ilu_drop_tol', 1e-3),
                ilu_fill_factor=opts.get('ilu_fill_factor', 5.0),
                algorithm=opts.get('algorithm', 'bicgstab'),
                gmres_restart=opts.get('gmres_restart', 100),
            )
            pm.mark('solution calculation')
            print(f'\t# of iteration steps = {iters}')
            print(f'\trelative residual error = {relres}')
            print(f'solve for: {opts["eqtype"].f}-field')
            print('maxwell_run finishes.\n')

        elif opts['method'] in ('gpu', 'aws'):
            # Apply spatial inversion for E-field based equation
            d_prim = flip_vec({w: grid3d.dl[(w, GT.DUAL)] for w in Axis.elems()})
            d_dual = flip_vec({w: grid3d.dl[(w, GT.PRIM)] for w in Axis.elems()})
            s_prim = flip_vec({w: s_factor[(w, GT.DUAL)] for w in Axis.elems()})
            s_dual = flip_vec({w: s_factor[(w, GT.PRIM)] for w in Axis.elems()})
            mu_flipped = flip_vec(mu)
            eps_flipped = flip_vec(eps)
            J_flipped = neg_vec(flip_vec(J))  # pseudovector

            ds_prim = mult_vec(d_prim, s_prim)
            ds_dual = mult_vec(d_dual, s_dual)
            F0 = {w: np.zeros(tuple(int(n) for n in grid3d.N)) for w in Axis.elems()}

            # fds expects separate d and s; since ds already combines them,
            # pass ds as d and ones as s (identity scaling).
            ones_vec = {w: np.ones(int(grid3d.N[int(w)])) for w in Axis.elems()}

            if opts['method'] == 'gpu':
                E, H, _ = fds(
                    osc.in_omega0(), ds_prim, ds_dual,
                    ones_vec, ones_vec,
                    mu_flipped, eps_flipped,
                    F0, J_flipped,
                    opts['maxit'], opts['tol'], 'plot',
                )
            else:  # 'aws'
                E, H, _ = fds(
                    osc.in_omega0(), ds_prim, ds_dual,
                    ones_vec, ones_vec,
                    mu_flipped, eps_flipped,
                    F0, J_flipped,
                    opts['maxit'], opts['tol'], 'none',
                )

            E = neg_vec(flip_vec(E))  # pseudovector
            J = neg_vec(flip_vec(J))  # pseudovector
            H = flip_vec(H)
            pm.mark('solution calculation')
            print(f'solve for: {opts["eqtype"].f}-field')
            print('maxwell_run finishes.\n')

    # ---- Return A and b if requested ----
    if opts['returnAandb']:
        eq = MatrixEquation(
            opts['eqtype'], opts['pml'], osc.in_omega0(),
            eps, mu, s_factor, J, M, grid3d,
        )
        extra['A'], extra['b'] = eq.matrix_op()
        _, _, extra['GfromF'] = eq.matrixfree_op()

    # ---- Return divergence operators if requested ----
    if opts['returnDiv']:
        Dive, Divm = create_divs(ge, s_factor, grid3d)
        extra['Dive'] = Dive
        extra['Divm'] = Divm

    # ---- Construct Scalar3d objects ----
    E_cell: Dict[Axis, Scalar3d] = {}
    H_cell: Dict[Axis, Scalar3d] = {}
    J_cell: Dict[Axis, Scalar3d] = {}
    M_cell: Dict[Axis, Scalar3d] = {}

    for w in Axis.elems():
        if E:  # non-empty
            E_cell[w] = array2scalar(E[w], grid3d, ge, FT.E, w, osc, PhysQ.E)
        if H:  # non-empty
            H_cell[w] = array2scalar(H[w], grid3d, ge, FT.H, w, osc, PhysQ.H)
        J_cell[w] = array2scalar(J[w], grid3d, ge, FT.E, w, osc, PhysQ.J)
        M_cell[w] = array2scalar(M[w], grid3d, ge, FT.H, w, osc, PhysQ.M)

    extra['grid3d'] = grid3d
    extra['J'] = J_cell
    extra['M'] = M_cell
    extra['eps'] = eps_node
    extra['mu'] = mu_node

    return E_cell, H_cell, obj_array, src_array, extra
