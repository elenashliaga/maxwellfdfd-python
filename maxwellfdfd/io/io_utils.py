"""
File I/O utility functions for MaxwellFDFD.

This module provides functions for reading FDFD solution output files
(``read_output``), writing simulation input files (``write_input``), and
interfacing with the external lightlabsFDS iterative solver server
(``fds``).
"""

from typing import Dict, List, Tuple, Union, Optional, TYPE_CHECKING

import numpy as np

from maxwellfdfd.base import (
    Axis, FT, GT, PhysQ, EquationType, Oscillation,
    chkarg, istypesizeof,
)

from .utils import collapse_complex, expand_complex, cell2array, array2scalar
from .grid_utils import read_grid

if TYPE_CHECKING:
    from maxwellfdfd.grid.grid3d import Grid3d
    from .scalar3d import Scalar3d


# ======================================================================
# read_output — read E/H solution from HDF5 files
# ======================================================================
def read_output(
    filenamebase: str,
) -> Tuple[Dict[Axis, 'Scalar3d'], Dict[Axis, 'Scalar3d']]:
    """
    Read E- and H-field solutions from HDF5 output files.

    Reads ``<filenamebase>.E.h5`` and ``<filenamebase>.H.h5``, collapses
    the interleaved real/imaginary data into complex arrays, and
    constructs Scalar3d objects for each component.

    Parameters
    ----------
    filenamebase : str
        Base filename (without extension).

    Returns
    -------
    E_cell : dict
        ``{Axis: Scalar3d}`` for the E-field components.
    H_cell : dict
        ``{Axis: Scalar3d}`` for the H-field components.

    Raises
    ------
    ValueError
        If ``filenamebase`` is not a string.
    ImportError
        If ``h5py`` is not available.
    """
    chkarg(isinstance(filenamebase, str), '"filenamebase" should be string.')

    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            'h5py is required for read_output(). '
            'Install it with: pip install h5py'
        ) from exc

    # Read the grid
    grid3d, ge, osc = read_grid(filenamebase)

    efile = f'./{filenamebase}.E.h5'
    hfile = f'./{filenamebase}.H.h5'

    with h5py.File(efile, 'r') as fe, h5py.File(hfile, 'r') as fh:
        E_array = fe['/E'][()]  # shape (ri, w, i, j, k)
        H_array = fh['/H'][()]  # shape (ri, w, i, j, k)

    assert E_array.shape == H_array.shape, 'E and H arrays should have the same shape.'

    # Collapse interleaved real/imag: (ri, w, i, j, k) -> (w, i, j, k)
    E_complex = collapse_complex(E_array)  # leading singleton dim removed
    H_complex = collapse_complex(H_array)

    # Remove leading singleton dimension if present
    if E_complex.ndim == 4 and E_complex.shape[0] == 1:
        E_complex = E_complex[0]
    if H_complex.ndim == 4 and H_complex.shape[0] == 1:
        H_complex = H_complex[0]

    # Split into per-component arrays
    E_cell: Dict[Axis, 'Scalar3d'] = {}
    H_cell: Dict[Axis, 'Scalar3d'] = {}
    for w in Axis.elems():
        wi = int(w)
        E_cell[w] = array2scalar(
            E_complex[wi], grid3d, ge, FT.E, w, osc, PhysQ.E,
        )
        H_cell[w] = array2scalar(
            H_complex[wi], grid3d, ge, FT.H, w, osc, PhysQ.H,
        )

    return E_cell, H_cell


# ======================================================================
# write_input — write simulation input to HDF5 file
# ======================================================================
def write_input(
    filenamebase: str,
    eqtype: EquationType,
    osc: Oscillation,
    grid3d: 'Grid3d',
    s_factor_cell: Dict[Tuple[Axis, GT], np.ndarray],
    eps_node_array: np.ndarray,
    eps_cell: Dict[Axis, np.ndarray],
    mu_cell: Dict[Axis, np.ndarray],
    J_cell: Dict[Axis, np.ndarray],
    M_cell: Dict[Axis, np.ndarray],
    F0_cell: Union[str, Dict[Axis, np.ndarray]],
    tol: float,
    maxit: int,
) -> None:
    """
    Write FDFD simulation input to an HDF5 file.

    Writes all parameters needed by an external solver (grid, materials,
    sources, PML factors) to ``<filenamebase>.h5``.

    Parameters
    ----------
    filenamebase : str
        Base filename (without extension).
    eqtype : EquationType
        Equation type (field type + grid type).
    osc : Oscillation
        Oscillation parameters.
    grid3d : Grid3d
        The 3D grid.
    s_factor_cell : dict
        ``{(Axis, GT): np.ndarray}`` PML s-factors.
    eps_node_array : np.ndarray
        Permittivity at grid nodes (``Nx × Ny × Nz``).
    eps_cell : dict
        ``{Axis: np.ndarray}`` permittivity at E-field positions.
    mu_cell : dict
        ``{Axis: np.ndarray}`` permeability at H-field positions.
    J_cell : dict
        ``{Axis: np.ndarray}`` electric current sources.
    M_cell : dict
        ``{Axis: np.ndarray}`` magnetic current sources.
    F0_cell : str or dict
        Initial guess type (``'zero'``, ``'rand'``) or explicit field dict.
    tol : float
        Solver tolerance (positive).
    maxit : int
        Maximum solver iterations (positive integer).

    Raises
    ------
    ValueError
        If any argument fails validation.
    ImportError
        If ``h5py`` is not available.
    """
    # Validate arguments
    chkarg(isinstance(filenamebase, str), '"filenamebase" should be string.')
    chkarg(
        istypesizeof(eqtype, 'EquationType'),
        '"eqtype" should be instance of EquationType.',
    )
    chkarg(
        istypesizeof(osc, 'Oscillation'),
        '"osc" should be instance of Oscillation.',
    )
    chkarg(
        istypesizeof(grid3d, 'Grid3d'),
        '"grid3d" should be instance of Grid3d.',
    )
    chkarg(isinstance(tol, (int, float)) and tol > 0, '"tol" should be positive.')
    chkarg(isinstance(maxit, int) and maxit > 0, '"maxit" should be positive integer.')

    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            'h5py is required for write_input(). '
            'Install it with: pip install h5py'
        ) from exc

    filename = f'./{filenamebase}.h5'

    with h5py.File(filename, 'w') as f:
        # Compression level
        cl = 5

        # Write simple scalar fields
        f.create_dataset('/f', data=float(int(eqtype.f)))
        f.create_dataset('/ge', data=float(int(eqtype.ge)))
        f.create_dataset('/omega', data=float(osc.in_omega0()))
        f.create_dataset('/lambda', data=float(osc.in_L0()))
        f.create_dataset('/L0', data=float(osc.unit.value(PhysQ.L)))

        # Initial guess type
        GEN_ZERO, GEN_RAND, GEN_GIVEN = 0, 1, 2
        if isinstance(F0_cell, str):
            if F0_cell == 'zero':
                x0_type = GEN_ZERO
            else:
                assert F0_cell == 'rand'
                x0_type = GEN_RAND
        else:
            x0_type = GEN_GIVEN
        f.create_dataset('/x0_type', data=float(x0_type))

        # Grid parameters
        f.create_dataset('/N', data=np.asarray(grid3d.N, dtype=float))
        bc_vals = np.array([[int(grid3d.bc[int(w)])] for w in Axis.elems()], dtype=float)
        f.create_dataset('/bc', data=bc_vals)
        f.create_dataset('/Npml', data=np.asarray(grid3d.Npml, dtype=float))
        f.create_dataset('/tol', data=float(tol))
        f.create_dataset('/maxit', data=float(maxit))

        # Bloch phase factors
        e_ikL = np.exp(-1j * (grid3d.kBloch * grid3d.L))
        f.create_dataset('/e_ikL', data=expand_complex(np.asarray(e_ikL, dtype=complex)))

        # Per-axis grid and PML data
        for w in Axis.elems():
            wi = int(w)
            # Primary grid locations (for visualization)
            lprim_arr = grid3d.lall[(w, GT.PRIM)]
            f.create_dataset(f'/{w}_prim', data=lprim_arr)

            for g in GT.elems():
                g_str = str(g)[:4]  # 'prim' or 'dual'
                # Grid cell sizes
                dl_arr = grid3d.dl[(w, g)]
                f.create_dataset(
                    f'/d{w}_{g_str}',
                    data=expand_complex(np.asarray(dl_arr, dtype=complex)),
                )
                # PML s-factors
                s_arr = s_factor_cell[(w, g)]
                f.create_dataset(
                    f'/s{w}_{g_str}',
                    data=expand_complex(np.asarray(s_arr, dtype=complex)),
                )

        # Write field arrays (eps, J, M, F0)
        dims = (2, Axis.count()) + tuple(int(n) for n in grid3d.N)

        eps_array = expand_complex(
            cell2array(eps_cell, 3).astype(complex)
        )
        f.create_dataset('/eps', data=eps_array)

        J_array = expand_complex(
            cell2array(J_cell, 3).astype(complex)
        )
        f.create_dataset('/J', data=J_array)

        M_array = expand_complex(
            cell2array(M_cell, 3).astype(complex)
        )
        f.create_dataset('/M', data=M_array)

        if not isinstance(F0_cell, str):
            F0_array = expand_complex(
                cell2array(F0_cell, 3).astype(complex)
            )
            f.create_dataset('/F0', data=F0_array)


# ======================================================================
# fds — server-based iterative solver interface
# ======================================================================
def fds(
    omega: complex,
    d_prim: Dict[Axis, np.ndarray],
    d_dual: Dict[Axis, np.ndarray],
    s_prim: Dict[Axis, np.ndarray],
    s_dual: Dict[Axis, np.ndarray],
    mu: Dict[Axis, np.ndarray],
    epsilon: Dict[Axis, np.ndarray],
    E: Dict[Axis, np.ndarray],
    J: Dict[Axis, np.ndarray],
    max_iters: int,
    err_thresh: float,
    view_progress: str = 'none',
) -> Tuple[Dict[Axis, np.ndarray], Dict[Axis, np.ndarray], float]:
    """
    Interface to the lightlabsFDS server-based iterative solver.

    Solves the time-harmonic Maxwell's equation:

    .. math::

        \\nabla \\times \\mu^{-1} \\nabla \\times E - \\omega^2 \\epsilon E
        = -i \\omega J

    by communicating with an external lightlabsFDS server via FTP and
    HDF5 files.

    Parameters
    ----------
    omega : complex
        Angular frequency (complex scalar).
    d_prim, d_dual : dict
        Length factors for primary and dual grid points.
        ``{Axis: np.ndarray}`` with 3 elements.
    s_prim, s_dual : dict
        Scaling factors for primary and dual grid points.
        ``{Axis: np.ndarray}`` with 3 elements.
    mu, epsilon, E, J : dict
        Permeability, permittivity, initial E-field, and current sources.
        ``{Axis: np.ndarray}`` with 3D arrays.
    max_iters : int
        Maximum number of iterations (positive integer).
    err_thresh : float
        Convergence threshold (between 0 and 1).
    view_progress : str, optional
        Progress display mode: ``'plot'``, ``'text'``, or ``'none'``.
        Default is ``'none'``.

    Returns
    -------
    E : dict
        ``{Axis: np.ndarray}`` solved electric field.
    H : dict
        ``{Axis: np.ndarray}`` solved magnetic field.
    err : float
        Final residual error.

    Raises
    ------
    ValueError
        If inputs fail validation.
    ImportError
        If ``h5py`` is not available.
    NotImplementedError
        If the lightlabsFDS server is not reachable.

    Notes
    -----
    This function requires network access to a lightlabsFDS server.
    The server address and credentials are configured internally. If the
    server is unavailable, a :class:`NotImplementedError` is raised with
    guidance on how to proceed.
    """
    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            'h5py is required for fds(). Install it with: pip install h5py'
        ) from exc

    import time

    # ---- Validate inputs ----
    if not np.isscalar(omega) or omega == 0:
        raise ValueError('OMEGA must be a non-zero scalar.')

    for name, cell in [('d_prim', d_prim), ('d_dual', d_dual),
                       ('s_prim', s_prim), ('s_dual', s_dual),
                       ('mu', mu), ('epsilon', epsilon),
                       ('E', E), ('J', J)]:
        if not isinstance(cell, dict) or len(cell) != Axis.count():
            raise ValueError(
                f'{name.upper()} must be a dict with {Axis.count()} elements.'
            )

    if not isinstance(max_iters, int) or max_iters <= 0:
        raise ValueError('MAX_ITERS must be a positive integer.')

    if not isinstance(err_thresh, (int, float)) or err_thresh <= 0 or err_thresh >= 1:
        raise ValueError('ERR_THRESH must be a positive number between 0 and 1.')

    if view_progress not in ('plot', 'text', 'none'):
        raise ValueError("VIEW_PROGRESS must be 'plot', 'text', or 'none'.")

    # ---- Construct HDF5 input file ----
    timestamp = time.strftime('%Y%m%d%H%M%S')
    filename = f'.fds.{timestamp}.h5'
    status_file = f'{filename}.status'
    out_file = f'{filename}.out'

    def _write_field(file, name, field_cell):
        """Write a 3-component field dict to HDF5."""
        for w in Axis.elems():
            wi = int(w)
            file.create_dataset(f'{name}_{w}', data=np.asarray(field_cell[w], dtype=complex))

    with h5py.File(filename, 'w') as file:
        file.create_dataset('omega', data=complex(omega))
        _write_field(file, 'd_prim', d_prim)
        _write_field(file, 'd_dual', d_dual)
        _write_field(file, 's_prim', s_prim)
        _write_field(file, 's_dual', s_dual)
        _write_field(file, 'mu', mu)
        _write_field(file, 'epsilon', epsilon)
        _write_field(file, 'E', E)
        _write_field(file, 'J', J)
        file.create_dataset('max_iters', data=max_iters)
        file.create_dataset('err_thresh', data=float(err_thresh))

    # ---- Server communication ----
    # NOTE: The lightlabsFDS server communication is environment-specific.
    # The original MATLAB code uses FTP to upload/download files.
    # This Python conversion uses ftplib but the server must be configured.
    servername = 'lightlabsFDS-server'  # configurable

    try:
        import ftplib
    except ImportError:
        raise ImportError(
            'ftplib is required for fds() server communication.'
        )

    try:
        ftp = ftplib.FTP(servername, 'fds-user', 'maxwell')
        with open(filename, 'rb') as f:
            ftp.storbinary(f'STOR {filename}', f)
    except Exception:
        raise NotImplementedError(
            f'Could not connect to the lightlabsFDS server ({servername}). '
            'The server-based solver requires network access to a configured '
            'lightlabsFDS instance. Consider using solve_eq_direct or '
            'solve_eq_iterative for local solving instead.'
        )

    import os
    os.remove(filename)

    # ---- Poll for results ----
    err_len = 0
    err = float('inf')

    while True:
        # Check if simulation completed
        files = ftp.nlst()
        sim_done = out_file in files

        if status_file in files:
            try:
                with open(status_file, 'wb') as f:
                    ftp.retrbinary(f'RETR {status_file}', f.write)
                with h5py.File(status_file, 'r') as sf:
                    err_arr = sf['/err'][()]
                if len(err_arr) > err_len:
                    err_len = len(err_arr)
                    err = float(err_arr[-1])
                    if view_progress == 'text':
                        print(f'  iteration progress: err = {err}')
            except Exception:
                pass  # partial read; retry next cycle

        if sim_done:
            break
        time.sleep(1)

    # ---- Download and parse results ----
    with open(out_file, 'wb') as f:
        ftp.retrbinary(f'RETR {out_file}', f.write)
    ftp.quit()

    E_result: Dict[Axis, np.ndarray] = {}
    H_result: Dict[Axis, np.ndarray] = {}

    with h5py.File(out_file, 'r') as of:
        for w in Axis.elems():
            E_result[w] = np.asarray(of[f'E_{w}'][()])
            H_result[w] = np.asarray(of[f'H_{w}'][()])

    os.remove(out_file)
    if os.path.exists(status_file):
        os.remove(status_file)

    return E_result, H_result, err
