"""
Grid and PML utility functions for MaxwellFDFD.

This module provides functions for generating PML stretched-coordinate
s-factors and reading grid information from HDF5 files.
"""

from typing import Dict, Tuple, Union, TYPE_CHECKING

import numpy as np

from maxwellfdfd.base import (
    Axis, BC, GT, Sign, Oscillation, PhysUnit,
    chkarg, istypesizeof, expand2mat,
)

if TYPE_CHECKING:
    from maxwellfdfd.grid.grid3d import Grid3d


def _calc_s_factor(
    omega: complex,
    depth: np.ndarray,
    Lpml: float,
    m: float,
    lnR: float,
) -> np.ndarray:
    """
    Calculate the PML s-factor for points inside the PML region.

    Uses the standard polynomial grading for the conductivity ``sigma``,
    the scaling factor ``kappa``, and the auxiliary parameter ``a``.

    Parameters
    ----------
    omega : complex
        Angular frequency.
    depth : np.ndarray
        Depth of each point inside the PML (distance from PML interface).
    Lpml : float
        Total PML thickness.
    m : float
        Polynomial grading degree (must be >= 1).
    lnR : float
        Natural logarithm of the target reflectance.

    Returns
    -------
    np.ndarray
        Complex s-factor values.
    """
    sigma_max = -(m + 1) * lnR / (2 * Lpml)
    sigma = sigma_max * (depth / Lpml) ** m

    kappa_max = 1.0
    kappa = 1.0 + (kappa_max - 1) * (depth / Lpml) ** m

    ma = m
    amax = 0.0
    a = amax * (1.0 - depth / Lpml) ** ma

    return kappa + sigma / (a + 1j * omega)


def generate_s_factor(
    omega: complex,
    grid3d: 'Grid3d',
    m: Union[float, np.ndarray] = 4,
    R: Union[float, np.ndarray] = np.exp(-16),
) -> Dict[Tuple[Axis, GT], np.ndarray]:
    """
    Generate PML stretched-coordinate s-factors for the grid.

    For example, ``s_factor_cell[(Axis.X, GT.DUAL)]`` is the s-factor
    multiplied to E_x (or eps_ww).

    Parameters
    ----------
    omega : complex
        Angular frequency.
    grid3d : Grid3d
        The 3D grid providing PML interface locations and thicknesses.
    m : float or np.ndarray, optional
        Polynomial grading degrees. Can be scalar, length-3 vector, or
        3×2 matrix. Default is 4.
    R : float or np.ndarray, optional
        Target reflectances. Can be scalar, length-3 vector, or 3×2
        matrix. Default is ``exp(-16) ≈ 1e-7``.

    Returns
    -------
    dict
        ``{(Axis, GT): np.ndarray}`` mapping each (axis, grid-type) pair
        to a length-``Nw`` s-factor array.

    Raises
    ------
    ValueError
        If inputs fail validation.
    """
    chkarg(istypesizeof(omega, 'complex'), '"omega" should be complex.')
    chkarg(
        istypesizeof(grid3d, 'Grid3d'),
        '"grid3d" should be instance of Grid3d.',
    )

    # Expand m to 3x2 matrix [axis, sign]
    m_arr = expand2mat(m, Axis.count(), Sign.count())
    chkarg(
        np.all(m_arr >= 1),
        'element of "m" should be real and at least 1.0.',
    )

    # Expand R to 3x2 matrix
    R_arr = expand2mat(R, Axis.count(), Sign.count())
    chkarg(
        np.all(R_arr > 0) and np.all(R_arr <= 1),
        'element of "R" should be real and between 0.0 and 1.0.',
    )
    lnR = np.log(R_arr)

    s_factor_cell: Dict[Tuple[Axis, GT], np.ndarray] = {}

    for w in Axis.elems():
        wi = int(w)
        Nw = int(grid3d.N[wi])

        # PML interface locations and thicknesses for this axis
        lpml = grid3d.lpml[wi, :]  # [lpml_n, lpml_p]
        Lpml = grid3d.Lpml[wi, :]  # [Lpml_n, Lpml_p]

        for g in GT.elems():
            l = grid3d.l[(w, g)]  # length == Nw

            # Indices of locations inside PML
            ind_pml_n = l < lpml[int(Sign.N)]
            ind_pml_p = l > lpml[int(Sign.P)]

            s_factor = np.ones(Nw, dtype=complex)

            for s in Sign.elems():
                si = int(s)
                ind_pml = ind_pml_n if s == Sign.N else ind_pml_p
                if np.any(ind_pml):
                    depth = np.abs(lpml[si] - l[ind_pml])
                    s_factor[ind_pml] = _calc_s_factor(
                        omega, depth, Lpml[si], m_arr[wi, si], lnR[wi, si],
                    )

            s_factor_cell[(w, g)] = s_factor

    return s_factor_cell


def read_grid(filenamebase: str) -> Tuple['Grid3d', GT, Oscillation]:
    """
    Read grid information from an HDF5 input file.

    Parameters
    ----------
    filenamebase : str
        Base filename (without extension). The file ``<filenamebase>.h5``
        is read.

    Returns
    -------
    grid3d : Grid3d
        The reconstructed 3D grid.
    ge : GT
        The E-field grid type.
    osc : Oscillation
        The oscillation parameters.

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
            'h5py is required for read_grid(). Install it with: pip install h5py'
        ) from exc

    from maxwellfdfd.grid.grid3d import Grid3d

    inputfile = f'./{filenamebase}.h5'

    with h5py.File(inputfile, 'r') as f:
        wvlen = float(f['/lambda'][()])
        L0 = float(f['/L0'][()])
        Npml = f['/Npml'][()].T  # transpose for axis-major
        ge_idx = int(f['/ge'][()])
        bc_arr = f['/bc'][()].T  # transpose for axis-major

        # Read primary grid locations per axis
        lprim: Dict[Axis, np.ndarray] = {}
        for w in Axis.elems():
            dataset_name = f'/{w}_prim'
            lprim[w] = np.asarray(f[dataset_name][()]).flatten()

    # Convert grid type index to GT enum
    ge = GT.elems(ge_idx)

    # Convert bc array to BC enums (negative end per axis)
    bc = []
    for w in Axis.elems():
        wi = int(w)
        bc_val = int(bc_arr[wi, int(Sign.N)])
        bc.append(BC.elems(bc_val))

    unit = PhysUnit(L0)
    osc = Oscillation(wvlen, unit)
    grid3d = Grid3d(unit, lprim, Npml, bc)

    return grid3d, ge, osc
