"""
Scalar3d class for MaxwellFDFD.

This module provides the Scalar3d class, which represents a 3D array of
scalar values. It can represent a single directional component of a 3D
vector field (e.g., E_x of the electric field E).

The fields are always evaluated at the primary grid points
(grid kind = GT.PRIM) unless a different ``gt_array`` is specified.
"""

from typing import Dict, List, Tuple, Union, Optional, TYPE_CHECKING

import numpy as np
from scipy.interpolate import interpn

from maxwellfdfd.base import (
    Axis, GT, PhysQ, Oscillation,
    chkarg, istypesizeof, isphysQcell,
)

if TYPE_CHECKING:
    from maxwellfdfd.grid.grid3d import Grid3d


class Scalar3d:
    """
    3D array of scalar values representing a field component.

    Scalar3d wraps a 3D NumPy array together with grid, oscillation, and
    physical-quantity metadata. It supports ghost-point expansion,
    data extraction at original grid points, and interpolation along
    Cartesian lines.

    Parameters
    ----------
    array : np.ndarray
        3D complex array of size ``(Nx+gx, Ny+gy, Nz+gz)`` where
        ``gw = int(gt_array[w])`` (0 for primary, 1 for dual).
    grid3d : Grid3d
        The 3D grid on which the data is defined.
    gt_array : list of GT
        Length-3 list ``[gt_x, gt_y, gt_z]`` specifying the grid type
        along each axis.
    osc : Oscillation
        Oscillation parameters (wavelength, frequency).
    physQcell : PhysQ or list, optional
        Physical quantity representation. Defaults to ``PhysQ.ARBITRARY``.
    name : str, optional
        Name of this quantity (e.g., ``"E_x"``). Defaults to ``''``.

    Attributes
    ----------
    array : np.ndarray
        The underlying 3D data array.
    grid3d : Grid3d
        The associated 3D grid.
    gt_array : list of GT
        Grid types per axis.
    osc : Oscillation
        Oscillation parameters.
    physQcell : list
        Physical quantity cell ``[PhysQ, power]``.
    unitvalue : float
        Unit value of the physical quantity.
    name : str
        Name of this quantity.

    Examples
    --------
    >>> from maxwellfdfd.base import Axis, GT, PhysQ, Oscillation, PhysUnit
    >>> from maxwellfdfd.grid import Grid3d
    >>> from maxwellfdfd.io import Scalar3d
    >>> unit = PhysUnit(1e-9)
    >>> osc = Oscillation(1550, unit)
    >>> # ... build grid3d ...
    >>> # scalar = Scalar3d(data, grid3d, [GT.PRIM]*3, osc, PhysQ.E, 'E_x')
    """

    def __init__(
        self,
        array: np.ndarray,
        grid3d: 'Grid3d',
        gt_array: List[GT],
        osc: Oscillation,
        physQcell: Union[PhysQ, List] = None,
        name: str = '',
    ) -> None:
        """
        Initialize Scalar3d.

        Parameters
        ----------
        array : np.ndarray
            3D complex array.
        grid3d : Grid3d
            The 3D grid.
        gt_array : list of GT
            Length-3 list of grid types per axis.
        osc : Oscillation
            Oscillation parameters.
        physQcell : PhysQ or list, optional
            Physical quantity. Defaults to ``PhysQ.ARBITRARY``.
        name : str, optional
            Quantity name. Defaults to ``''``.

        Raises
        ------
        ValueError
            If any argument fails validation.
        """
        chkarg(
            istypesizeof(grid3d, 'Grid3d'),
            '"grid3d" should be instance of Grid3d.',
        )
        self._grid3d: 'Grid3d' = grid3d

        # Validate gt_array
        chkarg(
            isinstance(gt_array, list) and len(gt_array) == Axis.count()
            and all(isinstance(g, GT) for g in gt_array),
            f'"gt_array" should be length-{Axis.count()} list with GT as elements.',
        )
        self._gt_array: List[GT] = gt_array

        # Validate array size: N + int(gt_array) per axis
        N = grid3d.N + np.array([int(g) for g in gt_array])
        chkarg(
            isinstance(array, np.ndarray) and array.ndim == 3
            and array.shape == tuple(N.astype(int)),
            f'"array" should be {int(N[0])}-by-{int(N[1])}-by-{int(N[2])} '
            f'array with complex elements.',
        )
        self._array: np.ndarray = np.asarray(array, dtype=complex)

        chkarg(
            istypesizeof(osc, 'Oscillation'),
            '"osc" should be instance of Oscillation.',
        )
        self._osc: Oscillation = osc

        # Handle physQcell
        if physQcell is None:
            physQcell = PhysQ.ARBITRARY
        chkarg(
            isinstance(physQcell, PhysQ) or isphysQcell(physQcell),
            '"physQcell" should be instance of PhysQ, or cell array '
            '{PhysQ, int; PhysQ, int; ...}.',
        )
        if isinstance(physQcell, PhysQ):
            # MATLAB {physQcell, 1} is a 1x2 cell (2-D); keep the same shape
            # here so isphysQcell() and PhysUnit.value() accept it.
            physQcell = [[physQcell, 1]]
        self._physQcell: List = physQcell
        self._unitvalue: float = osc.unit.value(self._physQcell)

        chkarg(isinstance(name, str), '"name" should be string.')
        self._name: str = name

    # ------------------------------------------------------------------
    # Read-only properties
    # ------------------------------------------------------------------
    @property
    def array(self) -> np.ndarray:
        """Get the underlying 3D data array."""
        return self._array

    @property
    def grid3d(self) -> 'Grid3d':
        """Get the 3D grid."""
        return self._grid3d

    @property
    def gt_array(self) -> List[GT]:
        """Get the grid types per axis."""
        return self._gt_array

    @property
    def osc(self) -> Oscillation:
        """Get the oscillation parameters."""
        return self._osc

    @property
    def physQcell(self) -> List:
        """Get the physical quantity cell."""
        return self._physQcell

    @property
    def unitvalue(self) -> float:
        """Get the unit value of the physical quantity."""
        return self._unitvalue

    @property
    def name(self) -> str:
        """Get the name of this quantity."""
        return self._name

    # ------------------------------------------------------------------
    # Location accessors
    # ------------------------------------------------------------------
    def lall(self) -> Dict[Axis, np.ndarray]:
        """
        Return all grid locations (including ghost points) for this scalar.

        Returns
        -------
        dict
            ``{Axis: np.ndarray}`` mapping each axis to its expanded
            location array.
        """
        return {
            w: self._grid3d.lall[(w, self._gt_array[int(w)])]
            for w in Axis.elems()
        }

    def lg(self) -> Dict[Axis, np.ndarray]:
        """
        Return grid locations including one layer of ghost points.

        Returns
        -------
        dict
            ``{Axis: np.ndarray}`` mapping each axis to its ghost-expanded
            location array.
        """
        return {
            w: self._grid3d.lg[(w, self._gt_array[int(w)])]
            for w in Axis.elems()
        }

    def l(self) -> Dict[Axis, np.ndarray]:
        """
        Return the original (non-ghost) grid locations for this scalar.

        Returns
        -------
        dict
            ``{Axis: np.ndarray}`` mapping each axis to its original
            location array.
        """
        return {
            w: self._grid3d.l[(w, self._gt_array[int(w)])]
            for w in Axis.elems()
        }

    # ------------------------------------------------------------------
    # Data extraction methods
    # ------------------------------------------------------------------
    def data_expanded(self) -> Tuple[np.ndarray, Dict[Axis, np.ndarray]]:
        """
        Return the full (expanded) array and its locations.

        Returns
        -------
        array : np.ndarray
            The full data array including ghost points.
        l_cell : dict
            ``{Axis: np.ndarray}`` of expanded locations (same as
            :meth:`lall`).
        """
        return self._array, self.lall()

    def data_ghost_expanded(self) -> Tuple[np.ndarray, Dict[Axis, np.ndarray]]:
        """
        Return the array sliced to ghost-expanded size.

        Returns
        -------
        array : np.ndarray
            Data sliced to ``(Nw+1)`` along each axis.
        l_cell : dict
            ``{Axis: np.ndarray}`` of ghost-expanded locations.
        """
        l_cell = self.lg()
        slices = tuple(
            slice(0, int(self._grid3d.N[int(w)] + 1)) for w in Axis.elems()
        )
        return self._array[slices], l_cell

    def data_original(self) -> Tuple[np.ndarray, Dict[Axis, np.ndarray]]:
        """
        Return the array at the original (non-ghost) grid points.

        For primary axes, the first ``N`` points are taken; for dual axes,
        points ``1`` through ``N`` (0-based) are taken.

        Returns
        -------
        array : np.ndarray
            Data at original grid points.
        l_cell : dict
            ``{Axis: np.ndarray}`` of original locations.
        """
        l_cell = self.l()
        idx: List[Union[slice, np.ndarray]] = []
        for w in Axis.elems():
            Nw = int(self._grid3d.N[int(w)])
            if self._gt_array[int(w)] == GT.PRIM:
                idx.append(slice(0, Nw))
            else:  # GT.DUAL
                idx.append(slice(1, Nw + 1))
        return self._array[tuple(idx)], l_cell

    # ------------------------------------------------------------------
    # Value extraction with interpolation
    # ------------------------------------------------------------------
    def value(
        self,
        x: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        z: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Return field values along a Cartesian line.

        At least two of ``x``, ``y``, ``z`` must be scalars (or ``None``),
        so that the requested points lie along a single Cartesian direction.

        Parameters
        ----------
        x, y, z : np.ndarray or None
            Coordinates along each axis. ``None`` means "use all available
            locations" for that axis.

        Returns
        -------
        np.ndarray
            Interpolated field values at the requested points.

        Raises
        ------
        ValueError
            If the points do not lie along a Cartesian direction.
        """
        loc_raw: Dict[Axis, Optional[np.ndarray]] = {Axis.X: x, Axis.Y: y, Axis.Z: z}
        lavail = self.l()
        loc: Dict[Axis, np.ndarray] = {}
        num_len1 = 0

        for w in Axis.elems():
            coord_w = loc_raw[w]
            if coord_w is None or (isinstance(coord_w, np.ndarray) and coord_w.size == 0):
                coord_w = lavail[w]
            else:
                coord_w = np.sort(np.asarray(coord_w, dtype=float))
                chkarg(
                    self._grid3d.comp[int(w)].contains(coord_w),
                    f'"{w}" should be inside grid.',
                )
            loc[w] = coord_w
            if coord_w.size == 1:
                num_len1 += 1

        chkarg(num_len1 >= 2, 'Points should be along Cartesian direction.')

        lall = self.lall()
        ind: Dict[Axis, np.ndarray] = {}

        for w in Axis.elems():
            loc_min = loc[w][0]
            lall_w = lall[w]
            # Find index of loc_min in lall_w
            matches = np.where(np.isclose(lall_w, loc_min))[0]
            if matches.size > 0:
                indw_min = int(matches[0])
            else:
                below = np.where(lall_w < loc_min)[0]
                indw_min = int(below[-1]) if below.size > 0 else 0

            loc_max = loc[w][-1]
            matches = np.where(np.isclose(lall_w, loc_max))[0]
            if matches.size > 0:
                indw_max = int(matches[0])
            else:
                above = np.where(lall_w > loc_max)[0]
                indw_max = int(above[0]) if above.size > 0 else len(lall_w) - 1

            ind[w] = np.arange(indw_min, indw_max + 1)

        isindlen1 = [ind[Axis.X].size == 1, ind[Axis.Y].size == 1, ind[Axis.Z].size == 1]

        if all(isindlen1):
            return self._array[int(ind[Axis.X][0]), int(ind[Axis.Y][0]), int(ind[Axis.Z][0])]

        # Expand single-element index ranges for interpolation
        for i, w in enumerate(Axis.elems()):
            if isindlen1[i]:
                indw = int(ind[w][0])
                if indw >= 1:
                    ind[w] = np.array([indw - 1, indw])
                else:
                    ind[w] = np.array([indw, indw + 1])

        # Build interpolation grid and query points
        points = [lall[w][ind[w][0]:ind[w][-1] + 1] for w in Axis.elems()]
        V = self._array[
            int(ind[Axis.X][0]):int(ind[Axis.X][-1]) + 1,
            int(ind[Axis.Y][0]):int(ind[Axis.Y][-1]) + 1,
            int(ind[Axis.Z][0]):int(ind[Axis.Z][-1]) + 1,
        ]
        xi = np.stack(np.meshgrid(
            loc[Axis.X], loc[Axis.Y], loc[Axis.Z], indexing='ij'
        ), axis=-1).reshape(-1, 3)

        val = interpn(points, V, xi)
        return val.reshape(
            loc[Axis.X].size, loc[Axis.Y].size, loc[Axis.Z].size
        ).squeeze()
