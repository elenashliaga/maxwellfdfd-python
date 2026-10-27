"""
Scalar2d class for MaxwellFDFD.

This module provides the Scalar2d class, which represents a 2D array of
scalar values. It can represent a single directional component of a 3D
vector field (e.g., E_x of E) on a 2D slice of the 3D grid.
"""

from typing import Dict, List, Tuple, Union, Optional, TYPE_CHECKING

import numpy as np
from scipy.interpolate import interpn

from maxwellfdfd.base import (
    Axis, Dir, GT, PhysQ, Oscillation,
    chkarg, istypesizeof, isphysQcell,
)

if TYPE_CHECKING:
    from maxwellfdfd.grid.grid2d import Grid2d


class Scalar2d:
    """
    2D array of scalar values representing a field component on a slice.

    Scalar2d wraps a 2D NumPy array together with grid, oscillation, and
    physical-quantity metadata. It includes an ``intercept`` value that
    specifies where in the normal axis the 2D slice is taken.

    Parameters
    ----------
    array : np.ndarray
        2D complex array of size ``(Nh+gh, Nv+gv)``.
    grid2d : Grid2d
        The 2D grid on which the data is defined.
    gt_array : list of GT
        Length-2 list ``[gt_h, gt_v]`` specifying the grid type along the
        horizontal and vertical directions.
    osc : Oscillation
        Oscillation parameters.
    physQcell : PhysQ or list, optional
        Physical quantity representation. Defaults to ``PhysQ.ARBITRARY``.
    name : str, optional
        Name of this quantity. Defaults to ``''``.
    intercept : float, optional
        Intercept of the plane of this 2D data in the normal axis.
        Defaults to ``NaN``.

    Attributes
    ----------
    array : np.ndarray
        The underlying 2D data array.
    grid2d : Grid2d
        The associated 2D grid.
    gt_array : list of GT
        Grid types per direction ``[gt_h, gt_v]``.
    osc : Oscillation
        Oscillation parameters.
    physQcell : list
        Physical quantity cell ``[PhysQ, power]``.
    unitvalue : float
        Unit value of the physical quantity.
    name : str
        Name of this quantity.
    intercept : float
        Intercept of the slice in the normal axis.
    """

    def __init__(
        self,
        array: np.ndarray,
        grid2d: 'Grid2d',
        gt_array: List[GT],
        osc: Oscillation,
        physQcell: Union[PhysQ, List] = None,
        name: str = '',
        intercept: float = float('nan'),
    ) -> None:
        """
        Initialize Scalar2d.

        Raises
        ------
        ValueError
            If any argument fails validation.
        """
        chkarg(
            istypesizeof(grid2d, 'Grid2d'),
            '"grid2d" should be instance of Grid2d.',
        )
        self._grid2d: 'Grid2d' = grid2d

        # Validate gt_array: length Dir.count() with GT elements
        chkarg(
            isinstance(gt_array, list) and len(gt_array) == Dir.count()
            and all(isinstance(g, GT) for g in gt_array),
            f'"gt_array" should be length-{Dir.count()} list with GT as elements.',
        )
        self._gt_array: List[GT] = gt_array

        # Validate array size: N + int(gt_array) per direction
        N = np.array(grid2d.N) + np.array([int(g) for g in gt_array])
        chkarg(
            isinstance(array, np.ndarray) and array.ndim == 2
            and array.shape == tuple(N.astype(int)),
            f'"array" should be {int(N[0])}-by-{int(N[1])} array '
            f'with complex elements.',
        )
        self._array: np.ndarray = np.asarray(array, dtype=complex)

        chkarg(
            istypesizeof(osc, 'Oscillation'),
            '"osc" should be instance of Oscillation.',
        )
        self._osc: Oscillation = osc

        if physQcell is None:
            physQcell = PhysQ.ARBITRARY
        chkarg(
            isinstance(physQcell, PhysQ) or isphysQcell(physQcell),
            '"physQcell" should be instance of PhysQ, or cell array.',
        )
        if isinstance(physQcell, PhysQ):
            physQcell = [physQcell, 1]
        self._physQcell: List = physQcell
        self._unitvalue: float = osc.unit.value(self._physQcell)

        chkarg(isinstance(name, str), '"name" should be string.')
        self._name: str = name

        chkarg(
            np.isreal(intercept) or np.isnan(intercept),
            '"intercept" should be real.',
        )
        self._intercept: float = intercept

    # ------------------------------------------------------------------
    # Read-only properties
    # ------------------------------------------------------------------
    @property
    def array(self) -> np.ndarray:
        """Get the underlying 2D data array."""
        return self._array

    @property
    def grid2d(self) -> 'Grid2d':
        """Get the 2D grid."""
        return self._grid2d

    @property
    def gt_array(self) -> List[GT]:
        """Get the grid types per direction."""
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

    @property
    def intercept(self) -> float:
        """Get the intercept of the slice in the normal axis."""
        return self._intercept

    # ------------------------------------------------------------------
    # Location accessors
    # ------------------------------------------------------------------
    def lall(self) -> Dict[Dir, np.ndarray]:
        """Return all grid locations (including ghost points)."""
        return {
            d: self._grid2d.lall[(d, self._gt_array[int(d)])]
            for d in Dir.elems()
        }

    def lg(self) -> Dict[Dir, np.ndarray]:
        """Return grid locations including one layer of ghost points."""
        return {
            d: self._grid2d.lg[(d, self._gt_array[int(d)])]
            for d in Dir.elems()
        }

    def l(self) -> Dict[Dir, np.ndarray]:
        """Return the original (non-ghost) grid locations."""
        return {
            d: self._grid2d.l[(d, self._gt_array[int(d)])]
            for d in Dir.elems()
        }

    # ------------------------------------------------------------------
    # Data extraction methods
    # ------------------------------------------------------------------
    def data_expanded(self) -> Tuple[np.ndarray, Dict[Dir, np.ndarray]]:
        """Return the full (expanded) array and its locations."""
        return self._array, self.lall()

    def data_ghost_expanded(self) -> Tuple[np.ndarray, Dict[Dir, np.ndarray]]:
        """Return the array sliced to ghost-expanded size."""
        l_cell = self.lg()
        slices = tuple(
            slice(0, int(self._grid2d.N[int(d)] + 1)) for d in Dir.elems()
        )
        return self._array[slices], l_cell

    def data_original(self) -> Tuple[np.ndarray, Dict[Dir, np.ndarray]]:
        """Return the array at the original (non-ghost) grid points."""
        l_cell = self.l()
        idx: List[slice] = []
        for d in Dir.elems():
            Nd = int(self._grid2d.N[int(d)])
            if self._gt_array[int(d)] == GT.PRIM:
                idx.append(slice(0, Nd))
            else:  # GT.DUAL
                idx.append(slice(1, Nd + 1))
        return self._array[tuple(idx)], l_cell

    # ------------------------------------------------------------------
    # Value extraction with interpolation
    # ------------------------------------------------------------------
    def value(
        self,
        h: Optional[np.ndarray] = None,
        v: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Return field values along a Cartesian line in the 2D plane.

        At least one of ``h``, ``v`` must be a scalar (or ``None``), so
        that the requested points lie along a single Cartesian direction.

        Parameters
        ----------
        h, v : np.ndarray or None
            Coordinates along horizontal and vertical directions. ``None``
            means "use all available locations".

        Returns
        -------
        np.ndarray
            Interpolated field values at the requested points.
        """
        loc_raw: Dict[Dir, Optional[np.ndarray]] = {Dir.H: h, Dir.V: v}
        lavail = self.l()
        loc: Dict[Dir, np.ndarray] = {}
        num_len1 = 0

        for d in Dir.elems():
            coord_d = loc_raw[d]
            if coord_d is None or (isinstance(coord_d, np.ndarray) and coord_d.size == 0):
                coord_d = lavail[d]
            else:
                coord_d = np.sort(np.asarray(coord_d, dtype=float))
                chkarg(
                    self._grid2d.comp[int(d)].contains(coord_d),
                    f'"{d}" should be inside grid.',
                )
            loc[d] = coord_d
            if coord_d.size == 1:
                num_len1 += 1

        chkarg(num_len1 >= 1, 'Points should be along Cartesian direction.')

        lall = self.lall()
        ind: Dict[Dir, np.ndarray] = {}

        for d in Dir.elems():
            loc_min = loc[d][0]
            lall_d = lall[d]
            matches = np.where(np.isclose(lall_d, loc_min))[0]
            if matches.size > 0:
                indw_min = int(matches[0])
            else:
                below = np.where(lall_d < loc_min)[0]
                indw_min = int(below[-1]) if below.size > 0 else 0

            loc_max = loc[d][-1]
            matches = np.where(np.isclose(lall_d, loc_max))[0]
            if matches.size > 0:
                indw_max = int(matches[0])
            else:
                above = np.where(lall_d > loc_max)[0]
                indw_max = int(above[0]) if above.size > 0 else len(lall_d) - 1

            ind[d] = np.arange(indw_min, indw_max + 1)

        isindlen1 = [ind[Dir.H].size == 1, ind[Dir.V].size == 1]

        if all(isindlen1):
            return self._array[int(ind[Dir.H][0]), int(ind[Dir.V][0])]

        for i, d in enumerate(Dir.elems()):
            if isindlen1[i]:
                indw = int(ind[d][0])
                if indw >= 1:
                    ind[d] = np.array([indw - 1, indw])
                else:
                    ind[d] = np.array([indw, indw + 1])

        points = [lall[d][ind[d][0]:ind[d][-1] + 1] for d in Dir.elems()]
        V = self._array[
            int(ind[Dir.H][0]):int(ind[Dir.H][-1]) + 1,
            int(ind[Dir.V][0]):int(ind[Dir.V][-1]) + 1,
        ]
        xi = np.stack(np.meshgrid(
            loc[Dir.H], loc[Dir.V], indexing='ij'
        ), axis=-1).reshape(-1, 2)

        val = interpn(points, V, xi)
        return val.reshape(loc[Dir.H].size, loc[Dir.V].size).squeeze()
