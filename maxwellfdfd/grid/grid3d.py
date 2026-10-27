"""
Grid3d class for 3D staggered grids.

This module implements the Grid3d class which represents a 3D staggered grid
used in finite-difference frequency-domain (FDFD) simulations. The grid is
composed of three 1D grids (one for each Cartesian axis).
"""

from typing import Dict, List, Tuple, Union
import numpy as np

from maxwellfdfd.base import Axis, BC, GT, Sign, PhysUnit, expand2mat, expand2row
from .grid1d import Grid1d


class Grid3d:
    """
    3D staggered grid for finite-difference discretization.
    
    Grid3d contains all information related to the 3D Yee grid. It is composed
    of three Grid1d instances, one for each Cartesian axis (x, y, z). The grid
    does not include frequency-dependent physical quantities such as omega,
    eps, mu, and PML s-factors.
    
    Parameters
    ----------
    unit : PhysUnit
        Physical unit instance for length conversions.
    lprim_cell : Dict[Axis, np.ndarray]
        Dictionary mapping each axis to its primary grid vertex locations.
        Each array must be monotonically increasing with at least 2 elements.
    Npml : int, float, or np.ndarray, optional
        PML cell counts. Can be:
        - Scalar: same count for all axes and both ends
        - Length-3 vector: different counts per axis, same for both ends
        - 3x2 matrix: different counts per axis and per end
        Default is 0 (no PML).
    bc : BC or list of BC, optional
        Boundary conditions. Can be:
        - Scalar: same BC for all axes
        - Length-3 vector: different BC per axis
        Default is BC.P (periodic).
    
    Attributes
    ----------
    comp : List[Grid1d]
        Component 1D grids [x, y, z].
    unit : PhysUnit
        Physical unit instance.
    unitvalue : float
        Unit value of length.
    l : Dict[Tuple[Axis, GT], np.ndarray]
        Grid locations {primary, dual} for each axis.
    dl : Dict[Tuple[Axis, GT], np.ndarray]
        Grid cell sizes {primary, dual} for each axis.
    bc : List[BC]
        Boundary conditions [bc_x, bc_y, bc_z].
    N : np.ndarray
        Number of grid cells [Nx, Ny, Nz].
    Ncell : List[List[int]]
        Cell counts as list of lists for function arguments.
    Ntot : int
        Total number of cells (Nx * Ny * Nz).
    L : np.ndarray
        Domain size [Lx, Ly, Lz].
    Npml : np.ndarray
        PML cell counts [[Npml_xn, Npml_xp], [Npml_yn, Npml_yp], [Npml_zn, Npml_zp]].
    lpml : np.ndarray
        PML interface locations.
    Lpml : np.ndarray
        PML thicknesses.
    center : np.ndarray
        Center of grid without PML [center_x, center_y, center_z].
    kBloch : np.ndarray
        Bloch wave vector [kBloch_x, kBloch_y, kBloch_z].
    
    Notes
    -----
    The grid uses 0-based indexing for all arrays, unlike MATLAB's 1-based
    indexing. Cell arrays from MATLAB are converted to Python dictionaries
    with tuple keys (Axis, GT).
    
    Examples
    --------
    >>> from maxwellfdfd.base import Axis, BC, PhysUnit
    >>> unit = PhysUnit()
    >>> lprim_cell = {
    ...     Axis.X: np.array([0.0, 0.5, 1.0]),
    ...     Axis.Y: np.array([0.0, 0.5, 1.0]),
    ...     Axis.Z: np.array([0.0, 0.5, 1.0])
    ... }
    >>> grid = Grid3d(unit, lprim_cell)
    >>> grid.N
    array([2, 2, 2])
    >>> grid.Ntot
    8
    """
    
    def __init__(
        self,
        unit: PhysUnit,
        lprim_cell: Dict[Axis, np.ndarray],
        Npml: Union[int, float, np.ndarray] = 0,
        bc: Union[BC, List[BC]] = BC.P
    ):
        """Initialize Grid3d instance."""
        # Validate unit
        if not isinstance(unit, PhysUnit):
            raise ValueError('"unit" should be instance of PhysUnit.')
        
        # Validate lprim_cell
        if not isinstance(lprim_cell, dict):
            raise ValueError('"lprim_cell" should be a dictionary.')
        if len(lprim_cell) != Axis.count():
            raise ValueError(
                f'"lprim_cell" should have {Axis.count()} elements (one for each axis).'
            )
        for axis in Axis.elems():
            if axis not in lprim_cell:
                raise ValueError(f'"lprim_cell" missing axis {axis}.')
            lprim_array = np.asarray(lprim_cell[axis])
            if lprim_array.ndim != 1:
                raise ValueError(
                    f'"lprim_cell[{axis}]" should be 1D array with real elements.'
                )
            if len(lprim_array) < 2:
                raise ValueError(
                    f'"lprim_cell[{axis}]" should have at least 2 elements.'
                )
            if not np.all(np.isreal(lprim_array)):
                raise ValueError(
                    f'"lprim_cell[{axis}]" should have real elements.'
                )
        
        # Process Npml
        Npml = np.asarray(Npml)
        if Npml.ndim == 0:
            # Scalar: expand to 3x2 matrix
            Npml = np.full((Axis.count(), Sign.count()), int(Npml))
        elif Npml.ndim == 1:
            # Vector: expand to 3x2 matrix
            if len(Npml) == Axis.count():
                # Different per axis, same for both ends
                Npml = np.tile(Npml.reshape(-1, 1), (1, Sign.count()))
            elif len(Npml) == Sign.count():
                # Same for all axes, different per end
                Npml = np.tile(Npml, (Axis.count(), 1))
            else:
                raise ValueError(
                    f'"Npml" vector should have length {Axis.count()} or {Sign.count()}.'
                )
        elif Npml.ndim == 2:
            # Matrix: should be 3x2
            if Npml.shape != (Axis.count(), Sign.count()):
                raise ValueError(
                    f'"Npml" matrix should be {Axis.count()}x{Sign.count()}.'
                )
        else:
            raise ValueError('"Npml" should be scalar, vector, or matrix.')
        
        # Ensure Npml is integer
        if not np.issubdtype(Npml.dtype, np.integer):
            Npml = Npml.astype(int)
        
        # Process bc
        if isinstance(bc, BC):
            # Scalar: expand to length-3 vector
            bc_list = [bc] * Axis.count()
        elif isinstance(bc, list):
            if len(bc) != Axis.count():
                raise ValueError(f'"bc" list should have length {Axis.count()}.')
            bc_list = bc
        else:
            raise ValueError('"bc" should be BC instance or list of BC instances.')
        
        # Validate bc_list
        for i, bc_i in enumerate(bc_list):
            if not isinstance(bc_i, BC):
                raise ValueError(f'Element {i} of "bc" should be BC instance.')
        
        # Set comp (component 1D grids)
        self._comp = []
        for axis in Axis.elems():
            grid1d = Grid1d(
                axis,
                unit,
                lprim_cell[axis],
                Npml[int(axis), :],
                bc_list[int(axis)]
            )
            self._comp.append(grid1d)
    
    @property
    def comp(self) -> List[Grid1d]:
        """Get component 1D grids [x, y, z]."""
        return self._comp
    
    @property
    def unit(self) -> PhysUnit:
        """Get the physical unit instance."""
        return self._comp[0].unit
    
    @property
    def unitvalue(self) -> float:
        """Get the unit value of length."""
        return self._comp[0].unitvalue
    
    @property
    def l(self) -> Dict[Tuple[Axis, GT], np.ndarray]:
        """
        Get grid locations {primary, dual} for each axis.
        
        Returns
        -------
        Dict[Tuple[Axis, GT], np.ndarray]
            Dictionary with (Axis, GT) tuple keys, containing grid
            locations for each axis and grid type.
        """
        result = {}
        for axis in Axis.elems():
            for gt in GT.elems():
                result[(axis, gt)] = self._comp[int(axis)].l[gt]
        return result
    
    @property
    def lg(self) -> Dict[Tuple[Axis, GT], np.ndarray]:
        """
        Get grid locations with ghost vertices for each axis.
        
        Returns
        -------
        Dict[Tuple[Axis, GT], np.ndarray]
            Dictionary with (Axis, GT) tuple keys, containing grid
            locations with ghost vertices.
        """
        result = {}
        for axis in Axis.elems():
            for gt in GT.elems():
                result[(axis, gt)] = self._comp[int(axis)].lg[gt]
        return result
    
    @property
    def lall(self) -> Dict[Tuple[Axis, GT], np.ndarray]:
        """
        Get grid locations with all vertices for each axis.
        
        Returns
        -------
        Dict[Tuple[Axis, GT], np.ndarray]
            Dictionary with (Axis, GT) tuple keys, containing grid
            locations with all vertices including extended dual vertices.
        """
        result = {}
        for axis in Axis.elems():
            for gt in GT.elems():
                result[(axis, gt)] = self._comp[int(axis)].lall[gt]
        return result
    
    @property
    def bound(self) -> np.ndarray:
        """
        Get grid boundaries for each axis.
        
        Returns
        -------
        np.ndarray
            3x2 array of boundary locations [[x_min, x_max], [y_min, y_max], [z_min, z_max]].
        """
        result = np.full((Axis.count(), Sign.count()), np.nan)
        for axis in Axis.elems():
            result[int(axis), :] = self._comp[int(axis)].bound
        return result
    
    @property
    def dl(self) -> Dict[Tuple[Axis, GT], np.ndarray]:
        """
        Get grid cell sizes {primary, dual} for each axis.
        
        Returns
        -------
        Dict[Tuple[Axis, GT], np.ndarray]
            Dictionary with (Axis, GT) tuple keys, containing grid
            cell sizes for each axis and grid type.
        """
        result = {}
        for axis in Axis.elems():
            for gt in GT.elems():
                result[(axis, gt)] = self._comp[int(axis)].dl[gt]
        return result
    
    @property
    def bc(self) -> List[BC]:
        """Get boundary conditions [bc_x, bc_y, bc_z]."""
        return [self._comp[int(axis)].bc for axis in Axis.elems()]
    
    @property
    def N(self) -> np.ndarray:
        """
        Get number of grid cells [Nx, Ny, Nz].
        
        Returns
        -------
        np.ndarray
            Length-3 array of cell counts.
        """
        return np.array([self._comp[int(axis)].N for axis in Axis.elems()])
    
    @property
    def Ncell(self) -> List[List[int]]:
        """
        Get cell counts as list of lists for function arguments.
        
        Returns
        -------
        List[List[int]]
            List of lists [[Nx], [Ny], [Nz]].
        """
        return [[self._comp[int(axis)].N] for axis in Axis.elems()]
    
    @property
    def Ntot(self) -> int:
        """
        Get total number of cells.
        
        Returns
        -------
        int
            Nx * Ny * Nz.
        """
        return np.prod(self.N)
    
    @property
    def L(self) -> np.ndarray:
        """
        Get domain size [Lx, Ly, Lz].
        
        Returns
        -------
        np.ndarray
            Length-3 array of domain sizes.
        """
        return np.array([self._comp[int(axis)].L for axis in Axis.elems()])
    
    @property
    def Npml(self) -> np.ndarray:
        """
        Get PML cell counts.
        
        Returns
        -------
        np.ndarray
            3x2 array of PML cell counts [[Npml_xn, Npml_xp], [Npml_yn, Npml_yp], [Npml_zn, Npml_zp]].
        """
        result = np.full((Axis.count(), Sign.count()), np.nan)
        for axis in Axis.elems():
            result[int(axis), :] = self._comp[int(axis)].Npml
        return result
    
    @property
    def lpml(self) -> np.ndarray:
        """
        Get PML interface locations.
        
        Returns
        -------
        np.ndarray
            3x2 array of PML interface locations.
        """
        result = np.full((Axis.count(), Sign.count()), np.nan)
        for axis in Axis.elems():
            result[int(axis), :] = self._comp[int(axis)].lpml
        return result
    
    @property
    def Lpml(self) -> np.ndarray:
        """
        Get PML thicknesses.
        
        Returns
        -------
        np.ndarray
            3x2 array of PML thicknesses.
        """
        result = np.full((Axis.count(), Sign.count()), np.nan)
        for axis in Axis.elems():
            result[int(axis), :] = self._comp[int(axis)].Lpml
        return result
    
    @property
    def center(self) -> np.ndarray:
        """
        Get center of grid without PML.
        
        Returns
        -------
        np.ndarray
            Length-3 array of grid centers [center_x, center_y, center_z].
        """
        return np.array([self._comp[int(axis)].center for axis in Axis.elems()])
    
    @property
    def kBloch(self) -> np.ndarray:
        """
        Get Bloch wave vector.
        
        Returns
        -------
        np.ndarray
            Length-3 array of Bloch wave vectors [kBloch_x, kBloch_y, kBloch_z].
        """
        return np.array([self._comp[int(axis)].kBloch for axis in Axis.elems()])
    
    def set_kBloch(self, plane_src) -> None:
        """
        Set Bloch wave vector from source.
        
        Parameters
        ----------
        plane_src : WithBloch
            Source with Bloch boundary conditions.
        
        Raises
        ------
        ValueError
            If plane_src is not an instance of WithBloch.
        """
        # Note: WithBloch is from source module, which may not be available
        # during grid module initialization. We'll use duck typing here.
        if not hasattr(plane_src, 'kBloch'):
            raise ValueError('"plane_src" should have kBloch method.')
        for axis in Axis.elems():
            self._comp[int(axis)].set_kBloch(plane_src)
    
    def contains(
        self,
        x: Union[float, np.ndarray],
        y: Union[float, np.ndarray],
        z: Union[float, np.ndarray]
    ) -> Union[bool, np.ndarray]:
        """
        Check if point(s) are within the grid.
        
        Parameters
        ----------
        x : float or np.ndarray
            X coordinate(s).
        y : float or np.ndarray
            Y coordinate(s).
        z : float or np.ndarray
            Z coordinate(s).
        
        Returns
        -------
        bool or np.ndarray
            True if point(s) are within grid boundaries, False otherwise.
        
        Raises
        ------
        ValueError
            If x, y, z have different shapes or contain non-real values.
        """
        x_array = np.asarray(x)
        y_array = np.asarray(y)
        z_array = np.asarray(z)
        
        if not np.all(np.isreal(x_array)):
            raise ValueError('"x" should be array with real elements.')
        if not np.all(np.isreal(y_array)):
            raise ValueError('"y" should be array with real elements.')
        if not np.all(np.isreal(z_array)):
            raise ValueError('"z" should be array with real elements.')
        
        if x_array.shape != y_array.shape or x_array.shape != z_array.shape:
            raise ValueError('"x", "y", "z" should have same size.')
        
        truth = np.ones(x_array.shape, dtype=bool)
        loc = [x_array, y_array, z_array]
        for axis in Axis.elems():
            truth = truth & self._comp[int(axis)].contains(loc[int(axis)])
        
        return truth
    
    def bound_plot(self, withpml: bool) -> np.ndarray:
        """
        Get boundaries for plotting.
        
        Parameters
        ----------
        withpml : bool
            If True, return full boundaries including PML.
            If False, return PML interface boundaries.
        
        Returns
        -------
        np.ndarray
            3x2 array of boundary locations.
        """
        result = np.full((Axis.count(), Sign.count()), np.nan)
        for axis in Axis.elems():
            result[int(axis), :] = self._comp[int(axis)].bound_plot(withpml)
        return result
    
    def lplot(
        self,
        g: Union[GT, List[GT]],
        withinterp: bool,
        withpml: bool
    ) -> List[np.ndarray]:
        """
        Get locations where data are evaluated for plotting.
        
        If the data do not include the boundaries of the simulation domain
        (or the PML interfaces for withpml=False), the boundary points are
        added to ensure that the plot is drawn from boundary to boundary.
        
        Parameters
        ----------
        g : GT or list of GT
            Grid type(s) (primary or dual). If single GT, use for all axes.
        withinterp : bool
            Whether to interpolate dual grid data to boundaries.
        withpml : bool
            Whether to include PML regions.
        
        Returns
        -------
        List[np.ndarray]
            List of location arrays for each axis.
        """
        if isinstance(g, GT):
            g_list = [g] * Axis.count()
        elif isinstance(g, list):
            if len(g) != Axis.count():
                raise ValueError(f'"g" list should have length {Axis.count()}.')
            g_list = g
        else:
            raise ValueError('"g" should be GT instance or list of GT instances.')
        
        if not isinstance(withinterp, bool):
            raise ValueError('"withinterp" should be logical.')
        if not isinstance(withpml, bool):
            raise ValueError('"withpml" should be logical.')
        
        result = []
        for axis in Axis.elems():
            result.append(self._comp[int(axis)].lplot(g_list[int(axis)], withinterp, withpml))
        
        return result
    
    def lvoxelbound(
        self,
        g: Union[GT, List[GT]],
        withpml: bool
    ) -> List[np.ndarray]:
        """
        Get locations of boundaries of voxels drawn.
        
        For data at primary grid points, the voxel centers are within the
        simulation domain including the boundary. For data at dual grid
        points, the voxel centers are within the simulation domain excluding
        the boundary.
        
        Parameters
        ----------
        g : GT or list of GT
            Grid type(s) (primary or dual). If single GT, use for all axes.
        withpml : bool
            Whether to include PML regions.
        
        Returns
        -------
        List[np.ndarray]
            List of voxel boundary arrays for each axis.
        """
        if isinstance(g, GT):
            g_list = [g] * Axis.count()
        elif isinstance(g, list):
            if len(g) != Axis.count():
                raise ValueError(f'"g" list should have length {Axis.count()}.')
            g_list = g
        else:
            raise ValueError('"g" should be GT instance or list of GT instances.')
        
        if not isinstance(withpml, bool):
            raise ValueError('"withpml" should be logical.')
        
        result = []
        for axis in Axis.elems():
            result.append(self._comp[int(axis)].lvoxelbound(g_list[int(axis)], withpml))
        
        return result