"""
Grid2d class for 2D staggered grids.

This module implements the Grid2d class which represents a 2D staggered grid
used in finite-difference frequency-domain (FDFD) simulations. The grid is
created by slicing a 3D grid along a normal axis.
"""

from typing import Dict, List, Tuple, Union
import numpy as np

from maxwellfdfd.base import Axis, BC, GT, Sign, Dir
from .grid3d import Grid3d


class Grid2d:
    """
    2D staggered grid for finite-difference discretization.
    
    Grid2d contains all information related to the 2D Yee grid. It is created
    by slicing a Grid3d along a specified normal axis. For example, if the
    normal axis is Y, the 2D grid will have horizontal (Z) and vertical (X)
    components.
    
    The grid does not include frequency-dependent physical quantities such as
    omega, eps, mu, and PML s-factors.
    
    Parameters
    ----------
    grid3d : Grid3d
        3D grid to slice.
    normal_axis : Axis
        Normal axis for the 2D slice. The 2D grid will be perpendicular to
        this axis.
    
    Attributes
    ----------
    normal_axis : Axis
        Normal axis of the 2D grid.
    comp : List[Grid1d]
        Component 1D grids [horizontal, vertical].
    unitvalue : float
        Unit value of length.
    axis : List[Axis]
        [h_axis, v_axis] for the horizontal and vertical directions.
    unit : PhysUnit
        Physical unit instance.
    l : Dict[Tuple[Dir, GT], np.ndarray]
        Grid locations {primary, dual} for each direction.
    lg : Dict[Tuple[Dir, GT], np.ndarray]
        Grid locations with ghost vertices.
    lall : Dict[Tuple[Dir, GT], np.ndarray]
        Grid locations with all vertices.
    bound : np.ndarray
        Grid boundaries [[h_min, h_max], [v_min, v_max]].
    dl : Dict[Tuple[Dir, GT], np.ndarray]
        Grid cell sizes {primary, dual} for each direction.
    bc : List[BC]
        Boundary conditions [bc_h, bc_v].
    N : np.ndarray
        Number of grid cells [Nh, Nv].
    Ncell : List[List[int]]
        Cell counts as list of lists for function arguments.
    Ntot : int
        Total number of cells (Nh * Nv).
    L : np.ndarray
        Domain size [Lh, Lv].
    Npml : np.ndarray
        PML cell counts [[Npml_hn, Npml_hp], [Npml_vn, Npml_vp]].
    lpml : np.ndarray
        PML interface locations.
    Lpml : np.ndarray
        PML thicknesses.
    center : np.ndarray
        Center of grid without PML [center_h, center_v].
    kBloch : np.ndarray
        Bloch wave vector [kBloch_h, kBloch_v].
    
    Notes
    -----
    The grid uses 0-based indexing for all arrays, unlike MATLAB's 1-based
    indexing. Cell arrays from MATLAB are converted to Python dictionaries
    with tuple keys (Dir, GT).
    
    The horizontal and vertical axes are determined by the normal axis using
    the cycle() method: if normal_axis is Y, then h_axis is Z and v_axis is X.
    
    Examples
    --------
    >>> from maxwellfdfd.base import Axis, BC, PhysUnit
    >>> from maxwellfdfd.grid import Grid3d, Grid2d
    >>> unit = PhysUnit()
    >>> lprim_cell = {
    ...     Axis.X: np.array([0.0, 0.5, 1.0]),
    ...     Axis.Y: np.array([0.0, 0.5, 1.0]),
    ...     Axis.Z: np.array([0.0, 0.5, 1.0])
    ... }
    >>> grid3d = Grid3d(unit, lprim_cell)
    >>> grid2d = Grid2d(grid3d, Axis.Y)
    >>> grid2d.N
    array([2, 2])
    >>> grid2d.Ntot
    4
    """
    
    def __init__(self, grid3d: Grid3d, normal_axis: Axis):
        """Initialize Grid2d instance by slicing Grid3d."""
        # Validate grid3d
        if not isinstance(grid3d, Grid3d):
            raise ValueError('"grid3d" should be instance of Grid3d.')
        
        # Validate normal_axis
        if not isinstance(normal_axis, Axis):
            raise ValueError('"normal_axis" should be instance of Axis.')
        
        # Get horizontal and vertical axes using cycle
        # MATLAB: [h, v] = cycle(normal_axis)
        # cycle returns (p, q, r) where r == normal_axis
        # For normal_axis=Y: cycle returns (Z, X, Y), so h=Z, v=X
        p, q, r = normal_axis.cycle()
        h_axis = p
        v_axis = q
        
        self._normal_axis = normal_axis
        self._comp = [
            grid3d.comp[int(h_axis)],
            grid3d.comp[int(v_axis)]
        ]
    
    @property
    def normal_axis(self) -> Axis:
        """Get the normal axis of the 2D grid."""
        return self._normal_axis
    
    @property
    def comp(self) -> List:
        """Get component 1D grids [horizontal, vertical]."""
        return self._comp
    
    @property
    def unitvalue(self) -> float:
        """Get the unit value of length."""
        return self._comp[0].unitvalue
    
    @property
    def axis(self) -> List[Axis]:
        """
        Get axes [h_axis, v_axis].
        
        Returns
        -------
        List[Axis]
            List of horizontal and vertical axes.
        """
        return [self._comp[0].axis, self._comp[1].axis]
    
    @property
    def unit(self):
        """Get the physical unit instance."""
        return self._comp[0].unit
    
    @property
    def l(self) -> Dict[Tuple[Dir, GT], np.ndarray]:
        """
        Get grid locations {primary, dual} for each direction.
        
        Returns
        -------
        Dict[Tuple[Dir, GT], np.ndarray]
            Dictionary with (Dir, GT) tuple keys, containing grid
            locations for each direction and grid type.
        """
        result = {}
        for d in Dir.elems():
            for gt in GT.elems():
                result[(d, gt)] = self._comp[int(d)].l[gt]
        return result
    
    @property
    def lg(self) -> Dict[Tuple[Dir, GT], np.ndarray]:
        """
        Get grid locations with ghost vertices for each direction.
        
        Returns
        -------
        Dict[Tuple[Dir, GT], np.ndarray]
            Dictionary with (Dir, GT) tuple keys, containing grid
            locations with ghost vertices.
        """
        result = {}
        for d in Dir.elems():
            for gt in GT.elems():
                result[(d, gt)] = self._comp[int(d)].lg[gt]
        return result
    
    @property
    def lall(self) -> Dict[Tuple[Dir, GT], np.ndarray]:
        """
        Get grid locations with all vertices for each direction.
        
        Returns
        -------
        Dict[Tuple[Dir, GT], np.ndarray]
            Dictionary with (Dir, GT) tuple keys, containing grid
            locations with all vertices including extended dual vertices.
        """
        result = {}
        for d in Dir.elems():
            for gt in GT.elems():
                result[(d, gt)] = self._comp[int(d)].lall[gt]
        return result
    
    @property
    def bound(self) -> np.ndarray:
        """
        Get grid boundaries for each direction.
        
        Returns
        -------
        np.ndarray
            2x2 array of boundary locations [[h_min, h_max], [v_min, v_max]].
        """
        result = np.full((Dir.count(), Sign.count()), np.nan)
        for d in Dir.elems():
            result[int(d), :] = self._comp[int(d)].bound
        return result
    
    @property
    def dl(self) -> Dict[Tuple[Dir, GT], np.ndarray]:
        """
        Get grid cell sizes {primary, dual} for each direction.
        
        Returns
        -------
        Dict[Tuple[Dir, GT], np.ndarray]
            Dictionary with (Dir, GT) tuple keys, containing grid
            cell sizes for each direction and grid type.
        """
        result = {}
        for d in Dir.elems():
            for gt in GT.elems():
                result[(d, gt)] = self._comp[int(d)].dl[gt]
        return result
    
    @property
    def bc(self) -> List[BC]:
        """Get boundary conditions [bc_h, bc_v]."""
        return [self._comp[int(d)].bc for d in Dir.elems()]
    
    @property
    def N(self) -> np.ndarray:
        """
        Get number of grid cells [Nh, Nv].
        
        Returns
        -------
        np.ndarray
            Length-2 array of cell counts.
        """
        return np.array([self._comp[int(d)].N for d in Dir.elems()])
    
    @property
    def Ncell(self) -> List[List[int]]:
        """
        Get cell counts as list of lists for function arguments.
        
        Returns
        -------
        List[List[int]]
            List of lists [[Nh], [Nv]].
        """
        return [[self._comp[int(d)].N] for d in Dir.elems()]
    
    @property
    def Ntot(self) -> int:
        """
        Get total number of cells.
        
        Returns
        -------
        int
            Nh * Nv.
        """
        return np.prod(self.N)
    
    @property
    def L(self) -> np.ndarray:
        """
        Get domain size [Lh, Lv].
        
        Returns
        -------
        np.ndarray
            Length-2 array of domain sizes.
        """
        return np.array([self._comp[int(d)].L for d in Dir.elems()])
    
    @property
    def Npml(self) -> np.ndarray:
        """
        Get PML cell counts.
        
        Returns
        -------
        np.ndarray
            2x2 array of PML cell counts [[Npml_hn, Npml_hp], [Npml_vn, Npml_vp]].
        """
        result = np.full((Dir.count(), Sign.count()), np.nan)
        for d in Dir.elems():
            result[int(d), :] = self._comp[int(d)].Npml
        return result
    
    @property
    def lpml(self) -> np.ndarray:
        """
        Get PML interface locations.
        
        Returns
        -------
        np.ndarray
            2x2 array of PML interface locations.
        """
        result = np.full((Dir.count(), Sign.count()), np.nan)
        for d in Dir.elems():
            result[int(d), :] = self._comp[int(d)].lpml
        return result
    
    @property
    def Lpml(self) -> np.ndarray:
        """
        Get PML thicknesses.
        
        Returns
        -------
        np.ndarray
            2x2 array of PML thicknesses.
        """
        result = np.full((Dir.count(), Sign.count()), np.nan)
        for d in Dir.elems():
            result[int(d), :] = self._comp[int(d)].Lpml
        return result
    
    @property
    def center(self) -> np.ndarray:
        """
        Get center of grid without PML.
        
        Returns
        -------
        np.ndarray
            Length-2 array of grid centers [center_h, center_v].
        """
        return np.array([self._comp[int(d)].center for d in Dir.elems()])
    
    @property
    def kBloch(self) -> np.ndarray:
        """
        Get Bloch wave vector.
        
        Returns
        -------
        np.ndarray
            Length-2 array of Bloch wave vectors [kBloch_h, kBloch_v].
        """
        return np.array([self._comp[int(d)].kBloch for d in Dir.elems()])
    
    def contains(
        self,
        h: Union[float, np.ndarray],
        v: Union[float, np.ndarray]
    ) -> Union[bool, np.ndarray]:
        """
        Check if point(s) are within the grid.
        
        Parameters
        ----------
        h : float or np.ndarray
            Horizontal coordinate(s).
        v : float or np.ndarray
            Vertical coordinate(s).
        
        Returns
        -------
        bool or np.ndarray
            True if point(s) are within grid boundaries, False otherwise.
        
        Raises
        ------
        ValueError
            If h and v have different shapes or contain non-real values.
        """
        h_array = np.asarray(h)
        v_array = np.asarray(v)
        
        if not np.all(np.isreal(h_array)):
            raise ValueError('"h" should be array with real elements.')
        if not np.all(np.isreal(v_array)):
            raise ValueError('"v" should be array with real elements.')
        
        if h_array.shape != v_array.shape:
            raise ValueError('"h" and "v" should have same size.')
        
        truth = np.ones(h_array.shape, dtype=bool)
        loc = [h_array, v_array]
        for d in Dir.elems():
            truth = truth & self._comp[int(d)].contains(loc[int(d)])
        
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
            2x2 array of boundary locations.
        """
        result = np.full((Dir.count(), Sign.count()), np.nan)
        for d in Dir.elems():
            result[int(d), :] = self._comp[int(d)].bound_plot(withpml)
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
            Grid type(s) (primary or dual). If single GT, use for all directions.
        withinterp : bool
            Whether to interpolate dual grid data to boundaries.
        withpml : bool
            Whether to include PML regions.
        
        Returns
        -------
        List[np.ndarray]
            List of location arrays for each direction.
        """
        if isinstance(g, GT):
            g_list = [g] * Dir.count()
        elif isinstance(g, list):
            if len(g) != Dir.count():
                raise ValueError(f'"g" list should have length {Dir.count()}.')
            g_list = g
        else:
            raise ValueError('"g" should be GT instance or list of GT instances.')
        
        if not isinstance(withinterp, bool):
            raise ValueError('"withinterp" should be logical.')
        if not isinstance(withpml, bool):
            raise ValueError('"withpml" should be logical.')
        
        result = []
        for d in Dir.elems():
            result.append(self._comp[int(d)].lplot(g_list[int(d)], withinterp, withpml))
        
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
            Grid type(s) (primary or dual). If single GT, use for all directions.
        withpml : bool
            Whether to include PML regions.
        
        Returns
        -------
        List[np.ndarray]
            List of voxel boundary arrays for each direction.
        """
        if isinstance(g, GT):
            g_list = [g] * Dir.count()
        elif isinstance(g, list):
            if len(g) != Dir.count():
                raise ValueError(f'"g" list should have length {Dir.count()}.')
            g_list = g
        else:
            raise ValueError('"g" should be GT instance or list of GT instances.')
        
        if not isinstance(withpml, bool):
            raise ValueError('"withpml" should be logical.')
        
        result = []
        for d in Dir.elems():
            result.append(self._comp[int(d)].lvoxelbound(g_list[int(d)], withpml))
        
        return result