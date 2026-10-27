"""
Grid1d class for 1D staggered grids.

This module implements the Grid1d class which represents a 1D staggered grid
used in finite-difference frequency-domain (FDFD) simulations. The grid supports
primary and dual grid points, PML regions, and various boundary conditions.
"""

from typing import Dict, List, Union
import numpy as np
from scipy.optimize import brentq

from maxwellfdfd.base import Axis, BC, GT, Sign, PhysUnit


class Grid1d:
    """
    1D staggered grid for finite-difference discretization.
    
    Grid1d contains all information related to the staggered grid in a single
    Cartesian axis. It does not include frequency-dependent physical quantities
    such as omega, eps, mu, and PML s-factors.
    
    The grid implements a staggered Yee scheme where:
    - Primary grid points (GT.PRIM) store electric field components
    - Dual grid points (GT.DUAL) store magnetic field components
    - Grid locations are stored as dictionaries with GT enum keys
    
    Parameters
    ----------
    axis : Axis
        Cartesian axis of this grid (x, y, or z).
    unit : PhysUnit
        Physical unit instance for length conversions.
    lprim_array : array_like
        Row vector of primary grid vertex locations. Must be monotonically
        increasing with at least 2 elements.
    Npml_array : array_like
        Length-2 row vector [Npml_negative, Npml_positive] specifying the
        number of primary grid cells inside PML at each end.
    bc : BC
        Boundary condition for this axis (PEC, PMC, or periodic).
    
    Attributes
    ----------
    axis : Axis
        Cartesian axis of this grid.
    unit : PhysUnit
        Physical unit instance.
    unitvalue : float
        Unit value of length.
    l : Dict[GT, np.ndarray]
        Grid locations {primary, dual} for vertices.
    dl : Dict[GT, np.ndarray]
        Grid cell sizes {primary, dual}.
    bc : BC
        Boundary condition.
    N : int
        Number of grid cells.
    L : float
        Length of the grid (domain).
    Npml : np.ndarray
        PML cell counts [negative, positive].
    lpml : np.ndarray
        PML interface locations.
    Lpml : np.ndarray
        PML thicknesses.
    center : float
        Center of grid without PML.
    
    Notes
    -----
    The grid uses 0-based indexing for all arrays, unlike MATLAB's 1-based
    indexing. Cell arrays from MATLAB are converted to Python dictionaries
    with enum keys.
    
    Examples
    --------
    >>> from maxwellfdfd.base import Axis, BC, PhysUnit
    >>> unit = PhysUnit()
    >>> lprim = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    >>> Npml = np.array([5, 5])
    >>> grid = Grid1d(Axis.X, unit, lprim, Npml, BC.P)
    >>> grid.N
    4
    >>> grid.L
    2.0
    """
    
    def __init__(
        self,
        axis: Axis,
        unit: PhysUnit,
        lprim_array: np.ndarray,
        Npml_array: np.ndarray,
        bc: BC
    ):
        """Initialize Grid1d instance."""
        # Validate axis
        if not isinstance(axis, Axis):
            raise ValueError('"axis" should be instance of Axis.')
        self._axis = axis
        
        # Validate unit
        if not isinstance(unit, PhysUnit):
            raise ValueError('"unit" should be instance of PhysUnit.')
        self._unit = unit
        self._unitvalue = unit.value
        
        # Validate Npml_array
        Npml_array = np.asarray(Npml_array)
        if Npml_array.shape != (2,):
            raise ValueError(
                f'"Npml" should be length-{Sign.count()} row vector with integer elements.'
            )
        if not np.issubdtype(Npml_array.dtype, np.integer):
            raise ValueError('"Npml" should have integer elements.')
        self._Npml = Npml_array
        
        # Validate bc
        if not isinstance(bc, BC):
            raise ValueError('"bc" should be instance of BC.')
        self._bc = bc
        
        # Validate lprim_array
        lprim_array = np.asarray(lprim_array)
        if lprim_array.ndim != 1:
            raise ValueError('"lprim_array" should be 1D array with real elements.')
        if len(lprim_array) < 2:
            raise ValueError('"lprim_array" should have at least 2 elements.')
        if not np.all(np.isreal(lprim_array)):
            raise ValueError('"lprim_array" should have real elements.')
        
        # Set N and L
        lprim = lprim_array.astype(float)
        self._N = len(lprim) - 1  # # of grid cells in the axis
        self._L = lprim[-1] - lprim[0]
        
        # Set loc and dl
        ldual = np.full(self._N + 1, np.nan)
        ldual[1:] = (lprim[:-1] + lprim[1:]) / 2
        
        if self._bc == BC.P:
            # Periodic boundary condition
            ldual[0] = ldual[-1] - (lprim[-1] - lprim[0])
            self._ldual_ext = ldual[1] + (lprim[-1] - lprim[0])
        else:
            # PEC or PMC boundary condition
            ldual[0] = lprim[0] - (ldual[1] - lprim[0])
            self._ldual_ext = lprim[-1] + (lprim[-1] - ldual[-1])
        
        # Store grid locations (MATLAB: l = {lprim(1:end-1), ldual(2:end)})
        # Convert to 0-based indexing: lprim(1:end-1) -> lprim[:-1]
        self._l = {
            GT.PRIM: lprim[:-1],
            GT.DUAL: ldual[1:]
        }
        
        # Store ghost vertex locations
        # MATLAB: lghost = [lprim(end), ldual(1)]
        self._lghost = np.array([lprim[-1], ldual[0]])
        
        # Store cell sizes
        # MATLAB: dl = {diff(ldual), diff(lprim)}
        self._dl = {
            GT.PRIM: np.diff(ldual),
            GT.DUAL: np.diff(lprim)
        }
        
        # Set lpml, Lpml, and center
        # MATLAB: lpml = [lprim(1 + Npml(Sign.n)), lprim(end - Npml(Sign.p))]
        # Convert to 0-based: Sign.n -> 0, Sign.p -> 1
        self._lpml = np.array([
            lprim[0 + self._Npml[0]],
            lprim[-1 - self._Npml[1]]
        ])
        
        # MATLAB: Lpml = [lpml(Sign.n) - lprim(1), lprim(end) - lpml(Sign.p)]
        self._Lpml = np.array([
            self._lpml[0] - lprim[0],
            lprim[-1] - self._lpml[1]
        ])
        
        # MATLAB: center = mean(lpml)
        self._center = np.mean(self._lpml)
        
        # Initialize kBloch
        self._kBloch = 0.0
    
    @property
    def axis(self) -> Axis:
        """Get the Cartesian axis of this grid."""
        return self._axis
    
    @property
    def unit(self) -> PhysUnit:
        """Get the physical unit instance."""
        return self._unit
    
    @property
    def unitvalue(self) -> float:
        """Get the unit value of length."""
        return self._unitvalue
    
    @property
    def l(self) -> Dict[GT, np.ndarray]:
        """Get grid locations {primary, dual} for vertices."""
        return self._l
    
    @property
    def dl(self) -> Dict[GT, np.ndarray]:
        """Get grid cell sizes {primary, dual}."""
        return self._dl
    
    @property
    def bc(self) -> BC:
        """Get the boundary condition."""
        return self._bc
    
    @property
    def N(self) -> int:
        """Get the number of grid cells."""
        return self._N
    
    @property
    def L(self) -> float:
        """Get the length of the grid (domain)."""
        return self._L
    
    @property
    def Npml(self) -> np.ndarray:
        """Get PML cell counts [negative, positive]."""
        return self._Npml
    
    @property
    def lpml(self) -> np.ndarray:
        """Get PML interface locations."""
        return self._lpml
    
    @property
    def Lpml(self) -> np.ndarray:
        """Get PML thicknesses."""
        return self._Lpml
    
    @property
    def center(self) -> float:
        """Get the center of grid without PML."""
        return self._center
    
    @property
    def lg(self) -> Dict[GT, np.ndarray]:
        """
        Get grid locations with ghost vertices.
        
        Returns
        -------
        Dict[GT, np.ndarray]
            Dictionary with GT.PRIM and GT.DUAL keys, containing grid
            locations with ghost vertices beyond the boundaries.
        """
        return {
            GT.PRIM: np.concatenate([self._l[GT.PRIM], [self._lghost[0]]]),
            GT.DUAL: np.concatenate([[self._lghost[1]], self._l[GT.DUAL]])
        }
    
    @property
    def lall(self) -> Dict[GT, np.ndarray]:
        """
        Get grid locations with all vertices.
        
        Returns
        -------
        Dict[GT, np.ndarray]
            Dictionary with GT.PRIM and GT.DUAL keys, containing grid
            locations with all vertices including extended dual vertices.
        """
        return {
            GT.PRIM: self.lg[GT.PRIM],
            GT.DUAL: np.concatenate([self.lg[GT.DUAL], [self._ldual_ext]])
        }
    
    @property
    def bound(self) -> np.ndarray:
        """
        Get grid boundaries.
        
        Returns
        -------
        np.ndarray
            Length-2 array [lall_prim[0], lall_prim[-1]].
        """
        return np.array([self.lall[GT.PRIM][0], self.lall[GT.PRIM][-1]])
    
    @property
    def kBloch(self) -> float:
        """Get the Bloch wave vector."""
        return self._kBloch
    
    def set_kBloch(self, bloch_src) -> None:
        """
        Set Bloch wave vector from source.
        
        Parameters
        ----------
        bloch_src : WithBloch
            Source with Bloch boundary conditions.
        
        Raises
        ------
        ValueError
            If bloch_src is not an instance of WithBloch.
        """
        # Note: WithBloch is from source module, which may not be available
        # during grid module initialization. We'll use duck typing here.
        if not hasattr(bloch_src, 'kBloch'):
            raise ValueError('"bloch_src" should have kBloch method.')
        self._kBloch = bloch_src.kBloch_for_axis(self._axis)
    
    def contains(self, l: Union[float, np.ndarray]) -> Union[bool, np.ndarray]:
        """
        Check if location(s) are within the grid.
        
        Parameters
        ----------
        l : float or np.ndarray
            Location(s) to check.
        
        Returns
        -------
        bool or np.ndarray
            True if location(s) are within grid boundaries, False otherwise.
        
        Notes
        -----
        This function can handle arrays of locations.
        """
        l_array = np.asarray(l)
        if not np.all(np.isreal(l_array)):
            raise ValueError('"l" should be array with real elements.')
        return (l_array >= self.lall[GT.PRIM][0]) & (l_array <= self.lall[GT.PRIM][-1])
    
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
            Length-2 array of boundary locations.
        """
        if withpml:
            return self.bound
        else:
            return self._lpml
    
    def lplot(
        self,
        g: GT,
        withinterp: bool,
        withpml: bool
    ) -> np.ndarray:
        """
        Get locations where data are evaluated for plotting.
        
        If the data do not include the boundaries of the simulation domain
        (or the PML interfaces for withpml=False), the boundary points are
        added to ensure that the plot is drawn from boundary to boundary.
        
        Parameters
        ----------
        g : GT
            Grid type (primary or dual).
        withinterp : bool
            Whether to interpolate dual grid data to boundaries.
        withpml : bool
            Whether to include PML regions.
        
        Returns
        -------
        np.ndarray
            Array of locations for plotting.
        """
        if not isinstance(g, GT):
            raise ValueError('"g" should be instance of GT.')
        if not isinstance(withinterp, bool):
            raise ValueError('"withinterp" should be logical.')
        if not isinstance(withpml, bool):
            raise ValueError('"withpml" should be logical.')
        
        if g == GT.PRIM:
            lplot = self.lall[g]
        else:  # g == GT.DUAL
            lplot = self._l[g]
        
        if not withpml:
            # MATLAB: lplot(1+Npml(Sign.n):end-Npml(Sign.p))
            # Convert to 0-based indexing
            start_idx = self._Npml[0]
            end_idx = len(lplot) - self._Npml[1]
            lplot = lplot[start_idx:end_idx]
        
        if g == GT.DUAL and withinterp:
            lbound = self.bound_plot(withpml)
            lplot = np.concatenate([[lbound[0]], lplot, [lbound[1]]])
        
        return lplot
    
    def lvoxelbound(self, g: GT, withpml: bool) -> np.ndarray:
        """
        Get locations of boundaries of voxels drawn.
        
        For data at primary grid points, the voxel centers are within the
        simulation domain including the boundary. For data at dual grid
        points, the voxel centers are within the simulation domain excluding
        the boundary.
        
        Parameters
        ----------
        g : GT
            Grid type (primary or dual).
        withpml : bool
            Whether to include PML regions.
        
        Returns
        -------
        np.ndarray
            Array of voxel boundary locations.
        """
        if not isinstance(g, GT):
            raise ValueError('"g" should be instance of GT.')
        if not isinstance(withpml, bool):
            raise ValueError('"withpml" should be logical.')
        
        lvoxelbound = self.lall[g.alter()]
        if not withpml:
            # MATLAB: lvoxelbound(1+Npml(Sign.n):end-Npml(Sign.p))
            start_idx = self._Npml[0]
            end_idx = len(lvoxelbound) - self._Npml[1]
            lvoxelbound = lvoxelbound[start_idx:end_idx]
        
        return lvoxelbound