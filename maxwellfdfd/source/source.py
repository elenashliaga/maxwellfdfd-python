"""
Source base class for MaxwellFDFD.

This module defines the abstract base class for all electromagnetic sources.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, TYPE_CHECKING
import numpy as np

from maxwellfdfd.base import Axis, GT, chkarg
from maxwellfdfd.shape import Shape

if TYPE_CHECKING:
    from maxwellfdfd.grid import Grid3d


class Source(ABC):
    """
    Abstract base class for all electromagnetic sources.
    
    Source is the superclass for all electric current sources J (or magnetic
    current sources M). It defines the interface that all concrete source
    implementations must follow.
    
    Parameters
    ----------
    lgrid_cell : list of ndarray
        List of grid locations [x_array, y_array, z_array] for the grid type
        specified by set_gridtype(). These are the locations of grid planes
        where dipole sources are defined.
    laltgrid_cell : list of ndarray
        List of grid locations [x_array, y_array, z_array] for the alternate
        grid type (alter(gt)).
    shape : Shape
        Shape of the source, used for visualization.
    forceprim : bool, optional
        If True, force primary grid assignment. If False, grid type is
        determined by set_gridtype(). Default is False.
    
    Attributes
    ----------
    lgrid : list of ndarray
        Grid locations for the primary grid type.
    laltgrid : list of ndarray
        Grid locations for the alternate grid type.
    shape : Shape
        Shape of the source.
    forceprim : bool
        Flag controlling behavior of l property.
    gt : GT or None
        Grid type (GT.PRIM or GT.DUAL) for this source.
    
    Notes
    -----
    - If this is SRCJ and E-field grid is primary, then gt = GT.PRIM.
    - If this is SRCJ and E-field grid is dual, then gt = GT.DUAL.
    - If this is SRCM and E-field grid is primary, then gt = GT.DUAL.
    - If this is SRCM and E-field grid is dual, then gt = GT.PRIM.
    
    Examples
    --------
    >>> import numpy as np
    >>> from maxwellfdfd.base import Axis, GT
    >>> from maxwellfdfd.shape import Point
    >>> from maxwellfdfd.source import Source
    >>> 
    >>> # Concrete source implementations should inherit from Source
    >>> # and implement the generate_kernel method.
    """
    
    def __init__(
        self,
        lgrid_cell: List[np.ndarray],
        laltgrid_cell: List[np.ndarray],
        shape: Shape,
        forceprim: bool = False
    ):
        """
        Initialize a Source.
        
        Parameters
        ----------
        lgrid_cell : list of ndarray
            Grid locations for the primary grid type.
        laltgrid_cell : list of ndarray
            Grid locations for the alternate grid type.
        shape : Shape
            Shape of the source.
        forceprim : bool, optional
            Force primary grid assignment. Default is False.
        """
        # Validate inputs
        chkarg(
            len(lgrid_cell) == Axis.count() and
            all(np.isrealobj(arr) for arr in lgrid_cell),
            f'"lgrid_cell" should be length-{Axis.count()} list with real arrays.'
        )
        chkarg(
            len(laltgrid_cell) == Axis.count() and
            all(np.isrealobj(arr) for arr in laltgrid_cell),
            f'"laltgrid_cell" should be length-{Axis.count()} list with real arrays.'
        )
        chkarg(isinstance(shape, Shape), '"shape" should be instance of Shape.')
        chkarg(isinstance(forceprim, bool), '"forceprim" should be bool.')
        
        self._lgrid = lgrid_cell
        self._laltgrid = laltgrid_cell
        self._shape = shape
        self._gt: GT | None = None
        self._forceprim = forceprim
    
    @property
    def lgrid(self) -> List[np.ndarray]:
        """Grid locations for the primary grid type."""
        return self._lgrid
    
    @property
    def laltgrid(self) -> List[np.ndarray]:
        """Grid locations for the alternate grid type."""
        return self._laltgrid
    
    @property
    def shape(self) -> Shape:
        """Shape of the source."""
        return self._shape
    
    @property
    def forceprim(self) -> bool:
        """Flag controlling behavior of l property."""
        return self._forceprim
    
    @property
    def gt(self) -> GT | None:
        """Grid type (GT.PRIM or GT.DUAL) for this source."""
        return self._gt
    
    @property
    def l(self) -> Dict[GT, List[np.ndarray]]:
        """
        Grid locations for all grid types.
        
        Returns a dictionary mapping grid types to lists of grid locations.
        The structure is {GT: [x_array, y_array, z_array]}.
        
        Returns
        -------
        dict
            Dictionary with GT keys and list of ndarray values.
        """
        l_dict: Dict[GT, List[np.ndarray]] = {}
        
        if self._forceprim:
            l_dict[GT.PRIM] = self._lgrid.copy()
            l_dict[GT.DUAL] = self._laltgrid.copy()
        else:
            if self._gt is None:
                raise RuntimeError('Grid type not set. Call set_gridtype() first.')
            l_dict[self._gt] = self._lgrid.copy()
            l_dict[self._gt.alter()] = self._laltgrid.copy()
        
        return l_dict
    
    def set_gridtype(self, gt: GT) -> None:
        """
        Set the grid type for this source.
        
        Parameters
        ----------
        gt : GT
            Grid type (GT.PRIM or GT.DUAL).
        
        Raises
        ------
        ValueError
            If gt is not an instance of GT.
        """
        chkarg(isinstance(gt, GT), '"gt" should be instance of GT.')
        self._gt = gt
    
    @abstractmethod
    def generate_kernel(
        self,
        w_axis: Axis,
        grid3d: 'Grid3d'
    ) -> Tuple[List[np.ndarray | slice | str], np.ndarray]:
        """
        Generate the source kernel for a specific axis.
        
        This is an abstract method that must be implemented by concrete
        source classes. It generates the indices and current density
        values for the source.
        
        Parameters
        ----------
        w_axis : Axis
            Axis for which to generate the source.
        grid3d : Grid3d
            3D grid instance.
        
        Returns
        -------
        index_cell : list
            List of indices for each axis. Each element can be an ndarray,
            slice, or ':' string.
        JMw_patch : ndarray
            Current density values for the source.
        
        Raises
        ------
        NotImplementedError
            If not implemented by a concrete subclass.
        """
        pass
    
    def generate(
        self,
        w_axis: Axis,
        grid3d: 'Grid3d'
    ) -> Tuple[List[np.ndarray | slice | str], np.ndarray]:
        """
        Generate the source for a specific axis.
        
        This is a wrapper around generate_kernel() that provides error
        handling and validation.
        
        Parameters
        ----------
        w_axis : Axis
            Axis for which to generate the source.
        grid3d : Grid3d
            3D grid instance.
        
        Returns
        -------
        index_cell : list
            List of indices for each axis.
        JMw_patch : ndarray
            Current density values for the source.
        
        Raises
        ------
        RuntimeError
            If source assignment fails.
        """
        chkarg(isinstance(w_axis, Axis), '"w_axis" should be instance of Axis.')
        # Note: Grid3d validation will be done when grid module is available
        # chkarg(isinstance(grid3d, Grid3d), '"grid3d" should be instance of Grid3d.')
        
        try:
            index_cell, JMw_patch = self.generate_kernel(w_axis, grid3d)
        except Exception as err:
            raise RuntimeError('Source assignment failed.') from err
        
        return index_cell, JMw_patch