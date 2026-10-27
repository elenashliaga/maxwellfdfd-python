"""
WithBloch mixin for MaxwellFDFD.

This module defines the WithBloch mixin class for sources that support
Bloch boundary conditions.
"""

from abc import abstractmethod
from typing import TYPE_CHECKING
import numpy as np

from maxwellfdfd.base import Axis

if TYPE_CHECKING:
    pass


class WithBloch:
    """
    Mixin class for sources that can set Bloch boundary conditions.
    
    This mixin provides an abstract interface for sources that need to
    specify Bloch wave vectors. Sources that inherit from this mixin
    can be used with periodic boundary conditions and support oblique
    incidence.
    
    Notes
    -----
    This is a mixin class and should be used with multiple inheritance.
    Concrete classes must implement the kBloch property.
    
    Examples
    --------
    >>> from maxwellfdfd.source import Source, WithBloch
    >>> 
    >>> class PlaneSrcWithBloch(Source, WithBloch):
    >>>     @property
    >>>     def kBloch(self) -> np.ndarray:
    >>>         # Return Bloch wave vector [kx, ky, kz]
    >>>         return self._kBloch
    """
    
    @property
    @abstractmethod
    def kBloch(self) -> np.ndarray:
        """
        Bloch wave vector.
        
        Returns
        -------
        ndarray
            Bloch wave vector [kx, ky, kz].
        
        Raises
        ------
        NotImplementedError
            If not implemented by a concrete class.
        """
        raise NotImplementedError('kBloch property must be implemented.')
    
    def kBloch_for_axis(self, axis: Axis) -> float:
        """
        Get the Bloch wave vector component for a specific axis.
        
        Parameters
        ----------
        axis : Axis
            The axis for which to get the Bloch wave vector component.
        
        Returns
        -------
        float
            The Bloch wave vector component for the specified axis.
        """
        return self.kBloch[int(axis)]