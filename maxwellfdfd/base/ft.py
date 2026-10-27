"""
Field type enumeration for MaxwellFDFD.

This module defines the field types (E and H) used in MaxwellFDFD simulations.
"""

from typing import List, Union
from ._enumerated import Enumerated


class FT(Enumerated):
    """
    Enumeration class for field types.
    
    Defines the two field types used in MaxwellFDFD:
    - E: Electric field
    - H: Magnetic field
    
    Examples
    --------
    >>> from maxwellfdfd.base import FT
    >>> FT.count()
    2
    >>> [ft.name for ft in FT.elems()]
    ['e', 'h']
    >>> int(FT.E)
    0
    >>> str(FT.E)
    'e'
    >>> FT.E.alter()
    <FT.H: 'h'>
    """
    
    E = 'e'  # Electric field
    H = 'h'  # Magnetic field
    
    @classmethod
    def elems(cls, ind: int | None = None) -> Union['FT', List['FT']]:
        """
        Return all instances of FT.
        
        Parameters
        ----------
        ind : int or None, optional
            If provided, return only the element at this index (0-based).
            If None, return all elements.
        
        Returns
        -------
        FT or list of FT
            All instances or the specific instance at index `ind`.
        """
        all_elems = [FT.E, FT.H]
        if ind is not None:
            return all_elems[ind]
        return all_elems
    
    @classmethod
    def count(cls) -> int:
        """
        Return the number of FT instances.
        
        Returns
        -------
        int
            Number of instances (always 2).
        """
        return 2
    
    def alter(self) -> 'FT':
        """
        Get the opposite field type.
        
        Returns
        -------
        FT
            The opposite field type (E -> H, H -> E).
        
        Raises
        ------
        ValueError
            If called on empty or invalid instance.
        """
        if self is None:
            raise ValueError("alter() cannot be called on None")
        all_elems = [FT.E, FT.H]
        return all_elems[FT.count() - 1 - int(self)]