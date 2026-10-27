"""
Direction enumeration for MaxwellFDFD.

This module defines the horizontal and vertical directions used in MaxwellFDFD.
"""

from typing import List, Union
from ._enumerated import Enumerated


class Dir(Enumerated):
    """
    Enumeration class for horizontal and vertical directions.
    
    Defines the two directions used in MaxwellFDFD:
    - H: Horizontal direction
    - V: Vertical direction
    
    Examples
    --------
    >>> from maxwellfdfd.base import Dir
    >>> Dir.count()
    2
    >>> [dir.name for dir in Dir.elems()]
    ['h', 'v']
    >>> int(Dir.H)
    0
    >>> str(Dir.H)
    'h'
    >>> Dir.H.alter()
    <Dir.V: 'v'>
    """
    
    H = 'h'  # horizontal
    V = 'v'  # vertical
    
    @classmethod
    def elems(cls, ind: int | None = None) -> Union['Dir', List['Dir']]:
        """
        Return all instances of Dir.
        
        Parameters
        ----------
        ind : int or None, optional
            If provided, return only the element at this index (0-based).
            If None, return all elements.
        
        Returns
        -------
        Dir or list of Dir
            All instances or the specific instance at index `ind`.
        """
        all_elems = [Dir.H, Dir.V]
        if ind is not None:
            return all_elems[ind]
        return all_elems
    
    @classmethod
    def count(cls) -> int:
        """
        Return the number of Dir instances.
        
        Returns
        -------
        int
            Number of instances (always 2).
        """
        return 2
    
    def alter(self) -> 'Dir':
        """
        Get the orthogonal direction.
        
        Returns
        -------
        Dir
            The orthogonal direction (H -> V, V -> H).
        
        Raises
        ------
        ValueError
            If called on empty or invalid instance.
        """
        if not isinstance(self, Dir):
            raise ValueError('"this" should have Dir as elements.')
        all_elems = [Dir.H, Dir.V]
        return all_elems[Dir.count() - 1 - int(self)]