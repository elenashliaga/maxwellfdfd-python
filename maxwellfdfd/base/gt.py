"""
Grid type enumeration for MaxwellFDFD.

This module defines the grid types (primary and dual) used in MaxwellFDFD.
"""

from typing import List, Union
from ._enumerated import Enumerated


class GT(Enumerated):
    """
    Enumeration class for grid types.
    
    Defines the two grid types used in MaxwellFDFD:
    - PRIM: Primary grid
    - DUAL: Dual grid
    
    Examples
    --------
    >>> from maxwellfdfd.base import GT
    >>> GT.count()
    2
    >>> [gt.name for gt in GT.elems()]
    ['prim', 'dual']
    >>> int(GT.PRIM)
    0
    >>> str(GT.PRIM)
    'prim'
    >>> GT.PRIM.alter()
    <GT.DUAL: 'dual'>
    """
    
    PRIM = 'prim'  # primary grid
    DUAL = 'dual'  # dual grid
    
    @classmethod
    def elems(cls, ind: int | None = None) -> Union['GT', List['GT']]:
        """
        Return all instances of GT.
        
        Parameters
        ----------
        ind : int or None, optional
            If provided, return only the element at this index (0-based).
            If None, return all elements.
        
        Returns
        -------
        GT or list of GT
            All instances or the specific instance at index `ind`.
        """
        all_elems = [GT.PRIM, GT.DUAL]
        if ind is not None:
            return all_elems[ind]
        return all_elems
    
    @classmethod
    def count(cls) -> int:
        """
        Return the number of GT instances.
        
        Returns
        -------
        int
            Number of instances (always 2).
        """
        return 2
    
    def alter(self) -> 'GT':
        """
        Get the opposite grid type.
        
        Returns
        -------
        GT
            The opposite grid type (PRIM -> DUAL, DUAL -> PRIM).
        
        Raises
        ------
        ValueError
            If called on empty or invalid instance.
        """
        if not isinstance(self, GT):
            raise ValueError('"this" should have GT as elements.')
        all_elems = [GT.PRIM, GT.DUAL]
        return all_elems[GT.count() - 1 - int(self)]