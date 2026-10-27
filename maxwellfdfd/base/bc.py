"""
Boundary condition enumeration for MaxwellFDFD.

This module defines the boundary condition types used in MaxwellFDFD simulations.
"""

from typing import List, Union
from ._enumerated import Enumerated


class BC(Enumerated):
    """
    Enumeration class for boundary conditions.
    
    Defines the three types of boundary conditions used in MaxwellFDFD:
    - E: Perfect electric conductor (PEC) - tangential E-field = 0
    - M: Perfect magnetic conductor (PMC) - normal E-field = 0
    - P: Periodic boundary condition
    
    Examples
    --------
    >>> from maxwellfdfd.base import BC
    >>> BC.count()
    3
    >>> [bc.name for bc in BC.elems()]
    ['e', 'm', 'p']
    >>> int(BC.E)
    0
    >>> str(BC.E)
    'e'
    """
    
    E = 'e'  # PEC: tangential component of E-field = 0
    M = 'm'  # PMC: normal component of E-field = 0
    P = 'p'  # periodic boundary condition
    
    @classmethod
    def elems(cls, ind: int | None = None) -> Union['BC', List['BC']]:
        """
        Return all instances of BC.
        
        Parameters
        ----------
        ind : int or None, optional
            If provided, return only the element at this index (0-based).
            If None, return all elements.
        
        Returns
        -------
        BC or list of BC
            All instances or the specific instance at index `ind`.
        """
        all_elems = [BC.E, BC.M, BC.P]
        if ind is not None:
            return all_elems[ind]
        return all_elems
    
    @classmethod
    def count(cls) -> int:
        """
        Return the number of BC instances.
        
        Returns
        -------
        int
            Number of instances (always 3).
        """
        return 3