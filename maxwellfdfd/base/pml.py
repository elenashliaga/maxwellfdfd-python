"""
PML (Perfectly Matched Layer) kind enumeration for MaxwellFDFD.

This module defines the types of PML used in MaxwellFDFD simulations.
"""

from typing import List, Union
from ._enumerated import Enumerated


class PML(Enumerated):
    """
    Enumeration class for PML kinds.
    
    Defines the two types of PML used in MaxwellFDFD:
    - SC: Stretched-coordinate PML
    - U: Uniaxial PML
    
    Examples
    --------
    >>> from maxwellfdfd.base import PML
    >>> PML.count()
    2
    >>> [pml.name for pml in PML.elems()]
    ['sc', 'u']
    >>> int(PML.SC)
    0
    >>> str(PML.SC)
    'sc'
    """
    
    SC = 'sc'  # stretched-coordinate PML
    U = 'u'    # uniaxial PML
    
    @classmethod
    def elems(cls, ind: int | None = None) -> Union['PML', List['PML']]:
        """
        Return all instances of PML.
        
        Parameters
        ----------
        ind : int or None, optional
            If provided, return only the element at this index (0-based).
            If None, return all elements.
        
        Returns
        -------
        PML or list of PML
            All instances or the specific instance at index `ind`.
        """
        all_elems = [PML.SC, PML.U]
        if ind is not None:
            return all_elems[ind]
        return all_elems
    
    @classmethod
    def count(cls) -> int:
        """
        Return the number of PML instances.
        
        Returns
        -------
        int
            Number of instances (always 2).
        """
        return 2