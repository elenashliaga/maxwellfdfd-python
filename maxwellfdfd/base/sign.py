"""
Sign enumeration for MaxwellFDFD.

This module defines the sign types (negative and positive) used in MaxwellFDFD.
"""

from typing import List, Union
from ._enumerated import Enumerated


class Sign(Enumerated):
    """
    Enumeration class for negative and positive signs.
    
    Defines the two sign types used in MaxwellFDFD:
    - N: Negative sign
    - P: Positive sign
    
    Examples
    --------
    >>> from maxwellfdfd.base import Sign
    >>> Sign.count()
    2
    >>> [sign.name for sign in Sign.elems()]
    ['n', 'p']
    >>> int(Sign.N)
    0
    >>> str(Sign.N)
    'n'
    >>> Sign.N.alter()
    <Sign.P: 'p'>
    """
    
    N = 'n'  # negative
    P = 'p'  # positive
    
    @classmethod
    def elems(cls, ind: int | None = None) -> Union['Sign', List['Sign']]:
        """
        Return all instances of Sign.
        
        Parameters
        ----------
        ind : int or None, optional
            If provided, return only the element at this index (0-based).
            If None, return all elements.
        
        Returns
        -------
        Sign or list of Sign
            All instances or the specific instance at index `ind`.
        """
        all_elems = [Sign.N, Sign.P]
        if ind is not None:
            return all_elems[ind]
        return all_elems
    
    @classmethod
    def count(cls) -> int:
        """
        Return the number of Sign instances.
        
        Returns
        -------
        int
            Number of instances (always 2).
        """
        return 2
    
    def alter(self) -> 'Sign':
        """
        Get the opposite sign.
        
        Returns
        -------
        Sign
            The opposite sign (N -> P, P -> N).
        
        Raises
        ------
        ValueError
            If called on empty or invalid instance.
        """
        if self is None:
            raise ValueError("alter() cannot be called on None")
        all_elems = [Sign.N, Sign.P]
        return all_elems[Sign.count() - 1 - int(self)]