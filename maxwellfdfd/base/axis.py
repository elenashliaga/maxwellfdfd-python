"""
Axis enumeration for MaxwellFDFD.

This module defines the Cartesian axes (x, y, z) used in MaxwellFDFD.
"""

from typing import List, Tuple, Union
from ._enumerated import Enumerated


class Axis(Enumerated):
    """
    Enumeration class for Cartesian axes.
    
    Defines the three Cartesian axes used in MaxwellFDFD:
    - X: x-axis
    - Y: y-axis
    - Z: z-axis
    
    Examples
    --------
    >>> from maxwellfdfd.base import Axis
    >>> Axis.count()
    3
    >>> [axis.name for axis in Axis.elems()]
    ['x', 'y', 'z']
    >>> int(Axis.X)
    0
    >>> str(Axis.X)
    'x'
    >>> p, q, r = Axis.Y.cycle()
    >>> [str(p), str(q), str(r)]
    ['z', 'x', 'y']
    """
    
    X = 'x'  # x-axis
    Y = 'y'  # y-axis
    Z = 'z'  # z-axis
    
    @classmethod
    def elems(cls, ind: int | None = None) -> Union['Axis', List['Axis']]:
        """
        Return all instances of Axis.
        
        Parameters
        ----------
        ind : int or None, optional
            If provided, return only the element at this index (0-based).
            If None, return all elements.
        
        Returns
        -------
        Axis or list of Axis
            All instances or the specific instance at index `ind`.
        """
        all_elems = [Axis.X, Axis.Y, Axis.Z]
        if ind is not None:
            return all_elems[ind]
        return all_elems
    
    @classmethod
    def count(cls) -> int:
        """
        Return the number of Axis instances.
        
        Returns
        -------
        int
            Number of instances (always 3).
        """
        return 3
    
    def cycle(self) -> Tuple['Axis', 'Axis', 'Axis']:
        """
        Return cyclic permutation [p, q, r] with r == self.
        
        For example, if self is Axis.Y, returns (Axis.Z, Axis.X, Axis.Y).
        
        Returns
        -------
        tuple of Axis
            Cyclic permutation (p, q, r) where r == self.
        """
        r = self
        idx = int(self)
        all_elems = [Axis.X, Axis.Y, Axis.Z]
        # Cyclic permutation [p, q, r] with r == self and all three distinct.
        # e.g. Axis.Y -> (Axis.Z, Axis.X, Axis.Y). The previous code used
        # idx % count for p, which made p == self (a non-distinct permutation)
        # and broke every shape/source relying on three distinct axes.
        p = all_elems[(idx + 1) % Axis.count()]
        q = all_elems[(idx + 2) % Axis.count()]
        return p, q, r
    
    def alter(self) -> 'Axis':
        """
        Get the next axis in cyclic order.
        
        Returns
        -------
        Axis
            The next axis (X -> Y, Y -> Z, Z -> X).
        """
        all_elems = [Axis.X, Axis.Y, Axis.Z]
        return all_elems[(int(self) + 1) % Axis.count()]