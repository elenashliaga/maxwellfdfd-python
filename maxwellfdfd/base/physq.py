"""
Physical quantity enumeration for MaxwellFDFD.

This module defines the physical quantities used in MaxwellFDFD simulations.
"""

from typing import List, Union
from enum import Enum, auto


# Mapping from member name to (symbol, SI unit). Defined at MODULE level
# (not inside the Enum body) because in Python 3.11+ a class-body dict
# assignment like ``_SYMBOLS = {...}`` is consumed by the Enum metaclass and
# turned into a spurious member, breaking the ``symbol``/``SIunit`` lookups.
_PHYSQ_SYMBOLS = {
    'ARBITRARY': ('', 'AU'),
    'L': ('L', 'm'),
    'OMEGA': ('\\omega', 'rad/s'),
    'EPS': ('\\epsilon', 'F/m'),
    'MU': ('\\mu', 'H/m'),
    'E': ('E', 'V/m'),
    'H': ('H', 'A/m'),
    'J': ('J', 'A/m^2'),
    'M': ('M', 'V/m^2'),
    'I': ('I', 'A'),
    'V': ('V', 'V'),
    'S': ('S', 'W/m^2'),
    'P': ('P', 'W'),
    'U': ('u', 'W/m^3'),
}


class PhysQ(Enum):
    """
    Enumeration class for physical quantities.
    
    Defines the physical quantities used in MaxwellFDFD with their
    mathematical symbols and SI units.
    
    Examples
    --------
    >>> from maxwellfdfd.base import PhysQ
    >>> PhysQ.count()
    14
    >>> PhysQ.L.symbol
    'L'
    >>> PhysQ.L.SIunit
    'm'
    >>> int(PhysQ.L)
    1
    >>> str(PhysQ.L)
    'L'
    """
    
    ARBITRARY = auto()
    L = auto()
    OMEGA = auto()
    EPS = auto()
    MU = auto()
    E = auto()
    H = auto()
    J = auto()
    M = auto()
    I = auto()
    V = auto()
    S = auto()
    P = auto()
    U = auto()
    
    # NOTE: symbol/SIunit mapping is the module-level ``_PHYSQ_SYMBOLS`` dict
    # above (kept outside the Enum body to avoid metaclass consumption).

    def __init__(self, value):
        """
        Initialize physical quantity.

        Parameters
        ----------
        value : auto
            Auto-generated value.
        """
        # NOTE: The symbol/SIunit are looked up lazily via the properties
        # below. They must NOT be set here, because during enum construction
        # the class name (PhysQ) is not yet bound, so referencing other
        # members (e.g. PhysQ.ARBITRARY) raises NameError.
        self._value_ = value

    @property
    def symbol(self) -> str:
        """
        Get the mathematical symbol of this physical quantity.

        Returns
        -------
        str
            Mathematical symbol.
        """
        return _PHYSQ_SYMBOLS[self.name][0]

    @property
    def SIunit(self) -> str:
        """
        Get the SI unit of this physical quantity.

        Returns
        -------
        str
            SI unit string.
        """
        return _PHYSQ_SYMBOLS[self.name][1]
    
    @classmethod
    def elems(cls, ind: int | None = None) -> Union['PhysQ', List['PhysQ']]:
        """
        Return all instances of PhysQ.
        
        Parameters
        ----------
        ind : int or None, optional
            If provided, return only the element at this index (0-based).
            If None, return all elements.
        
        Returns
        -------
        PhysQ or list of PhysQ
            All instances or the specific instance at index `ind`.
        """
        all_elems = list(cls)
        if ind is not None:
            return all_elems[ind]
        return all_elems
    
    @classmethod
    def count(cls) -> int:
        """
        Return the number of PhysQ instances.
        
        Returns
        -------
        int
            Number of instances (always 14).
        """
        return len(cls)
    
    def __int__(self) -> int:
        """
        Return the integer representation of this enumeration.
        
        Returns
        -------
        int
            0-based index of this enumeration value.
        """
        return list(type(self)).index(self)
    
    def __str__(self) -> str:
        """
        Return the string representation of this enumeration.
        
        Returns
        -------
        str
            Name of this enumeration value.
        """
        return self.name
    
    def __lt__(self, other: 'PhysQ') -> bool:
        """
        Less than comparison.
        
        Parameters
        ----------
        other : PhysQ
            Another PhysQ instance.
        
        Returns
        -------
        bool
            True if self < other.
        """
        if type(self) != type(other):
            raise TypeError(f"Cannot compare {type(self).__name__} with {type(other).__name__}")
        return int(self) < int(other)
    
    def __le__(self, other: 'PhysQ') -> bool:
        """
        Less than or equal comparison.
        
        Parameters
        ----------
        other : PhysQ
            Another PhysQ instance.
        
        Returns
        -------
        bool
            True if self <= other.
        """
        if type(self) != type(other):
            raise TypeError(f"Cannot compare {type(self).__name__} with {type(other).__name__}")
        return int(self) <= int(other)
    
    def __gt__(self, other: 'PhysQ') -> bool:
        """
        Greater than comparison.
        
        Parameters
        ----------
        other : PhysQ
            Another PhysQ instance.
        
        Returns
        -------
        bool
            True if self > other.
        """
        if type(self) != type(other):
            raise TypeError(f"Cannot compare {type(self).__name__} with {type(other).__name__}")
        return int(self) > int(other)
    
    def __ge__(self, other: 'PhysQ') -> bool:
        """
        Greater than or equal comparison.
        
        Parameters
        ----------
        other : PhysQ
            Another PhysQ instance.
        
        Returns
        -------
        bool
            True if self >= other.
        """
        if type(self) != type(other):
            raise TypeError(f"Cannot compare {type(self).__name__} with {type(other).__name__}")
        return int(self) >= int(other)