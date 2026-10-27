"""
Base enumeration class for MaxwellFDFD.

This module provides the base class for all enumerations in MaxwellFDFD.
It implements common functionality for enumeration classes including:
- Element access and counting
- Integer and string conversion
- Comparison operators
- Sorting support
"""

from enum import Enum
from typing import List, Tuple, TypeVar, Any, Union

T = TypeVar('T', bound='Enumerated')


class Enumerated(Enum):
    """
    Abstract base class for enumerated quantities.
    
    This class provides common functionality for all enumeration classes in
    MaxwellFDFD. It supports conversion to strings and integers, comparison
    operators, and sorting.
    
    Notes
    -----
    This is an abstract base class. Concrete enumeration classes should inherit
    from this class and implement the `elems()` and `count()` class methods.
    
    Examples
    --------
    >>> # Example using Axis (concrete subclass)
    >>> from maxwellfdfd.base import Axis
    >>> Axis.count()
    3
    >>> [axis.name for axis in Axis.elems()]
    ['x', 'y', 'z']
    >>> int(Axis.X)
    0
    >>> str(Axis.X)
    'x'
    """
    
    def __init__(self, name: str):
        """
        Initialize enumeration instance.
        
        Parameters
        ----------
        name : str
            The name of the enumeration value.
        """
        self._name_ = name
    
    @classmethod
    def elems(cls, ind: int | None = None) -> Union[T, List[T]]:
        """
        Return all instances of the enumeration.
        
        Parameters
        ----------
        ind : int or None, optional
            If provided, return only the element at this index (0-based).
            If None, return all elements.
        
        Returns
        -------
        Enumerated or list of Enumerated
            All instances or the specific instance at index `ind`.
        
        Raises
        ------
        NotImplementedError
            If not implemented by subclass.
        """
        raise NotImplementedError(f"{cls.__name__}.elems() must be implemented by subclass")
    
    @classmethod
    def count(cls) -> int:
        """
        Return the number of instances in the enumeration.
        
        Returns
        -------
        int
            Number of instances.
        
        Raises
        ------
        NotImplementedError
            If not implemented by subclass.
        """
        raise NotImplementedError(f"{cls.__name__}.count() must be implemented by subclass")
    
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
    
    def __lt__(self, other: T) -> bool:
        """
        Less than comparison.
        
        Parameters
        ----------
        other : Enumerated
            Another enumeration of the same type.
        
        Returns
        -------
        bool
            True if self < other.
        
        Raises
        ------
        TypeError
            If other is not the same type.
        """
        if type(self) != type(other):
            raise TypeError(f"Cannot compare {type(self).__name__} with {type(other).__name__}")
        return int(self) < int(other)
    
    def __le__(self, other: T) -> bool:
        """
        Less than or equal comparison.
        
        Parameters
        ----------
        other : Enumerated
            Another enumeration of the same type.
        
        Returns
        -------
        bool
            True if self <= other.
        
        Raises
        ------
        TypeError
            If other is not the same type.
        """
        if type(self) != type(other):
            raise TypeError(f"Cannot compare {type(self).__name__} with {type(other).__name__}")
        return int(self) <= int(other)
    
    def __gt__(self, other: T) -> bool:
        """
        Greater than comparison.
        
        Parameters
        ----------
        other : Enumerated
            Another enumeration of the same type.
        
        Returns
        -------
        bool
            True if self > other.
        
        Raises
        ------
        TypeError
            If other is not the same type.
        """
        if type(self) != type(other):
            raise TypeError(f"Cannot compare {type(self).__name__} with {type(other).__name__}")
        return int(self) > int(other)
    
    def __ge__(self, other: T) -> bool:
        """
        Greater than or equal comparison.
        
        Parameters
        ----------
        other : Enumerated
            Another enumeration of the same type.
        
        Returns
        -------
        bool
            True if self >= other.
        
        Raises
        ------
        TypeError
            If other is not the same type.
        """
        if type(self) != type(other):
            raise TypeError(f"Cannot compare {type(self).__name__} with {type(other).__name__}")
        return int(self) >= int(other)