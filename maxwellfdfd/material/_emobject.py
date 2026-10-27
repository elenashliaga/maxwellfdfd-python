"""
EMObject stub for material module.

This is a temporary placeholder for the EMObject class that will be
provided by the io module. It is used only for type checking and
should be replaced when the io module is converted.

Note: This is a private module (prefixed with _) and should not be
imported directly by users.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .material import Material
    from ..shape.shape import Shape


class EMObject:
    """
    Stub class for electromagnetic object.
    
    This is a temporary placeholder combining Shape and Material.
    Will be replaced by the full EMObject from the io module.
    
    Parameters
    ----------
    shape : Shape
        The shape of the object.
    material : Material
        The material of the object.
    """
    
    def __init__(self, shape: 'Shape', material: 'Material'):
        """
        Initialize EMObject stub.
        
        Parameters
        ----------
        shape : Shape
            The shape of the object.
        material : Material
            The material of the object.
        """
        self.shape = shape
        self.material = material
    
    def __repr__(self) -> str:
        """String representation of EMObject."""
        return f"EMObject(shape={self.shape}, material={self.material})"