"""
EMObject class for MaxwellFDFD.

This module provides the EMObject class, which represents the combination
of a :class:`~maxwellfdfd.shape.shape.Shape` (or list of shapes) and a
:class:`~maxwellfdfd.material.material.Material`. EMObject is the basic
building block for defining objects placed in the simulation domain.
"""

from typing import List, Union, TYPE_CHECKING

from maxwellfdfd.base import chkarg, istypesizeof

if TYPE_CHECKING:
    from maxwellfdfd.shape.shape import Shape
    from maxwellfdfd.material.material import Material


class EMObject:
    """
    Electromagnetic object: the combination of Shape and Material.

    An EMObject binds one or more shapes to a single material. When
    constructed with a list of shapes, it behaves as a collection whose
    individual elements can be accessed via indexing.

    Parameters
    ----------
    shape : Shape or list of Shape
        A single shape or a row-vector-like list of shapes.
    material : Material
        The material filling the shape(s).

    Attributes
    ----------
    shape : Shape or list of Shape
        The shape(s) of this object (read-only).
    material : Material
        The material of this object (read-only).

    Notes
    -----
    In the MATLAB original, ``EMObject`` is a ``handle`` array class. In
    Python, when multiple shapes are provided, a list of EMObject
    instances is returned via :meth:`from_shapes`. For a single shape,
    the EMObject holds that shape directly.

    Examples
    --------
    >>> from maxwellfdfd.io import EMObject
    >>> from maxwellfdfd.shape import Box
    >>> from maxwellfdfd.material import Material
    >>> mat = Material('dielectric', 'b', 4.0)
    >>> box = Box([[-50, 50], [-50, 50], [-200, 1700]], [2, 2, 20])
    >>> obj = EMObject(box, mat)
    >>> obj.material.name
    'dielectric'
    """

    def __init__(
        self,
        shape: Union['Shape', List['Shape']],
        material: 'Material',
    ) -> None:
        """
        Initialize EMObject with shape(s) and material.

        Parameters
        ----------
        shape : Shape or list of Shape
            A single shape or a list of shapes.
        material : Material
            The material filling the shape(s).

        Raises
        ------
        ValueError
            If ``shape`` is not a Shape or list of Shape, or if
            ``material`` is not a Material instance.
        """
        # Validate shape: must be a Shape or a list of Shape.
        if isinstance(shape, list):
            chkarg(
                all(istypesizeof(s, 'Shape') for s in shape),
                '"shape" should be a list with Shape as elements.',
            )
        else:
            chkarg(
                istypesizeof(shape, 'Shape'),
                '"shape" should be instance of Shape or list of Shape.',
            )

        chkarg(
            istypesizeof(material, 'Material'),
            '"material" should be instance of Material.',
        )

        self._shape: Union['Shape', List['Shape']] = shape
        self._material: 'Material' = material

    # ------------------------------------------------------------------
    # Read-only properties
    # ------------------------------------------------------------------
    @property
    def shape(self) -> Union['Shape', List['Shape']]:
        """Get the shape(s) of this object."""
        return self._shape

    @property
    def material(self) -> 'Material':
        """Get the material of this object."""
        return self._material

    # ------------------------------------------------------------------
    # Collection helpers (for the MATLAB array-of-EMObject behaviour)
    # ------------------------------------------------------------------
    @classmethod
    def from_shapes(
        cls,
        shapes: List['Shape'],
        material: 'Material',
    ) -> List['EMObject']:
        """
        Create a list of EMObject instances from a list of shapes.

        This mirrors the MATLAB behaviour where ``EMObject(shape_array,
        material)`` produces an array of EMObject, one per shape, all
        sharing the same material.

        Parameters
        ----------
        shapes : list of Shape
            List of shapes.
        material : Material
            The material for all objects.

        Returns
        -------
        list of EMObject
            One EMObject per shape.
        """
        return [cls(s, material) for s in shapes]

    def __repr__(self) -> str:
        """Return a string representation of the EMObject."""
        return f'EMObject(shape={self._shape!r}, material={self._material!r})'
