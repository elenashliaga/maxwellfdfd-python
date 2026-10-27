"""
MaxwellFDFD material module.

This module provides classes and functions for defining and assigning
electromagnetic materials to grid nodes in MaxwellFDFD simulations.

Public API
----------
Classes:
    Material : Electromagnetic material with permittivity and permeability

Functions:
    assign_material_node : Assign material properties to grid nodes
    mean_material_node : Compute mean material at grid nodes

Examples
--------
Create a simple material:

>>> from maxwellfdfd.material import Material
>>> vacuum = Material('vacuum', 'none', 1.0)
>>> dielectric = Material('dielectric', 'b', 4.0)

Create a material from data file:

>>> from maxwellfdfd.base import Oscillation
>>> from maxwellfdfd.material import Material
>>> osc = Oscillation(wvlen=1.55)  # 1.55 um wavelength
>>> gold = Material.fromtable(osc, 'Au', 'y')

Assign materials to grid:

>>> from maxwellfdfd.grid import Grid3d
>>> from maxwellfdfd.shape import Box
>>> from maxwellfdfd.material import Material, assign_material_node
>>> from maxwellfdfd.material._emobject import EMObject
>>> grid = Grid3d(...)
>>> mat = Material('dielectric', 'b', 4.0)
>>> shape = Box(...)
>>> obj = EMObject(shape, mat)
>>> eps, mu = assign_material_node(grid, [obj])
"""

# Import public classes and functions
from .material import Material
from .assign_material_node import assign_material_node
from .mean_material_node import mean_material_node

# Define public API
__all__ = [
    'Material',
    'assign_material_node',
    'mean_material_node',
]

# Module version
__version__ = '0.1.0'