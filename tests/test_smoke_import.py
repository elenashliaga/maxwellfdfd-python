"""Tier 0 smoke test: every public sub-module imports without error.

This catches gross import-time failures (syntax errors, circular imports,
missing names) before any numerical code runs.
"""
import importlib

import pytest

# (module path, list of public names expected to be importable)
MODULES = [
    "maxwellfdfd",
    "maxwellfdfd.base",
    "maxwellfdfd.shape",
    "maxwellfdfd.source",
    "maxwellfdfd.grid",
    "maxwellfdfd.material",
    "maxwellfdfd.diff",
    "maxwellfdfd.io",
]


@pytest.mark.parametrize("modpath", MODULES, ids=lambda m: m)
def test_module_imports(modpath):
    """Each top-level sub-package must import cleanly."""
    importlib.import_module(modpath)


def test_public_api_names():
    """The documented public API names must all be present on the package."""
    import maxwellfdfd as mf

    expected = [
        "Axis", "BC", "Dir", "GT", "Sign", "FT", "PML", "PhysQ",
        "Oscillation", "EquationType",
        "Shape", "Box", "Material",
        "PointSrc", "Grid1d", "Grid2d", "Grid3d",
        "maxwell_run",
    ]
    missing = [name for name in expected if not hasattr(mf, name)]
    assert missing == [], f"Missing public API names: {missing}"
