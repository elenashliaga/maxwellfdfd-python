"""Tier 1 unit tests: base enums (Axis, GT, BC, FT, PhysQ).

Verifies the integer mapping, cycling, alteration, and iteration contracts
that the rest of the codebase relies on. These are cheap and catch the most
common MATLAB->Python porting mistakes (off-by-one, string-vs-int values).
"""
import numpy as np

from maxwellfdfd.base import Axis, GT, BC, FT, PhysQ


# ---------------------------------------------------------------- Axis
def test_axis_count_and_elems():
    assert Axis.count() == 3
    elems = Axis.elems()
    assert elems == [Axis.X, Axis.Y, Axis.Z]


def test_axis_int_mapping():
    assert int(Axis.X) == 0
    assert int(Axis.Y) == 1
    assert int(Axis.Z) == 2


def test_axis_cycle():
    # cycle() returns a 3-tuple of Axis elements (the cyclic neighbours).
    cyc = Axis.X.cycle()
    assert len(cyc) == 3
    assert all(isinstance(c, Axis) for c in cyc)


def test_axis_alter_cycles():
    # alter() advances to the next axis cyclically: X->Y->Z->X.
    assert Axis.X.alter() == Axis.Y
    assert Axis.Y.alter() == Axis.Z
    assert Axis.Z.alter() == Axis.X


# ------------------------------------------------------------------ GT
def test_gt_values_and_alter():
    assert int(GT.PRIM) == 0
    assert int(GT.DUAL) == 1
    assert GT.PRIM.alter() == GT.DUAL
    assert GT.DUAL.alter() == GT.PRIM


# ------------------------------------------------------------------ BC
def test_bc_distinct_members():
    # P (periodic), E (PEC), M (PMC) must be distinct.
    assert BC.P != BC.E
    assert BC.E != BC.M


# ------------------------------------------------------------------ FT
def test_ft_members():
    assert FT.E != FT.H


# --------------------------------------------------------------- PhysQ
def test_physq_symbol_lookup():
    # Regression for the _SYMBOLS-as-enum-member bug (Python 3.11+).
    assert PhysQ.E.symbol == 'E'
    assert PhysQ.EPS.symbol == '\\epsilon'
    assert PhysQ.L.SIunit == 'm'


def test_physq_int_mapping():
    # auto() values start at 0 in this port.
    assert int(PhysQ.ARBITRARY) == 0
    assert int(PhysQ.E) == 5


def test_physq_count():
    # 14 documented physical quantities.
    assert PhysQ.count() == 14
