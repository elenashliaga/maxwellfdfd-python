# MaxwellFDFD Python Examples

Runnable example for the `maxwellfdfd` package: a Python conversion of a 2D example script from the original MaxwellFDFD MATLAB library.

## Files

| File | Description | MATLAB Source |
|------|-------------|---------------|
| [`slit_2d.py`](slit_2d.py) | Ag metal slit — TM slit mode excited by a magnetic plane source | `2d/slit_2d.m` |
| [`vis_utils.py`](vis_utils.py) | Shared visualization & power-flux helpers (matplotlib), used by `slit_2d.py` | — |
| `slit_2d_Ex.png` | Pre-generated output figure: \|E_x\| of the TM slit mode | — |

## Prerequisites

```bash
python3 -m pip install numpy scipy matplotlib
```

The `maxwellfdfd` package must be importable. Run the example from the `maxwellfdfd-python/` directory:

```bash
cd maxwellfdfd-python
python3 examples/slit_2d.py
```

## What it does

Two silver blocks (constant eps = −129.2 − 3.3j, Ag at 1550 nm) separated by a 160 nm vacuum gap form a subwavelength metallic slit. A z-polarized *magnetic* plane source (`SRCM`, `PlaneSrc`) is injected at y = −500 nm and excites the TM slit mode; the power transmitted through the slit is measured at y = +500 nm. This is the magnetic-source (TM) counterpart of the MATLAB `ag_slot_2d.m` electric-source setup.

**Output:** `examples/slit_2d_Ex.png` — |E_x| field magnitude (abs, 'hot' colormap) with the structure overlay. Runtime ≈ 10 s.
