# Reference — H₂O / STO-3G restricted Hartree–Fock total energy

## Method

Restricted Hartree–Fock (RHF), minimal STO-3G Gaussian basis, single-point
total electronic + nuclear-repulsion energy on a fixed water geometry. Driven
by the concrete `RHF` method through the public Frame API.

**One regression case, multi-device** (PyTorch-style): run
`reg_h2o_rhf_sto3g --device=cpu|cuda|both` (default `both` when built with
CUDA). Same geometry and literature pin on every device — not a separate
`*_cuda` sibling case.

## Reference value

**E_RHF/STO-3G(H₂O) = −74.961754 Hartree** at R(O–H) = 0.95 Å, ∠H–O–H = 104.5°.

Source: LibreTexts, *Live Textbook of Physical Chemistry* (R. Peverati),
§28.2 "Hartree–Fock Calculation for Water", which reports −74.961754063 Ha at
exactly this geometry. Cross-checked against PySCF 2.13 (`RHF`/`STO-3G` at the
same geometry reproduces the value to six decimals). The STO-3G basis itself is
from W. J. Hehre, R. F. Stewart, J. A. Pople, *J. Chem. Phys.* **51**, 2657
(1969), doi:10.1063/1.1672392. This is a cited literature value at the case
geometry, not a golden snapshot of Atomiverse's own output.

## Geometry

O at the origin; the two H symmetric about the +z axis in the xz-plane, so the
O–H bonds subtend the reference angle:

- R(O–H) = 0.95 Å
- ∠H–O–H = 104.5°

Derived Cartesian coordinates [Å], with the half-angle θ = 52.25° and
x_H = R sin θ, z_H = R cos θ — the same two formulas `reg_h2o_rhf_sto3g.cpp`
evaluates:

| Atom | x | y | z |
|---|---|---|---|
| O | 0.000000 | 0.0 | 0.000000 |
| H | +0.751155 | 0.0 | 0.581606 |
| H | −0.751155 | 0.0 | 0.581606 |

These are rounded for reading only. The case computes them in double
precision from R and the angle (0.7511550950566511 / 0.5816064160327268);
do not retype the rounded table into a consumer.

The molecule sits in a 20 Å cubic box (gas-phase; the box only supplies a
non-degenerate cell and does not enter the isolated-molecule SCF).

## Tolerance and its rationale

**1 × 10⁻³ Ha.** The citation geometry **coincides** with the case geometry, so
a tight milli-Hartree sanity window is honest: this checks that the public RHF
path reproduces the reference water energy at the very geometry the reference
quotes. (The engine in fact matches PySCF 2.13 at this geometry to below
1 × 10⁻⁵ Ha; the 1 × 10⁻³ window is a deliberately generous end-to-end sanity
band, not a high-precision benchmark — that belongs to an acceptance-time
external-engine comparison run outside this tree.)

## Provenance note (do not re-pair the old figure)

An earlier draft of this case cited **−74.9420 Ha** as a "Szabo & Ostlund"
water value. That pairing is wrong on two counts, recorded here so it does not
come back:

1. Szabo & Ostlund's worked minimal-basis STO-3G SCF example is **HeH⁺**, not
   water — the water number was misattributed.
2. −74.9420 Ha is the **Crawford Programming Project #3** water energy at a
   *stretched* geometry, R(O–H) = 1.1 Å, ∠H–O–H = 104.0° (reproduced to the
   quoted digits by PySCF 2.13). It is not an equilibrium-geometry literature
   value and does not match this case's 0.95 Å / 104.5° geometry.

The accepted equilibrium-geometry RHF/STO-3G water energy is ≈ −74.96 Ha, as
used above.

## What is checked

For each selected device, `RHF{STO3G}.run(frame)` converges and
the Frame energy `frame.meta.get(kEnergy)` [eV], converted to Hartree via
`units::kHartreeInEv`, lies within 1 × 10⁻³ Ha of the cited −74.961754 Ha.
Non-convergence, a non-finite energy, or an out-of-tolerance value exits
non-zero with `[device] got / want / tol`.

No public integral pack / solver bag: Frame + concrete RHF only.
