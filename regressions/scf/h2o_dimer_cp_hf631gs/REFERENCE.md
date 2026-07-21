# Reference — H₂O dimer HF/6-31G* Boys–Bernardi counterpoise

## Method

Restricted Hartree–Fock (RHF), Pople 6-31G\* (Cartesian), single-point
Boys–Bernardi counterpoise (CP) correction on a fixed water-dimer geometry.
Driven by the public `cpu::scf::Counterpoise` orchestrator over
`cpu::scf::RHF` (five serial SCFs: dimer, free A, free B, A-with-B-ghost,
B-with-A-ghost). Frame is the sole geometry currency; ghost markers live on
the method (`ghost_atoms`), never on Frame.

## Literature

Boys, S. F. & Bernardi, F. *Mol. Phys.* **19**, 553 (1970).
DOI [10.1080/00268977000101561](https://doi.org/10.1080/00268977000101561).

Sign convention (locked in code, tests, and tutorial):

\[
\mathrm{bsse} = (E_A^{AB} - E_A^{A}) + (E_B^{AB} - E_B^{B}) \le 0
\]

\[
\Delta E_{\mathrm{CP}} = E_{AB}^{AB} - E_A^{AB} - E_B^{AB}
\]

Monomer energy lowers in the larger dimer basis, so BSSE is non-positive;
\(|\mathrm{bsse}|\) is the magnitude of the basis-set superposition error.

## Geometry

Fixed water-dimer Cartesian coordinates [Å]:

| Atom | role | x | y | z |
|---|---|---:|---:|---:|
| O | A | −1.551007 | −0.114520 | 0.000000 |
| H | A | −1.934259 | 0.762503 | 0.000000 |
| H | A | −0.599677 | 0.040712 | 0.000000 |
| O | B | 1.350625 | 0.111469 | 0.000000 |
| H | B | 1.680398 | −0.373741 | −0.758561 |
| H | B | 1.680398 | −0.373741 | 0.758561 |

Monomer A = indices `{0,1,2}`; monomer B = `{3,4,5}`. Gas-phase 30 Å cubic
box (cell does not enter the isolated-molecule SCF).

## Externally pinned PySCF values

Oracle engine: **PySCF 2.14.0**, `scf.RHF`, basis `'6-31g*'`, **`cart=True`**
(matching Atomiverse Pople Cartesian convention; PySCF's default is spherical
and would disagree by ~0.01 mHa on the CP energy), `unit='Angstrom'`, ghost
atoms via the `ghost:O` / `ghost:H` element prefix. Convergence default.
Dimer AO count = 38 (Cartesian 6-31G\*).

```python
from pyscf import gto, scf

geom = """
O  -1.551007  -0.114520   0.000000
H  -1.934259   0.762503   0.000000
H  -0.599677   0.040712   0.000000
O   1.350625   0.111469   0.000000
H   1.680398  -0.373741  -0.758561
H   1.680398  -0.373741   0.758561
"""
e_dimer = scf.RHF(gto.M(atom=geom, basis="6-31g*", unit="Angstrom",
                        cart=True, verbose=0)).kernel()
# free monomers, then ghost: prefixes for A-in-AB and B-in-AB
# → e_a, e_b, e_a_ghost, e_b_ghost  (all with cart=True)
e_int_cp = e_dimer - e_a_ghost - e_b_ghost
bsse = (e_a_ghost - e_a) + (e_b_ghost - e_b)
```

Pinned scalars [Hartree] (cart=True):

| Quantity | PySCF 2.14.0 |
|---|---:|
| \(E_{AB}^{AB}\) | −152.02982896658477 |
| \(E_A^{A}\) | −76.01034691284323 |
| \(E_B^{B}\) | −76.01045593149992 |
| \(E_A^{AB}\) (B ghost) | −76.01072014609187 |
| \(E_B^{AB}\) (A ghost) | −76.01156637927771 |
| \(\Delta E_{\mathrm{raw}}\) | −0.00902612224162 |
| \(\Delta E_{\mathrm{CP}}\) | **−0.00754244121519** |
| bsse | **−0.00148368102643** (≈ 1.48 mHa) |

These are **external** pins, not golden snapshots of Atomiverse output.

## Tolerance and its rationale

**1 × 10⁻⁶ Ha** on `e_int_cp` and `bsse`. Same-geometry, same Cartesian
basis, pure HF integral path — micro-Hartree agreement with PySCF is the
honest acceptance window for this chain link. The CP correction magnitude
(~1.5 mHa) sits in the published milli-Hartree class for HF/6-31G\* water
dimer BSSE.

## What is checked

`cpu::scf::Counterpoise({0,1,2},{3,4,5}).run(dimer, rhf)` with
`rhf.basis = Pople631Gs` yields `e_int_cp` and `bsse` within 1e-6 Ha of the
PySCF pins above; `bsse ≤ 0`; `|bsse|` in [0.1, 50] mHa. Non-finite or
out-of-tolerance values exit non-zero with `got / want / tol`.
