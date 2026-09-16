# Reference — H₂O molecular DFT-D4 dispersion energy (PBE params)

## Method

Grimme / Caldeweyher **DFT-D4** with Becke–Johnson (rational) damping
evaluated as a geometry-only additive correction on the fixed water geometry
below, using the **PBE** reference-functional parameters. Basis STO-3G is
recorded for protocol consistency with the molecular KS path; E_disp does not
depend on the electronic density or the basis.

Public API: `cpu::scf::D4` (`include/atomiverse/cpu/scf/dftd4.h`). Optional
fold-in on mean-field methods via `method.d4 = true` then
`method.energy_disp()` after `run()`.

Two-body formula (BJ damping; Caldeweyher et al., *J. Chem. Phys.* **150**,
154122 (2019)):

\[
E_{\mathrm{disp}}^{(2)} = -\sum_{A<B}\left[
  \frac{s_6\,C_6^{AB}}{R_{AB}^6 + f(R_0^{AB})^6}
  + \frac{s_8\,C_8^{AB}}{R_{AB}^8 + f(R_0^{AB})^8}
\right],
\quad
f(R_0) = a_1 R_0 + a_2,\quad
R_0^{AB}=\sqrt{3\,r_4^{A} r_4^{B}},\quad
C_8^{AB}=3\,C_6^{AB}\,r_4^{A} r_4^{B}.
\]

C6 coefficients are charge- and CN-dependent (EEQ2019 partial charges +
EN-weighted erf coordination numbers + reference polarizability
interpolation). PBE-D4 BJ parameters (SAW190103, BJ+ATM fit set):
\(s_6=1\), \(s_8=0.95948085\), \(a_1=0.38574991\), \(a_2=4.80688534\) Bohr.
Three-body ATM is **not** included in Atomiverse; for this H₂O geometry the
ATM contribution is ~3×10⁻¹¹ Ha and is absorbed by the 10⁻⁶ Ha gate.

## Reference value

**E_disp (PBE-D4, H₂O) = −0.000195303719 Hartree**

at the pin geometry below.

Source: offline PySCF **2.14.0** + `pyscf-dispersion` / libdftd4 backend with
`DFTD4Dispersion(mol, xc='pbe', version='d4', atm=True)` (PySCF default).
External-engine oracle — not a golden snapshot of Atomiverse's own output.
Primary literature: E. Caldeweyher, S. Ehlert, A. Hansen, H. Neugebauer,
S. Spicher, C. Bannwarth, S. Grimme, *J. Chem. Phys.* **150**, 154122 (2019).

## Geometry

PySCF oracle preset `water` from `tools/pyscf_oracle/pin_scf.py` [Å]:

| Atom | x | y | z |
|---|---|---|---|
| O | 0.000000 | 0.000000 | 0.117300 |
| H | 0.000000 | 0.757200 | −0.469200 |
| H | 0.000000 | −0.757200 | −0.469200 |

Gas-phase isolated molecule (no periodic images). A 20 Å cubic box may be
set on the Frame for SCF consumers; D4 ignores the box.

## Basis / angular convention

- Basis: **STO-3G** (protocol; E_disp is basis-independent)
- Angular: **Cartesian** (`cart=True`)

## Oracle: PySCF 2.14.0 (offline — nothing in CMake/ctest imports PySCF)

Interpreter: `~/work/.x86_64/bin/python` (pyscf **2.14.0**,
`pyscf-dispersion`).

### Exact command (pin helper)

```bash
~/work/.x86_64/bin/python tools/pyscf_oracle/pin_scf.py d4 \
  --geom water --basis sto-3g --xc pbe
```

### Pinned JSON excerpt (2026-07-23)

```json
{
  "basis": "sto-3g",
  "cart": true,
  "e_disp_ha": -0.00019530371933102527,
  "geom": "water",
  "method": "d4",
  "pyscf": "2.14.0",
  "xc": "pbe"
}
```

## Tolerance and its rationale

**1 × 10⁻⁶ Ha.** Job gate for the pure dispersion energy. With matching
geometry, PBE-D4 parameters, EEQ2019 charges, and in-tree reference
polarizability tables, Atomiverse tracks the two-body term of simple-dftd4 /
PySCF to well below micro-Hartree (ATM on H₂O is ~3×10⁻¹¹ Ha).

## What is checked

`cpu::scf::D4{}.energy(frame)` returns E_disp [Hartree] within 1 × 10⁻⁶ Ha
of the pinned −0.000195303719 Ha. Non-finite or out-of-tolerance exits
non-zero with `got / want / tol`.

## Perf (fair wall twin)

Protocol: `tools/pyscf_oracle/fair_bench.py --method d4 --geom water
--basis sto-3g --xc pbe --repeat 5 -- ./build/regressions/reg_dft-d4-h2o-pbe 2000`

Both sides loop **2000** evaluations (ATV via argv; PySCF in-process) so
process startup does not dominate the microsecond-scale kernel.

| side | median wall (5×) |
|---|---|
| Atomiverse `D4::energy` × 2000 | 0.02485 s |
| PySCF `DFTD4Dispersion.get_dispersion` × 2000 | 0.01143 s |
| **ratio** T_atv / T_pyscf | **2.17** |
| within_10x (≤ 10) | **true** |

Accuracy gate (single eval): PASS at 1e-6 Ha vs offline pin.
