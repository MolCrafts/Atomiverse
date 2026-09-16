# DFT-D4 dispersion — H₂O / PBE parameters

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `dft-d4-h2o-pbe` |
| Regression | `regressions/scf/dft-d4-h2o-pbe/` |
| Public API | `cpu::scf::D4` (`cpu/scf/dftd4.h`); optional `method.d4 = true` + `energy_disp()` |
| Roadmap | `docs/roadmap/dft.md` §8 Dispersion corrections (DFT-D4 ✓ energy) |

## Summary

Grimme / Caldeweyher **DFT-D4** is a geometry-only additive London dispersion
correction with charge- and coordination-number-dependent \(C_6\) coefficients
and Becke–Johnson (rational) damping. This page pins the **two-body**
molecular energy on gas-phase water using the **PBE** D4 parameter set. The
correction is independent of the electronic density and basis; STO-3G is
recorded only for protocol consistency with the molecular KS path. Public
path: concrete `cpu::scf::D4` with parameters as members (PBE defaults), or
fold-in on a mean-field method via `d4 = true` then `energy_disp()` after
`run()`. Product gate: `regressions/scf/dft-d4-h2o-pbe`
(`reg_dft-d4-h2o-pbe`). Fair wall ratio
\(r = T_{\mathrm{ATV}}/T_{\mathrm{PySCF}} = 2.17\) (≤10×).

## Method

**Two-body DFT-D4 with BJ damping** (Caldeweyher et al., *J. Chem. Phys.*
**150**, 154122 (2019)):

\[
E_{\mathrm{disp}}^{(2)} = -\sum_{A<B}\left[
  \frac{s_6\,C_6^{AB}}{R_{AB}^6 + f(R_0^{AB})^6}
  + \frac{s_8\,C_8^{AB}}{R_{AB}^8 + f(R_0^{AB})^8}
\right],
\quad
f(R_0) = a_1 R_0 + a_2,
\]

\[
R_0^{AB}=\sqrt{3\,r_4^{A} r_4^{B}},\qquad
C_8^{AB}=3\,C_6^{AB}\,r_4^{A} r_4^{B}.
\]

**Charge / CN model.** Partial charges from EEQ2019; coordination numbers are
EN-weighted erf CNs; \(C_6\) from reference polarizability interpolation
(in-tree tables, \(Z\le 94\)). Defaults: \(g_a=3\), \(g_c=2\), \(w_f=6\).

**PBE-D4 BJ parameters** (SAW190103, BJ+ATM fit set; two-body scales used
here):

| Parameter | Value | Unit |
| --------- | ----: | ---- |
| \(s_6\) | 1 | dimensionless |
| \(s_8\) | 0.95948085 | dimensionless |
| \(a_1\) | 0.38574991 | dimensionless |
| \(a_2\) | 4.80688534 | Bohr |

**Key exclusions.**

- **Three-body ATM is not included** in Atomiverse. On this H₂O geometry the
  ATM term is \(\sim 3\times 10^{-11}\) Ha and is absorbed by the \(10^{-6}\) Ha
  gate. PySCF’s `DFTD4Dispersion(..., atm=True)` is the offline twin; residual
  is far below the product tolerance.
- Isolated (non-periodic) evaluation only — the Frame box is ignored.
- No D4 analytic forces / stress on this foothold (roadmap forces ○).
- No other reference-functional parameter sets exercised by this regression
  (defaults are PBE; members are public if a caller changes them).

## Implementation

Public surface only (`cpu::Frame` + concrete method members; no `*Config`, no
factories):

```cpp
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/dftd4.h"

// H₂O geometry [Å] — pin_scf.py water / REFERENCE.md
ATV::cpu::Frame frame(3);
frame.span(ATV::kAtom_type)[0] = 8; // O
frame.span(ATV::kAtom_x)[0] = 0.0;
frame.span(ATV::kAtom_y)[0] = 0.0;
frame.span(ATV::kAtom_z)[0] = 0.1173;
frame.span(ATV::kAtom_type)[1] = 1; // H
frame.span(ATV::kAtom_x)[1] = 0.0;
frame.span(ATV::kAtom_y)[1] = 0.7572;
frame.span(ATV::kAtom_z)[1] = -0.4692;
frame.span(ATV::kAtom_type)[2] = 1; // H
frame.span(ATV::kAtom_x)[2] = 0.0;
frame.span(ATV::kAtom_y)[2] = -0.7572;
frame.span(ATV::kAtom_z)[2] = -0.4692;
frame.box = ATV::cpu::Box::cubic(20.0); // ignored by D4

const ATV::cpu::scf::D4 d4{};              // PBE-D4 defaults as members
const double e_disp = d4.energy(frame);    // [Hartree]
```

Optional fold-in on a mean-field method (not required by the regression gate):

```cpp
ATV::cpu::scf::RKS rks(ATV::cpu::scf::XCType::PBE);
rks.d4 = true;
rks.run(frame);
const double ed = rks.energy_disp();       // [Hartree]
```

Build and run the regression:

```bash
bash -lc '
  module load buildenv-gcc/2026.03-mpich
  export OPENBLAS_ROOT=$OPENBLASDIR
  export RUSTUP_TOOLCHAIN=stable-x86_64-unknown-linux-gnu
  export LD_LIBRARY_PATH=$LIBRARY_PATH:$LD_LIBRARY_PATH
  cmake -B build -DCMAKE_BUILD_TYPE=Release \
        -DATV_BUILD_TESTS=ON -DATV_BUILD_HF=ON \
        -DATV_BUILD_REGRESSIONS=ON -DATV_BUILD_CUDA=OFF \
        -DBLA_VENDOR=OpenBLAS
  cmake --build build --target reg_dft-d4-h2o-pbe -j$(nproc)
  ./build/regressions/reg_dft-d4-h2o-pbe
  # fair_bench multi-iter wall twin:
  ./build/regressions/reg_dft-d4-h2o-pbe 2000
'
```

| Path | Role |
| ---- | ---- |
| `include/atomiverse/cpu/scf/dftd4.h` | Public `cpu::scf::D4` |
| `include/atomiverse/cpu/scf/mean_field_method.h` | `d4` flag, `energy_disp()` |
| `regressions/scf/dft-d4-h2o-pbe/reg_dft-d4-h2o-pbe.cpp` | Product gate |
| `regressions/scf/dft-d4-h2o-pbe/REFERENCE.md` | Offline pins + fair_bench numbers |

## Validation (vs PySCF)

Source:
[`regressions/scf/dft-d4-h2o-pbe/REFERENCE.md`](../../../regressions/scf/dft-d4-h2o-pbe/REFERENCE.md).
Oracle: offline PySCF **2.14.0** + `pyscf-dispersion` / libdftd4
(`DFTD4Dispersion(mol, xc='pbe', version='d4', atm=True)`). External-engine
pin — not a golden of Atomiverse’s own output; **never** live-import PySCF in
ctest.

**Setup:**

| Item | Value |
| ---- | ----- |
| Molecule | H₂O, charge 0 |
| Geometry [Å] | O (0, 0, 0.1173); H (0, ±0.7572, −0.4692) |
| Box | 20 Å cubic on Frame (D4 ignores box) |
| Basis | STO-3G (protocol only; \(E_{\mathrm{disp}}\) basis-independent) |
| Angular | Cartesian (`cart=True`) |
| XC params | PBE-D4 |
| Pin command | `python tools/pyscf_oracle/pin_scf.py d4 --geom water --basis sto-3g --xc pbe` |

| Quantity | Atomiverse / pin | PySCF (offline) | Tol | Notes |
| -------- | ---------------: | --------------: | --- | ----- |
| \(E_{\mathrm{disp}}\) [Ha] | −0.000195303719 | −0.00019530371933102527 | \(1\times 10^{-6}\) Ha | Two-body ATV vs full D4 (ATM \(\sim 3\times 10^{-11}\) Ha) |

**Tolerance rationale.** Matching geometry, PBE-D4 parameters, EEQ2019
charges, and in-tree polarizability tables track the two-body term of
simple-dftd4 / PySCF well below micro-Hartree. Gate is \(1\times 10^{-6}\) Ha
on the pure dispersion energy.

## Performance

Source: `REFERENCE.md` fair wall twin
(`tools/pyscf_oracle/fair_bench.py --method d4 --geom water --basis sto-3g
--xc pbe --repeat 5 -- ./build/regressions/reg_dft-d4-h2o-pbe 2000`).

Both sides loop **2000** evaluations (ATV via argv; PySCF in-process) so
process startup does not dominate the microsecond-scale kernel. Median of 5
repeats.

| Twin | \(T_{\mathrm{PySCF}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | -------------------------: | -----------------------: | ----: | ---- | ------ |
| D4 energy × 2000 | 0.01143 | 0.02485 | **2.17** | ✓ | Micro-kernel; multi-iter required |

Gate: \(r = T_{\mathrm{ATV}}/T_{\mathrm{PySCF}} \le 10\). Accuracy gate (single
eval): PASS at \(1\times 10^{-6}\) Ha vs offline pin.

## Limitations

- Two-body only; three-body ATM not implemented (negligible on H₂O; may
  matter for larger systems or dense solids).
- Molecular / non-periodic path; no GPW / plane-wave D4 on this regression
  (see CP2K queue for periodic D4).
- Energy only — no D4 nuclear forces or cell stress in the product gate.
- Single-system pin (H₂O + PBE parameters); other \(Z\) and reference XC
  sets rely on tables but are not e2e-gated here.
- Wall ratio ~2× PySCF is within gate; not a speed optimization target for
  this wave.

## References

- Offline pin: `regressions/scf/dft-d4-h2o-pbe/REFERENCE.md`
- Roadmap: `docs/roadmap/dft.md` (dispersion §)
- E. Caldeweyher, S. Ehlert, A. Hansen, H. Neugebauer, S. Spicher,
  C. Bannwarth, S. Grimme, *J. Chem. Phys.* **150**, 154122 (2019)
- PySCF oracle protocol: `docs/tutorial/appendix-pyscf-oracle.md`
