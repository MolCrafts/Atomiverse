# Molecular DFT dispersion corrections (D3BJ, D4)

Grimme-type dispersion corrections add a geometry-only London dispersion
energy to a molecular SCF; the correction is independent of the electronic
density and basis. Two corrections are public as concrete types with their
parameters as members (PBE defaults): `cpu::scf::D3BJ` and `cpu::scf::D4`
(`cpu/scf/dftd4.h`), each evaluated with `energy(frame)` [Hartree]. A
mean-field method folds them in with the `d3bj` / `d4` flags, after which
`energy()` includes \(E_{\mathrm{disp}}\) and `energy_disp()` returns it.

All sections run on CPU and evaluate isolated (non-periodic) molecules — the
Frame box is ignored. Besides the D4 section below, the case carries
`d3bj-molecular-pbe` (H₂O D3(BJ) energy with PBE parameters,
`cpu::scf::D3BJ{}.energy(frame)`), whose oracle record lives in the case
`REFERENCE.md`. Periodic D3 / D4 live under GPW
([gpw-d3.md](gpw-d3.md), [gpw-d4.md](gpw-d4.md)).

Case: `ctest -R '^scf/dispersion$'` — sources and oracle record in [`regressions/scf/dispersion/`](../../../regressions/scf/dispersion/). One section of the case per `##` heading below: `atv_regressions --case scf/dispersion --section <name>`.

## Common setup and oracle

- System: H₂O, charge 0, O (0, 0, 0.1173); H (0, 0.7572, −0.4692);
  H (0, −0.7572, −0.4692) Å; `Box::cubic(20.0)` on the Frame (ignored). STO-3G with Cartesian shells is recorded
  only for protocol consistency with the molecular KS path.
- Oracle: offline PySCF **2.14.0** + `pyscf-dispersion` (an external-engine
  pin, not a golden of Atomiverse output); never imported by ctest. Protocol:
  `docs/tutorial/appendix-pyscf-oracle.md`.
- Build: `-DATV_BUILD_HF=ON -DATV_BUILD_REGRESSIONS=ON` (CPU build environment
  `buildenv-gcc/2026.03-mpich`, see the repository build notes).
- Roadmap: `docs/roadmap/dft.md` §8 Dispersion corrections.

## dft-d4-h2o-pbe — DFT-D4 dispersion, H₂O / PBE parameters {#dft-d4-h2o-pbe}

### Summary

Grimme / Caldeweyher **DFT-D4** uses charge- and coordination-number-dependent
\(C_6\) coefficients and Becke–Johnson (rational) damping. This section pins
the **two-body** molecular energy on gas-phase water with the **PBE** D4
parameter set (roadmap: DFT-D4 ✓ energy). Fair wall ratio
\(r = T_{\mathrm{ATV}}/T_{\mathrm{PySCF}} = 2.17\) (≤10×).

### Method

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

- **Three-body ATM is not included.** On this H₂O geometry the ATM term is
  \(\sim 3\times 10^{-11}\) Ha and is absorbed by the \(10^{-6}\) Ha gate;
  PySCF’s `DFTD4Dispersion(..., atm=True)` is the offline twin.
- No D4 analytic forces / stress on the molecular path (roadmap forces ○).
- No other reference-functional parameter sets are exercised (defaults are
  PBE; members are public if a caller changes them).

### Implementation

```cpp
#include "atomiverse/cpu/scf/dftd4.h"

const ATV::cpu::scf::D4 d4{};              // PBE-D4 defaults as members
const double e_disp = d4.energy(frame);    // [Hartree]
```

Optional fold-in on a mean-field method (not used by the section):

```cpp
ATV::cpu::scf::RKS rks{ATV::cpu::scf::BasisSet::STO3G, ATV::cpu::scf::XCType::PBE};
rks.d4 = true;
rks.run(frame);
const double ed = rks.energy_disp();       // [Hartree]
```

The section binary takes an optional repeat count (`2000` for the wall twin).

| Path | Role |
| ---- | ---- |
| `dftd4.h` | Public `cpu::scf::D4` |
| `mean_field_method.h` | `d4` flag, `energy_disp()` |

### Validation (vs PySCF)

Oracle call: `DFTD4Dispersion(mol, xc='pbe', version='d4', atm=True)`
(libdftd4).

| Quantity | Atomiverse / pin | PySCF (offline) | Tol | Notes |
| -------- | ---------------: | --------------: | --- | ----- |
| \(E_{\mathrm{disp}}\) [Ha] | −0.000195303719 | −0.00019530371933102527 | \(1\times 10^{-6}\) Ha | Two-body ATV vs full D4 (ATM \(\sim 3\times 10^{-11}\) Ha) |

**Tolerance rationale.** Matching geometry, PBE-D4 parameters, EEQ2019
charges, and in-tree polarizability tables track the two-body term of
simple-dftd4 / PySCF well below a micro-Hartree. The gate is
\(1\times 10^{-6}\) Ha on the pure dispersion energy.

### Performance

Both sides loop **2000** evaluations (ATV via argv; PySCF in-process) so
process startup does not dominate the microsecond-scale kernel. Median of 5
repeats.

| Twin | \(T_{\mathrm{PySCF}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | -------------------------: | -----------------------: | ----: | ---- | ------ |
| D4 energy × 2000 | 0.01143 | 0.02485 | **2.17** | ✓ | Micro-kernel; multi-iter required |

Gate: \(r = T_{\mathrm{ATV}}/T_{\mathrm{PySCF}} \le 10\). Accuracy gate (single
eval): PASS at \(1\times 10^{-6}\) Ha vs the offline pin.

### Limitations

- Two-body only; three-body ATM is not implemented (negligible on H₂O; may
  matter for larger systems or dense solids).
- Molecular / non-periodic path only.
- Energy only — no D4 nuclear forces or cell stress.
- Single-system pin (H₂O + PBE parameters); other \(Z\) and reference XC
  sets rely on tables but are not end-to-end gated here.
- The ~2× PySCF wall ratio is within the gate and not a speed optimization
  target.

### References

- E. Caldeweyher, S. Ehlert, A. Hansen, H. Neugebauer, S. Spicher,
  C. Bannwarth, S. Grimme, *J. Chem. Phys.* **150**, 154122 (2019)
