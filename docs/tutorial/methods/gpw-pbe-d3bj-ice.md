# GPW PBE-D3(BJ) — periodic water dimer (ice-relevant H-bond)

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `gpw-pbe-d3bj-ice` |
| Regression | `regressions/gpw/gpw-pbe-d3bj-ice/` |
| Public API | `cpu::gpw::GpwDft::d3bj` / `cpu::gpw::GPWEvaluator::d3bj`; `GpwResult::e_disp` [Ha] |
| Roadmap | `docs/roadmap/dft.md` §8 Dispersion |

## Summary

Grimme **DFT-D3 with Becke–Johnson (rational) damping** is wired into the
**Γ-point GPW** path as an additive London correction on top of PBE. This
product pin uses a **periodic water-dimer cell** (ice-relevant H-bonded pair in
a 12 Å cubic box — not a full ice Ih supercell) and gates the
**dispersion energy** against offline **CP2K 2026.1** `TYPE DFTD3(BJ)`.

Public surface: bool members `d3bj` on `cpu::gpw::GpwDft` and
`cpu::gpw::GPWEvaluator` (no `*Config` bag). Dispersion is reported on
`GpwResult::e_disp` [Hartree] and folded into `e_total`. Mutually exclusive
with zero-damping `d3`. Product gate:
`regressions/gpw/gpw-pbe-d3bj-ice` (`reg_gpw-pbe-d3bj-ice`). Wall ratio is
**perf_exempt** (E_disp pin; full QS twin wall unfair — see Performance).

## Method

**Two-body DFT-D3 with BJ damping** (Grimme et al., *J. Comput. Chem.* **32**,
1456 (2011)), periodic lattice sum over image vectors \(\mathbf{T}\):

\[
E_{\mathrm{disp}} = -\sum_{A<B,\,T}\left[
  \frac{s_6\,C_6^{AB}}{R_{AB,T}^6 + f^6}
  + \frac{s_8\,C_8^{AB}}{R_{AB,T}^8 + f^8}
\right],
\quad
f = a_1\sqrt{C_8^{AB}/C_6^{AB}} + a_2.
\]

**PBE-D3(BJ) parameters** (CP2K `REFERENCE_FUNCTIONAL PBE`):

| Parameter | Value | Unit |
| --------- | ----: | ---- |
| \(s_6\) | 1.0 | dimensionless |
| \(s_8\) | 0.7875 | dimensionless |
| \(a_1\) | 0.4289 | dimensionless |
| \(a_2\) | 4.4407 | Bohr |

**CN form (GPW / CP2K parity).** The GPW BJ path uses **CP2K coordination
numbers** (`cnparam_d3`, including the long-range
\(\tfrac12(1-\tanh(R-2 r_{\mathrm{cov}}))\) factor) so \(E_{\mathrm{disp}}\)
matches offline `TYPE DFTD3(BJ)`. Molecular `cpu::scf::D3BJ` keeps the original
Grimme CN (PySCF / simple-dftd3 twin) and is unchanged by this slug.

**Pair cutoff.** Atomiverse energy lattice sum uses
**rcut = 2 × R_CUTOFF 20 Å ≈ 75.59 Bohr**. CP2K prints
`Cutoff Radius [Bohr]: 37.79` for `R_CUTOFF 20` (Å→Bohr); the 2× convention
matches the zero-damping GPW D3 pin. Three-body **ATM off**
(`CALCULATE_C9_TERM .FALSE.`).

**Electronic host.** Dispersion is geometry-only; the regression still runs a
short Γ-RKS GPW (PBE, `DZVP-MOLOPT-SR-GTH` + `GTH-PBE-q*`) so `e_disp` is
materialized on a real `GpwResult` and included in `e_total`. Absolute QS
\(E_{\mathrm{tot}}\) micro-Ha parity is **out of scope** (OT/grid/DIIS
differences).

**Key exclusions (from closed spec).**

- Full ice Ih supercell / phonons
- DFT-D4 / ATM three-body
- Molecular `D3BJ` / PySCF pin changes
- k-points, UKS, cell stress
- CUDA GPW
- Absolute \(e_{\mathrm{tot}}\) pin vs CP2K QS

## Implementation

Public surface only (`GpwDft` / `GPWEvaluator` members; no `*Config`, no
`make_*` / `build_*` factories). Regression uses the concrete solver:

```cpp
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/molrs.hpp"

using ATV::cpu::gpw::GaussianBasis;
using ATV::cpu::gpw::GpwAtom;
using ATV::cpu::gpw::GpwDft;
using ATV::cpu::gpw::GthPotential;
using ATV::cpu::gpw::XcKind;

// Geometry [Å] → Bohr — CP2K water dimer (REFERENCE.md).
constexpr double kA2B = 1.0 / 0.52917720859;
// … fill 6 GpwAtom rows (2×O + 4×H) with DZVP-MOLOPT-SR-GTH + GTH-PBE-q* …

GpwDft gpw;
const double L = 12.0 * kA2B;
gpw.cell[0] = gpw.cell[1] = gpw.cell[2] = L;
gpw.cutoff_ry = 200.0; // e_disp is geometry-only; SCF only needs to run
gpw.n_occ = 8;         // 16 valence e⁻ → 8 doubly occupied
gpw.max_iter = 8;
gpw.tol = 1e-6;
gpw.mix = 0.4;
gpw.xc = XcKind::PBE;
gpw.nonperiodic = false;
gpw.d3bj = true;       // mutually exclusive with gpw.d3

gpw.build(atoms);
const auto& res = gpw.run(atoms);
// res.e_disp [Ha] within 1e-8 of offline CP2K pin
// res.e_total includes e_disp
```

Driver / Frame path (component; same member):

```cpp
// kinds = {O, H} Kind{Element, GaussianBasis, GthPotential}
ATV::cpu::gpw::GPWEvaluator eval(kinds, /*cutoff_ry=*/300.0, /*max_iter=*/200,
                                 /*tol=*/1e-7, /*mix=*/0.3,
                                 ATV::cpu::gpw::XcKind::PBE,
                                 /*nonperiodic=*/false);
eval.d3bj = true;
// single_point(frame, eval) or bind on Evaluate / MD / Minimize force slot
// eval.last_result().e_disp  [Hartree]
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
        -DATV_BUILD_PW=ON -DATV_BUILD_GPW=ON \
        -DATV_BUILD_REGRESSIONS=ON -DATV_BUILD_CUDA=OFF \
        -DBLA_VENDOR=OpenBLAS
  cmake --build build --target reg_gpw-pbe-d3bj-ice -j$(nproc)
  ./build/regressions/reg_gpw-pbe-d3bj-ice
'
```

| Path | Role |
| ---- | ---- |
| `include/atomiverse/cpu/gpw/gpw_solver.h` | `GpwDft`, `GpwResult::e_disp`, `d3bj` |
| `include/atomiverse/cpu/gpw/gpw_evaluator.h` | `GPWEvaluator::d3bj` |
| `include/atomiverse/cpu/gpw/internal/dftd3.h` | CP2K-CN BJ engine (private) |
| `regressions/gpw/gpw-pbe-d3bj-ice/reg_gpw-pbe-d3bj-ice.cpp` | Product gate |
| `regressions/gpw/gpw-pbe-d3bj-ice/REFERENCE.md` | Offline pin + perf_exempt |
| `tools/cp2k_oracle/inputs/gpw-pbe-d3bj-ice/` | CP2K twin input |

## Validation (CP2K)

Source:
[`regressions/gpw/gpw-pbe-d3bj-ice/REFERENCE.md`](../../../regressions/gpw/gpw-pbe-d3bj-ice/REFERENCE.md).
Oracle: offline **CP2K 2026.1** (`git:5e54ba2`, module
`2026.1-gmpflf-2026.03` on Arrhenius), twin
`tools/cp2k_oracle/inputs/gpw-pbe-d3bj-ice/water_dimer_d3bj.inp`. External
engine pin — not a golden of Atomiverse’s own output; **never** launch live
CP2K from ctest.

**Setup:**

| Item | Value |
| ---- | ----- |
| System | Water dimer (ice-relevant H-bond pair) |
| Geometry [Å] | O (0,0,0); H (0.758602, 0, ±0.504284); O (3,0,0); H (3.758602, 0, ±0.504284) |
| Cell | ABC **12.0 12.0 12.0 Å**, PERIODIC XYZ |
| Basis / PP | `DZVP-MOLOPT-SR-GTH` + `GTH-PBE-q6` (O) / `GTH-PBE-q1` (H) |
| XC + disp | PBE + `TYPE DFTD3(BJ)`, `REFERENCE_FUNCTIONAL PBE` |
| R_CUTOFF | 20.0 (Å in twin; ATV rcut = 2× ≈ 75.59 Bohr) |
| Occupations | RKS, 16 valence e⁻ → `n_occ = 8` |
| Pin line | `Dispersion energy: -0.00137627769481` |

| Quantity | Atomiverse / pin | CP2K (offline) | Tol | Notes |
| -------- | ---------------: | -------------: | --- | ----- |
| \(E_{\mathrm{disp}}\) [Ha] | −0.00137627769481 | −0.00137627769481 | \(1\times 10^{-8}\) Ha | Hard gate |
| \(E_{\mathrm{tot}}\) QS [Ha] | (not gated) | −34.33962825563683 | — | Context only; OT/grid/DIIS |

**Tolerance rationale.** Matching geometry, PBE-BJ parameters, CP2K CN form, and
rcut 40 Bohr (2× printed CP2K cutoff) track CP2K dispersion to nano-Hartree.
Unit tests also check analytic BJ forces vs central FD and
\(e_{\mathrm{total}}(\mathrm{d3bj}) - e_{\mathrm{total}}(\mathrm{off}) = e_{\mathrm{disp}}\).

## Performance

Source: `REFERENCE.md` (queue / handoff: **perf_exempt**).

D3 evaluation itself is \(O(N^2)\) over a 6-atom cell (microseconds). Full GPW
SCF wall is dominated by the electronic structure, not D3. A fair
**dispersion-only** microbenchmark vs CP2K is not meaningful; a full twin SCF
ratio depends on cutoff/OT choices and is not the product gate.

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | ------------------------: | ----------------------: | ----: | ---- | ------ |
| Full QS (context) | ≈ 13.4 (1 rank × 4 OMP, pin host) | — (not fair-twinned) | **perf_exempt** | — | CP2K wall is full QS; hard gate is E_disp only |
| E_disp product gate | n/a | short SCF + D3 (reg) | **perf_exempt** | — | Geometry-only deliverable |

**perf_exempt = true.** Reason (from REFERENCE / closed acceptance ac-006):
primary deliverable is E_disp pin parity; CP2K wall includes full Quickstep
(~13 s) while the hard-gate regression only needs a short SCF to materialize
`e_disp`. Re-measure a strict full-SCF twin later if product policy requires
\(r = T_{\mathrm{ATV}}/T_{\mathrm{CP2K}} \le 10\).

## Limitations

- **Cell is a water dimer**, not bulk ice Ih — sufficient for periodic D3(BJ)
  lattice-sum parity; not a structural ice benchmark.
- **E_disp hard gate only** — absolute GPW \(e_{\mathrm{tot}}\) vs CP2K QS is
  not asserted (OT/grid/DIIS differences).
- **Two-body only** — no ATM / C9 three-body term.
- **PBE parameters only** on this pin; other reference functionals share tables
  but are not e2e-gated here.
- **CN split:** GPW uses CP2K CN; molecular `D3BJ` uses Grimme CN — do not mix
  the two when comparing to PySCF.
- **`d3` and `d3bj` mutually exclusive** — both true throws at run.
- Wall **perf_exempt** for this wave; not a speed optimization target.

## References

- Offline pin: `regressions/gpw/gpw-pbe-d3bj-ice/REFERENCE.md`
- Oracle inputs: `tools/cp2k_oracle/inputs/gpw-pbe-d3bj-ice/`
- Roadmap: `docs/roadmap/dft.md` §8 Dispersion
- GPW narrative chapter: `docs/tutorial/ch09-gpw.md`
- Closed spec (git history): `.claude/specs/gpw-pbe-d3bj-ice/` (deleted at done)
- S. Grimme, S. Ehrlich, L. Goerigk, *J. Comput. Chem.* **32**, 1456 (2011)
  (DFT-D3 with BJ damping)
