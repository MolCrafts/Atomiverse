# GPW PBE-D3 + ATM — isolated water dimer (three-body C9)

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `gpw-d3-atm` |
| Regression | `regressions/gpw/gpw-d3-atm/` |
| Public API | `cpu::gpw::GpwDft::d3` + `d3_atm`; `cpu::gpw::GPWEvaluator::d3` + `d3_atm`; `GpwResult::e_disp` [Ha] |
| Roadmap | `docs/roadmap/dft.md` §8 Dispersion |

## Summary

Grimme **DFT-D3 zero damping** (PBE) with the optional **Axilrod–Teller–Muto
(ATM) three-body C9** term is wired into the **GPW** path as an additive
London correction. This product pin uses an **isolated water dimer** in a
12 Å cubic box (`PERIODIC NONE` / MT host) and gates the **dispersion energy**
against offline **CP2K 2026.1** `TYPE DFTD3` with `CALCULATE_C9_TERM .TRUE.`.

Public surface: bool members `d3` and `d3_atm` on `cpu::gpw::GpwDft` and
`cpu::gpw::GPWEvaluator` (no `*Config` bag). `d3_atm` enables C9 on top of
either `d3` or `d3bj`; it is ignored when neither two-body flag is set.
Dispersion is reported on `GpwResult::e_disp` [Hartree] and folded into
`e_total`. Product gate: `regressions/gpw/gpw-d3-atm` (`reg_gpw-d3-atm`).
Wall ratio is **perf_exempt** (E_disp pin; full QS twin wall unfair — see
Performance).

## Method

**Two-body DFT-D3 zero damping** (Grimme et al., *J. Chem. Phys.* **132**,
154104 (2010)), matching CP2K `qs_dispersion_d3`:

\[
E_{\mathrm{disp}}^{(2)} = -\sum_{A<B}\sum_{n=6,8}
  s_n\,\frac{C_n^{AB}}{R_{AB}^{n}}\,f_{d,n}(R_{AB}),
\quad
f_{d,n}(R)=\frac{1}{1+6\bigl(R/(s_{r,n} R_0^{AB})\bigr)^{-\alpha_n}},
\]

with \(\alpha_6=14\), \(\alpha_8=16\),
\(C_8^{AB}=3\,C_6^{AB}\,r_{2r4}^{A} r_{2r4}^{B}\), and CN-interpolated
reference \(C_6\) (\(k_3=-4\)). Coordination numbers use the CP2K
`cnparam_d3` form (including the long-range
\(\tfrac12(1-\tanh(R-2 r_{\mathrm{cov}}))\) factor).

**PBE-D3 (zero) parameters** (CP2K `REFERENCE_FUNCTIONAL PBE`):

| Parameter | Value | Unit |
| --------- | ----: | ---- |
| \(s_6\) | 1.0 | dimensionless |
| \(s_8\) | 0.722 | dimensionless |
| \(s_{r,6}\) | 1.217 | dimensionless |
| \(s_{r,8}\) | 1.0 | dimensionless |

**Three-body ATM / C9** (CP2K `CALCULATE_C9_TERM`):

\[
C_9^{ABC} = -\sqrt{C_6^{AB}\,C_6^{BC}\,C_6^{CA}},
\quad
E_9 \;\text{contrib.}\; \propto -s_9\,f_d\,C_9\,\mathrm{ang}\big/
  (R_{AB}^{2} R_{BC}^{2} R_{CA}^{2})^{3/2},
\]

with \(s_9 = s_6\), damping `damping_d3(rabc, r0, 4/3, 16)`, and angle factor
from CP2K’s \(s_1\cdot s_2\cdot s_3\) form. A smooth fifth-order cutoff
zeroes each term over \([r_{\mathrm{cut}}-1,\, r_{\mathrm{cut}}]\).

**Pair / ABC cutoff.** Atomiverse uses **rcut = 2 × R_CUTOFF 20 Å ≈ 75.59 Bohr**
(CP2K prints `Cutoff Radius [Bohr]: 37.79` for `R_CUTOFF 20`; the 2×
convention matches the two-body GPW D3 pin). Twin:
`LONG_RANGE_CORRECTION .FALSE.`.

**Electronic host.** Dispersion is geometry-only; the regression still runs a
short Γ-RKS GPW (PBE, `DZVP-MOLOPT-SR-GTH` + `GTH-PBE-q*`, `nonperiodic = true`)
so `e_disp` is materialized on a real `GpwResult` and included in `e_total`.
Absolute QS \(E_{\mathrm{tot}}\) micro-Ha parity is **out of scope**
(OT/grid/DIIS differences).

**ATM magnitude on this pin.** Two-body-only baseline (same geometry, ATM off):
**−0.00081399213205 Ha**. With ATM:
**−0.00081414118193 Ha**. ATM Δ ≈ **−1.49×10⁻⁷ Ha** (attractive three-body on
the dimer).

**Key exclusions.**

- Periodic lattice-sum ATM product gate (this pin is **isolated** / MT)
- Full ice Ih supercell / phonons
- Dedicated D3 ATM force / stress e2e product gate (unit tests cover ATM
  analytic forces vs FD; force/stress QS twins are separate slugs)
- DFT-D4 ATM (`d4_atm` / `gpw-d4-atm`)
- Molecular `cpu::scf::D3BJ` / PySCF pin changes
- Absolute \(e_{\mathrm{tot}}\) pin vs CP2K QS
- k-points, UKS, cell stress as part of this slug
- CUDA GPW

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
gpw.nonperiodic = true; // isolated / MT — matches twin PERIODIC NONE
gpw.d3 = true;          // zero damping; exclusive vs d3bj / d4 / ts / …
gpw.d3_atm = true;      // CP2K CALCULATE_C9_TERM

gpw.build(atoms);
const auto& res = gpw.run(atoms);
// res.e_disp [Ha] within 1e-9 of offline CP2K pin
// res.e_total includes e_disp
```

Driver / Frame path (component; same members):

```cpp
// kinds = {O, H} Kind{Element, GaussianBasis, GthPotential}
ATV::cpu::gpw::GPWEvaluator eval(kinds, /*cutoff_ry=*/300.0, /*max_iter=*/200,
                                 /*tol=*/1e-7, /*mix=*/0.3,
                                 ATV::cpu::gpw::XcKind::PBE,
                                 /*nonperiodic=*/true);
eval.d3 = true;
eval.d3_atm = true;
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
  cmake --build build --target reg_gpw-d3-atm -j$(nproc)
  ./build/regressions/reg_gpw-d3-atm
'
```

| Path | Role |
| ---- | ---- |
| `include/atomiverse/cpu/gpw/gpw_solver.h` | `GpwDft`, `GpwResult::e_disp`, `d3`, `d3_atm` |
| `include/atomiverse/cpu/gpw/gpw_evaluator.h` | `GPWEvaluator::d3`, `d3_atm` |
| `include/atomiverse/cpu/gpw/internal/dftd3.h` | `dftd3_pbe(..., calculate_c9)` engine (private) |
| `include/atomiverse/cpu/gpw/internal/dftd3_params.h` | In-tree D3 tables (`D3Params{}`) |
| `regressions/gpw/gpw-d3-atm/reg_gpw-d3-atm.cpp` | Product gate |
| `regressions/gpw/gpw-d3-atm/REFERENCE.md` | Offline pin + perf_exempt |
| `tools/cp2k_oracle/inputs/gpw-d3-atm/` | CP2K twin input |

## Validation (CP2K)

Source:
[`regressions/gpw/gpw-d3-atm/REFERENCE.md`](../../../regressions/gpw/gpw-d3-atm/REFERENCE.md).
Oracle: offline **CP2K 2026.1** (`git` / module
`2026.1-gmpflf-2026.03` on Arrhenius), twin
`tools/cp2k_oracle/inputs/gpw-d3-atm/water_dimer_d3_atm.inp`. External
engine pin — not a golden of Atomiverse’s own output; **never** launch live
CP2K from ctest.

**Setup:**

| Item | Value |
| ---- | ----- |
| System | Isolated water dimer |
| Geometry [Å] | O (0,0,0); H (0.758602, 0, ±0.504284); O (3,0,0); H (3.758602, 0, ±0.504284) |
| Cell | ABC **12.0 12.0 12.0 Å**, **PERIODIC NONE** |
| Basis / PP | `DZVP-MOLOPT-SR-GTH` + `GTH-PBE-q6` (O) / `GTH-PBE-q1` (H) |
| XC + disp | PBE + `TYPE DFTD3`, `REFERENCE_FUNCTIONAL PBE` |
| ATM | `CALCULATE_C9_TERM .TRUE.` |
| R_CUTOFF | 20.0 (Å in twin; ATV rcut = 2× ≈ 75.59 Bohr) |
| Occupations | RKS, 16 valence e⁻ → `n_occ = 8` |
| Pin line | `Dispersion energy: -0.00081414118193` |

| Quantity | Atomiverse / pin | CP2K (offline) | Tol | Notes |
| -------- | ---------------: | -------------: | --- | ----- |
| \(E_{\mathrm{disp}}\) (D3+ATM) [Ha] | −0.00081414118193 | −0.00081414118193 | \(1\times 10^{-9}\) Ha | Hard gate |
| \(E_{\mathrm{disp}}\) (two-body only) [Ha] | −0.00081399213205 | (unit-test pin) | — | Context; ATM Δ ≈ −1.49×10⁻⁷ Ha |

**Tolerance rationale.** Matching geometry, PBE-D3 zero parameters, CP2K CN,
rcut 40 Bohr (2× printed CP2K cutoff), and ATM form track CP2K dispersion to
sub-nano-Hartree. Unit tests also check analytic ATM forces vs central FD and
that default `d3_atm = false` keeps the two-body pin.

## Performance

Source: `REFERENCE.md` (queue: **perf_exempt**).

D3 + ATM evaluation itself is \(O(N^2)\) + \(O(N^3)\) over a 6-atom cell
(microseconds). Full GPW SCF wall is dominated by the electronic structure,
not dispersion. A fair **dispersion-only** microbenchmark vs CP2K is not
meaningful; a full twin SCF ratio depends on cutoff/OT choices and is not the
product gate.

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | ------------------------: | ----------------------: | ----: | ---- | ------ |
| Full QS (context) | ≈ 10.8 (1 rank × 4 OMP, pin host) | — (not fair-twinned) | **perf_exempt** | — | CP2K wall is full QS; hard gate is E_disp only |
| E_disp product gate | n/a | short SCF + D3 ATM (reg) | **perf_exempt** | — | Geometry-only deliverable |

**perf_exempt = true.** Reason (from REFERENCE): primary deliverable is E_disp
pin parity; CP2K wall includes full Quickstep (~11 s) while the hard-gate
regression only needs a short SCF to materialize `e_disp`. Re-measure a
strict full-SCF twin later if product policy requires
\(r = T_{\mathrm{ATV}}/T_{\mathrm{CP2K}} \le 10\).

## Limitations

- **Isolated dimer**, not bulk ice or periodic ATM lattice-sum product gate.
- **E_disp hard gate only** — absolute GPW \(e_{\mathrm{tot}}\) vs CP2K QS is
  not asserted (OT/grid/DIIS differences).
- **ATM Δ is tiny** on this geometry (~10⁻⁷ Ha); the pin still locks the
  three-body form against CP2K.
- **`d3` and `d3bj` mutually exclusive** with each other and with `d4` / `ts`
  / `mbd` / `xdm` — both two-body flags true throws at run. `d3_atm` stacks
  only on an active D3 two-body flag.
- **Dedicated ATM force/stress e2e** is out of this slug’s product gate
  (unit FD coverage exists; QS force/stress twins are separate queue rows).
- Wall **perf_exempt** for this wave; not a speed optimization target.

## References

- Offline pin: `regressions/gpw/gpw-d3-atm/REFERENCE.md`
- Oracle inputs: `tools/cp2k_oracle/inputs/gpw-d3-atm/`
- Roadmap: `docs/roadmap/dft.md` §8 Dispersion
- Related two-body D3(BJ): `docs/tutorial/methods/gpw-pbe-d3bj-ice.md`
- Related D4 ATM: `regressions/gpw/gpw-d4-atm/`
- GPW narrative chapter: `docs/tutorial/ch09-gpw.md`
- S. Grimme, J. Antony, S. Ehrlich, H. Krieg, *J. Chem. Phys.* **132**, 154104
  (2010) (DFT-D3 zero damping)
- B. M. Axilrod, E. Teller, *J. Chem. Phys.* **11**, 299 (1943); Y. Muto,
  *Proc. Phys.-Math. Soc. Jpn.* **17**, 629 (1943) (three-body ATM)
