# GPW free-atom XDM — periodic water dimer (ice-relevant H-bond)

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `gpw-xdm-ice` |
| Regression | `regressions/gpw/gpw-xdm-ice/` |
| Public API | `cpu::gpw::GpwDft::xdm` / `cpu::gpw::GPWEvaluator::xdm`; `GpwResult::e_disp` [Ha] |
| Roadmap | `docs/roadmap/dft.md` §8 Dispersion |

## Summary

**Exchange-hole dipole moment (XDM)** pairwise dispersion is wired into the
**Γ-point GPW** path as an additive London correction on top of PBE. This
product pin is the **free-atom foothold**: multipole moments \(M_1,M_2,M_3\) and
static polarizabilities \(\alpha\) are tabulated free-atom values (no Hirshfeld
volume scaling, no density-dependent BR hole moments). The system is a
**periodic water-dimer cell** (ice-relevant H-bonded pair in a 12 Å cubic box —
not a full ice Ih supercell), matching `gpw-ts-ice` / `gpw-pbe-d3bj-ice`
geometry.

**Why not a stock CP2K QS pin?** CP2K 2026.1 `PAIR_POTENTIAL TYPE` accepts only
`DFTD2|DFTD3|DFTD3(BJ)|DFTD4` — there is **no** `TYPE XDM`. Canonical full
(density-dependent) XDM oracles are postg / critic2 / Quantum ESPRESSO
`xdm_dispersion`. This slug therefore gates against an **offline dual
pure-Python lattice sum** of the free-atom XDM formula (same policy as
free-atom TS when stock CP2K is unavailable). Twin note retained under
`tools/cp2k_oracle/inputs/gpw-xdm-ice/`.

Public surface: bool members `xdm` on `cpu::gpw::GpwDft` and
`cpu::gpw::GPWEvaluator` (no `*Config` bag). Dispersion is reported on
`GpwResult::e_disp` [Hartree] and folded into `e_total`. Mutually exclusive
with `d3` / `d3bj` / `d4` / `ts` / `mbd` / `vdw_df` / `vdw_df2` / `rvv10`.
Product gate: `regressions/gpw/gpw-xdm-ice` (`reg_gpw-xdm-ice`). Wall ratio is
**perf_exempt** (E_disp pin; full QS twin wall unfair — see Performance).

## Method

**Free-atom XDM** with Becke–Johnson combination rules and BJ damping
(Becke & Johnson, *J. Chem. Phys.* **127**, 154108 (2007); postg / QE
`PW/src/xdm_dispersion.f90`, Otero-de-la-Roza & Johnson). Periodic pure
real-space lattice sum over image vectors \(\mathbf{T}\):

\[
\mathrm{fac}=\frac{\alpha_i\alpha_j}{M_{1i}\alpha_j+M_{1j}\alpha_i},\quad
C_6=\mathrm{fac}\,M_{1i}M_{1j},
\]

\[
C_8=\tfrac{3}{2}\,\mathrm{fac}\,(M_{1i}M_{2j}+M_{2i}M_{1j}),
\]

\[
C_{10}=2\,\mathrm{fac}\,(M_{1i}M_{3j}+M_{3i}M_{1j})
+\tfrac{21}{5}\,\mathrm{fac}\,M_{2i}M_{2j},
\]

\[
R_c=\frac{1}{3}\Bigl(\sqrt{C_8/C_6}+(C_{10}/C_6)^{1/4}+\sqrt{C_{10}/C_8}\Bigr),
\quad R_{\mathrm{vdW}}=a_1 R_c+a_2,
\]

\[
E_{\mathrm{disp}}=-\sum_{i\le j}\sum_{\mathbf T}{}^{\prime}
\Biggl[
\frac{C_6}{R^6+R_{\mathrm{vdW}}^6}
+\frac{C_8}{R^8+R_{\mathrm{vdW}}^8}
+\frac{C_{10}}{R^{10}+R_{\mathrm{vdW}}^{10}}
\Biggr].
\]

The prime excludes the singular \(R=0\) self term.

**PBE damping (QE NC-PBE / postg `xdm.param`):**

| Parameter | Value | Unit |
| --------- | ----: | ---- |
| \(a_1\) | 0.4283 | dimensionless |
| \(a_2\) | 2.4690 | Å (\(\approx 4.66573\) Bohr in-tree) |

**Free-atom tables** (in-tree `basis_data/xdm_tables.h`; CRC Handbook 88th Ed.
\(\alpha\); \(M_n\) inverted from literature homo \(C_n\)):

| Z | \(\alpha\) [Bohr³] | \(M_1\) | \(M_2\) | \(M_3\) | Source \(C_n\) |
| - | ------------------ | ------- | ------- | ------- | -------------- |
| H | 4.49979 | 2.88859 | 18.4304 | 241.636 | Yan et al. PRA 54, 2824 |
| O | 5.41216 | 5.74631 | 28.9471 | 262.617 | Chu–Dalgarno JCP 121, 4083 (\(C_6\)); consistent \(C_8,C_{10}\) |

Only H and O are populated for this ice water-dimer foothold; unsupported \(Z\)
throws (no fallback).

**Pair cutoff.** Pure real-space orthorhombic lattice sum via
`pair_image_counts` with **rcut = 50 Å** (\(\approx 94.2\) Bohr default in
`xdm_pbe`).

**Electronic host.** Dispersion is geometry-only; the regression still runs a
short Γ-RKS GPW (PBE, `DZVP-MOLOPT-SR-GTH` + `GTH-PBE-q*`) so `e_disp` is
materialized on a real `GpwResult` and included in `e_total`. Absolute QS
\(E_{\mathrm{tot}}\) micro-Ha parity is **out of scope** (no CP2K XDM twin;
OT/grid/DIIS differences vs any other engine).

**Key exclusions.**

- Density-dependent Hirshfeld / BR exchange-hole moments (full XDM)
- Full ice Ih supercell / phonons
- Stock CP2K `TYPE XDM` (does not exist in 2026.1)
- k-points, UKS, cell stress as part of this slug
- CUDA GPW
- Absolute \(e_{\mathrm{tot}}\) pin vs any QS engine
- Free-atom tables beyond H/O

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

// Geometry [Å] → Bohr — periodic water dimer (REFERENCE.md).
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
gpw.xdm = true;        // mutually exclusive with d3/d3bj/d4/ts/mbd/…

gpw.build(atoms);
const auto& res = gpw.run(atoms);
// res.e_disp [Ha] within 1e-8 of offline dual-Python pin
// res.e_total includes e_disp
```

Driver / Frame path (component; same member):

```cpp
// kinds = {O, H} Kind{Element, GaussianBasis, GthPotential}
ATV::cpu::gpw::GPWEvaluator eval(kinds, /*cutoff_ry=*/300.0, /*max_iter=*/200,
                                 /*tol=*/1e-7, /*mix=*/0.3,
                                 ATV::cpu::gpw::XcKind::PBE,
                                 /*nonperiodic=*/false);
eval.xdm = true;
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
  cmake --build build --target reg_gpw-xdm-ice -j$(nproc)
  ./build/regressions/reg_gpw-xdm-ice
'
```

| Path | Role |
| ---- | ---- |
| `include/atomiverse/cpu/gpw/gpw_solver.h` | `GpwDft`, `GpwResult::e_disp`, `xdm` |
| `include/atomiverse/cpu/gpw/gpw_evaluator.h` | `GPWEvaluator::xdm` |
| `include/atomiverse/cpu/gpw/internal/xdm_dispersion.h` | `xdm_pbe` free-atom engine (private) |
| `include/atomiverse/cpu/gpw/basis_data/xdm_tables.h` | Free-atom \(\alpha,M_n\); QE NC-PBE \(a_1,a_2\) |
| `regressions/gpw/gpw-xdm-ice/reg_gpw-xdm-ice.cpp` | Product gate |
| `regressions/gpw/gpw-xdm-ice/REFERENCE.md` | Offline pin + perf_exempt |
| `tools/cp2k_oracle/inputs/gpw-xdm-ice/` | CP2K-loop twin **note** (no stock XDM input) |

## Validation (offline dual-Python pin; CP2K capability loop)

Source:
[`regressions/gpw/gpw-xdm-ice/REFERENCE.md`](../../../regressions/gpw/gpw-xdm-ice/REFERENCE.md).
Oracle context: **CP2K capability loop** (`tools/cp2k_oracle/`), but **no**
stock CP2K XDM energy — hard gate is dual pure-Python free-atom lattice sum
(postg / QE combination rules). Twin note:
`tools/cp2k_oracle/inputs/gpw-xdm-ice/water_dimer_xdm.note`. External offline
pin — not a golden of Atomiverse’s own output; **never** launch live CP2K (or
postg/QE) from ctest.

**Setup:**

| Item | Value |
| ---- | ----- |
| System | Water dimer (ice-relevant H-bond pair) |
| Geometry [Å] | O (0,0,0); H (0.758602, 0, ±0.504284); O (3,0,0); H (3.758602, 0, ±0.504284) |
| Cell | ABC **12.0 12.0 12.0 Å**, PERIODIC XYZ |
| Basis / PP | `DZVP-MOLOPT-SR-GTH` + `GTH-PBE-q6` (O) / `GTH-PBE-q1` (H) (E_disp basis-independent) |
| XC + disp | PBE + free-atom XDM (QE NC-PBE BJ \(a_1,a_2\)) |
| R_CUTOFF | 50 Å pure real-space |
| Occupations | RKS, 16 valence e⁻ → `n_occ = 8` |
| Pin line | \(E_{\mathrm{disp}} = -2.1148157612879029\times 10^{-3}\) Ha |

| Quantity | Atomiverse / pin | Offline dual-Python | Tol | Notes |
| -------- | ---------------: | ------------------: | --- | ----- |
| \(E_{\mathrm{disp}}\) periodic [Ha] | −2.114815761287903×10⁻³ | −2.114815761287903×10⁻³ | \(1\times 10^{-8}\) Ha | Hard gate |
| \(E_{\mathrm{disp}}\) isolated [Ha] | (unit test) | −2.102043599851622×10⁻³ | tight kernel | Cross-check |
| \(E_{\mathrm{tot}}\) QS [Ha] | (not gated) | — | — | Context only; no CP2K XDM twin |

**Tolerance rationale.** Matching free-atom \(\alpha,M_n\), QE NC-PBE damping,
and 50 Å real-space rcut tracks the dual-Python lattice sum to machine
precision on the kernel (\(\sim 10^{-16}\) Ha); regression hard gate is
\(1\times 10^{-8}\) Ha. Unit tests also pin isolated dual-Python and analytic
forces vs central FD.

## Performance

Source: `REFERENCE.md` (queue / handoff: **perf_exempt**).

XDM evaluation is \(O(N^2\times N_{\mathrm{images}})\) over a 6-atom cell
(microseconds). Full GPW SCF wall is dominated by the electronic structure, not
the dispersion sum. There is no stock CP2K XDM twin wall to fair-bench.

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | ------------------------: | ----------------------: | ----: | ---- | ------ |
| Stock CP2K XDM QS | n/a (no `TYPE XDM`) | — | **perf_exempt** | — | Twin note only |
| E_disp product gate | n/a | short SCF + XDM (reg) | **perf_exempt** | — | Geometry-only deliverable |

**perf_exempt = true.** Reason (from REFERENCE / handoff): primary deliverable
is E_disp pin parity; full twin SCF ratio is unfair for a geometry-only hard
gate (same policy as `gpw-ts-ice` / `gpw-pbe-d3bj-ice`). Re-measure only if a
future stock engine pin (or density-dependent full XDM) becomes the product
gate.

## Limitations

- **Free-atom foothold only** — no Hirshfeld / BR density-dependent moments
  (full postg/QE XDM remains out of scope).
- **Cell is a water dimer**, not bulk ice Ih — sufficient for periodic
  free-atom lattice-sum parity; not a structural ice benchmark.
- **No stock CP2K XDM** in 2026.1 — pin is dual pure-Python, not CP2K QS
  energy.
- **E_disp hard gate only** — absolute GPW \(e_{\mathrm{tot}}\) is not
  asserted against any external QS engine.
- **Tables cover H and O only** for this foothold; other \(Z\) throws.
- **`xdm` mutually exclusive** with other GPW dispersion / nonlocal flags —
  any second flag true throws at run.
- Wall **perf_exempt** for this wave; not a speed optimization target.

## References

- Offline pin: `regressions/gpw/gpw-xdm-ice/REFERENCE.md`
- Oracle note: `tools/cp2k_oracle/inputs/gpw-xdm-ice/water_dimer_xdm.note`
- Roadmap: `docs/roadmap/dft.md` §8 Dispersion
- GPW narrative chapter: `docs/tutorial/ch09-gpw.md`
- Closed spec (git history): `.claude/specs/gpw-xdm-ice/` (deleted at done)
- A. D. Becke, E. R. Johnson, *J. Chem. Phys.* **127**, 154108 (2007) (XDM)
- A. Otero-de-la-Roza, E. R. Johnson, postg / QE `xdm_dispersion` (NC-PBE
  \(a_1,a_2\); free-atom tables)
- Z.-C. Yan et al., *Phys. Rev. A* **54**, 2824 (1996) (H \(C_n\))
- X. Chu, A. Dalgarno, *J. Chem. Phys.* **121**, 4083 (2004) (O \(C_6\))
