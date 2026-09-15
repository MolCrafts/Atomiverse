# GPW multi-k Hirshfeld volumes — water dimer (PBE + MBD@rsSCS)

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `gpw-hirshfeld-kpoints` |
| Regression | `regressions/gpw/gpw-hirshfeld-kpoints/` |
| Public API | `cpu::gpw::GpwDft::hirshfeld` + `mbd` + `k_mesh`; `GpwResult::e_disp` [Ha], `GpwResult::volume_ratios`; same members on `GPWEvaluator` |
| Roadmap | `docs/roadmap/dft.md` §8 Dispersion |

## Summary

**Hirshfeld volume ratios** \(V/V_{\mathrm{free}}\) are computed from the final
**multi-k** (Monkhorst–Pack) **GPW PBE** valence density and used to
**density-scale** free-atom **MBD@rsSCS** (or TS) parameters. This product pin
extends Γ-only Hirshfeld (`reg_gpw-mbd-hirshfeld`) to the **k-summed density**
path: \(\rho=\sum_k w_k\rho_k\), then the same in-tree Hirshfeld kernel and
libMBD free-atom tables as the Γ product.

Public surface: bool members `mbd` / `ts` and `hirshfeld` on
`cpu::gpw::GpwDft` and `cpu::gpw::GPWEvaluator`, plus integer `k_mesh[3]` and
optional `k_time_reversal` (no `*Config` bag). Dispersion lands on
`GpwResult::e_disp` [Hartree]; per-atom ratios on `GpwResult::volume_ratios`.
Product gate: `regressions/gpw/gpw-hirshfeld-kpoints`
(`reg_gpw-hirshfeld-kpoints`). Wall ratio is **perf_exempt** (E_disp +
volume-ratio pin; SCF dominates wall — same policy as free-atom / Γ Hirshfeld
MBD).

## Method

**Hirshfeld effective volume** (Tkatchenko & Scheffler, *Phys. Rev. Lett.*
**102**, 073005 (2009)) on a uniform orthorhombic GPW density grid:

\[
w_A(\mathbf{r})
  = \frac{\rho_A^{\mathrm{free}}(|\mathbf{r}-\mathbf{R}_A|)}
         {\sum_B\rho_B^{\mathrm{free}}(|\mathbf{r}-\mathbf{R}_B|)},
\quad
V_A = \int w_A(\mathbf{r})\,\rho(\mathbf{r})\,|\mathbf{r}-\mathbf{R}_A|^3\,
  d^3r,
\quad
v_A = V_A / V_A^{\mathrm{free}}.
\]

Free-atom densities \(\rho_A^{\mathrm{free}}\) and \(V_A^{\mathrm{free}}\) use
the **in-tree exponential model** (`basis_data/hirshfeld_tables.h`) — the same
kernel as `gpw-mbd-hirshfeld`. Minimum-image distances when the cell is
periodic.

**Multi-k density.** With `k_mesh` all positive, the SCF accumulates the
k-summed valence density on the real-space grid. Hirshfeld runs **once** after
the multi-k SCF (not per k). With time-reversal fold, a \(2\times2\times2\) mesh
yields **4** irreducible k-points on this orthorhombic cell (`n_kpts = 4`).

**Density-scaled MBD@rsSCS** (Ambrosetti et al., *J. Chem. Phys.* **140**,
18A508 (2014)). Free-atom polarizability / C₆ / van der Waals radius scale as

\[
\alpha \propto v,\qquad C_6 \propto v^2,\qquad R \propto v^{1/3}.
\]

Default `mbd_k_mesh = {0,0,0}` keeps **isolated** (geometry-cell) MBD@rsSCS —
only the **Hirshfeld volumes** come from multi-k density. Periodic Ewald MBD
is a separate product (`reg_gpw-mbd-periodic`).

**Electronic host.** Short multi-k RKS GPW (PBE, `DZVP-MOLOPT-SR-GTH` +
`GTH-PBE-q*`, CUTOFF 200 Ry) materializes \(\rho\) and folds \(E_{\mathrm{disp}}\)
into `e_total`. Absolute QS \(E_{\mathrm{tot}}\) micro-Ha parity vs any external
engine is **out of scope**.

**Key exclusions.**

- Stock **CP2K 2026.1** `PAIR_POTENTIAL TYPE` is only
  `DFTD2|DFTD3|DFTD3(BJ)|DFTD4` — **no TS/MBD**; hard pin is the ATV product
  under fixed settings, dual-gated vs free-atom MBD and Γ Hirshfeld
- **UKS + hirshfeld** throws (closed-shell RKS only, Γ and multi-k)
- Standalone Hirshfeld without `ts` or `mbd` throws
- Hirshfeld does **not** apply to D3 / D3(BJ) / D4 / XDM / vdW-DF*
- Periodic Ewald MBD mesh (`mbd_k_mesh`) is not required for this pin
- CUDA GPW, analytic Hirshfeld forces as a separate deliverable
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

// Geometry [Å] → Bohr — same water dimer as gpw-mbd-ice / gpw-mbd-hirshfeld.
constexpr double kA2B = 1.0 / 0.52917720859;
// … fill 6 GpwAtom rows (2×O + 4×H) with DZVP-MOLOPT-SR-GTH + GTH-PBE-q* …

GpwDft gpw;
const double L = 12.0 * kA2B;
gpw.cell[0] = gpw.cell[1] = gpw.cell[2] = L;
gpw.cutoff_ry = 200.0;
gpw.n_occ = 8;          // 16 valence e⁻ → 8 doubly occupied
gpw.max_iter = 16;
gpw.tol = 1e-6;
gpw.mix = 0.4;
gpw.xc = XcKind::PBE;
gpw.nonperiodic = false;
gpw.mbd = true;
gpw.hirshfeld = true;   // requires ts or mbd; RKS only
gpw.k_mesh[0] = gpw.k_mesh[1] = gpw.k_mesh[2] = 2;
gpw.k_time_reversal = true;

gpw.build(atoms);
const auto& res = gpw.run(atoms);
// res.n_kpts >= 2 (product: 4 with TR fold)
// res.e_disp [Ha] within 1e-8 of offline multi-k Hirshfeld-MBD pin
// res.volume_ratios size 6, each in (0.3, 1.5)
// res.e_total includes e_disp
```

Driver / Frame path (component; same members):

```cpp
// kinds = {O, H} Kind{Element, GaussianBasis, GthPotential}
ATV::cpu::gpw::GPWEvaluator eval(kinds, /*cutoff_ry=*/200.0, /*max_iter=*/16,
                                 /*tol=*/1e-6, /*mix=*/0.4,
                                 ATV::cpu::gpw::XcKind::PBE,
                                 /*nonperiodic=*/false);
eval.mbd = true;
eval.hirshfeld = true;
// set k-mesh on the owned GpwDft path used by the evaluator when multi-k is
// configured for the run (same product members as GpwDft)
// single_point(frame, eval) or bind on Evaluate / MD / Minimize force slot
// eval.last_result().e_disp  [Hartree]
// eval.last_result().volume_ratios
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
  cmake --build build -j$(nproc) --target atomiverse reg_gpw-hirshfeld-kpoints
  ./build/regressions/reg_gpw-hirshfeld-kpoints
'
```

| Path | Role |
| ---- | ---- |
| `include/atomiverse/cpu/gpw/gpw_solver.h` | `GpwDft::{mbd,hirshfeld,k_mesh}`, `GpwResult::{e_disp,volume_ratios,n_kpts}` |
| `include/atomiverse/cpu/gpw/gpw_evaluator.h` | `GPWEvaluator::hirshfeld` / `mbd` |
| `include/atomiverse/cpu/gpw/internal/hirshfeld_volumes.h` | \(V/V_{\mathrm{free}}\) kernel (private) |
| `include/atomiverse/cpu/gpw/basis_data/hirshfeld_tables.h` | Free-atom exponential tables |
| `regressions/gpw/gpw-hirshfeld-kpoints/reg_gpw-hirshfeld-kpoints.cpp` | Product gate |
| `regressions/gpw/gpw-hirshfeld-kpoints/REFERENCE.md` | Offline pin + perf_exempt |
| `tools/cp2k_oracle/inputs/gpw-hirshfeld-kpoints/` | Oracle note (no stock CP2K TS/MBD) |
| `tests/cpu/gpw/test_cpu_gpw_hirshfeld.cpp` | Unit gates (volumes + UKS throw) |

## Validation (CP2K capability / offline product pin)

Source:
[`regressions/gpw/gpw-hirshfeld-kpoints/REFERENCE.md`](../../../regressions/gpw/gpw-hirshfeld-kpoints/REFERENCE.md).
Oracle notes:
[`tools/cp2k_oracle/inputs/gpw-hirshfeld-kpoints/water_dimer_k_hirshfeld.note`](../../../tools/cp2k_oracle/inputs/gpw-hirshfeld-kpoints/water_dimer_k_hirshfeld.note).
**Never** launch live CP2K (or recompute pins) from ctest. Because stock
CP2K 2026.1 has no TS/MBD `PAIR_POTENTIAL`, the hard gate freezes the **ATV
product** under fixed settings and cross-checks against free-atom MBD and the
Γ Hirshfeld dual kernel (same tables / scaling as `gpw-mbd-hirshfeld`).

**Setup:**

| Item | Value |
| ---- | ----- |
| System | Water dimer (same geometry as `gpw-mbd-ice` / `gpw-mbd-hirshfeld` / `gpw-ts-ice`) |
| Geometry [Å] | O (0,0,0); H (0.758602, 0, ±0.504284); O (3,0,0); H (3.758602, 0, ±0.504284) |
| Cell | ABC **12.0 12.0 12.0 Å**, PERIODIC XYZ |
| k-mesh | Monkhorst–Pack **2×2×2**, time-reversal → **4** irreducible k |
| Basis / PP | `DZVP-MOLOPT-SR-GTH` + `GTH-PBE-q6` (O) / `GTH-PBE-q1` (H) |
| XC + disp | PBE + MBD@rsSCS with Hirshfeld \(v_A\) from multi-k \(\rho\) |
| CUTOFF | **200 Ry** |
| Occupations | RKS, 16 valence e⁻ → `n_occ = 8` |
| SCF | max_iter 16, tol \(1\times10^{-6}\), mix 0.4 |
| Pin line | \(E_{\mathrm{disp}} = -1.1479849553026611\times10^{-3}\) Ha |

| Quantity | Atomiverse / pin | Cross-check | Tol | Notes |
| -------- | ---------------: | ----------: | --- | ----- |
| \(E_{\mathrm{disp}}\) multi-k Hirshfeld-MBD [Ha] | **−1.1479849553026611×10⁻³** | dual product gate | \(1\times10^{-8}\) Ha | Hard regression gate |
| \(E_{\mathrm{disp}}\) Γ Hirshfeld-MBD [Ha] | −1.1479852469005181×10⁻³ | `reg_gpw-mbd-hirshfeld` | context | \(\Delta\sim3\times10^{-10}\) Ha on this cell |
| \(E_{\mathrm{disp}}\) free-atom MBD [Ha] | −1.4928804171407961×10⁻³ | `reg_gpw-mbd-ice` | context | \(v_A=1\) (no density scale) |
| \(v_A\) (O H H O H H) | ≈ 0.7403, 0.6529, 0.6529, 0.7153, 0.5784, 0.5784 | product range (0.3, 1.5) | qualitative | Reported on `volume_ratios` |
| \(n_{\mathrm{kpts}}\) | ≥ 2 (product 4 with TR) | multi-k path | integer | Fails if Γ-only |

**Tolerance rationale.** On the 12 Å molecular cell the multi-k vs Γ density
difference is tiny, so \(E_{\mathrm{disp}}\) nearly matches the Γ Hirshfeld pin;
the hard gate still freezes the **multi-k product path** (k-summed \(\rho\) →
Hirshfeld → MBD). Free-atom MBD is more attractive (no volume contraction).
Unit tests: multi-k path reports volumes + attractive \(E_{\mathrm{disp}}\);
UKS + `hirshfeld` still throws.

## Performance

Source: `REFERENCE.md` / handoff harness wave 6 — **perf_exempt**.

Hirshfeld volumes are a single post-SCF grid integral; MBD is \(O(N^3)\) over
a 6-atom cell. Full multi-k GPW SCF wall dominates. A fair
**dispersion-only** microbenchmark vs CP2K is not available (no stock TS/MBD
in CP2K 2026.1); a full twin QS ratio is not the product gate.

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | ------------------------: | ----------------------: | ----: | ---- | ------ |
| Full QS twin | n/a (no stock TS/MBD) | multi-k SCF + Hirshfeld + MBD (reg) | **perf_exempt** | — | Hard gate is E_disp + volumes |
| E_disp product gate | n/a | short multi-k SCF (reg) | **perf_exempt** | — | Same policy as `gpw-mbd-hirshfeld` / `gpw-mbd-ice` |

**perf_exempt = true.** Reason: primary deliverable is multi-k \(E_{\mathrm{disp}}\)
+ volume-ratio parity under fixed product settings; SCF dominates wall. Handoff:
the case `REFERENCE.md` (wave 6 row `gpw-hirshfeld-kpoints`).

## Limitations

- **Cell is a water dimer**, not bulk ice — sufficient to gate multi-k density
  → Hirshfeld → MBD; not a condensed-phase structural benchmark.
- **E_disp hard gate only** — absolute GPW \(e_{\mathrm{tot}}\) vs external QS
  is not asserted.
- **No stock CP2K TS/MBD twin** — pin is ATV product dual-gated vs free-atom /
  Γ Hirshfeld paths (oracle note under `tools/cp2k_oracle/`).
- **RKS only** — UKS + `hirshfeld` throws by design.
- **Isolated MBD geometry** on this pin (`mbd_k_mesh = {0,0,0}`); periodic
  Ewald MBD is `reg_gpw-mbd-periodic`.
- **`ts`/`mbd` mutually exclusive** with D3/D3(BJ)/D4/XDM/vdW-DF*; `hirshfeld`
  requires `ts` or `mbd`.
- Multi-k vs Γ \(\Delta E_{\mathrm{disp}}\) is \(\sim 3\times10^{-10}\) Ha on this
  dilute cell — the value of the slug is the **k-summed density path**, not a
  large numerical shift.
- Wall **perf_exempt** for this wave; not a speed optimization target.

## References

- Offline pin: `regressions/gpw/gpw-hirshfeld-kpoints/REFERENCE.md`
- Oracle note: `tools/cp2k_oracle/inputs/gpw-hirshfeld-kpoints/water_dimer_k_hirshfeld.note`
- Related pins: `regressions/gpw/gpw-mbd-hirshfeld/REFERENCE.md` (Γ Hirshfeld),
  `regressions/gpw/gpw-mbd-ice/REFERENCE.md` (free-atom MBD),
  `regressions/gpw/gpw-mbd-periodic/REFERENCE.md` (Ewald MBD)
- Oracle rules: `tools/cp2k_oracle/README.md` (offline pins only; never live
  in ctest)
- Roadmap: `docs/roadmap/dft.md` §8 Dispersion
- GPW narrative chapter: `docs/tutorial/ch09-gpw.md`
- Closed spec (git history): `.claude/specs/gpw-hirshfeld-kpoints/` (deleted at done)
- A. Tkatchenko, M. Scheffler, *Phys. Rev. Lett.* **102**, 073005 (2009)
  (TS / Hirshfeld volumes)
- A. Ambrosetti et al., *J. Chem. Phys.* **140**, 18A508 (2014) (MBD@rsSCS)
