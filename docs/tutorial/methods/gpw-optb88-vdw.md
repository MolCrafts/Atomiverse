# GPW optB88-vdW — periodic water dimer (optB88 exchange + DRSLL)

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `gpw-optb88-vdw` |
| Regression | `regressions/gpw/gpw-optb88-vdw/` |
| Public API | `cpu::gpw::XcKind::OptB88VdW` + `GpwDft::vdw_df` / `GPWEvaluator::vdw_df`; `GpwResult::e_disp` [Ha] |
| Roadmap | `docs/roadmap/dft.md` §8 Dispersion |

## Summary

**optB88-vdW** (Klimeš–Bowler–Michaelides) is the product pairing of
**optimized B88 exchange** (libxc `GGA_X_OPTB88_VDW`) with **PW92 LDA
correlation** and **DRSLL / vdW-DF DF1 nonlocal correlation**. On Atomiverse
GPW the semi-local piece is `XcKind::OptB88VdW`; the nonlocal piece reuses the
shipped `vdw_df` (post-SCF DRSLL on the final density). This product pin uses
the same **periodic water-dimer cell** as `gpw-vdw-df-ice` / `gpw-rvv10-ice`
(12 Å cubic, ice-relevant H-bonded pair — not bulk ice Ih).

Public surface: enum member `xc = XcKind::OptB88VdW` and bool `vdw_df` on
`cpu::gpw::GpwDft` and `cpu::gpw::GPWEvaluator` (no `*Config` bag). Nonlocal
energy is reported on `GpwResult::e_disp` [Hartree] and folded into `e_total`.
Public Frame energy remains eV via the evaluator path. Product gate:
`regressions/gpw/gpw-optb88-vdw` (`reg_gpw-optb88-vdw`). Wall ratio is
**perf_exempt** (XC combo + E_disp pin; full CP2K QS twin wall unavailable on
the pin host — see Performance).

## Method

**optB88-vdW functional split** (Klimeš, Bowler, Michaelides, *J. Phys.:
Condens. Matter* **22**, 022201 (2010); CP2K Hutter recipe):

| Piece | Role | Atomiverse |
| ----- | ---- | ---------- |
| Exchange | Optimized B88 | `XCType::OptB88VdW` / `XcKind::OptB88VdW` |
| Local correlation | PW92 LDA only | same `OptB88VdW` semi-local (no GGA-C) |
| Nonlocal correlation | DRSLL / vdW-DF DF1 | `GpwDft::vdw_df = true` |

**Exchange parameters** (libxc `GGA_X_OPTB88_VDW`, id 139):

| Parameter | Value | Unit |
| --------- | ----: | ---- |
| \(\beta\) | 0.003368671161034469 | dimensionless |
| \(\gamma\) | 6.981317007977318 | dimensionless |

(Stock B88 uses \(\beta=0.0042\), \(\gamma=6\); the optB88 reparametrization
is the activation of this XC vs PBE.)

**Nonlocal correlation** (Dion et al., *Phys. Rev. Lett.* **92**, 246401
(2004); Román-Pérez & Soler, *Phys. Rev. Lett.* **103**, 096102 (2009)):

\[
E_c^{\mathrm{nl}}
  = \tfrac12 \iint \rho(\mathbf r)\,\phi(\mathbf r,\mathbf r')\,\rho(\mathbf r')
    \,\mathrm d\mathbf r\,\mathrm d\mathbf r'
\]

with DF1 \(q_0(\rho,|\nabla\rho|^2)\) (\(Z_{ab}=-0.8491\), \(q_{\mathrm{cut}}=5\))
and community RPS kernel tables (libvdwxc / QE). On this product path
evaluation is **post-SCF** on the final total density (same kernel as
`gpw-vdw-df-ice`; density-driven difference only vs PBE+DRSLL).

**CP2K twin surface** (offline input; never live in ctest):

```
&XC_FUNCTIONAL
  &LIBXC
    FUNCTIONAL XC_GGA_X_OPTB88_VDW
  &END LIBXC
  &PW92
  &END PW92
&END XC_FUNCTIONAL
&vdW_POTENTIAL
  DISPERSION_FUNCTIONAL NON_LOCAL
  &NON_LOCAL
    TYPE DRSLL
  &END NON_LOCAL
&END vdW_POTENTIAL
```

Input: `tools/cp2k_oracle/inputs/gpw-optb88-vdw/water_dimer_optb88_vdw.inp`.

**Electronic host.** Γ-RKS GPW, `DZVP-MOLOPT-SR-GTH` + `GTH-PBE-q*`, CUTOFF
200 Ry, \(n_{\mathrm{occ}}=8\) (16 valence e⁻), mix 0.4, tol \(10^{-7}\),
max_iter 40. Semi-local XC is self-consistent; DRSLL is additive post-SCF on
the converged density for this pin (see also `gpw-vdw-scf-potential` for the
self-consistent nonlocal potential productization).

**Key exclusions.**

- Full ice Ih supercell / phonons
- Stock VV10 (distinct from rVV10; CP2K has no `TYPE VV10` in 2026.1)
- Absolute CP2K QS \(E_{\mathrm{tot}}\) micro-Ha digits on the pin host
  (Arrhenius `cp2k.psmp` **without LIBXC** — twin retained for LIBXC-enabled
  hosts)
- Pairwise D3/D4/TS/MBD/XDM (mutually exclusive with `vdw_df`)
- k-points / UKS / cell stress as part of this slug
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
gpw.cutoff_ry = 200.0;
gpw.n_occ = 8;         // 16 valence e⁻ → 8 doubly occupied
gpw.max_iter = 40;
gpw.tol = 1e-7;
gpw.mix = 0.4;
gpw.xc = XcKind::OptB88VdW;
gpw.nonperiodic = false;
gpw.vdw_df = true;     // DRSLL nonlocal; exclusive vs d3/d3bj/d4/ts/mbd/…

gpw.build(atoms);
const auto& res = gpw.run(atoms);
// res.e_disp [Ha] — DRSLL on optB88 density
// res.e_total includes e_disp [Ha]
```

Driver / Frame path (component; same members):

```cpp
// kinds = {O, H} Kind{Element, GaussianBasis, GthPotential}
ATV::cpu::gpw::GPWEvaluator eval(kinds, /*cutoff_ry=*/200.0, /*max_iter=*/40,
                                 /*tol=*/1e-7, /*mix=*/0.4,
                                 ATV::cpu::gpw::XcKind::OptB88VdW,
                                 /*nonperiodic=*/false);
eval.vdw_df = true;
// single_point(frame, eval) or bind on Evaluate / MD / Minimize force slot
// eval.last_result().e_disp  [Hartree]
// frame.meta.get(kEnergy)    [eV] public surface
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
  cmake --build build --target reg_gpw-optb88-vdw -j$(nproc)
  ./build/regressions/reg_gpw-optb88-vdw
'
```

| Path | Role |
| ---- | ---- |
| `include/atomiverse/cpu/gpw/gpw_solver.h` | `GpwDft`, `XcKind::OptB88VdW`, `vdw_df`, `GpwResult::e_disp` |
| `include/atomiverse/cpu/gpw/gpw_evaluator.h` | `GPWEvaluator::vdw_df` |
| `include/atomiverse/cpu/scf/xc_functional.h` | `XCType::OptB88VdW` (semi-local kernel) |
| `include/atomiverse/cpu/gpw/internal/xc_grid_gga.h` | `xc_grid_optb88_vdw` / spin |
| `include/atomiverse/cpu/gpw/internal/vdw_df.h` | DRSLL engine (private) |
| `regressions/gpw/gpw-optb88-vdw/reg_gpw-optb88-vdw.cpp` | Product gate |
| `regressions/gpw/gpw-optb88-vdw/REFERENCE.md` | Offline pins + perf_exempt |
| `tools/cp2k_oracle/inputs/gpw-optb88-vdw/` | CP2K twin input + notes |

## Validation (CP2K / offline pins)

Source:
[`regressions/gpw/gpw-optb88-vdw/REFERENCE.md`](../../../regressions/gpw/gpw-optb88-vdw/REFERENCE.md).
Oracle design: offline **CP2K** twin under
`tools/cp2k_oracle/inputs/gpw-optb88-vdw/` (Hutter recipe:
`XC_GGA_X_OPTB88_VDW` + `&PW92` + `TYPE DRSLL`). On the pin host, **CP2K is
present but built without LIBXC** (`unknown subsection LIBXC`), so absolute
CP2K QS digits are **not** invented. Hard gates use (1) a **semi-local dual
gate** vs PBE, (2) a **first-green ATV freeze** for full optB88-vdW, and (3)
unit-kernel pins already shipped for optB88 exchange parameters and DRSLL vs
libvdwxc. External pins — not a golden of Atomiverse’s own output; **never**
launch live CP2K from ctest. When a LIBXC-enabled CP2K is available offline,
re-run the twin and replace absolute pins (cite CP2K version).

**Setup:**

| Item | Value |
| ---- | ----- |
| System | Water dimer (ice-relevant H-bond pair) |
| Geometry [Å] | O (0,0,0); H (0.758602, 0, ±0.504284); O (3,0,0); H (3.758602, 0, ±0.504284) |
| Cell | ABC **12.0 12.0 12.0 Å**, PERIODIC XYZ |
| Basis / PP | `DZVP-MOLOPT-SR-GTH` + `GTH-PBE-q6` (O) / `GTH-PBE-q1` (H) |
| XC + nonlocal | `XcKind::OptB88VdW` + `vdw_df` (DRSLL) |
| CUTOFF | 200 Ry |
| Occupations | RKS, 16 valence e⁻ → `n_occ = 8` |
| SCF | mix 0.4, tol \(10^{-7}\), max_iter 40 |

| Quantity | Atomiverse / pin | Oracle / notes | Tol | Notes |
| -------- | ---------------: | -------------: | --- | ----- |
| \(\lvert E_{\mathrm{OptB88}} - E_{\mathrm{PBE}}\rvert\) [Ha] (semi-local, `vdw_df=false`) | **0.303** | dual-gate floor \(>10^{-4}\) | \(10^{-4}\) Ha floor | Activation of optB88-x + PW92 vs PBE |
| \(E_{\mathrm{disp}}\) [Ha] (optB88 density + DRSLL) | **1.5207282287995946×10⁻¹** | first-green freeze; kernel vs libvdwxc in `gpw-vdw-df-ice` | \(5\times 10^{-5}\) Ha | Hard gate |
| \(E_{\mathrm{tot}}\) [Ha] (optB88-vdW) | **−34.481701207152035** | first-green freeze until LIBXC CP2K | \(1\times 10^{-3}\) Ha | Setup integrity |
| \(E_{\mathrm{OptB88}}\) semi-local [Ha] | −34.633774030031404 | context | — | no nonlocal |
| \(E_{\mathrm{PBE}}\) semi-local [Ha] | −34.330331404832023 | context | — | dual-gate baseline |
| SCF iters (full) | 11 (converged) | context | — | regression host |

**Cross-check.** \(E_{\mathrm{disp}}\) is close to the PBE+DRSLL pin of
`gpw-vdw-df-ice` (**1.519439×10⁻¹ Ha**) — density-driven difference only. The
**semi-local** 0.303 Ha shift is the functional activation for optB88 exchange
+ PW92. Pure \(E_c^{\mathrm{nl}}\) on a molecular density can be positive;
dispersion *binding* is recovered in interaction differences (same note as
vdW-DF).

**Tolerance rationale.** Dual gate fails closed if `OptB88VdW` is a no-op alias
of PBE. \(E_{\mathrm{disp}}\) tol \(5\times 10^{-5}\) Ha matches the DRSLL
product pin family. \(E_{\mathrm{tot}}\) is a loose integrity bound until an
offline LIBXC-enabled CP2K twin replaces the freeze.

## Performance

Source: `REFERENCE.md` (queue / handoff: **perf_exempt**).

Nonlocal RPS is \(O(N_g \log N_g \times n_\alpha)\) with \(n_\alpha=20\);
wall on the 6-atom cell is dominated by the electronic SCF, not the
dispersion kernel. A fair full-QS twin wall vs CP2K is **unavailable** on the
pin host (no LIBXC). Same policy as `gpw-vdw-df-ice` / `gpw-rvv10-ice`.

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | ------------------------: | ----------------------: | ----: | ---- | ------ |
| Full QS (context) | n/a (LIBXC missing on pin host) | — | **perf_exempt** | — | Twin input retained for LIBXC hosts |
| E_disp / XC product gate | n/a | short SCF + DRSLL (reg) | **perf_exempt** | — | Hard gate is energy pins + dual gate |

**perf_exempt = true.** Reason (from REFERENCE / handoff): primary deliverable
is XC combo + E_disp pin parity; no fair CP2K QS wall without LIBXC. Re-measure
a strict full-SCF twin later if product policy requires
\(r = T_{\mathrm{ATV}}/T_{\mathrm{CP2K}} \le 10\).

## Limitations

- **Cell is a water dimer**, not bulk ice Ih — sufficient for periodic
  optB88-vdW + DRSLL parity; not a structural ice benchmark.
- **Absolute CP2K QS digits deferred** until a LIBXC-enabled offline twin
  replaces the first-green ATV freeze (input already under
  `tools/cp2k_oracle/inputs/gpw-optb88-vdw/`).
- **Post-SCF DRSLL on this pin** — self-consistent nonlocal potential/forces
  are productized under `gpw-vdw-scf-potential`, not re-gated here.
- **Mutually exclusive dispersion flags** — `vdw_df` cannot combine with
  `d3` / `d3bj` / `d4` / `ts` / `mbd` / `xdm` / `vdw_df2` / `rvv10`.
- **Semi-local is optB88 + PW92 only** — do not confuse with PBE+DRSLL
  (`gpw-vdw-df-ice`) or rVV10 (`gpw-rvv10-ice`).
- Wall **perf_exempt** for this wave; not a speed optimization target.
- **CUDA GPW** out of scope.

## References

- Offline pin: `regressions/gpw/gpw-optb88-vdw/REFERENCE.md`
- Oracle inputs: `tools/cp2k_oracle/inputs/gpw-optb88-vdw/`
- Oracle rules: `tools/cp2k_oracle/README.md` (offline pins only; never in ctest)
- Roadmap: `docs/roadmap/dft.md` §8 Dispersion
- Related product pins: `gpw-vdw-df-ice` (DRSLL kernel),
  `gpw-vdw-scf-potential` (SC nonlocal potential + forces)
- GPW narrative chapter: `docs/tutorial/ch09-gpw.md`
- Closed spec (git history): `.claude/specs/gpw-optb88-vdw/` (deleted at done)
- J. Klimeš, D. R. Bowler, A. Michaelides, *J. Phys.: Condens. Matter* **22**,
  022201 (2010) (optB88-vdW)
- M. Dion, H. Rydberg, E. Schröder, D. C. Langreth, B. I. Lundqvist,
  *Phys. Rev. Lett.* **92**, 246401 (2004) (vdW-DF / DRSLL)
- G. Román-Pérez, J. M. Soler, *Phys. Rev. Lett.* **103**, 096102 (2009)
  (efficient RPS evaluation)
