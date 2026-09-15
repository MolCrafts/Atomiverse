# GPW PBE + rVV10 — periodic water dimer (ice-relevant H-bond)

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `gpw-rvv10-ice` |
| Regression | `regressions/gpw/gpw-rvv10-ice/` |
| Public API | `cpu::gpw::GpwDft::rvv10` / `cpu::gpw::GPWEvaluator::rvv10`; `GpwResult::e_disp` [Ha] |
| Roadmap | `docs/roadmap/dft.md` §8 Dispersion |

## Summary

**Nonlocal rVV10 correlation** (Sabatini–Gorni–de Gironcoli revised VV10) is
wired into the **Γ-point GPW** path as a density-dependent nonlocal correlation
correction on top of PBE. This product pin uses a **periodic water-dimer cell**
(ice-relevant H-bonded pair in a 12 Å cubic box — same nuclei as
`gpw-pbe-d3bj-ice` / `gpw-vdw-df-ice` / `gpw-ts-ice`, not bulk ice Ih) and gates
the **nonlocal energy** \(E_c^{\mathrm{nl}}\) on `GpwResult::e_disp` against a
**dual-frozen Atomiverse post-SCF pin**, with offline **CP2K 2026.1**
`TYPE RVV10` as the literature-aligned cross-check (self-consistent density).

Public surface: bool members `rvv10` on `cpu::gpw::GpwDft` and
`cpu::gpw::GPWEvaluator` (no `*Config` bag). Nonlocal energy is reported on
`GpwResult::e_disp` [Hartree] and folded into `e_total`. Mutually exclusive with
pairwise / MBD / vdW-DF / vdW-DF2 flags (`d3`, `d3bj`, `d4`, `ts`, `mbd`,
`vdw_df`, `vdw_df2`). Product gate: `regressions/gpw/gpw-rvv10-ice`
(`reg_gpw-rvv10-ice`). Wall ratio is **perf_exempt** (E_disp pin; full QS twin
wall unfair — see Performance).

## Method

**Revised VV10 nonlocal correlation** (Sabatini et al., *Phys. Rev. B* **87**,
041108(R) (2013)), evaluated with the Román–Pérez–Soler FFT method
(Román-Pérez & Soler, *Phys. Rev. Lett.* **103**, 096102 (2009)):

\[
E_c^{\mathrm{nl}}
  = \beta\int\rho\,\mathrm d\mathbf r
  + \tfrac12\iint \theta(\mathbf r)\,\phi(\mathbf r,\mathbf r')\,\theta(\mathbf r')\,.
\]

Local \(q_0 = w_0/\kappa\) is saturated to the community q-mesh;
\(\theta_\alpha = c\,\rho^{3/4} p_\alpha(q_0)\); \(\phi_{\alpha\beta}(k)\) comes
from the **rVV10 kernel table** (CP2K `rVV10_kernel_table.dat`, nqs = 20,
nr = 1024, \(r_{\max}\) = 100 Bohr). Atomiverse embeds the table as
float32+zlib (`rvv10_kernel_data.cpp`) — **no runtime file I/O**.

**Default parameters** (CP2K `PARAMETERS 6.3 0.0093`; ATV
`kRvv10B` / `kRvv10C`):

| Parameter | Value | Unit |
| --------- | ----: | ---- |
| \(b\) | 6.3 | dimensionless (range separation) |
| \(C\) | 0.0093 | a.u. (gradient coefficient) |
| \(q_{\mathrm{cut}}\) | 0.5 | 1/Bohr (soft upper q) |
| \(q_{\min}\) | \(1\times 10^{-4}\) | 1/Bohr (soft lower q) |

**Product evaluation policy.** ATV evaluates rVV10 **post-SCF** on the final
total density (spectral \(|\nabla\rho|^2\) from the GPW FFT gradient). CP2K
applies rVV10 **self-consistently** inside the XC potential. The dual-frozen
regression pin is the post-SCF ATV \(E_c^{\mathrm{nl}}\); the CP2K SC value is
the cross-check for **sign and scale** (expected Δ ∼ 10⁻⁵ Ha from density
policy + float32 kernel tables).

**Electronic host.** Γ-RKS GPW PBE, `DZVP-MOLOPT-SR-GTH` + `GTH-PBE-q*`,
CUTOFF 200 Ry, 12 Å cubic **PERIODIC XYZ**, 16 valence e⁻ → `n_occ = 8`.
Absolute QS \(E_{\mathrm{tot}}\) micro-Ha parity vs CP2K is **out of scope**
(OT/grid/DIIS + post-SCF vs SC density).

**Key exclusions.**

- Self-consistent rVV10 potential in the KS Hamiltonian (post-SCF energy only)
- Stock **VV10** (distinct from rVV10 — still roadmap ○)
- Analytic rVV10 forces / stress product gate
- Full ice Ih supercell / phonons
- k-points, UKS, multigrid, nonperiodic MT with this flag
- Pairwise D3/D4/TS, MBD, vdW-DF / vdW-DF2 (mutually exclusive)
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
gpw.cutoff_ry = 200.0;
gpw.n_occ = 8;         // 16 valence e⁻ → 8 doubly occupied
gpw.max_iter = 40;
gpw.tol = 1e-7;
gpw.mix = 0.4;
gpw.xc = XcKind::PBE;
gpw.nonperiodic = false;
gpw.rvv10 = true;      // exclusive vs d3/d3bj/d4/ts/mbd/vdw_df/vdw_df2

gpw.build(atoms);
const auto& res = gpw.run(atoms);
// res.e_disp [Ha] within 5e-5 of dual-frozen pin
// res.e_total includes e_disp
```

Driver / Frame path (component; same member):

```cpp
// kinds = {O, H} Kind{Element, GaussianBasis, GthPotential}
ATV::cpu::gpw::GPWEvaluator eval(kinds, /*cutoff_ry=*/200.0, /*max_iter=*/40,
                                 /*tol=*/1e-7, /*mix=*/0.4,
                                 ATV::cpu::gpw::XcKind::PBE,
                                 /*nonperiodic=*/false);
eval.rvv10 = true;
// single_point(frame, eval) or bind on Evaluate / MD / Minimize force slot
// eval.last_result().e_disp  [Hartree]
// Public Frame energy remains eV on frame.meta (kEnergy); internal Ha OK when cited.
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
  cmake --build build --target reg_gpw-rvv10-ice -j$(nproc)
  ./build/regressions/reg_gpw-rvv10-ice
'
```

| Path | Role |
| ---- | ---- |
| `include/atomiverse/cpu/gpw/gpw_solver.h` | `GpwDft`, `GpwResult::e_disp`, `rvv10` |
| `include/atomiverse/cpu/gpw/gpw_evaluator.h` | `GPWEvaluator::rvv10` |
| `include/atomiverse/cpu/gpw/internal/rvv10.h` | RPS energy kernel (private) |
| `src/cpu/gpw/internal/rvv10.cpp` | Post-SCF evaluation |
| `src/cpu/gpw/internal/rvv10_kernel_data.cpp` | Embedded CP2K kernel table |
| `regressions/gpw/gpw-rvv10-ice/reg_gpw-rvv10-ice.cpp` | Product gate |
| `regressions/gpw/gpw-rvv10-ice/REFERENCE.md` | Offline pin + perf_exempt |
| `tools/cp2k_oracle/inputs/gpw-rvv10-ice/` | CP2K twin input + notes |

## Validation (CP2K)

Source:
[`regressions/gpw/gpw-rvv10-ice/REFERENCE.md`](../../../regressions/gpw/gpw-rvv10-ice/REFERENCE.md).
Oracle: offline **CP2K 2026.1** (Arrhenius tree install `work/cp2k/install`,
GCC 11.5 + libxc + FFTW3), twin
`tools/cp2k_oracle/inputs/gpw-rvv10-ice/water_dimer_rvv10.inp`. External
engine pin — not a golden of Atomiverse’s own output; **never** launch live
CP2K from ctest.

**Setup:**

| Item | Value |
| ---- | ----- |
| System | Water dimer (ice-relevant H-bond pair) |
| Geometry [Å] | O (0,0,0); H (0.758602, 0, ±0.504284); O (3,0,0); H (3.758602, 0, ±0.504284) |
| Cell | ABC **12.0 12.0 12.0 Å**, PERIODIC XYZ |
| Basis / PP | `DZVP-MOLOPT-SR-GTH` + `GTH-PBE-q6` (O) / `GTH-PBE-q1` (H) |
| XC + nl | PBE + `TYPE RVV10`, PARAMETERS **6.3 0.0093**, SCALE 1.0 |
| CUTOFF / REL_CUTOFF | 200 / 40 Ry (CP2K twin) |
| Occupations | RKS, 16 valence e⁻ → `n_occ = 8` |
| CP2K pin line | `Dispersion energy = 0.06116467960721` Ha (SC) |
| ATV dual-frozen pin | \(E_c^{\mathrm{nl}} = +6.11534922097146\times 10^{-2}\) Ha (post-SCF) |

| Quantity | Atomiverse / pin | CP2K (offline) | Tol | Notes |
| -------- | ---------------: | -------------: | --- | ----- |
| \(E_{\mathrm{disp}}\) [Ha] | **+6.11534922097146×10⁻²** | **+6.116467960721×10⁻²** | \(5\times 10^{-5}\) Ha | Hard gate on dual-frozen ATV post-SCF pin |
| \(\Delta\) ATV − CP2K [Ha] | −1.1×10⁻⁵ | — | — | Post-SCF vs SC density + float32 kernel |
| \(E_{\mathrm{tot}}\) QS [Ha] | (not gated) | −34.27712583714492 | — | Context only; OT/grid/DIIS |
| Sign | \(E_{\mathrm{disp}} > 0\) | \(E_{\mathrm{disp}} > 0\) | — | Asserted in regression (isolated dimer density) |

**Tolerance rationale.** The regression freezes the ATV post-SCF value at
\(5\times 10^{-5}\) Ha — wider than pairwise D3 nano-Hartree gates because the
kernel tables are float32 and CP2K’s SC density is not reproduced. CP2K SC
agrees to ~1.1×10⁻⁵ Ha on this geometry, confirming sign and community-kernel
scale.

## Performance

Source: `REFERENCE.md` (queue / handoff: **perf_exempt**).

rVV10 evaluation is an FFT-based nonlocal kernel on the density grid
(\(O(N_g \log N_g)\) per q-channel product). Full GPW SCF wall is still
dominated by the electronic structure, not the post-SCF kernel. A fair
**dispersion-only** microbenchmark vs CP2K is not meaningful; a full twin SCF
ratio depends on cutoff/OT choices and is not the product gate.

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | ------------------------: | ----------------------: | ----: | ---- | ------ |
| Full QS (context) | ≈ 121 (OMP=8, pin host) | — (not fair-twinned) | **perf_exempt** | — | CP2K wall is full SC QS; hard gate is E_disp only |
| E_disp product gate | n/a | SCF + post-SCF kernel (reg) | **perf_exempt** | — | Primary deliverable is E_disp pin parity |

**perf_exempt = true.** Reason (from REFERENCE / handoff): primary deliverable
is \(E_{\mathrm{disp}}\) pin parity (same policy as `gpw-vdw-df-ice` /
`gpw-vdw-df2-ice` / `gpw-ts-ice`). CP2K wall includes full Quickstep with
self-consistent nonlocal potential (~121 s OMP=8) while the hard-gate
regression materializes post-SCF `e_disp` after a short DIIS SCF. Re-measure a
strict full-SCF twin later if product policy requires
\(r = T_{\mathrm{ATV}}/T_{\mathrm{CP2K}} \le 10\).

## Limitations

- **Cell is a water dimer**, not bulk ice Ih — sufficient for periodic
  nonlocal kernel parity; not a structural ice benchmark.
- **Post-SCF energy only** — no self-consistent rVV10 potential, no analytic
  rVV10 forces/stress product gate.
- **E_disp hard gate only** — absolute GPW \(e_{\mathrm{tot}}\) vs CP2K QS is
  not asserted (OT/grid/DIIS + density policy).
- **Stock VV10** (non-revised) remains roadmap ○; do not equate with rVV10.
- **Mutual exclusion** with `d3` / `d3bj` / `d4` / `ts` / `mbd` / `vdw_df` /
  `vdw_df2` — both true throws at run.
- **Periodic single-grid** path only for the nonlocal kernel.
- Wall **perf_exempt** for this wave; not a speed optimization target.
- Internal energies cited in Ha/Ry; public Frame energy remains **eV** on
  `frame.meta.get(kEnergy)`.

## References

- Offline pin: `regressions/gpw/gpw-rvv10-ice/REFERENCE.md`
- Oracle inputs: `tools/cp2k_oracle/inputs/gpw-rvv10-ice/`
- Roadmap: `docs/roadmap/dft.md` §8 Dispersion
- GPW narrative chapter: `docs/tutorial/ch09-gpw.md`
- Sibling nonlocal / dispersion pins: `gpw-vdw-df-ice`, `gpw-vdw-df2-ice`,
  `gpw-pbe-d3bj-ice`, `gpw-ts-ice`
- R. Sabatini, T. Gorni, S. de Gironcoli, *Phys. Rev. B* **87**, 041108(R)
  (2013) (rVV10)
- J. Román-Pérez, J. M. Soler, *Phys. Rev. Lett.* **103**, 096102 (2009)
  (RPS FFT nonlocal kernel)
- O. A. Vydrov, T. Van Voorhis, *J. Chem. Phys.* **133**, 244103 (2010)
  (original VV10; stock VV10 not implemented here)
