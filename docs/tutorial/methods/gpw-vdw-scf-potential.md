# GPW nonlocal vdW — self-consistent \(V_{\mathrm{nl}}\) + forces (rVV10 product pin)

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `gpw-vdw-scf-potential` |
| Regression | `regressions/gpw/gpw-vdw-scf-potential/` |
| Public API | `cpu::gpw::GpwDft::rvv10` / `vdw_df` / `vdw_df2` (and `GPWEvaluator` twins); `GpwResult::e_disp` [Ha]; forces via `gpw_forces_from_scf` |
| Roadmap | `docs/roadmap/dft.md` §8 Dispersion |

## Summary

Earlier ice pins (`gpw-vdw-df-ice`, `gpw-vdw-df2-ice`, `gpw-rvv10-ice`)
delivered **post-SCF** nonlocal correlation energy \(E_c^{\mathrm{nl}}\) on a
frozen PBE density. This slug productizes the **self-consistent** path: each
SCF iteration adds the nonlocal functional derivative

\[
V_{\mathrm{nl}}[\rho]
  = \frac{\delta E_c^{\mathrm{nl}}[\rho]}{\delta\rho}
\]

to the KS effective potential, so the density responds to the nonlocal
correlation; the final \(E_{\mathrm{disp}}=E_c^{\mathrm{nl}}\) is evaluated on
that SC density and folded into \(e_{\mathrm{total}}\). Analytic forces reuse
the same \(V_{\mathrm{nl}}\) inside the veff contribution of
`assemble_gpw_forces` / `gpw_forces_from_scf`.

The **product regression** gates **Γ-point GPW PBE + rVV10** on the same
periodic water-dimer cell as `gpw-rvv10-ice` (12 Å cubic, ice-relevant H-bond
pair) against offline **CP2K 2026.1** `TYPE RVV10` (self-consistent by default).
The SC machinery is shared for **DRSLL** (`vdw_df`), **LMKLL** (`vdw_df2`),
and **rVV10** (`rvv10`); only the rVV10 twin is e2e-gated here (energy + force
FD). Public surface remains bool members on `GpwDft` / `GPWEvaluator` — no
`*Config` bag, no factories. Wall ratio is **perf_exempt** (SC RPS costs
\(O(N_{\mathrm{SCF}}\times n_\alpha)\) FFTs; primary deliverable is
potential/force correctness).

## Method

### Nonlocal functionals (shared SC product path)

Román–Pérez–Soler (RPS) FFT evaluation of density-based nonlocal correlation
on the GPW uniform grid (periodic single-grid only):

| Flag | CP2K `NON_LOCAL TYPE` | Formula family | Defaults / notes |
| ---- | --------------------- | -------------- | ---------------- |
| `vdw_df` | DRSLL (vdW-DF1) | Dion \(E_c^{\mathrm{nl}}\); \(Z_{ab}=-0.8491\) | libvdwxc DF1 kernel tables |
| `vdw_df2` | LMKLL (vdW-DF2) | same kernel; \(Z_{ab}=-1.887\) | libvdwxc DF2 |
| `rvv10` | RVV10 | Sabatini revised VV10 | \(b=6.3\), \(C=0.0093\); CP2K `rVV10_kernel_table.dat` |

**rVV10 energy** (Sabatini, Gorni, de Gironcoli, *Phys. Rev. B* **87**,
041108(R) (2013); RPS FFT of Román-Pérez & Soler, *Phys. Rev. Lett.* **103**,
096102 (2009)):

\[
E_c^{\mathrm{nl}}
  = \beta\int\rho\,\mathrm d\mathbf r
  + \tfrac12\iint \theta(\mathbf r)\,\phi(\mathbf r,\mathbf r')\,\theta(\mathbf r')\,
  \mathrm d\mathbf r\,\mathrm d\mathbf r',
\]

with \(q_0=w_0/\kappa\) saturated to the q-mesh, \(\theta_\alpha = c\,\rho^{3/4}
p_\alpha(q_0)\), and the community rVV10 kernel \(\phi_{\alpha\beta}(k)\).
Spectral \(|\nabla\rho|^2\) is taken from the GPW FFT gradient.

**vdW-DF / DF2** use the Dion form \(E_c^{\mathrm{nl}}=\tfrac12\iint\rho\,\phi\,\rho\)
with \(q_0(\rho,|\nabla\rho|^2)\) and the shared RPS kernel table; variants
differ only in \(Z_{ab}\) (see `internal/vdw_df.h`).

### Self-consistency vs post-SCF

| Mode | Density for \(E_c^{\mathrm{nl}}\) | \(V_{\mathrm{nl}}\) in SCF | Product status |
| ---- | -------------------------------- | ------------------------- | -------------- |
| Post-SCF (historical ice pins) | Final PBE density only | off | `gpw-rvv10-ice` etc. (energy pin) |
| **SC (this slug)** | SC density after \(V_{\mathrm{nl}}\) in every iteration | **on** (`potential=true`) | `gpw-vdw-scf-potential` |

Product steps (CP2K `qs_dispersion_nonloc` / Soler eq. 10 + spectral
divergence of the gradient flux):

1. **Each SCF iteration:** evaluate RPS \(E_c^{\mathrm{nl}}\) and
   \(V_{\mathrm{nl}}=\delta E_c^{\mathrm{nl}}/\delta\rho\); add \(V_{\mathrm{nl}}\)
   onto KS \(V_{\mathrm{eff}}\) (UKS: both spins share the same total-\(\rho\)
   potential).
2. **After convergence:** recompute \(E_{\mathrm{disp}}=E_c^{\mathrm{nl}}\) on
   the SC total density → `GpwResult::e_disp` [Hartree]; fold into
   `e_total`.
3. **Forces:** rebuild \(V_{\mathrm{nl}}\) on the SC density and include it in
   the veff piece of analytic GPW forces; regression gates analytic \(F_{O0,x}\)
   against central finite difference of \(E_{\mathrm{total}}\)
   (\(h=5\times10^{-4}\) Bohr).

CP2K twin surface: `&VDW_POTENTIAL` / `POTENTIAL_TYPE NON_LOCAL` /
`&NON_LOCAL TYPE RVV10` with `PARAMETERS 6.3 0.0093` (self-consistent by
default; `RUN_TYPE ENERGY_FORCE`).

### Electronic host

Γ-RKS GPW: **PBE** semi-local XC + GTH-PBE + `DZVP-MOLOPT-SR-GTH`, cubic
**12 Å** cell, **CUTOFF 200 Ry**, 16 valence e⁻ → `n_occ = 8`, periodic XYZ.
Nonlocal vdW is **single-grid only** (multigrid / production multigrid path
rejected when these flags are set — same constraint as the ice energy pins).

### Key exclusions

- Absolute \(E_{\mathrm{tot}}\) micro-Ha parity vs CP2K QS (OT vs DIIS/mix,
  multigrid residual) — **not** gated
- Multi-k / UKS product gate for SC \(V_{\mathrm{nl}}\) (UKS force assembly
  shares \(V_{\mathrm{nl}}\); no e2e pin)
- Analytic **stress** for nonlocal vdW (forces only here)
- Pairwise D3/D4/TS/MBD/XDM combinations with nonlocal flags (mutually exclusive)
- Stock VV10 (unrevised) — separate residual if any
- CUDA GPW
- Live CP2K / live kernel files at ctest time (kernels are in-tree static tables)

## Implementation

Public surface only (`GpwDft` / `GPWEvaluator` members; no `*Config`, no
`make_*` / `build_*` factories). Regression uses the concrete solver + force
helper:

```cpp
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/cpu/gpw/internal/gpw_forces.h"
#include "atomiverse/molrs.hpp"

using ATV::cpu::gpw::GaussianBasis;
using ATV::cpu::gpw::GpwAtom;
using ATV::cpu::gpw::GpwDft;
using ATV::cpu::gpw::GthPotential;
using ATV::cpu::gpw::XcKind;
using ATV::cpu::gpw::gpw_forces_from_scf;

// Geometry [Å] → Bohr — same water dimer as gpw-rvv10-ice (REFERENCE.md).
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
gpw.rvv10 = true;      // SC V_nl each SCF step; exclusive vs vdw_df / vdw_df2 / D*

gpw.build(atoms);
const auto res = gpw.run(atoms);
// res.e_disp [Ha] within 1e-4 of offline CP2K SC pin
// res.e_total includes e_disp on the SC density

const auto f_ana = gpw_forces_from_scf(atoms, gpw, res);
// f_ana[i][k] [Ha/Bohr]; FD of e_total vs O0 x gates |ΔF| ≤ 5e-4
```

Same flags on the Frame / component path:

```cpp
// kinds = {O, H} Kind{Element, GaussianBasis, GthPotential}
ATV::cpu::gpw::GPWEvaluator eval(kinds, /*cutoff_ry=*/200.0, /*max_iter=*/40,
                                 /*tol=*/1e-7, /*mix=*/0.4,
                                 ATV::cpu::gpw::XcKind::PBE,
                                 /*nonperiodic=*/false);
eval.rvv10 = true;   // or eval.vdw_df / eval.vdw_df2
// single_point(frame, eval) or bind on Evaluate / MD / Minimize force slot
// eval.last_result().e_disp  [Hartree]
// public frame energy remains eV via kEnergy
```

Build and run the regression (scoped targets only):

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
  cmake --build build -j$(nproc) --target atomiverse reg_gpw-vdw-scf-potential
  ./build/regressions/reg_gpw-vdw-scf-potential
'
```

| Path | Role |
| ---- | ---- |
| `include/atomiverse/cpu/gpw/gpw_solver.h` | `GpwDft::{rvv10,vdw_df,vdw_df2}`, `GpwResult::e_disp` |
| `include/atomiverse/cpu/gpw/gpw_evaluator.h` | `GPWEvaluator` twin members |
| `include/atomiverse/cpu/gpw/internal/vdw_df.h` | DRSLL/LMKLL + `nonlocal_vdw_on_grid` (energy + optional \(V_{\mathrm{nl}}\)) |
| `include/atomiverse/cpu/gpw/internal/rvv10.h` | rVV10 RPS energy + optional \(V_{\mathrm{nl}}\) |
| `include/atomiverse/cpu/gpw/internal/gpw_forces.h` | `gpw_forces_from_scf` (reuses SC \(V_{\mathrm{nl}}\) in veff) |
| `src/cpu/gpw/gpw_solver.cpp` | SC loop: `nonlocal_vdw_on_grid(..., potential=true)` each iter |
| `src/cpu/gpw/internal/gpw_forces.cpp` | Force assembly with nonlocal veff |
| `regressions/gpw/gpw-vdw-scf-potential/reg_gpw-vdw-scf-potential.cpp` | Product gate |
| `regressions/gpw/gpw-vdw-scf-potential/REFERENCE.md` | Offline pin + perf_exempt |
| `tools/cp2k_oracle/inputs/gpw-vdw-scf-potential/water_dimer_rvv10_force.inp` | CP2K ENERGY_FORCE twin |

## Validation (CP2K)

Source:
[`regressions/gpw/gpw-vdw-scf-potential/REFERENCE.md`](../../../regressions/gpw/gpw-vdw-scf-potential/REFERENCE.md).
Oracle: offline **CP2K 2026.1** (Arrhenius `work/cp2k/install`), twin
`tools/cp2k_oracle/inputs/gpw-vdw-scf-potential/water_dimer_rvv10_force.inp`
(and energy twin under `gpw-rvv10-ice`). External engine pin — **not** a golden
of Atomiverse’s own output; **never** launch live CP2K from ctest.

**Setup:**

| Item | Value |
| ---- | ----- |
| System | Water dimer (ice-relevant H-bond pair) |
| Geometry [Å] | O (0,0,0); H (0.758602, 0, ±0.504284); O (3,0,0); H (3.758602, 0, ±0.504284) |
| Cell | ABC **12.0 12.0 12.0 Å**, PERIODIC XYZ |
| Basis / PP | `DZVP-MOLOPT-SR-GTH` + `GTH-PBE-q6` (O) / `GTH-PBE-q1` (H) |
| XC + nonlocal | PBE + `TYPE RVV10`, PARAMETERS **6.3 0.0093** |
| Grid | CUTOFF **200** Ry, REL_CUTOFF 40, NGRIDS 4 (CP2K multigrid; ATV single-grid product) |
| Occupations | RKS, 16 valence e⁻ → `n_occ = 8` |
| CP2K run | `RUN_TYPE ENERGY_FORCE`, OT + OUTER_SCF, `EPS_SCF 1e-7` |
| Pin line | `Dispersion energy = 0.06116467960721 Ha` |

| Quantity | Atomiverse (product) | CP2K (offline SC) | Tol | Notes |
| -------- | -------------------: | ----------------: | --- | ----- |
| \(E_{\mathrm{disp}}\) SC [Ha] | \(+6.11479080989958\times10^{-2}\) | \(+6.116467960721\times10^{-2}\) | \(1\times10^{-4}\) Ha | Hard gate; \(\Delta\approx -1.7\times10^{-5}\) Ha (single-grid vs CP2K multigrid residual) |
| \(E_{\mathrm{disp}}\) post-SCF ATV [Ha] | \(+6.115349220971\times10^{-2}\) (`gpw-rvv10-ice`) | (same CP2K SC pin) | \(5\times10^{-5}\) Ha (ice pin) | Context: post-SCF vs SC density |
| \(E_{\mathrm{tot}}\) QS [Ha] | finite; includes \(e_{\mathrm{disp}}\) | −34.27712583714492 | — | Context only; not absolute-parity gated |
| \(F_{O0,x}\) analytic vs FD [Ha/Bohr] | \(\|F_{\mathrm{ana}}-F_{\mathrm{FD}}\|\approx 1\times10^{-6}\) | — (ATV FD of \(E_{\mathrm{total}}\)) | \(5\times10^{-4}\) | Non-trivial \(\|F_{\mathrm{ana}}\|>10^{-5}\); \(h=5\times10^{-4}\) Bohr |

**Tolerance rationale.** SC \(E_{\mathrm{disp}}\) is expected to track CP2K at
the \(10^{-4}\) Ha level on this cell with matching RVV10 parameters; residual
vs CP2K is dominated by **single-grid vs multigrid** density (and secondary
OT/DIIS differences), not a wrong kernel. The force gate is an internal
analytic-vs-FD consistency check on the SC total energy surface (nonlocal
veff included), not a CP2K force-vector pin.

Offline re-pin (human only; never ctest):

```bash
# Host with CP2K 2026.1 + rVV10_kernel_table.dat on BASIS/potential path
cd tools/cp2k_oracle/inputs/gpw-vdw-scf-potential
# cp2k.psmp water_dimer_rvv10_force.inp
# → Dispersion energy = 0.06116467960721 Ha
# → Total energy      = -34.27712583714492 Ha
# wall ≈ 121 s (OMP=8) on pin host
```

## Performance

Source: `REFERENCE.md` (queue / handoff: **perf_exempt**).

SC nonlocal evaluation invokes the RPS FFT stack **once per SCF iteration**
(plus force rebuild), so wall scales as
\(O(N_{\mathrm{SCF}}\times n_\alpha)\) relative to a post-SCF single shot.
CP2K full QS wall for the force twin is ≈ **121 s** (OMP=8) on the pin host.
A fair full-SCF twin ratio is not the product gate: the deliverable is SC
energy pin + force FD correctness.

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | ------------------------: | ----------------------: | ----: | ---- | ------ |
| Full QS ENERGY_FORCE (context) | ≈ 121 (OMP=8) | — (not fair-twinned) | **perf_exempt** | — | Multigrid OT vs ATV single-grid DIIS/mix |
| Product gate (SC E_disp + FD) | n/a | 3× SCF (equil + 2 FD) + forces | **perf_exempt** | — | Potential/force correctness |

**perf_exempt = true.** Reason (from REFERENCE / queue): primary deliverable is
SC \(E_{\mathrm{disp}}\) pin + analytic force FD; SC RPS FFT cost makes a
strict \(r\le10\) full-QS twin unfair / uninformative for this wave (same
policy family as ice E_disp pins). Re-measure a strict full-SCF twin later if
product policy requires \(r=T_{\mathrm{ATV}}/T_{\mathrm{CP2K}}\le10\).

## Limitations

- **Cell is a water dimer**, not bulk ice Ih — sufficient for SC nonlocal
  parity and force FD; not a structural ice benchmark.
- **rVV10 product pin only** for e2e; `vdw_df` / `vdw_df2` share the SC path
  but are gated as post-SCF energy pins on their ice regressions (not re-gated
  here for SC forces).
- **Single-grid product** — multigrid nonlocal SC is rejected; residual vs
  CP2K multigrid sits in the \(10^{-5}\) Ha band on this pin.
- **Forces only** — nonlocal vdW **stress** is out of scope for this slug.
- **Absolute \(e_{\mathrm{tot}}\)** vs CP2K QS is not asserted (OT/grid/DIIS).
- **Exclusivity:** at most one of `d3` / `d3bj` / `d4` / `ts` / `xdm` / `mbd` /
  `vdw_df` / `vdw_df2` / `rvv10` may be true (throws at run).
- **Periodic single-grid only** for nonlocal flags (`nonperiodic` rejected).
- **Post-SCF ice pins remain valid** energy footholds; they do **not** exercise
  \(V_{\mathrm{nl}}\) self-consistency or force FD.
- Wall **perf_exempt** for this wave; not a speed optimization target.
- Public Frame energy is still **eV** on `kEnergy`; internal/regression pins
  cite **Ha** for \(E_{\mathrm{disp}}\) and forces in **Ha/Bohr**.

## References

- Offline pin: `regressions/gpw/gpw-vdw-scf-potential/REFERENCE.md`
- Oracle inputs: `tools/cp2k_oracle/inputs/gpw-vdw-scf-potential/`
- Oracle notes: `tools/cp2k_oracle/README.md`
- Roadmap: `docs/roadmap/dft.md` §8 Dispersion; `docs/roadmap/index.md` GPW residual
- Related post-SCF energy pins: `gpw-rvv10-ice`, `gpw-vdw-df-ice`,
  `gpw-vdw-df2-ice` (regressions under `regressions/gpw/`)
- Related exchange + nonlocal combo: optB88-vdW (`gpw-optb88-vdw`)
- Pairwise dispersion siblings: [gpw-pbe-d3bj-ice](gpw-pbe-d3bj-ice.md),
  [gpw-d4-water](gpw-d4-water.md)
- GPW narrative chapter: `docs/tutorial/ch09-gpw.md`
- Closed spec (git history): `.claude/specs/gpw-vdw-scf-potential/` (deleted at
  done; wave-8 merge 2026-07-26)
- O. A. Vydrov, T. Van Voorhis, *J. Chem. Phys.* **133**, 244103 (2010) (VV10)
- R. Sabatini, T. Gorni, S. de Gironcoli, *Phys. Rev. B* **87**, 041108(R)
  (2013) (rVV10)
- G. Román-Pérez, J. M. Soler, *Phys. Rev. Lett.* **103**, 096102 (2009) (RPS FFT)
- M. Dion et al., *Phys. Rev. Lett.* **92**, 246401 (2004) (vdW-DF)
- K. Lee et al., *Phys. Rev. B* **82**, 081101(R) (2010) (vdW-DF2 / LMKLL)
- CP2K Quickstep nonlocal vdW (`qs_dispersion_nonloc`, `TYPE RVV10` / DRSLL /
  LMKLL)
