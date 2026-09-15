# GPW free-atom XDM — analytic ionic forces (water dimer)

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `gpw-xdm-forces` |
| Regression | `regressions/gpw/gpw-xdm-forces/` |
| Public API | `cpu::gpw::GpwDft::xdm` / `cpu::gpw::GPWEvaluator::xdm` with force assembly; `xdm_pbe` → `XdmResult::{energy,forces}`; `GpwResult::e_disp` [Ha] |
| Related energy pin | `regressions/gpw/gpw-xdm-ice/` (same geometry / tables; energy-only) |
| Roadmap | `docs/roadmap/dft.md` §8 Dispersion |

## Summary

**Exchange-hole dipole moment (XDM)** free-atom pairwise dispersion is folded
into **Γ-point GPW** product **ionic forces**: electronic Hellmann–Feynman +
Pulay terms plus geometry-only \(F_A^{\mathrm{XDM}} = -\partial E_{\mathrm{disp}}/\partial R_A\).
The hard gate is **dual pure-Python** energy + force dual-gate (central FD,
\(h=10^{-5}\) Bohr) on the ice-relevant water dimer used by `gpw-xdm-ice` /
`gpw-ts-ice` — **not** a stock CP2K Quickstep `ENERGY_FORCE` twin.

**Why not CP2K QS?** CP2K 2026.1 stock `PAIR_POTENTIAL TYPE` is only
`DFTD2|DFTD3|DFTD3(BJ)|DFTD4` — there is **no** `TYPE XDM`. Canonical full
(density-dependent) XDM oracles are postg / critic2 / QE. Atomiverse ships the
**free-atom foothold** (fixed multipole moments + CRC \(\alpha\); no Hirshfeld
volume scaling) with the same dual pure-Python lattice-sum policy as
`gpw-xdm-ice` and the force dual-gate policy as `gpw-mbd-forces` /
`gpw-ts-forces`.

Public surface: bool members `xdm` on `cpu::gpw::GpwDft` and
`cpu::gpw::GPWEvaluator` (no `*Config` bag). When `xdm=true`, `assemble_gpw_forces`
adds XDM forces to the electronic force; `GpwResult::e_disp` reports the same
periodic energy as the energy-only path. Product gate:
`regressions/gpw/gpw-xdm-forces` (`reg_gpw-xdm-forces`). Wall ratio is
**perf_exempt** (E_disp + force dual-gate; no fair CP2K ENERGY_FORCE without
stock XDM).

## Method

**Free-atom XDM** (Becke & Johnson, *J. Chem. Phys.* **127**, 154108 (2007);
postg / QE `xdm_dispersion` combination rules + BJ damping):

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
E_{\mathrm{disp}}=-\sum_{i\le j}\sum_{\mathbf T}'
\Biggl[
\frac{C_6}{R^6+R_{\mathrm{vdW}}^6}
+\frac{C_8}{R^8+R_{\mathrm{vdW}}^8}
+\frac{C_{10}}{R^{10}+R_{\mathrm{vdW}}^{10}}
\Biggr].
\]

Analytic ionic forces from the same kernel:

\[
\mathbf{F}_A^{\mathrm{XDM}}
= -\frac{\partial E_{\mathrm{disp}}}{\partial \mathbf{R}_A}
\quad\text{[Ha/Bohr]}.
\]

**Product ionic force** (`assemble_gpw_forces` / multi-k twin path):

\[
\mathbf{F}_A
=
\mathbf{F}_A^{\mathrm{HF+Pulay+grid/core}}
+ \mathbf{F}_A^{\mathrm{XDM}}.
\]

Geometry-only XDM does not change the SCF density; the dispersion contribution
is additive on top of electronic forces (same wiring pattern as free-atom TS /
D3(BJ) / D4 / MBD).

**PBE BJ damping (QE NC-PBE):**

| Parameter | Value | Unit |
| --------- | ----: | ---- |
| \(a_1\) | 0.4283 | dimensionless |
| \(a_2\) | 2.4690 | Å (= **4.6657338220946531** Bohr) |

**Free-atom tables** (in-tree `basis_data/xdm_tables.h`; foothold Z = H, O only):

| Z | \(\alpha\) [Bohr³] | \(M_1\) | \(M_2\) | \(M_3\) | \(C_n\) source |
| - | -----------------: | ------: | ------: | ------: | -------------- |
| H | 4.49979 | 2.88859 | 18.4304 | 241.636 | Yan et al., *Phys. Rev. A* **54**, 2824 (1996) |
| O | 5.41216 | 5.74631 | 28.9471 | 262.617 | Chu–Dalgarno, *J. Chem. Phys.* **121**, 4083 (2004) (\(C_6\)); consistent \(C_8,C_{10}\) |

**Twin settings (binding):**

| Knob | Value |
| ---- | ----- |
| System | Water dimer (ice-relevant H-bond; not bulk ice Ih) |
| Atom order | O H H O H H |
| Geometry [Å] | O (0,0,0); H (0.758602, 0, ±0.504284); O (3,0,0); H (3.758602, 0, ±0.504284) |
| Cell (periodic pin / short SCF) | ABC **12.0 12.0 12.0 Å**, PERIODIC XYZ |
| Isolated force pin | Same nuclei; **no** lattice images |
| Pair cutoff | **50 Å** pure real-space orthorhombic (`pair_image_counts`) |
| Å→Bohr | \(1/0.52917720859\) (ATV) |
| XC host | PBE |
| Basis / PP (short SCF) | `DZVP-MOLOPT-SR-GTH` + `GTH-PBE-q6` (O) / `GTH-PBE-q1` (H) |
| Occupations | RKS, 16 valence e⁻ → `n_occ = 8` |
| SCF (product materialization) | cutoff 200 Ry, `max_iter=8`, `tol=1e-6` (E_disp / F_XDM are basis-independent) |

**Key exclusions (this page / acceptance):**

- Density-dependent XDM (Hirshfeld / BR exchange-hole moments) — free-atom
  foothold only.
- Stock CP2K `TYPE XDM` ENERGY_FORCE twin — does not exist in 2026.1.
- Absolute QS \(E_{\mathrm{tot}}\) micro-Ha parity vs CP2K — out of scope
  (grid/OT/DIIS residual; dispersion is geometry-only).
- XDM stress / cell derivatives — not gated here.
- Full ice Ih supercell / phonons.
- CUDA GPW.
- Mutual exclusivity: `d3` / `d3bj` / `d4` / `ts` / `xdm` / `mbd` (and
  nonlocal `vdw_df` / `vdw_df2` / `rvv10`) — any two true throws at run.

## Implementation

Public surface only (`GpwDft` / `GPWEvaluator` members; no `*Config`, no
`make_*` / `build_*` factories). Product regression gates the internal kernel
`xdm_pbe` (same pattern as `gpw-mbd-forces`) plus a short `GpwDft` SCF with
`xdm=true`:

```cpp
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/cpu/gpw/internal/xdm_dispersion.h"
#include "atomiverse/molrs.hpp"

using ATV::cpu::gpw::GaussianBasis;
using ATV::cpu::gpw::GpwAtom;
using ATV::cpu::gpw::GpwDft;
using ATV::cpu::gpw::GthPotential;
using ATV::cpu::gpw::xdm_pbe;
using ATV::cpu::gpw::XcKind;

constexpr double kA2B = 1.0 / 0.52917720859;
// Z = {8,1,1,8,1,1}; pos in Bohr from REFERENCE geometry …

// Geometry-only energy + analytic forces (isolated and periodic).
const auto xdm_iso = xdm_pbe(Z, pos, /*cell*/{0, 0, 0}, /*periodic=*/false);
const double L = 12.0 * kA2B;
const double cellp[3] = {L, L, L};
const double rcut = 50.0 * kA2B;
const auto xdm_per = xdm_pbe(Z, pos, cellp, /*periodic=*/true, rcut);
// xdm_*.energy [Ha]; xdm_*.forces[i][d] [Ha/Bohr]
// Dual-gate: independent central FD of energy vs product forces (h=1e-5 Bohr)

// Product SCF materializes the same periodic E_disp on GpwResult.
GpwDft gpw;
gpw.cell[0] = gpw.cell[1] = gpw.cell[2] = L;
gpw.cutoff_ry = 200.0;
gpw.n_occ = 8;
gpw.max_iter = 8;
gpw.tol = 1e-6;
gpw.mix = 0.4;
gpw.xc = XcKind::PBE;
gpw.nonperiodic = false;
gpw.xdm = true; // exclusive vs d3/d3bj/d4/ts/mbd/vdw_*

gpw.build(atoms);
const auto& res = gpw.run(atoms);
// res.e_disp within 1e-8 of periodic dual-Python pin
// assemble_gpw_forces(..., cfg with xdm) adds F_XDM to electronic forces
```

Driver / Frame path (component; same member):

```cpp
// kinds = {O, H} Kind{Element, GaussianBasis, GthPotential}
ATV::cpu::gpw::GPWEvaluator eval(kinds, /*cutoff_ry=*/300.0, /*max_iter=*/200,
                                 /*tol=*/1e-7, /*mix=*/0.3,
                                 ATV::cpu::gpw::XcKind::PBE,
                                 /*nonperiodic=*/false,
                                 /*compute_forces=*/true);
eval.xdm = true;
// single_point(frame, eval) or bind on Evaluate / MD / Minimize force slot
// eval.last_result().e_disp [Hartree]
// Frame kAtom_f* [eV/Å] includes electronic + XDM forces
```

Build and run the regression (requires GPW + HF + PW + regressions):

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
  cmake --build build --target reg_gpw-xdm-forces -j$(nproc)
  ./build/regressions/reg_gpw-xdm-forces
'
```

| Path | Role |
| ---- | ---- |
| `include/atomiverse/cpu/gpw/gpw_solver.h` | `GpwDft::xdm`, `GpwResult::e_disp` |
| `include/atomiverse/cpu/gpw/gpw_evaluator.h` | `GPWEvaluator::xdm` |
| `include/atomiverse/cpu/gpw/internal/xdm_dispersion.h` | `xdm_pbe` / `XdmResult` |
| `include/atomiverse/cpu/gpw/basis_data/xdm_tables.h` | Free-atom \(\alpha,M_1{-}M_3\); PBE \(a_1,a_2\) |
| `src/cpu/gpw/internal/xdm_dispersion.cpp` | Energy + analytic force kernel |
| `src/cpu/gpw/internal/gpw_forces.cpp` | `assemble_gpw_forces` folds `xdm` forces |
| `tests/cpu/gpw/test_cpu_gpw_xdm.cpp` | Unit energy/force dual-gate |
| `tests/cpu/gpw/test_cpu_gpw_forces.cpp` | Product Γ RKS + XDM analytic vs multi-SCF FD |
| `regressions/gpw/gpw-xdm-forces/reg_gpw-xdm-forces.cpp` | Product e2e gate |
| `regressions/gpw/gpw-xdm-forces/REFERENCE.md` | Offline pins + perf_exempt |
| `tools/cp2k_oracle/inputs/gpw-xdm-forces/` | Twin policy note (no stock XDM) |
| `regressions/gpw/gpw-xdm-ice/` | Energy-only sibling pin |

## Validation (dual pure-Python / FD dual-gate)

Source:
[`regressions/gpw/gpw-xdm-forces/REFERENCE.md`](../../../regressions/gpw/gpw-xdm-forces/REFERENCE.md).
Oracle: offline **dual pure-Python** lattice sum (same tables as ATV
`xdm_tables.h`) + **independent central FD** of \(E_{\mathrm{disp}}\) vs
product analytic forces. Twin notes under
`tools/cp2k_oracle/inputs/gpw-xdm-forces/`. Hard-coded pins in the regression
binary — **never** live CP2K / postg / QE from ctest.

**Setup:**

| Item | Value |
| ---- | ----- |
| System | Water dimer, 6 atoms (order O H H O H H) |
| Geometry | See Method table (same as `gpw-xdm-ice`) |
| Cell | Isolated (force sample + iso energy); periodic 12 Å (per energy + per FD) |
| Damping | QE NC-PBE \(a_1=0.4283\), \(a_2=2.4690\) Å |
| rcut | 50 Å (`pair_image_counts`) |
| FD step | \(h = 10^{-5}\) Bohr (central difference) |
| Atomiverse path | `xdm_pbe` + short `GpwDft` with `xdm=true` |

**Offline hard pins:**

| Quantity | Pin | Tol (ATV gate) | Notes |
| -------- | --: | -------------- | ----- |
| \(E_{\mathrm{disp}}\) isolated [Ha] | **−2.1020435998516220×10⁻³** | \(1\times 10^{-8}\) Ha | Dual pure-Python |
| \(E_{\mathrm{disp}}\) periodic [Ha] | **−2.1148157612879029×10⁻³** | \(1\times 10^{-8}\) Ha | Same as `gpw-xdm-ice` |
| max\|F_product − F_fd\| [Ha/Bohr] | ≤ **1×10⁻⁶** | dual-gate (iso + periodic) | Independent central FD |
| \(\|\sum_A\mathbf{F}_A\|\) [Ha/Bohr] | ≤ **1×10⁻⁸** | translation residual | Free-atom pairwise |
| max\|F_iso − F_dual_python sample\| [Ha/Bohr] | ≤ **1×10⁻¹⁰** | hard component pin | Offline analytic ≡ ATV tables |
| \(GpwDft\) `e_disp` [Ha] | = periodic pin | \(1\times 10^{-8}\) Ha | Product SCF materialization |
| Cross-check dual-Python vs ATV forces | \(\sim 10^{-16}\) Ha/Bohr | offline only | Never in ctest |

**Isolated free-atom force sample** [Ha/Bohr] (dual pure-Python ≡ ATV;
REFERENCE / regression hard pin):

| Atom | \(F_x\) | \(F_y\) | \(F_z\) |
| ---- | ------: | ------: | ------: |
| O0 | +1.6875829936003448×10⁻⁴ | 0 | 0 |
| H0 | +6.9736686381575340×10⁻⁵ | 0 | −1.3172272218329521×10⁻⁵ |
| H1 | +6.9736686381575340×10⁻⁵ | 0 | +1.3172272218329521×10⁻⁵ |
| O1 | −1.1378113807647229×10⁻⁴ | 0 | 0 |
| H2 | −9.7225267023356434×10⁻⁵ | 0 | −1.5448899346549854×10⁻⁵ |
| H3 | −9.7225267023356434×10⁻⁵ | 0 | +1.5448899346549854×10⁻⁵ |

**Tolerance rationale.** Matching free-atom tables, BJ combination rules, and
the same real-space image enumeration track dual pure-Python to machine
epsilon on energy and forces. The product dual-gate (FD \(h=10^{-5}\) Bohr)
uses \(1\times 10^{-6}\) Ha/Bohr to cover central-difference truncation while
remaining tight vs physical \(|F|\) (\(\sim 10^{-4}\) Ha/Bohr on O0). Component
pin vs dual-Python at \(1\times 10^{-10}\) Ha/Bohr is the hard offline force
sample (not a live engine re-run).

## Performance

Source: `REFERENCE.md` (queue / handoff: **perf_exempt**).

XDM evaluation is \(O(N^2\times N_{\mathrm{images}})\) over a 6-atom cell
(microseconds). Full GPW SCF wall dominates if present. There is **no** fair
CP2K `ENERGY_FORCE` twin without stock XDM.

| Twin | \(T_{\mathrm{oracle}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | --------------------------: | ----------------------: | ----: | ---- | ------ |
| Full QS ENERGY_FORCE | n/a (no stock `TYPE XDM`) | — | **perf_exempt** | — | Twin note retained under `tools/cp2k_oracle/inputs/gpw-xdm-forces/` |
| E_disp + force dual-gate | n/a | `reg_gpw-xdm-forces` | **perf_exempt** | — | Geometry-only primary deliverable |

**perf_exempt = true.** Same policy as `gpw-xdm-ice` / `gpw-ts-forces` /
`gpw-mbd-forces`: primary acceptance is E_disp + analytic force dual-gate
parity, not \(r = T_{\mathrm{ATV}}/T_{\mathrm{CP2K}} \le 10\).

## Limitations

- **Free-atom foothold only** — no Hirshfeld / density-dependent multipole
  scaling (contrast density-scaled TS/MBD via `GpwDft::hirshfeld`).
- **H and O tables only** in `xdm_tables.h` for the water-dimer foothold;
  unsupported Z throws (no silent fallback).
- **Cell is a water dimer**, not bulk ice Ih — sufficient for pairwise force
  / lattice-sum parity; not a structural ice benchmark.
- **No stock CP2K XDM twin** — pins are dual pure-Python + FD dual-gate, not
  QS FORCE_EVAL components.
- **No XDM stress pin** on this slug (ionic forces only).
- **`xdm` mutually exclusive** with other GPW dispersion / nonlocal flags.
- **Short SCF** in the regression materializes `e_disp` only; absolute
  \(e_{\mathrm{tot}}\) vs a high-cutoff QS twin is not gated.
- Frame forces remain eV/Å on the public surface; pin comparison is Ha/Bohr
  (`units::kEvPerAngstromInHartreePerBohr` when converting Frame columns).
- Unit product analytic-vs-FD for full GPW+XDM lives under
  `tests/cpu/gpw/test_cpu_gpw_forces.cpp` (not the regression dual-gate of
  geometry-only XDM).

## References

- Offline pin: `regressions/gpw/gpw-xdm-forces/REFERENCE.md`
- Energy sibling: `regressions/gpw/gpw-xdm-ice/REFERENCE.md`
- Twin notes: `tools/cp2k_oracle/inputs/gpw-xdm-forces/water_dimer_xdm_forces.note`
- Oracle policy: `tools/cp2k_oracle/README.md`
- Roadmap: `docs/roadmap/dft.md` §8 Dispersion
- Narrative GPW: `docs/tutorial/ch09-gpw.md`
- Related force pins: `gpw-ts-forces`, `gpw-mbd-forces`, `gpw-d4-forces`,
  `gpw-forces-water`
- A. D. Becke, E. R. Johnson, *J. Chem. Phys.* **127**, 154108 (2007)
- A. Otero-de-la-Roza, E. R. Johnson, *J. Chem. Phys.* **136**, 174109 (2012)
  (postg / damping practice; QE NC-PBE \(a_1,a_2\))
- J. VandeVondele et al., *Comput. Phys. Commun.* **167**, 103 (2005) (GPW host)
