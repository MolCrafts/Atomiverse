# Constant-potential MD NVT short — parallel-plate NaCl toy (E1 product)

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `const-pot-md-nvt-short` |
| Regression | `regressions/md/const-pot-md-nvt-short/` |
| Public API | `cpu::ConstPotentialBO` + `cpu::md::FixFreeze` on `cpu::md::MD` (with `CoulLong` / `KSpacePME`, `NoseHoover`, `VelocityVerlet`) |
| Roadmap / queue | `docs/developer/metalwalls-method-queue.md` → **done** (wave 1, 2026-07-23); map E1 P0 |

## Summary

**E1 product** for classical electrochemical MD: fluctuating electrode charges at
**fixed potential** (Born–Oppenheimer / Siepmann–Sprik / Reed / Scalfi form),
**frozen walls**, and a short **NVT** trajectory of a neutral electrolyte between
parallel plates. Everything is ordinary Components on `cpu::md::MD` — not a new
Driver category and not a charge solve inside `KSpace*`. The dense Gaussian
matrix \(A\) is rebuilt and inverted each `pre_force` step; force evaluation
remains point-charge `CoulLong` + `KSpacePME` (full Gaussian force kernels are
the separate slug `gaussian-electrode-charge`). Product gate
`reg_const-pot-md-nvt-short` asserts electrode polarity, electroneutrality,
freeze immobility, finite thermo/charges over 400 steps, and a soft \|Q\| pin
(~5 %) against **MetalWalls c8ad175d** (`num_pbc 3` twin; ATV uses PME).
Fair wall-clock ratio remains **`perf_exempt`** (product trajectory; not a
kernel microbench).

## Method

**Constant-potential electrodes.** Electrode sites carry Gaussian charge
densities of width \(w\). At fixed applied potentials \(\Psi\), charges \(Q\)
minimise the quadratic energy (Siepmann & Sprik 1995; Reed, Lanning, Madden
2007; Scalfi *et al.* 2022; MetalWalls JOSS 2020):

\[
U = \tfrac12 Q^{\mathsf T} A Q - Q^{\mathsf T}(\Psi - \Phi_{\mathrm{electrolyte}}).
\]

Stationarity plus global electroneutrality yields the dense linear system

\[
A Q = \Psi - \Phi_{\mathrm{electrolyte}},\qquad
\sum_i Q_i = -Q_{\mathrm{electrolyte}},
\]

with Gaussian matrix elements

\[
A_{ii} = k_e\sqrt{2/\pi}/w,\quad
A_{ij} = k_e\,\mathrm{erf}\!\big(r_{ij}/(w\sqrt2)\big)/r_{ij}\ (i\neq j).
\]

Here \(k_e = \texttt{units::coulomb} = 14.399645\,\mathrm{eV·Å}/e^2\).
\(\Phi_i\) is the min-image potential at electrode site \(i\) from
**non-electrode** (point) charges. For neutral electrolyte,
\(\sum_i Q_i = 0\).

**Force path (this slug).** After \(Q\) is written to electrode
`kAtom_charge`, real-space Coulomb is `CoulLong` and reciprocal is
`KSpacePME` — **point-charge** PME, not Gaussian electrode force kernels.
Walls are held immobile by `FixFreeze` (force zero + velocity zero + position
restore; LAMMPS `fix setforce 0` family). Mobile ions run Velocity-Verlet
NVT with Nose–Hoover.

**What this pin includes**

| Item | Setting |
| ---- | ------- |
| Geometry | Parallel plates + 4 Na⁺ + 4 Cl⁻ (see Validation) |
| \(\Psi_L, \Psi_R\) | −0.5 V, +0.5 V |
| Gaussian width \(w\) | 0.5 Å |
| Long-range | `CoulLong` + `KSpacePME` (\(\alpha=0.3\) Å⁻¹, cutoff 6 Å) |
| Soft LJ | \(\varepsilon=0.05\) eV, \(\sigma=2.0\) Å, cut 5 Å |
| Ensemble | NVT, \(T=300\) K, \(\tau_T=100\) fs, \(dt=0.5\) fs, 400 steps |
| Freeze | wall types 2 and 3 |

**Key exclusions (non-goals of this slug)**

- Live MetalWalls or LAMMPS `ELECTRODE` numeric \(Q\) bit-match (no binary on
  worker; algorithm + polarity / neutrality gates only).
- Gaussian electrode force kernels (`gaussian-electrode-charge`).
- Dense → CG charge solve for large walls (`const-pot-cg-large-electrode`).
- Charge observer / charging-work thermodynamics (sibling wave-1 slugs).
- 2D Ewald / slab corrections (`slab-2d-ewald-foothold`), external field
  (`external-efield-md`), polarizable / TF electrodes (E4–E7).
- New Driver category; charge solve welded into `KSpace*`.
- Public `*Config` bags or `make_*` / `build_*` factories.

## Implementation

Public surface only: constructor arguments / members on Components; composition
on `cpu::md::MD`. No `*Config` bag.

```cpp
#include "atomiverse/cpu/dynamics/const_potential.h"
#include "atomiverse/cpu/dynamics/fix_freeze.h"
#include "atomiverse/cpu/dynamics/init.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/nose_hoover.h"
#include "atomiverse/cpu/dynamics/velocity_verlet.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/longrange/pme.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/coul_long.h"
#include "atomiverse/cpu/potentials/pair_lj_cut.h"

using namespace ATV;

// Parallel-plate toy: types 2/3 = walls, 0/1 = Na/Cl (see regression).
cpu::Frame frame(/*N*/ 16);
// … fill positions, type, charge, mass, tag; orthorhombic box …

cpu::ConstPotentialBO cpot(
    /*electrode_types*/ std::vector<int>{2, 3},
    /*potentials_V*/    std::vector<double>{-0.5, +0.5},
    /*gaussian_width_A*/ 0.5);

cpu::md::MD sim;
sim.set_observer(cpu::md::InitVelocity{/*T=*/300.0, /*seed=*/7u});
sim.set_integrator(cpu::md::VelocityVerlet{/*dt [fs]=*/0.5});
sim.set_thermostat(cpu::md::NoseHoover{/*T=*/300.0, /*tau_T=*/100.0, /*dt=*/0.5});
sim.set_evaluator(cpu::PairLJCut{/*eps=*/0.05, /*sigma=*/2.0, /*cut=*/5.0});
sim.set_neighbor(cpu::md::NeighborList<>{/*cut=*/6.0, /*skin=*/0.5});
sim.on(md::Phase::force).add(cpu::CoulLong{/*alpha=*/0.3, /*cutoff=*/6.0});
sim.set_evaluator(cpu::KSpacePME{/*alpha=*/0.3, /*spacing=*/0.5, /*order=*/5});
sim.set_observer(std::move(cpot));           // default: pre_force @ -500
sim.set_observer(cpu::md::FixFreeze{2});     // left wall type
sim.set_observer(cpu::md::FixFreeze{3});     // right wall type
sim.build();
sim.run(frame, 400);

// Electrode totals: sum kAtom_charge over type 2 / 3 (or cpot.total_charge
// if retained before move). Expect Q_L < 0 < Q_R for Ψ_L < Ψ_R.
```

Build and run the regression (**MD-only** configure; modules as in
`AGENTS.md`):

```bash
bash -lc '
  module load buildenv-gcc/2026.03-mpich
  export OPENBLAS_ROOT="$OPENBLASDIR"
  export RUSTUP_TOOLCHAIN=stable-x86_64-unknown-linux-gnu
  export LD_LIBRARY_PATH="$LIBRARY_PATH:$LD_LIBRARY_PATH"
  cmake -B build -DCMAKE_BUILD_TYPE=Release \
        -DATV_BUILD_MD=ON -DATV_BUILD_KSPACE=ON \
        -DATV_BUILD_TESTS=ON -DATV_BUILD_REGRESSIONS=ON \
        -DATV_BUILD_HF=OFF -DATV_BUILD_PW=OFF -DATV_BUILD_GPW=OFF \
        -DATV_BUILD_CUDA=OFF \
        -DBLA_VENDOR=OpenBLAS
  cmake --build build --target reg_const-pot-md-nvt-short -j$(nproc)
  ./build/regressions/reg_const-pot-md-nvt-short
'
```

| Path | Role |
| ---- | ---- |
| `include/atomiverse/cpu/dynamics/const_potential.h` | Public `ConstPotentialBO` |
| `include/atomiverse/cpu/dynamics/fix_freeze.h` | Public `FixFreeze` |
| `include/atomiverse/cpu/dynamics/md.h` | `cpu::md::MD` preset |
| `include/atomiverse/cpu/potentials/coul_long.h` | Real-space long Coulomb |
| `include/atomiverse/cpu/longrange/pme.h` | `KSpacePME` |
| `src/cpu/dynamics/` | BO solve + freeze impl (private workspace) |
| `regressions/md/const-pot-md-nvt-short/reg_const-pot-md-nvt-short.cpp` | Product gate |
| `tools/metalwalls_oracle/inputs/metalwalls/const-pot-md-nvt-short/` | Offline twin notes (no live dump yet) |

## Validation (vs MetalWalls algorithm pin)

Source:
[`regressions/md/const-pot-md-nvt-short/REFERENCE.md`](../../../regressions/md/const-pot-md-nvt-short/REFERENCE.md).  
Oracle: **MetalWalls-class** constant-potential BO (Siepmann–Sprik / Reed /
Scalfi). **No MetalWalls binary** and **no LAMMPS ELECTRODE package** on the
offline worker (LAMMPS 30 Mar 2026 packages on host: EXTRA-*, KSPACE, MANYBODY —
no ELECTRODE). Pins are **algorithm + continuum polarity / neutrality /
freeze**, not a live energy or \(Q\) dump. Never launch MetalWalls or LAMMPS
from ctest. Twin notes:
`tools/metalwalls_oracle/inputs/metalwalls/const-pot-md-nvt-short/README.md`.

**Setup (regression consumer; authoritative for numbers):**

| Item | Value |
| ---- | ----- |
| Species | Na⁺ type 0, Cl⁻ type 1, left wall type 2, right wall type 3 |
| Box | orthorhombic \(L_x=L_y=14\) Å, \(L_z=20\) Å |
| Walls | \(2\times 2\) sites at \(z=2\) Å (type 2) and \(z=18\) Å (type 3) |
| Electrolyte | 4 Na⁺ + 4 Cl⁻ on staggered mid-slab lattice (\(z\in\{6,14\}\) Å) |
| \(\Psi_L, \Psi_R\) | −0.5 V, +0.5 V |
| \(w\) | 0.5 Å |
| Soft LJ | \(\varepsilon=0.05\) eV, \(\sigma=2.0\) Å, cut 5 Å |
| Coulomb / PME | \(\alpha=0.3\) Å⁻¹, real-space cutoff 6 Å, PME spacing 0.5 Å, order 5 |
| NVT | `VelocityVerlet` + `NoseHoover`, \(T=300\) K, \(\tau_T=100\) fs, \(dt=0.5\) fs |
| Trajectory | 400 steps (8 chunks of 50 after 1 seed step) |
| Units | eV, Å, fs, amu, eV/Å, K, charge e, potential V (= eV/e) |

**Accuracy / acceptance gate (product):**

| Quantity | Atomiverse / pin | Oracle | Bound | Notes |
| -------- | ---------------: | ------ | ----- | ----- |
| Vacuum-style polarity (after first BO solve) | \(Q_L < 0 < Q_R\) | MW-class capacitor sign | hard | Higher-\(\Psi\) plate positive |
| Electroneutrality (step 1) | \(\lvert Q_L+Q_R\rvert\) | MW neutrality | \(\le 10^{-8}\) e | Neutral electrolyte |
| Neutrality during NVT chunks | \(\lvert Q_L+Q_R\rvert\) | MW neutrality | \(\le 10^{-6}\) e | Looser for PME / min-image \(\Phi\) noise |
| Thermo | \(E\), \(K\) | finite every chunk | finite | Smoke stability / no NaN |
| Charges | all \(q_i\) | finite | finite | Charge-solve stability |
| Freeze position | wall \(\Delta x\) | FixFreeze / MW `mobile=false` | \(\le 10^{-10}\) Å | Per wall atom vs build snapshot |
| Freeze velocity | wall \(\lvert v\rvert\) | same | \(\le 10^{-12}\) Å/fs | After NVT |
| End polarity | \(Q_L < 0 < Q_R\) | MW-class | hard | Polarity survives 400 steps |
| Fluid not frozen | ion \(\lvert v\rvert\) or \(K\) | selective freeze | any ion \(v^2>10^{-12}\) or \(K>0.01\) eV | Walls only |

**Continuum diagnostic (optional; not a hard gate).** Vacuum parallel-plate
capacitance \(C \approx A/(4\pi k_e d)\) with \(A=L_x L_y\), \(d\approx 16\) Å,
\(\Delta\Psi = 1\) V gives \(\lvert Q\rvert \sim 0.05\,e\) scale. Atomistic
Gaussian BO on this toy grid is the **same sign** and **similar magnitude**;
the product gate does **not** assert a numeric \(Q\) column against a live
MetalWalls dump.

When a MetalWalls (or LAMMPS ELECTRODE) build is available offline, place a
runtime twin under `tools/metalwalls_oracle/inputs/metalwalls/const-pot-md-nvt-short/`
and refresh `oracle_version` plus numeric \(Q\) columns in REFERENCE — do not
invent them here.

## Performance

Source:
[`REFERENCE.md`](../../../regressions/md/const-pot-md-nvt-short/REFERENCE.md)
(2026-07-23).

| Item | Value |
| ---- | ----- |
| `perf_exempt` | **true** |
| Reason | No MetalWalls binary and no LAMMPS ELECTRODE package on the offline worker; cannot fair-bench vs an oracle twin for this MW-only path |
| \(r = T_{\mathrm{ATV}}/T_{\mathrm{oracle}}\) | **n/a (exempt)** |
| `within_10x` | n/a (exempt) |

Campaign queue records the same: ratio **perf_exempt**
(`docs/developer/metalwalls-method-queue.md` → done). Gate when a twin exists:
\(r = T_{\mathrm{ATV}}/T_{\mathrm{oracle}} \le 10\) on the same host/core count
(single-rank), with MetalWalls as primary oracle for this MW-only algorithm
(LAMMPS ELECTRODE only if that package is the chosen offline twin).

| Twin | \(T_{\mathrm{oracle}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | --------------------------: | -----------------------: | ----: | ---- | ------ |
| Parallel-plate BO NVT short | — | — | **exempt** | n/a | No MW / ELECTRODE binary at pin; remeasure when available |

**No invented wall times or speed ratios.**

## Limitations

- **Short NVT smoke product**, not production EDL sampling: 16 atoms, 400 steps,
  soft LJ, toy \(2\times 2\) electrode grids.
- **Point-charge force path** after BO \(Q\) write — not full Gaussian electrode
  forces (`gaussian-electrode-charge`).
- **Dense \(A^{-1}\)** only — large electrodes need CG
  (`const-pot-cg-large-electrode`).
- **Oracle is algorithm-class MetalWalls**, not a live dump: polarity /
  neutrality / freeze gates only until offline MW (or ELECTRODE) pins exist.
- **`perf_exempt`**: no fair \(r\); do not invent a speed ratio.
- **CPU MD only** in this loop; no CUDA electrode path here.
- Sibling E1 pieces (charge observer, charging work) and E2–E10 map rows are
  separate queue slugs — not claimed by this page.

## References

- Offline pin: `regressions/md/const-pot-md-nvt-short/REFERENCE.md`
- Regression consumer: `regressions/md/const-pot-md-nvt-short/reg_const-pot-md-nvt-short.cpp`
- Oracle notes: `tools/metalwalls_oracle/inputs/metalwalls/const-pot-md-nvt-short/README.md`,
  `tools/metalwalls_oracle/README.md`
- Capability map: `.claude/notes/metalwalls-capability-map.md` (E1)
- Queue / loop / handoff: `docs/developer/metalwalls-method-queue.md`,
  `docs/developer/metalwalls-method-loop.md`,
  `docs/developer/metalwalls-method-handoff.md`
- Public API: `include/atomiverse/cpu/dynamics/const_potential.h`,
  `include/atomiverse/cpu/dynamics/fix_freeze.h`
- J. I. Siepmann, M. Sprik, *J. Chem. Phys.* **102**, 511 (1995) — Gaussian
  fluctuating electrode charges at fixed potential.
- S. K. Reed, O. J. Lanning, P. A. Madden, *J. Chem. Phys.* **126**, 084704
  (2007) — constant-potential MD electrodes.
- L. Scalfi *et al.*, *Chem. Rev.* **122**, 10899 (2022) — constant-potential
  and electrochemical MD review.
- A. Marin-Laflèche *et al.*, *J. Open Source Softw.* **5**, 2373 (2020)
  (10.21105/joss.02373) — MetalWalls.
