# Bulk polarizable IL under NPT — PIM virial reaches the barostat (map §9)

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `pim-npt-polarizable-il` |
| Regression | `regressions/md/pim-npt-polarizable-il/` |
| Public API | `cpu::PimInducedDipole` (`Potential`) + `cpu::md::NoseHooverNPT` on `cpu::md::MD`; pin legs on `cpu::Evaluate` |
| Roadmap | map §9 "Bulk polarizable IL / PIM-NPT" |
| Oracle | **MetalWalls** bulk-PIM Thole form (offline numpy pin; MW JOSS 2020, Thole 1981) — not LAMMPS |

## Summary

The **PIM-NPT product twin**: a bulk polarizable ionic melt barostats on the
**full** pressure including the polarization contribution. The existing
`cpu::PimInducedDipole` Component (E4, `pim-induced-dipole-bulk`) solves the
Thole-damped induced-dipole SCC and writes polarization energy + forces to
Frame; this slug pins that its **virial**

\[
W=\tfrac12\sum_{i\neq j}\mathbf r_{ij}\otimes\mathbf f_{ij}
\quad\text{(Hellmann–Feynman pair forces, Voigt, eV, LAMMPS sign)}
\]

lands on the shared stress meta (`kStress`) that **every barostat reads** —
so `cpu::md::NoseHooverNPT` (MTK, LAMMPS `fix npt iso` convention) sees the
polarization pressure \(P_{\mathrm{pol}}=\mathrm{tr}(W)/(3V)\) alongside the
LJ / Coulomb pair virials, with no PIM-specific plumbing. Three legs:
(1) hard pin of \(E_{\mathrm{pol}}\), all six Voigt virial components and
\(P_{\mathrm{pol}}\) against an offline numpy implementation of the published
MetalWalls bulk-PIM Thole form; (2) stress **additivity** — the
LJ+coul/cut+PIM stack's trace minus the LJ+coul/cut stack's trace equals the
pinned PIM trace; (3) a short **NPT product run** of a stretched toy ionic
melt that must contract, boundedly, with per-step SCC convergence and a
finite MTK conserved quantity. **Oracle is MetalWalls** (offline numpy pin,
no MW binary on the host); LAMMPS has no SCF bulk-PIM twin (its polarizable
paths are Drude / QEq — a different algorithm), hence **`perf_exempt`**.

## Method

**Physics.** Same Applequist/Thole SCC as `pim-induced-dipole-bulk`
(see that page for the induced-dipole equations): isotropic per-type
polarizability \(\alpha\), permanent field of Frame point charges, mutual
induction through the Thole-damped dipole tensor
(\(u=a\,r/(\alpha_i\alpha_j)^{1/6}\), Thole 1981), Hellmann–Feynman forces at
converged SCC. New in this slug is the **pressure coupling**: the pair-force
decomposition of those Hellmann–Feynman forces is accumulated into the Voigt
virial and summed into `kStress`, the single stress meta shared by all
`Potential` components. `NoseHooverNPT` (MTK equations of motion; Tuckerman
*J. Phys. A* **39**, 5629 (2006), Shinoda *Phys. Rev. B* **69**, 134103
(2004)) then barostats on configurational + kinetic stress — polarization
included — exactly as it would for any classical pair style.

**What this pin includes**

| Item | Setting |
| ---- | ------- |
| Geometry | 4×4×4 simple-cubic rock-salt toy IL, \(d=2.35\) Å (compressed start), cubic \(L=9.4\) Å, \(N=64\) |
| Charges / types | \((-1)^{i_x+i_y+i_z}\): +1 cation (type 0, \(\alpha=0\)), −1 anion (type 1, \(\alpha=3.0\) Å³) |
| Symmetry breaking | deterministic displacement `disp[a] = 0.15·sin(1.7·(i+1) + 2.3·a)` Å (a perfect rock-salt lattice has \(E_{\mathrm{pol}}\equiv 0\)) |
| PIM | cutoff 4.0 Å, `thole_a = 0.39`, `tt_b = 0`, `pol_tol = 10^{-13}` e·Å (pin uses 25 Jacobi iterations); \(k = 14.399645\) eV·Å/e² |
| Pin binding | `cpu::Evaluate` + `add_ref` at `evaluate::Phase::evaluate` |
| NPT leg | same lattice **stretched** to \(d=2.9\) Å (start tension \(P_0\approx-0.12\) eV/Å³); `PairLJCut{0.01 eV, 3.0 Å}` + `PairCoulCut` + PIM; `NoseHooverNPT{T=800 K, τ_T=100 fs, P=0, τ_P=500 fs, dt=1 fs}` + `Virial` + `InitVelocity`, 250 steps |

The NPT leg starts stretched, not compressed: a stretched ionic melt
contracts onto the stiff \(r^{-12}\) wall (self-limiting), while the
compressed pin-leg density (\(P_0\approx+0.88\) eV/Å³) would expand
ballistically through the soft attractive side.

**Key exclusions (non-goals of this slug)**

- New public surface — this is a **product twin** of existing Components
  (`PimInducedDipole`, `NoseHooverNPT`, pair styles); no new class.
- Ewald/PME polarization virial (real-space cutoff PIM only).
- Fumi–Tosi NPT twin: `PairBorn` / `PairBuck` do not write `kStress` (found
  gap, reported upstream; out of slug scope) — the product leg uses the
  virial-correct LJ + coul/cut stack instead.
- Live MetalWalls binary inside ctest (offline numpy pin only).
- LAMMPS Drude / QEq polarizable models as oracle or wall-clock twin.

## Implementation

Public surface only (`cpu::Frame` + ctor args / members; no Config bag).
Pin leg — PIM alone on `cpu::Evaluate`, virial read from `kStress`:

```cpp
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/potentials/pim_induced_dipole.h"

using namespace ATV;

cpu::Frame f = /* 4x4x4 rock-salt toy IL, d = 2.35 Å (see regression) */;

cpu::PimInducedDipole pim(/*cutoff [Å]=*/4.0,
                          /*alpha_by_type [Å³]=*/{0.0, 3.0},
                          /*thole_a=*/0.39,
                          /*pol_tol [e·Å]=*/1e-13,
                          /*pol_max_iter=*/1000);
cpu::Evaluate eval;
eval.on(evaluate::Phase::evaluate).add_ref(pim);
eval.build();
eval.run(f);

const double E_pol = f.meta.get(kEnergy);         // [eV]
const double* W = f.meta.get(kStress);            // Voigt virial [eV]
const double P_pol = (W[0] + W[1] + W[2]) / (3.0 * f.box.volume());
```

NPT product composition — the barostat reads the same `kStress` the PIM
virial lands on (plus an out-of-library `TrackVolume` Criterion, the
documented user-extensible surface):

```cpp
#include "atomiverse/cpu/dynamics/init.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/nose_hoover_npt.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/observe/virial_stress.h"
#include "atomiverse/cpu/potentials/pair_coul_cut.h"
#include "atomiverse/cpu/potentials/pair_lj_cut.h"

cpu::Frame f2 = /* same lattice stretched to d = 2.9 Å */;

cpu::PimInducedDipole pim2(4.0, {0.0, 3.0}, 0.39,
                           /*pol_tol=*/1e-10, /*max_iter=*/500);
cpu::md::MD sim;
sim.set_evaluator(cpu::PairLJCut{0.01, 3.0, 4.0});
sim.set_evaluator(cpu::PairCoulCut{4.0});
sim.set_evaluator_ref(pim2);
sim.set_neighbor(cpu::md::NeighborList<>{4.0, 0.3});
sim.set_integrator(cpu::md::NoseHooverNPT{
    /*T_target [K]=*/800.0, /*tau_T [fs]=*/100.0,
    /*P_target [eV/Å³]=*/0.0, /*tau_P [fs]=*/500.0, /*dt [fs]=*/1.0});
sim.add(cpu::md::InitVelocity{800.0, 20260727u});
sim.add(cpu::md::Virial{});                  // kinetic stress
sim.build();
sim.run(f2, 250);

const double H = f2.meta.get(kConserved);    // MTK conserved quantity [eV]
const double V1 = f2.box.volume();           // [Å³]
```

Architecture rules:

- **No new Driver, no new Component.** The slug is composition: `Potential`s
  stack on `cpu::md::MD`; the barostat is the `NoseHooverNPT` integrator.
- Parameters are **constructor arguments and public members** — no Config
  bag, no factory.
- Frame is the sole I/O currency: \(E_{\mathrm{pol}}\) on `kEnergy`, virial
  on `kStress`, conserved quantity on `kConserved`; the SCC throws on any
  unconverged step (no silent drift).

Build and run the regression (MD-only configure; CPU login pattern from
`AGENTS.md` / metalwalls loop):

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
        -DATV_BUILD_CUDA=OFF -DBLA_VENDOR=OpenBLAS
  cmake --build build --target reg_pim-npt-polarizable-il -j"$(nproc)"
  ./build/regressions/reg_pim-npt-polarizable-il
'
```

| Path | Role |
| ---- | ---- |
| `include/atomiverse/cpu/potentials/pim_induced_dipole.h` | Public `PimInducedDipole` (E4; virial into `kStress`) |
| `include/atomiverse/cpu/dynamics/nose_hoover_npt.h` | MTK NPT integrator+thermostat+barostat |
| `regressions/md/pim-npt-polarizable-il/reg_pim-npt-polarizable-il.cpp` | Product gate (3 legs) |
| `tools/metalwalls_oracle/inputs/pim_npt_polarizable_il_pin.py` | Offline numpy pin script (never run at ctest time) |

## Validation (vs MetalWalls-form offline numpy pin)

Source:
[`regressions/md/pim-npt-polarizable-il/REFERENCE.md`](../../../regressions/md/pim-npt-polarizable-il/REFERENCE.md)
and
[`tools/metalwalls_oracle/pin_notes_pim-npt-polarizable-il.md`](../../../tools/metalwalls_oracle/pin_notes_pim-npt-polarizable-il.md)
(offline pin; **never** live MetalWalls in ctest).

**Oracle:** MetalWalls JOSS **5**, 2373 (2020) bulk-PIM multipole form with
Thole (1981) exponential damping — an **independent numpy implementation**
(`pim_npt_polarizable_il_pin.py`, numpy 2.5.1) evaluated on the
deterministic regression start configuration. These are literature-formula
evaluations, not a golden dump of Atomiverse output.

### Accuracy

| Quantity | ATV | Oracle pin | Tol | Notes |
| -------- | --: | ---------: | --- | ----- |
| \(E_{\mathrm{pol}}\) [eV] | matches pin | **−8.695749870555906** | \(\lvert\Delta E\rvert \le 10^{-8}\) | dual: `kEnergy` and `last_pol_energy()` |
| \(W_{xx}\) [eV] | matches pin | **−10.66184373345355** | \(\lvert\Delta W\rvert \le 10^{-7}\) | Voigt on `kStress` |
| \(W_{yy}\) [eV] | matches pin | **−12.125914211727187** | \(\le 10^{-7}\) | |
| \(W_{zz}\) [eV] | matches pin | **−9.657852793291474** | \(\le 10^{-7}\) | |
| \(W_{xy}\) [eV] | matches pin | **−0.2325271422922529** | \(\le 10^{-7}\) | full off-diagonal, not just trace |
| \(W_{xz}\) [eV] | matches pin | **−2.2469237837238** | \(\le 10^{-7}\) | |
| \(W_{yz}\) [eV] | matches pin | **+1.375774440282161** | \(\le 10^{-7}\) | |
| \(P_{\mathrm{pol}}=\mathrm{tr}(W)/(3V)\) [eV/Å³] | matches pin | **−0.013021203851134545** | \(\lvert\Delta P\rvert \le 10^{-10}\) | the pressure the barostat sees |
| Stress additivity [eV] | \(\mathrm{tr}(\text{LJ+coul+PIM})-\mathrm{tr}(\text{LJ+coul})\) | pinned PIM trace | \(\le 10^{-6}\) | PIM shares the barostat's `kStress` |
| SCC | converged (25 iters in pin) | — | `pol_tol = 10^{-13}` e·Å | throws on non-convergence |

Tolerances cover `std::sin` vs `np.sin` position ulps, summation order, and
the \(10^{-13}\) SCC tail.

**NPT product leg (invariants, not pins):** 250-step MTK run from the
stretched start must show per-step SCC convergence, finite conserved
quantity \(H\), final \(E_{\mathrm{pol}} < 0\), contraction
(\(V_{\min} < 0.97\,V_0\)) that is bounded
(\(0.5\,V_0 < V_{\min}\), \(V_{\max} < 1.2\,V_0\)), and finite positive
final volume.

## Performance

Source: REFERENCE.md; handoff
the case `REFERENCE.md`
(wave 5, 2026-07-27).

| Twin | \(T_{\mathrm{oracle}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | --------------------------: | -----------------------: | ----: | ---- | ------ |
| bulk PIM + NPT polarizable IL | — | — | **perf_exempt** | n/a | No MetalWalls binary on PATH; LAMMPS has no SCF bulk-PIM twin (Drude/QEq are a different algorithm) |

| Field | Value |
| ----- | ----- |
| `perf_exempt` | **true** |
| `perf_exempt_reason` | No MetalWalls binary on PATH; LAMMPS has no SCF bulk-PIM twin (its polarizable paths are Drude/QEq — a different algorithm), so no fair wall-clock twin exists. Same exemption as `pim-induced-dipole-bulk`. |
| `ratio_atv_over_oracle` | **n/a** |
| `within_10x` | exempt |

Do **not** invent wall-clock ratios. When a MetalWalls twin binary is
available, record \(T_{\mathrm{atv}}/T_{\mathrm{MW}}\) in REFERENCE and
require \(\le 10\). Runtime of the regression itself is ≪ 60 s
(\(N=64\), \(O(N^2)\) SCC per step).

## Limitations

- **Real-space cutoff PIM only.** No Ewald/PME polarization virial; the toy
  melt's 4.0 Å cutoff is a deterministic pin choice, not production
  electrostatics.
- **Toy IL, not a force field.** 64-ion rock-salt lattice with generic
  LJ + coul/cut; a production polarizable-IL parameterization (e.g. CL&Pol,
  Fumi–Tosi + PIM) is downstream usage.
- **`PairBorn` / `PairBuck` are stress-blind** (found gap, reported
  upstream): a Fumi–Tosi NPT twin would barostat without the Born term
  until those pair styles write `kStress`.
- **Isotropic NPT** (`NoseHooverNPT` iso convention); no anisotropic /
  triclinic cell dynamics in this pin.
- **All-pairs \(O(N^2)\) SCC foothold** inherited from E4; production
  scaling deferred.
- **No fair speed ratio** until a MetalWalls bulk-PIM twin is on the pin
  host.

## References

- Offline pin: [`regressions/md/pim-npt-polarizable-il/REFERENCE.md`](../../../regressions/md/pim-npt-polarizable-il/REFERENCE.md)
- Pin notes: [`tools/metalwalls_oracle/pin_notes_pim-npt-polarizable-il.md`](../../../tools/metalwalls_oracle/pin_notes_pim-npt-polarizable-il.md)
- Product gate: [`regressions/md/pim-npt-polarizable-il/reg_pim-npt-polarizable-il.cpp`](../../../regressions/md/pim-npt-polarizable-il/reg_pim-npt-polarizable-il.cpp)
- Sibling foothold page: [`pim-induced-dipole-bulk.md`](pim-induced-dipole-bulk.md) (E4 — the SCC physics and equations)
- Oracle notes: [`tools/metalwalls_oracle/README.md`](../../../tools/metalwalls_oracle/README.md)
- Literature: Marin-Laflèche et al., *J. Open Source Softw.* **5**, 2373
  (2020) (MetalWalls classical PIM); Thole, *Chem. Phys.* **59**, 341
  (1981); Tuckerman et al., *J. Phys. A* **39**, 5629 (2006) and Shinoda
  et al., *Phys. Rev. B* **69**, 134103 (2004) (MTK NPT, LAMMPS
  `fix npt iso` convention)
