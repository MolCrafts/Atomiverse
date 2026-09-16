# Electrode charge observer — parallel-plate \(Q_{\mathrm{tot}}\) (fixed charge)

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `const-pot-charge-observer` |
| Regression | `regressions/md/const-pot-charge-observer/` |
| Public API | `cpu::ElectrodeChargeObserver` (`Observer`); Frame meta `kElectrodeCharge` |
| Roadmap | MetalWalls map E1 observe |

## Summary

Per-electrode **total charge** measurement for classical electrochemical MD.
`cpu::ElectrodeChargeObserver` is a Frame-immutable `Observer` that sums
`kAtom_charge` over owned atoms whose `kAtom_type` matches each configured
electrode type id. It does **not** solve constant-potential charges — it
measures whatever is already on the Frame (fixed charges, or charges written
by a prior const-pot Component such as `ConstPotBO`). Default bind is MD
`end_of_step`. Product gate `reg_const-pot-charge-observer` pins \(Q_L\) and
\(Q_R\) against an offline **LAMMPS** `compute reduce sum` twin on a 10-atom
parallel-plate foothold (constant-charge control). Fair wall vs LAMMPS is
**`perf_exempt`** (micro-sum vs binary startup).

## Method

**Observable.** For electrode \(e\) identified by atom type \(t_e\),

\[
Q_e = \sum_{i\,\in\,\mathrm{owned}:\;\mathrm{type}_i = t_e} q_i
\quad[\mathrm{e}].
\]

Only **owned** local atoms contribute; ghosts are excluded so domain
decomposition cannot double-count charge. With MPI, electrode totals are
`MPI_Allreduce`’d over the domain cart communicator. Electrolyte (and any
other non-listed types) never enter the sums.

**Control regime on this pin.** The regression uses **constant-charge**
walls (\(q_i\) fixed at construction), not a constant-potential charge
solve. That isolates the observer from BO/CG/MaZe physics and matches the
LAMMPS twin (`variable atom q` + `compute reduce sum` on type groups).
Constant-potential solves and charge *dynamics* live on other E1/E2 slugs
(`const-pot-bo-parallel-plate`, `const-pot-md-nvt-short`, …).

**Geometry (parallel-plate toy).**

| Quantity | Value |
| -------- | ----- |
| Box | orthorhombic \(20\times 10\times 10\) Å |
| Left wall | type 1, 4 atoms, \(q_i = +0.125\) e |
| Right wall | type 2, 4 atoms, \(q_i = -0.125\) e |
| Electrolyte placeholders | type 3, 2 atoms, \(q_i = +1\) e (excluded) |
| Ensemble | none (`run 0` twin) — pure charge accounting |

Analytic totals (identical to the reduce-sum pin):

\[
Q_L = 4\times 0.125 = +0.5\,\mathrm{e},\qquad
Q_R = 4\times(-0.125) = -0.5\,\mathrm{e}.
\]

**Frame publication.** After `launch`, totals are available as
`obs.total_charge(e)` and as Frame meta `kElectrodeCharge` (first ≤3
electrodes packed into `MetaF64x3` for thermo / write-back).

**Key exclusions (this slug).**

- Constant-potential BO/CG/MaZe charge **solve** → other E1/E2 regressions;
  this Component never writes \(q_i\).
- Capacitance / charge-fluctuation statistics → E10 (`capacitance-charge-fluctuation`).
- Potential-profile observers → E10.
- Live LAMMPS or MetalWalls at ctest time (offline pins only).
- Fair MD throughput vs LAMMPS on a force loop (not this micro-observer).

## Implementation

Public surface only (`cpu::Frame` + ctor args / members; no `*Config` bag, no
`make_*` / `build_*` factories). Header:
`include/atomiverse/cpu/observe/electrode_charge.h`; implementation
`src/cpu/observe/electrode_charge.cpp`.

Minimal sequence (matches `reg_const-pot-charge-observer.cpp`). Frame is the
only I/O token; the observer is bound on a driver — never constructed with a
user Context:

```cpp
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/observe/electrode_charge.h"

using namespace ATV;

// Parallel-plate twin: types 1/2 walls, type 3 electrolyte excluded.
cpu::Frame frame(10);
// … set x,y,z,type,charge,mass,tag as in the regression …
// wall q = ±0.125 e (4 atoms each); electrolyte q = +1 e (2 atoms)
frame.box = cpu::Box::orthorhombic(20.0, 10.0, 10.0);

cpu::ElectrodeChargeObserver obs({1, 2}); // left type, right type

// Deferred observer (default md::Phase::end_of_step, order +40) — after any
// charge-updating Component has written kAtom_charge:
cpu::md::MD sim;
// … potentials, neighbor, integrator, const-pot / freeze as needed …
sim.set_observer_ref(obs);
// or: sim.set_observer(cpu::ElectrodeChargeObserver{{1, 2}});
// or: sim.on(md::Phase::end_of_step).add_ref(obs);
sim.build();
sim.run(frame, /*nsteps=*/1);

const double Q_L = obs.total_charge(0);                 // [e]
const double Q_R = obs.total_charge(1);                 // [e]
const double meta_L = frame.meta.get(kElectrodeCharge)[0];
const double meta_R = frame.meta.get(kElectrodeCharge)[1];
```

Build and run the regression (MD-only flags; never enable HF/PW/GPW/CUDA
for this loop):

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
  cmake --build build --target reg_const-pot-charge-observer -j$(nproc)
  ./build/regressions/reg_const-pot-charge-observer
'
```

| Path | Role |
| ---- | ---- |
| `include/atomiverse/cpu/observe/electrode_charge.h` | Public `ElectrodeChargeObserver` |
| `src/cpu/observe/electrode_charge.cpp` | Owned-atom sum + optional MPI allreduce |
| `include/atomiverse/core/frame_schema.h` | `kElectrodeCharge` meta key |
| `regressions/md/const-pot-charge-observer/reg_const-pot-charge-observer.cpp` | Product gate |
| `tools/metalwalls_oracle/inputs/lammps/const_pot_charge_observer.in` | Offline LAMMPS twin (never ctest) |

## Validation (oracle: LAMMPS)

Source:
[`regressions/md/const-pot-charge-observer/REFERENCE.md`](../../../regressions/md/const-pot-charge-observer/REFERENCE.md).

**Oracle is LAMMPS, not MetalWalls.** Summing group charge is core LAMMPS
(`variable atom q` + `compute reduce sum`). The ELECTRODE package is not
required for this twin; MetalWalls remains the oracle for MW-only BO/CG/MaZe
charge *solves* on other E1 slugs. Offline pin only — **never** launch LAMMPS
from ctest.

| Item | Value |
| ---- | ----- |
| Oracle | `lammps` |
| Oracle version | LAMMPS (30 Mar 2026 - Development - `patch_30Mar2026-1074-g7f680de296`) |
| Twin input | `tools/metalwalls_oracle/inputs/lammps/const_pot_charge_observer.in` |
| Units | metal → charge in elementary charges [e] |
| Dynamics | `run 0` (no pair/kspace on the pin) |

Recorded pin lines:

```text
PIN_QL 0.5
PIN_QR -0.5
```

| Quantity | Atomiverse / pin | LAMMPS pin | Tol | Notes |
| -------- | ---------------: | ---------: | --- | ----- |
| \(Q_L\) [e] | \(+0.5\) | \(+0.5\) | \(10^{-12}\) | type-1 wall sum |
| \(Q_R\) [e] | \(-0.5\) | \(-0.5\) | \(10^{-12}\) | type-2 wall sum |
| \(Q_L + Q_R\) [e] | \(0\) | \(0\) | \(10^{-12}\) | electrodes alone (electrolyte excluded) |
| `meta` \(Q_L,Q_R\) | same as members | — | \(10^{-12}\) | `kElectrodeCharge[0,1]` |

Tolerance is machine-eps scale: the pin is an **exact finite sum** of fixed
charges (no force loop, no k-space). It catches wrong type selection, ghost
double-counting, or unit mistakes — not a golden of Atomiverse-only output.

Offline re-pin (developer host; not ctest):

```bash
lmp -in tools/metalwalls_oracle/inputs/lammps/const_pot_charge_observer.in
```

## Performance

| Twin | \(T_{\mathrm{LAMMPS}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r = T_{\mathrm{ATV}}/T_{\mathrm{oracle}}\) | Gate | Caveat |
| ---- | --------------------------: | ----------------------: | ----------------------------------------: | ---- | ------ |
| `const-pot-charge-observer` | — | — | **perf_exempt** | n/a | Micro-observer: single \(O(N)\) sum over 10 atoms. Fair wall vs LAMMPS is dominated by binary startup (~0.3 s for `run 0` with no pair/kspace), not the reduce itself. Not a force-loop twin; ratio would not measure product MD throughput. |

When a longer MD + electrode fair bench exists (`const-pot-md-nvt-short` and
follow-ons), re-record \(T_{\mathrm{ATV}}/T_{\mathrm{oracle}}\) there. No
invented wall times on this page.

## Limitations

- **Measure-only.** Does not impose electrode potential, write charges, or
  couple to PME/slab Ewald; pair with `ConstPotBO` / freeze / long-range
  Components for a full const-pot stack.
- **Type-keyed electrodes.** Electrode membership is by `kAtom_type` list;
  arbitrary group masks / tag lists are out of scope on this foothold.
- **Meta pack depth.** `kElectrodeCharge` publishes the first ≤3 electrodes
  for thermo; more electrodes remain available via `total_charges()`.
- **MPI.** Allreduce requires a domain cart communicator on
  `ctx.runtime.md.dd` when `mpi_size > 1`; serial / single-rank path needs no
  reduce.
- **Perf gate.** This slug is explicitly **perf_exempt**; do not treat a
  startup-dominated ratio as product MD speed.
- **Oracle choice.** Overlap physics with LAMMPS → LAMMPS pin (binding for this
  loop). Do not re-pin against MetalWalls for a plain reduce-sum.

## References

- Offline pin: `regressions/md/const-pot-charge-observer/REFERENCE.md`
- Regression: `regressions/md/const-pot-charge-observer/reg_const-pot-charge-observer.cpp`
- LAMMPS twin: `tools/metalwalls_oracle/inputs/lammps/const_pot_charge_observer.in`
- Oracle notes: `tools/metalwalls_oracle/README.md`
- Related E1: `const-pot-bo-parallel-plate`, `freeze-electrode-atoms`,
  `charging-work-energy`, `const-pot-md-nvt-short`
- Literature / context: MetalWalls electrode diagnostics (JOSS
  [10.21105/joss.02373](https://doi.org/10.21105/joss.02373)); LAMMPS
  `compute reduce` documentation
