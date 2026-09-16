# Electrode charging work \(-\sum_i \Psi_i Q_i\) — parallel-plate bookkeeping

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `charging-work-energy` |
| Regression | `regressions/md/charging-work-energy/` |
| Public API | `cpu::electrodes::ChargingWork` (`Potential` kind); bind via `set_evaluator` on `cpu::Evaluate` / `cpu::md::MD` |

## Summary

Under constant applied electrode potentials, the classical potential energy of
an electrochemical cell includes a **charging-work** term exchanged with the
external circuit: \(E_{\mathrm{charge}} = -\sum_i \Psi_i Q_i\). Atomiverse
implements this as the bookkeeping Component
`cpu::electrodes::ChargingWork` — a `Potential` that reads fixed per-type
applied potentials \(\Psi\) [V] and current per-atom charges \(Q\) [e] from the
Frame, and accumulates energy [eV] into `kEnergy` and the dedicated channel
`kEnergyChargingWork`. It does **not** solve for electrode charges (that is
the separate const-pot / BO Component stack) and writes **no** nuclear forces.
The product gate `reg_charging-work-energy` pins the closed-form MetalWalls /
Scalfi identity on a fixed parallel-plate assignment to \(-0.4\) eV within
\(1\times 10^{-12}\) eV. Oracle is **literature** (formula identity); fair wall
vs a MetalWalls/LAMMPS twin binary is **`perf_exempt`**.

## Method

**MetalWalls total energy under constant potential**
(Marin-Laflèche et al., *JOSS* **5**, 2373 (2020), eq. 6):

\[
V(Q) = V_c(Q) - \sum_{i=1}^{N} \Psi_i Q_i .
\]

Here \(V_c(Q)\) is the Coulomb energy of the full charge distribution and the
second term is the work to charge the electrodes at applied potentials
\(\Psi_i\). Scalfi et al. (*PCCP* **22**, 10480 (2020); arXiv:1911.09351 eq. 12)
identify \(q^{\mathrm{T}}\Psi = Q_L\Delta\psi\) as that work for a symmetric
two-electrode cell and place \(-\,q^{\mathrm{T}}\Psi\) in the
constant-potential Boltzmann weight.

This slug pins **only** the second term. Coulomb \(V_c\) and the Born–Oppenheimer
charge solve are other E1 regressions (`const-pot-bo-parallel-plate`,
`const-pot-md-nvt-short`, …).

**Units (ATV metal):** \(Q\) [e], \(\Psi\) [V], energy [eV] with
\(1\,e\cdot 1\,\mathrm{V} = 1\,\mathrm{eV}\).

**Closed-form pin geometry** (symmetric parallel plate, **fixed** charges as a
proxy for post-const-pot \(Q\)):

| Region | type | \(N\) | \(Q_i\) [e] | \(\Psi\) [V] |
| ------ | ---: | ----: | ----------: | -----------: |
| Left wall | 1 | 4 | \(+0.10\) | \(+0.50\) |
| Right wall | 2 | 4 | \(-0.10\) | \(-0.50\) |

Positions: left \(z = 0\), right \(z = 10\) Å; cubic box \(L = 20\) Å (unused by
the bookkeeping term).

\[
E_{\mathrm{charge}}
  = -\bigl(4\cdot 0.50\cdot 0.10 + 4\cdot(-0.50)\cdot(-0.10)\bigr)
  = -0.4~\mathrm{eV}.
\]

Scalfi check: \(Q_L = 0.4\,\mathrm{e}\), \(\Delta\psi = 1.0\,\mathrm{V}\),
\(q^{\mathrm{T}}\Psi = Q_L\Delta\psi = 0.4\),
\(E = -q^{\mathrm{T}}\Psi = -0.4\) eV.

**What this pin includes**

| Item | Setting |
| ---- | ------- |
| Operator | \(E = -\sum_i \Psi_i Q_i\) over electrode atoms only |
| Atom selection | by integer `kAtom_type` → fixed \(\Psi\) map |
| Charge source | Frame `kAtom_charge` (caller-owned; no solve) |
| Energy channels | `kEnergy` (add) + `kEnergyChargingWork` (add) |
| Forces | none |
| Domain | owned atoms (`nlocal`); global total via existing energy `Reduce` when multi-rank |

**Key exclusions (non-goals of this slug)**

- Const-pot / matrix charge solve for \(Q\) (`const-pot-bo-parallel-plate`,
  `const-pot-md-nvt-short`).
- Coulomb \(V_c\), Gaussian electrode smear, 2D Ewald, external \(\mathbf{E}\)
  (other wave-1 / E2–E3 slugs).
- Live MetalWalls or LAMMPS inside ctest.
- Public `*Config` bags or `make_*` factories.
- Fair wall twin (no MW/LAMMPS named binary for this energy term on the pin host).

## Implementation

Public surface only: constructor arguments (type list + \(\Psi\) list), then
`set_evaluator` on an Evaluate (or MD) preset. No `*Config` bag.

```cpp
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/electrodes/charging_work.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"

using namespace ATV;

// Parallel-plate pin (REFERENCE.md): 4 left + 4 right atoms.
cpu::Frame frame(8);
for (int i = 0; i < 4; ++i) {
    frame.span(kAtom_x)[i] = static_cast<double>(i);
    frame.span(kAtom_y)[i] = 0.0;
    frame.span(kAtom_z)[i] = 0.0;
    frame.span(kAtom_type)[i] = 1;
    frame.span(kAtom_charge)[i] = 0.10;   // e
    frame.span(kAtom_mass)[i] = 12.011;
    frame.span(kAtom_tag)[i] = i + 1;
}
for (int i = 4; i < 8; ++i) {
    frame.span(kAtom_x)[i] = static_cast<double>(i - 4);
    frame.span(kAtom_y)[i] = 0.0;
    frame.span(kAtom_z)[i] = 10.0;
    frame.span(kAtom_type)[i] = 2;
    frame.span(kAtom_charge)[i] = -0.10;  // e
    frame.span(kAtom_mass)[i] = 12.011;
    frame.span(kAtom_tag)[i] = i + 1;
}
frame.box = cpu::Box::cubic(20.0);

cpu::Evaluate eval;
eval.set_evaluator(
    cpu::electrodes::ChargingWork(/*types=*/{1, 2},
                                 /*psi_V=*/{0.5, -0.5}));  // V
eval.build();
eval.run(frame);

const double e_tot = frame.meta.get(kEnergy);                 // −0.4 eV
const double e_cw  = frame.meta.get(kEnergyChargingWork);     // −0.4 eV
```

On classical MD, bind the same Component after the const-pot charge update so
\(Q\) on the Frame is current:

```cpp
cpu::md::MD sim;
// … const-pot Component, Coulomb/PME, neighbor, integrator …
sim.set_evaluator(
    cpu::electrodes::ChargingWork({left_type, right_type}, {psi_L, psi_R}));
```

MD-only configure + build + run the product gate (CPU login; modules as in
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
  cmake --build build --target reg_charging-work-energy -j$(nproc)
  ./build/regressions/reg_charging-work-energy
'
```

| Path | Role |
| ---- | ---- |
| `include/atomiverse/cpu/electrodes/charging_work.h` | Public `ChargingWork` |
| `src/cpu/electrodes/charging_work.cpp` | Launch: \(-\sum\Psi_i Q_i\), meta write |
| `include/atomiverse/core/frame_schema.h` | `kEnergy`, `kEnergyChargingWork` |
| `tests/cpu/electrodes/test_charging_work.cpp` | Unit pins (parallel plate, Scalfi form, ZeroEnergy clear) |
| `regressions/md/charging-work-energy/reg_charging-work-energy.cpp` | Product gate |
| `regressions/md/charging-work-energy/REFERENCE.md` | Offline pin + oracle statement |
| `tools/metalwalls_oracle/pin_notes_charging-work-energy.md` | Pin provenance |

## Validation (vs literature)

Source:
[`regressions/md/charging-work-energy/REFERENCE.md`](../../../regressions/md/charging-work-energy/REFERENCE.md).
**Oracle: literature** — MetalWalls JOSS **5**(53) 2373 (2020) eq. 6 second term
and Scalfi et al. *PCCP* **22** 10480 (2020) / arXiv:1911.09351 eq. 12.
No live MetalWalls or LAMMPS binary is invoked in ctest; the pin is the
closed-form evaluation of the published identity on a fixed charge assignment.
Pin notes: `tools/metalwalls_oracle/pin_notes_charging-work-energy.md`.

**Setup:**

| Item | Value |
| ---- | ----- |
| System | Symmetric parallel-plate toy (8 electrode atoms) |
| Left | type 1, \(N=4\), \(Q_i=+0.10\) e, \(\Psi=+0.50\) V, \(z=0\) |
| Right | type 2, \(N=4\), \(Q_i=-0.10\) e, \(\Psi=-0.50\) V, \(z=10\) Å |
| Cell | cubic \(L=20\) Å (unused by bookkeeping) |
| Public path | `cpu::Evaluate` + `ChargingWork` only |
| Tolerance | absolute \(1\times 10^{-12}\) eV (exact rational arithmetic) |
| oracle | `literature` |
| oracle_version | MetalWalls JOSS 5(53) 2373 (2020) eq.6; Scalfi et al. PCCP 22 10480 (2020) / arXiv:1911.09351 eq.12 |

**Accuracy gate (product):**

| Quantity | Atomiverse / pin | Literature identity | Tol | Notes |
| -------- | ---------------: | ------------------: | --- | ----- |
| \(E_{\mathrm{charge}}\) (`kEnergy`) | **−0.4 eV** | **−0.4 eV** | **\(1\times 10^{-12}\) eV** | Product gate |
| \(E_{\mathrm{charge}}\) (`kEnergyChargingWork`) | **−0.4 eV** | **−0.4 eV** | **\(1\times 10^{-12}\) eV** | Dedicated channel |
| Scalfi \(E = -Q_L\Delta\psi\) | −0.4 eV | \(Q_L=0.4\), \(\Delta\psi=1.0\) | \(1\times 10^{-12}\) eV | Unit-test cross-check |

Asserted by `reg_charging-work-energy`:

```text
| frame.meta.get(kEnergy) − (−0.4) | ≤ 1e-12 eV
| frame.meta.get(kEnergyChargingWork) − (−0.4) | ≤ 1e-12 eV
```

## Performance

Source: [`REFERENCE.md`](../../../regressions/md/charging-work-energy/REFERENCE.md)
(2026-07-23); queue / handoff: **perf_exempt**.

| Item | Value |
| ---- | ----- |
| `perf_exempt` | **true** |
| Reason | No MetalWalls/LAMMPS twin binary for this named energy term in the agent environment; the Component is pure \(O(N_{\mathrm{electrode}})\) arithmetic with no force evaluation. Fair wall vs an oracle binary is not applicable until a MetalWalls install is wired under `tools/metalwalls_oracle/`. |
| \(r = T_{\mathrm{ATV}}/T_{\mathrm{oracle}}\) | **n/a (exempt)** |
| `within_10x` | exempt |

| Twin | \(T_{\mathrm{oracle}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | ------------------------: | ----------------------: | ----: | ---- | ------ |
| Parallel-plate \(-\sum\Psi Q\) | — | micro (bookkeeping) | **exempt** | n/a | No MW/LAMMPS named twin; literature pin only |

Do **not** invent a wall ratio. Campaign records the same exempt status in
the case `REFERENCE.md`.

## Limitations

- **Bookkeeping only.** Does not assign or update electrode charges; const-pot
  BO / MD loops remain separate E1 product paths.
- **No Coulomb \(V_c\).** MetalWalls total \(V(Q) = V_c - \sum\Psi Q\) is not
  assembled here; pair/PME/Gaussian-electrode terms live elsewhere.
- **Type-keyed \(\Psi\).** Applied potentials are fixed per atom type at
  construction; dynamic re-tagging of electrode membership mid-run is not a
  product surface of this Component.
- **No forces.** Charging work does not contribute nuclear forces; electrode
  freeze / piston / stress are other map items.
- **`perf_exempt`.** No fair \(r\) until an offline MetalWalls (or other)
  twin is timed for this named term; do not invent a speed ratio.
- **Literature oracle**, not a LAMMPS ELECTRODE binary pin — by design for this
  formula identity (LAMMPS priority applies only when the algorithm also exists
  in LAMMPS as a comparable twin).

## References

- Offline pin: [`regressions/md/charging-work-energy/REFERENCE.md`](../../../regressions/md/charging-work-energy/REFERENCE.md)
- Pin notes: `tools/metalwalls_oracle/pin_notes_charging-work-energy.md`
- Product gate: `regressions/md/charging-work-energy/reg_charging-work-energy.cpp`
- Unit tests: `tests/cpu/electrodes/test_charging_work.cpp`
- Public header: `include/atomiverse/cpu/electrodes/charging_work.h`
- Capability map: `.claude/notes/metalwalls-capability-map.md` (E1)
- Oracle tooling: `tools/metalwalls_oracle/README.md`
- A. Marin-Laflèche et al., *J. Open Source Softw.* **5**, 2373 (2020),
  DOI [10.21105/joss.02373](https://doi.org/10.21105/joss.02373) — eq. 6
- L. Scalfi et al., *Phys. Chem. Chem. Phys.* **22**, 10480 (2020);
  arXiv:1911.09351 — eq. 12
