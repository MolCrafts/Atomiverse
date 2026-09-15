# Post-solve ghost-charge refresh — const-pot MPI halo correctness

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `const-pot-charge-halo-mpi` |
| Regression | `regressions/md/const-pot-charge-halo-mpi/` |
| Public API | `Component::updates_owned_charges()` protocol + `ChargePacker` through `ctx.md.comm->forward_comm` after each const-pot solve |
| Roadmap | map §1 "Charge halo correctness after Q update" (**P0**) |
| Oracle | **literature** (self-consistency invariant; no oracle binary — LAMMPS ELECTRODE / MetalWalls forward-comm contract, citations below) |

## Summary

**Ghost-charge correctness after the constant-potential charge solve under MPI
domain decomposition.** Constant-potential MD rewrites electrode charges
**mid-step**: the Q solve runs at `md::Phase::pre_force` — *after* the step's
forward halo has already populated ghost atoms. Without a second, post-solve
charge communication, the force stages at rank boundaries read ghost charges
that are **one solve stale** — and identically **zero on the first step**,
since electrode charges start at 0. Forces are then silently wrong exactly at
the cuts.

This capability makes forgetting the refresh impossible: any Component that
declares `updates_owned_charges()` (all of `ConstPotBO`, `ConstPotential`,
`ConstPotCG`, `ConstPotMaZe`, `DipoleElectrodeJoint`) posts `ChargePacker`
through `ctx.md.comm->forward_comm` after writing owned charges, before any
force stage. This mirrors the documented LAMMPS ELECTRODE contract: `fix
electrode` forward-communicates the per-atom charge after every conp solve so
ghosts match owners before force evaluation (Ahrens-Iwers et al., *J. Chem.
Phys.* **157**, 084801 (2022)). The regression gates the **default** MPI
pipeline — nothing registers a Plan-level refresh component.

## Method

**The invariant.** Under domain decomposition each rank owns a slab of atoms
and holds read-only **ghost** copies of its neighbours' boundary atoms.
Per-atom state consumed by the force stages must be owner-fresh on every
ghost row when forces read it. The MD step already forward-communicates
positions at the step's halo exchange; a mid-step charge rewrite (the
constant-potential solve) invalidates ghost charges *after* that exchange.
The contract is therefore

\[
q_{\mathrm{ghost}}(\mathrm{tag}) \equiv q_{\mathrm{owner}}(\mathrm{tag})
\quad \text{at every force read, every step.}
\]

Because the halo transport is a memcpy of the owner value, the pin is
**bitwise** (tolerance 1e−15 e), not approximate.

**Why oracle = literature.** No external binary is run: the gate is a
self-consistency invariant whose *correctness contract* is documented
behaviour of the reference codes —

- A. P. Thompson et al., *Comput. Phys. Commun.* **271**, 108171 (2022),
  DOI [10.1016/j.cpc.2021.108171](https://doi.org/10.1016/j.cpc.2021.108171)
  — LAMMPS forward/reverse ghost communication semantics (owner → ghost
  refresh of per-atom state).
- L. J. V. Ahrens-Iwers, M. Janssen, S. R. Tee, R. H. Meißner, *ELECTRODE:
  An electrochemistry package for atomistic simulations*, *J. Chem. Phys.*
  **157**, 084801 (2022), DOI
  [10.1063/5.0099239](https://doi.org/10.1063/5.0099239) — per-step conp
  charge solve; solved charges are forward-communicated
  (`comm->forward_comm`) before force evaluation.
- A. Marin-Laflèche et al., *MetalWalls*, *J. Open Source Softw.* **5**, 2373
  (2020), DOI [10.21105/joss.02373](https://doi.org/10.21105/joss.02373) —
  the constant-potential MD loop this capability maps.

### System (child process, np = 2)

38-atom parallel-plate capacitor, box 20 × 10 × 16 Å (orthorhombic), forced
**2×1×1 decomposition along x** (`ATV_FORCE_DIMS=2,1,1`) so each rank owns
part of **both** plates and the halo carries electrode ghosts across the cut:

| Species | Type | Count | z [Å] | q₀ [e] |
| ------- | ---- | ----- | ----- | ------ |
| Bottom plate (Ψ = 0 V) | 1 | 15 (5×3 grid) | 5 | 0 (solved) |
| Top plate (Ψ = 1 V) | 2 | 15 (5×3 grid) | 11 | 0 (solved) |
| Electrolyte ions | 3 | 8 | 7–9.5 | ±0.25 (neutral) |

Pipeline (all defaults — the auto-injected refresh **is** the feature under
test): `VelocityVerlet{0.1 fs}` + `NeighborList{3.0, 0.3}` +
`PairCoulCut{3.0}` + `ConstPotBO{η = 2 Å⁻¹}` (electroneutrality on, matrix
rebuilt per step, `xy_image_range = 0` primary-cell solve). 5 MD steps. An
out-of-library `GhostChargeChecker` Component — the user-extension path,
defined in the regression file — runs at the force phase and compares every
ghost row's charge against its owner's current value (tag-matched via
allgather; small system).

### Key exclusions (non-goals of this slug)

- **DD-aware global electrode solve.** The per-rank `ConstPotBO` solve under
  DD is a rank-local partial solve; making the solve itself DD-global is a
  separate queued capability. This slug pins the **transport** invariant
  only: whatever charge the owner solved, every ghost copy carries it before
  forces read it.
- Live LAMMPS / MetalWalls inside ctest (no oracle binary at all here).
- Atom migration across ranks (dt = 0.1 fs, 5 steps — no migration).
- New Driver categories, public `*Config` bags, or manual user wiring of the
  refresh (auto-injection is the deliverable).

## Implementation

Nothing new is registered by the user: build the standard constant-potential
MD pipeline and the charge refresh auto-injects under `ATV_BUILD_MPI` because
`ConstPotBO::updates_owned_charges()` returns `true`.

```cpp
#include "atomiverse/cpu/dynamics/const_pot_bo.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/velocity_verlet.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/pair_coul_cut.h"

using namespace ATV;

cpu::md::MD sim;
sim.set_integrator(cpu::md::VelocityVerlet{/*dt [fs]=*/0.1});
sim.set_neighbor(cpu::md::NeighborList<>{/*cutoff [Å]=*/3.0, /*skin=*/0.3});
sim.set_evaluator(cpu::PairCoulCut{3.0});

cpu::md::ConstPotBO conp(/*eta [1/Å]=*/2.0);
conp.add_electrode(/*type=*/1, /*Psi [V]=*/0.0);
conp.add_electrode(/*type=*/2, /*Psi [V]=*/1.0);
conp.electroneutrality = true;
conp.rebuild_matrix_every_step = true;
sim.add_ref(conp);                 // default binding: pre_force @ -50

sim.build();
// ConstPotBO posts ChargePacker through ctx.md.comm->forward_comm after
// writing owned charges (updates_owned_charges()).
sim.run(frame, /*nsteps=*/5);      // Frame in / Frame out; solved q on kAtom_charge
```

Extending the protocol: any new charge-writing Component keeps the invariant
by overriding one virtual and posting `ChargePacker` after the write —

```cpp
class MyChargeSolver : public cpu::Component {
    bool updates_owned_charges() const override { return true; }
    // ... solve at pre_force; then ChargePacker::forward(ctx).
};
```

| Path | Role |
| ---- | ---- |
| `include/atomiverse/cpu/component/component.h` | `updates_owned_charges()` protocol (default `false`) |
| `include/atomiverse/cpu/dynamics/charge_packer.h` | `ChargePacker` extra-field forward through `ctx.md.comm` |
| `include/atomiverse/cpu/communication/comm.h` | `Comm::forward_comm(Packer&)` |
| `regressions/md/const-pot-charge-halo-mpi/reg_const-pot-charge-halo-mpi.cpp` | Product gate (self-mpiexec parent + 2-rank child) |

The regression is one binary in two modes: the parent re-launches itself
through the configured `mpiexec` (compile-time pin `ATV_MPIEXEC`) at np = 2;
the child runs the capacitor MD with the checker and rank 0 writes one result
line that the parent gates.

## Validation (literature-pinned invariant)

Source:
[`regressions/md/const-pot-charge-halo-mpi/REFERENCE.md`](../../../regressions/md/const-pot-charge-halo-mpi/REFERENCE.md).
No oracle binary is run — the pins are the transport invariant itself, whose
contract is the cited LAMMPS ELECTRODE / MetalWalls behaviour.

### Accuracy

| Quantity | ATV (gate) | Oracle pin (literature contract) | Tol | Notes |
| -------- | ---------: | -------------------------------: | --- | ----- |
| max \|q_ghost − q_owner\| over all ghosts, ranks, steps [e] | gated | **0** | **1e-15** | halo transport is a memcpy of the owner value — bitwise |
| electrode ghosts with \|q\| > 1e-12 e | gated | **> 0** | exact | non-vacuous: electrode charges start at 0; only the refreshed solve makes ghosts nonzero |
| unresolved ghost tags | gated | **0** | exact | every ghost must map to an owner |
| checker launches | gated | **≥ 5** (one per step) | exact | invariant enforced on every step |

Failure mode pinned: **without** the auto-injected refresh, the first gate
fails at the first step with max \|dq\| ≈ the solved electrode charge
magnitude (~1e-2 e in this geometry) — the stale-ghost bug this regression
makes unshippable.

## Performance

| Twin | \(T_{\mathrm{oracle}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | --------------------------: | -----------------------: | ----: | ---- | ------ |
| ghost-charge invariant, np = 2 | — | — | **perf_exempt** | n/a | No oracle binary is run (self-consistency invariant — there is no external twin to time); 38-atom 5-step child runs in seconds |

| Field | Value |
| ----- | ----- |
| `perf_exempt` | **true** |
| `perf_exempt_reason` | oracle = literature: the gate is a self-consistency transport invariant with no external binary run, so no fair wall-clock twin exists |
| `ratio_atv_over_oracle` | **n/a** |
| `within_10x` | exempt |

Do **not** invent wall-clock ratios for this slug — there is nothing external
to race.

## Limitations

- **Transport only.** The per-rank `ConstPotBO` solve under DD remains a
  rank-local partial solve; the DD-aware **global** electrode solve is a
  separate queued capability. Solved-charge *values* under DD are not pinned
  here — only that ghosts carry exactly what owners solved.
- **No migration exercised.** dt = 0.1 fs over 5 steps keeps every atom on
  its rank; the refresh composes with migration via the standard exchange
  machinery but this pin does not gate it.
- **2 ranks, forced 2×1×1.** One cut orientation; higher rank counts and 3D
  decompositions reuse the same halo endpoints but are not separately pinned.
- **Primary-cell solve** (`xy_image_range = 0`) in the child — the electrode
  physics here is deliberately minimal; accuracy of the conp solve itself is
  pinned by the sibling electrode slugs (E1 BO plate, CG, MaZe, …).

## References

- Pin + contract: [`regressions/md/const-pot-charge-halo-mpi/REFERENCE.md`](../../../regressions/md/const-pot-charge-halo-mpi/REFERENCE.md)
- Product gate: [`regressions/md/const-pot-charge-halo-mpi/reg_const-pot-charge-halo-mpi.cpp`](../../../regressions/md/const-pot-charge-halo-mpi/reg_const-pot-charge-halo-mpi.cpp)
- Protocol header: [`include/atomiverse/cpu/component/component.h`](../../../include/atomiverse/cpu/component/component.h) (`updates_owned_charges()`)
- Packer: [`include/atomiverse/cpu/dynamics/charge_packer.h`](../../../include/atomiverse/cpu/dynamics/charge_packer.h) (`ChargePacker`)
- Capability map: `.claude/notes/metalwalls-capability-map.md` §1 (charge halo correctness, P0)
- Literature: A. P. Thompson et al., *Comput. Phys. Commun.* **271**, 108171 (2022), DOI [10.1016/j.cpc.2021.108171](https://doi.org/10.1016/j.cpc.2021.108171); L. J. V. Ahrens-Iwers et al., *J. Chem. Phys.* **157**, 084801 (2022), DOI [10.1063/5.0099239](https://doi.org/10.1063/5.0099239); A. Marin-Laflèche et al., *J. Open Source Softw.* **5**, 2373 (2020), DOI [10.21105/joss.02373](https://doi.org/10.21105/joss.02373)
- Related siblings: [`const-pot-bo-parallel-plate.md`](const-pot-bo-parallel-plate.md), [`const-charge-electrode.md`](const-charge-electrode.md), [`maze-symplectic-charge-dynamics.md`](maze-symplectic-charge-dynamics.md)
