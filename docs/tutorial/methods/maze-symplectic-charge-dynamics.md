# True mass-zero constrained (MaZe) electrode charge dynamics

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `maze-symplectic-charge-dynamics` |
| Regression | `regressions/md/maze-symplectic-charge-dynamics/` |
| Public API | `cpu::md::ConstPotMaZe` members `symplectic` (default `true`), `maze_tol` [V], `max_maze_iters`; accessors `constraint_residual()`, `last_maze_iters()`, `max_dQ()`, `solver()` (Component on `cpu::md::MD`, default `pre_force`) |
| Roadmap / queue | `docs/developer/metalwalls-method-queue.md` wave 5; map §1 "MaZe mass-zero charge dynamics" ~ → ✓★ |
| Oracle | **MetalWalls** physics — literature pin (Coretti et al. 2020, DOI 10.1063/5.0007192; no MW binary on host → `perf_exempt`) |

## Summary

**Mass-zero constrained (MaZe) charge dynamics** evolves constant-potential
electrode charges as *auxiliary dynamical degrees of freedom with zero mass*,
constrained every step to the constant-potential condition — the SHAKE-like
formulation of Coretti, Scalfi, Bacon, Rotenberg, Vuilleumier, Salanne,
Ciccotti, Bonella, *J. Chem. Phys.* **152**, 194701 (2020). Wave 4 landed
`ConstPotMaZe` as an **adiabatic foothold** (CG + Kolafa ASPC warm start on
`ConstPotBO` each `pre_force`, `reg_maze-charge-dynamics`); this wave replaces
its default path with the **true symplectic mass-zero dynamics**:

- Verlet extrapolation \(\tilde Q = 2Q_n - Q_{n-1}\) provides the
  time-reversible initial state;
- SHAKE corrections **along the constraint gradients**
  (\(Q_{n+1} = \tilde Q + A\gamma\), CG on \(A^2\gamma = -\sigma(\tilde Q)\))
  drive the constant-potential constraint
  \(\sigma_i = (AQ - (\Psi - b))_i - \chi\) to `maze_tol` every step
  (\(\chi\) = uniform potential shift of the simultaneous electroneutrality
  constraint, applied along the cached \(u = A^{-1}\mathbf 1\) direction);
- the CG residual of the SHAKE system **is** the constraint violation, so the
  stopping rule bounds \(\lVert\sigma\rVert_\infty\) directly.

Solved tightly, the electrode charges are a function of the ionic positions
only: the NVE ionic dynamics is **time-reversible** and the
constant-potential grand energy \(KE + U - \sum_i\Psi_iQ_i\) shows **no
secular drift** — the paper's contrast against dissipative predictor /
loose-SCF adiabatic propagation. `symplectic = false` keeps the wave-4
CG+ASPC adiabatic twin (bitwise the old behaviour) for comparison studies.
Still a **Component** on `cpu::md::MD` — no new Driver category, no charge
solve inside `KSpace*`.

## Method

**Physics.** Coretti et al., *J. Chem. Phys.* **152**, 194701 (2020), DOI
[10.1063/5.0007192](https://doi.org/10.1063/5.0007192). For the *linear*
constant-potential constraint the SHAKE iteration reduces to a linear solve
that can be driven to (near) machine precision (their §II.C), which yields
three testable algorithmic invariants (exact statements, not fitted numbers):

1. **Constraint at solver precision every step:**
   \(\lVert AQ-(\Psi-b)-\chi\mathbf 1\rVert_\infty \le\) `maze_tol`.
2. **Time reversibility** (their §II.B, §III): an NVE velocity-flip
   palindrome retraces the ionic trajectory — unlike dissipative
   predictor / loose-SCF adiabatic propagation.
3. **No secular drift** of the constant-potential grand energy
   \(E = KE + U - \sum_i \Psi_i Q_i\) (MetalWalls JOSS **5**, 2373 (2020)
   eq. 6), while a loose-tolerance adiabatic twin drifts (their Fig. 2
   discussion).

The exactness of the converged charges themselves (MaZe = the dense BO
solution of the same KKT system) is pinned in the **unit** tests against the
dense Cholesky `ConstPotBO` path, which earlier waves pinned against the
MetalWalls / LAMMPS-ELECTRODE plate-capacitor algebra
(`reg_const-pot-bo-parallel-plate`).

**Why MetalWalls (not LAMMPS):** mass-zero constrained charge dynamics is
**MW-only** physics — LAMMPS ELECTRODE ships conp/conq matrix and CG solvers
but no SHAKE-like mass-zero charge dynamics. Oracle priority for this loop:
LAMMPS when the feature overlaps; **MetalWalls for MW-only physics** (this
slug). No MetalWalls binary is installed on the agent host → literature pin
with full citation, `perf_exempt`.

### System (parallel-plate NVE palindrome)

Orthorhombic box \(6 \times 6 \times 20\) Å, min-image electrostatics
(`PairGaussianCoul`, cutoff 12 Å > max min-image distance 10.9 Å — every
pair interacts exactly once, identical kernels to the elastance build):

| Region | type | N | z [Å] | η [1/Å] | Ψ [V] | mass [amu] |
| ------ | ---: | ---: | ---: | ---: | ---: | ---: |
| Bottom plate | 1 | 4 | 4 | 2.0 | 0.0 | static |
| Top plate | 2 | 4 | 16 | 2.0 | +1.0 | static |
| Ion + | 3 | 1 | 8.5 | point | — | 40.0 |
| Ion − | 3 | 1 | 11.5 | point | — | 40.0 |

Ions ±0.5 e at (2.1, 2.1, 8.5) and (4.2, 4.2, 11.5) Å (lateral min-image
separation 2.97 Å, offsets chosen away from the L/2 min-image force kink),
zero initial velocity. Explicit NVE velocity Verlet, dt = 0.25 fs, 400 steps
forward, velocity flip, 400 steps back (palindrome), grand-energy drift
sampled after a 5-step settle. The dissipative contrast is the identical
trajectory run with `symplectic = false` and a production-loose
`cg_tol = 1e-3` CG+ASPC twin.

### Key exclusions (non-goals of this slug)

- Live MetalWalls or LAMMPS inside ctest (literature pin only).
- Full electrode Ewald / MW reciprocal matrix (v1 min-image elastance).
- Electrolyte beyond the two probe ions; no thermostat (NVE by design).
- MaZe as a separate Driver or solver class — only members on
  `ConstPotMaZe` (thin owner of `ConstPotBO`).
- Charge solve inside PME / `KSpace*`.
- Public `*Config` bags or `make_*` / `build_*` factories.
- Fair wall-clock twin vs the MetalWalls binary (none on agent host).

## Implementation

Public surface only (`cpu::Frame` + `ConstPotMaZe` ctor / members; no Config
bag). MD-driver composition:

```cpp
cpu::md::ConstPotMaZe maze(/*eta [1/Å]=*/2.0);
maze.add_electrode(/*type=*/1, /*Psi [V]=*/0.0);
maze.add_electrode(/*type=*/2, /*Psi [V]=*/1.0);
maze.maze_tol = 1e-10;          // ‖σ‖∞ constraint tolerance [V]
// maze.symplectic = false;     // wave-4 CG+ASPC adiabatic twin, bitwise

cpu::md::MD sim;
sim.set_integrator(cpu::md::VelocityVerlet{0.25});
sim.set_evaluator(cpu::PairGaussianCoul{6.0, {0.0, 2.0, 2.0, 0.0}});
sim.set_neighbor(cpu::md::NeighborList<>{6.0, 0.5}); // cutoff < L/2
sim.add_ref(maze);              // pre_force −50 default binding
sim.add(cpu::md::FixFreeze{1});
sim.add(cpu::md::FixFreeze{2});
sim.build();
sim.run(frame, 400);
maze.constraint_residual();     // ≤ maze_tol [V]
maze.last_maze_iters();         // 0 when the Verlet guess already satisfies σ
maze.max_dQ();                  // max |Q_i(t) − Q_i(t−1)| [e]
```

The regression itself uses a specialized out-of-library NVE harness (not the
product `MD` driver) so the pair energy can see **every** min-image pair (the
elastance/RHS build has no cutoff) and so the velocity-flip palindrome is
exact — a production `NeighborList` (cutoff < L/2) cannot cover that in a
small cell. User code still binds MaZe on `cpu::md::MD` as above; Frame remains
the only I/O token.

Architecture rules:

- **Driver stays `cpu::md::MD`.** MaZe is a Component; never a new Driver
  category, never a charge solve inside `KSpace*`.
- Parameters are **constructor arguments and public members** — no
  `ConstPotMaZeConfig`. The underlying BO machinery (electrode registration,
  eta, images, neutrality) is reached through `solver()` — composition over
  duplication.
- Frame is the sole I/O currency; solved charges land on `kAtom_charge`.
- `max_maze_iters` is a hard cap — the constraint CG throws if exceeded (no
  silent fallback).

Build and run the regression (MD-only configure; CPU login pattern):

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
  cmake --build build --target reg_maze-symplectic-charge-dynamics -j"$(nproc)"
  ./build/regressions/reg_maze-symplectic-charge-dynamics
'
```

| Path | Role |
| ---- | ---- |
| `include/atomiverse/cpu/dynamics/const_pot_maze.h` | Public `ConstPotMaZe` (`symplectic`, `maze_tol`, `max_maze_iters`) |
| `src/cpu/dynamics/const_pot_maze.cpp` | Verlet extrapolation + SHAKE constraint CG (\(A^2\gamma = -\sigma\)) + χ neutrality shift |
| `regressions/md/maze-symplectic-charge-dynamics/reg_maze-symplectic-charge-dynamics.cpp` | Product gate (palindrome + conservation vs loose twin) |
| `tests/cpu/dynamics/test_const_pot_maze_symplectic.cpp` | Unit: dense-BO match per site, neutrality, zero-iteration re-solve, warm-start, foothold twin unchanged, validation |
| `tools/metalwalls_oracle/pin_notes_maze-symplectic-charge-dynamics.md` | Pin provenance |

## Validation (vs MetalWalls / Coretti 2020 MaZe)

Source:
[`regressions/md/maze-symplectic-charge-dynamics/REFERENCE.md`](../../../regressions/md/maze-symplectic-charge-dynamics/REFERENCE.md)
(literature pin; **never** live MetalWalls in ctest). Oracle tooling:
[`tools/metalwalls_oracle/pin_notes_maze-symplectic-charge-dynamics.md`](../../../tools/metalwalls_oracle/pin_notes_maze-symplectic-charge-dynamics.md).

**Oracle:** MetalWalls implements this MaZe path as one of its charge solvers
(Marin-Laflèche et al., JOSS **5**, 2373 (2020)); with no MW binary on the
host, the pins are the Coretti 2020 **algorithmic invariants** plus internal
twin discrimination on the identical 10-atom trajectory — not fitted numbers.

### Accuracy

Measured at pin time (2026-07-27, Release, GCC 14.3, x86_64 epyc); the gate
thresholds carry 10×+ margins over these measurements.

| Quantity | Atomiverse (MaZe, measured) | Oracle pin (Coretti 2020 invariant) | Tol (gate) | Notes |
| -------- | --------------------------: | ----------------------------------- | ---------- | ----- |
| Constraint residual \(\lVert\sigma\rVert_\infty\), every step [V] | **9.99e−11** (max) | constraint at solver precision (§II.C) | ≤ **1e−10** | CG stop rule bounds it directly |
| Electrode electroneutrality \(\lvert\sum Q\rvert\), every step [e] | **5.6e−17** (max) | exact (KKT constraint) | ≤ **1e−9** | χ shift along \(u=A^{-1}\mathbf 1\) |
| Palindrome ion-position return error [Å] | **1.4e−14** | time-reversible (§III) | ≤ **1e−6** | 2 × 400 steps, velocity flip |
| Palindrome electrode-charge return error [e] | **9.0e−13** | time-reversible (§III) | ≤ **1e−6** | vs first-solve charges |
| MaZe vs loose CG+ASPC twin, position return | 1.4e−14 vs **1.07e−05** | reversible vs dissipative | ratio < **0.1** | ~9 orders of magnitude |
| NVE grand-energy drift (after 5-step settle) [eV] | **7.57e−08** | no secular drift (Fig. 2) | ≤ **1e−6** | \(KE+U-\sum\Psi_iQ_i\) |
| MaZe drift vs loose (cg_tol 1e−3) twin drift [eV] | 7.57e−08 vs **1.76e−06** | conservative vs drifting | MaZe < twin | 23× better conserving |
| \(Q_{\mathrm{top}}\) first solve [e] | **+0.135822** | \(>0\) at ΔΨ = +1 V | polarity, exact | twin agrees: +0.135821 |
| Min ion–ion min-image separation [Å] | **3.798** | bound smooth trajectory | > **1.8** | scenario validity |

Geometry, η, Ψ, dt, step counts, and tolerances: **System** table and
REFERENCE (match `reg_maze-symplectic-charge-dynamics`). Units: eV, Å, fs,
amu, V, e (Atomiverse public surface).

## Performance

Source: REFERENCE.md runtime note; handoff
[`docs/developer/metalwalls-method-handoff.md`](../../developer/metalwalls-method-handoff.md).

| Twin | \(T_{\mathrm{MetalWalls}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | ------------------------------: | -----------------------: | ----: | ---- | ------ |
| 2 × 800-step NVE, 10-atom cell (8×8 elastance) | — | < 1 | **perf_exempt** | n/a | No MetalWalls binary on agent host for a fair MaZe twin wall |

| Field | Value |
| ----- | ----- |
| `perf_exempt` | **true** |
| `perf_exempt_reason` | no MetalWalls binary on agent host for a fair MaZe wall-clock twin; both 800-step regression runs complete in < 1 s |
| `ratio_atv_over_oracle` | **n/a** |
| `within_10x` | exempt |

Do **not** invent wall-clock ratios. When a MetalWalls binary is available,
re-run an offline fair bench, record version + walls, and replace
`perf_exempt` in REFERENCE.

## Limitations

- **Literature pin, not a numerical MW twin.** The gates are the Coretti
  2020 invariants + internal twin discrimination; the converged charges are
  pinned against the dense `ConstPotBO` KKT solution (itself pinned against
  the plate-capacitor algebra in earlier waves), not against MW output files.
- **v1 min-image elastance** on a small textbook cell; no full electrode
  Ewald / MW reciprocal matrix.
- **Two probe ions, NVE, static walls** — no electrolyte bath, thermostat,
  or piston in this slug (compose with sibling slugs:
  `const-pot-il-edl-box`, `piston-electrode-pz`).
- **Dense \(O(N_{\mathrm{elec}}^2)\) mat-vecs** in the constraint CG
  (\(A^2\gamma\) via two applications of \(A\)); large-electrode scaling is
  the CG slug's territory (`const-pot-cg-large-electrode`).
- **No fair speed ratio** until MetalWalls is on the pin host.
- Scope of this page is the **symplectic path only**; the CG+ASPC adiabatic
  foothold (`symplectic = false`) is `reg_maze-charge-dynamics` (wave 4) and
  the ASPC predictor page.

## References

- Literature pin: [`regressions/md/maze-symplectic-charge-dynamics/REFERENCE.md`](../../../regressions/md/maze-symplectic-charge-dynamics/REFERENCE.md)
- Product gate: [`regressions/md/maze-symplectic-charge-dynamics/reg_maze-symplectic-charge-dynamics.cpp`](../../../regressions/md/maze-symplectic-charge-dynamics/reg_maze-symplectic-charge-dynamics.cpp)
- Public header: [`include/atomiverse/cpu/dynamics/const_pot_maze.h`](../../../include/atomiverse/cpu/dynamics/const_pot_maze.h)
- Pin notes: [`tools/metalwalls_oracle/pin_notes_maze-symplectic-charge-dynamics.md`](../../../tools/metalwalls_oracle/pin_notes_maze-symplectic-charge-dynamics.md)
- Queue / handoff: [`docs/developer/metalwalls-method-queue.md`](../../developer/metalwalls-method-queue.md), [`docs/developer/metalwalls-method-handoff.md`](../../developer/metalwalls-method-handoff.md), [`docs/developer/metalwalls-method-loop.md`](../../developer/metalwalls-method-loop.md)
- Capability map: `.claude/notes/metalwalls-capability-map.md` (§1 MaZe mass-zero charge dynamics)
- Literature: Coretti, Scalfi, Bacon, Rotenberg, Vuilleumier, Salanne,
  Ciccotti, Bonella, *Mass-zero constrained molecular dynamics for electrode
  charges in simulations of electrochemical systems*, *J. Chem. Phys.*
  **152**, 194701 (2020), DOI
  [10.1063/5.0007192](https://doi.org/10.1063/5.0007192);
  Marin-Laflèche et al., *MetalWalls: A classical molecular dynamics software
  dedicated to the simulation of electrochemical systems*, JOSS **5**(53),
  2373 (2020), DOI [10.21105/joss.02373](https://doi.org/10.21105/joss.02373)
- Related pages: [`const-pot-bo-parallel-plate.md`](const-pot-bo-parallel-plate.md) (E1 BO solve), [`electrode-charge-predictor.md`](electrode-charge-predictor.md) (ASPC predictor), [`const-pot-cg-large-electrode.md`](const-pot-cg-large-electrode.md) (CG scaling)
