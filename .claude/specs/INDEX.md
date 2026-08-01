# Specs index

Live specs under `.claude/specs/`. Completed work is **deleted**; history is
the git log. Method census: `docs/roadmap/`.

## Closed — CP2K harness wave 6 (shift) (2026-08-01)

Merged / landed on `dev` (HEAD `8474914d` at harness close); specs deleted;
history is the git log. Incomplete staff kept `[~]` on unmerged
`wf/gpw-ts-kpoint-stress` only.

| Queue | Specs (deleted; see git) | Shipped highlights |
| ----- | ------------------------ | ------------------ |
| cp2k-method-loop harness wave 6 (shift) | gpw-ts-kpoint-forces, gpw-metallic-forces, gpw-uks-metallic-forces, gpw-vdw-kpoint-forces, gpw-optb88-forces, gpw-hirshfeld-kpoint-stress, gpw-bands-si, gpw-dos-si | multi-k free-atom TS forces; multi-k RKS/UKS free-energy metallic forces; multi-k nonlocal vdW forces; OptB88+DRSLL ENERGY_FORCE; multi-k Hirshfeld stress; GPW bands/DOS |

Do not re-create closed slugs without superseding intent.

## Closed — CP2K harness wave 5 (shift) (2026-08-01)

Merged / landed on `dev` (HEAD `350fe453` at harness close); specs deleted;
history is the git log. Incomplete staff kept `[~]` on unmerged
`wf/gpw-ts-kpoint-stress` and `wf/gpw-ts-kpoint-forces`.

| Queue | Specs (deleted; see git) | Shipped highlights |
| ----- | ------------------------ | ------------------ |
| cp2k-method-loop harness wave 5 (shift) | gpw-uks-pbesol, gpw-hirshfeld-kpoint-forces, gpw-dftu-forces (+ gpw-xdm-kpoint-forces / periodic-scs-ump2-kpoints / dfpt-gga-spin-finite-q-smearing product without live specs on parent) | multi-k free-atom XDM forces; Γ-UKS PBEsol; multi-k SCS-UMP2; four-way finite-q×spin×GGA×smear DFPT; multi-k Hirshfeld TS/MBD forces; Γ DFT+U forces |

Do not re-create closed slugs without superseding intent.

## Closed — CP2K harness wave 4 (shift) (2026-08-01)

Merged / landed on `dev` (HEAD `6f1218b3` at harness close); specs deleted;
history is the git log. Incomplete staff kept `[~]` on unmerged
`wf/gpw-ts-kpoint-stress` only.

| Queue | Specs (deleted; see git) | Shipped highlights |
| ----- | ------------------------ | ------------------ |
| cp2k-method-loop harness wave 4 (shift) | gpw-d3-atm-kpoint-forces, gpw-hirshfeld-forces, gpw-d3-kpoint-forces, gpw-d4-kpoint-forces, gpw-d4-atm-kpoint-stress, periodic-berry-gpw-uks, periodic-scs-ump2-si, dfpt-gga-spin-smearing (+ gpw-d3bj-kpoint-forces product without live spec on parent) | multi-k D3 ATM forces; Hirshfeld-scaled TS/MBD forces; multi-k D3/D3(BJ)/D4 two-body force pins; multi-k D4 ATM stress; GPW UKS Berry; Γ SCS-UMP2; joint DFPT spin×GGA×smear |

Do not re-create closed slugs without superseding intent.

## Closed — CP2K harness wave 3 (shift) (2026-08-01)

Merged / landed on `dev` (HEAD `9bd134e0` at harness close); specs deleted;
history is the git log. Incomplete staff kept `[~]` on unmerged
`wf/gpw-ts-kpoint-stress`, `wf/gpw-d3-atm-kpoint-forces`,
`wf/gpw-hirshfeld-forces`.

| Queue | Specs (deleted; see git) | Shipped highlights |
| ----- | ------------------------ | ------------------ |
| cp2k-method-loop harness wave 3 (shift) | periodic-berry-gpw, periodic-scs-mp2-kpoints (+ gpw-d4-atm-kpoint-forces / gpw-optb88-uks-kpoints product without live specs on parent) | multi-k D4 ATM force pin; multi-k UKS optB88-vdW; multi-k SCS-MP2 dual gate; GPW KSV Berry |

Do not re-create closed slugs without superseding intent.

## Closed — CP2K harness wave 2 (shift) (2026-08-01)

Merged / landed on `dev` (HEAD `79bcfa14` at harness close); specs deleted;
history is the git log. Incomplete staff kept `[~]` on unmerged
`wf/gpw-ts-kpoint-stress` only.

| Queue | Specs (deleted; see git) | Shipped highlights |
| ----- | ------------------------ | ------------------ |
| cp2k-method-loop harness wave 2 (shift) | gpw-d4-kpoint-stress, gpw-xdm-kpoint-stress, cpmd-uks-thermostat-kpoints, periodic-wannier-gpw-uks, periodic-scs-mp2-si, gpw-optb88-uks, gpw-vdw-uks-kpoint-multigrid (+ hirshfeld / optb88-kpoints product without live specs) | multi-k D4/XDM stress; multi-k UKS Hirshfeld MBD; multi-k UKS CPMD eNH; GPW UKS MLWF; Γ SCS-MP2; Γ-UKS + multi-k RKS optB88-vdW; multi-k UKS vdW×multigrid |

Do not re-create closed slugs without superseding intent.

## Closed — CP2K harness wave 1 (shift) (2026-08-01)

Merged into `dev` (six `--no-ff` feature merges + roadmap residual refresh
HEAD `424d6989`); specs deleted; history is the git log. Incomplete staff
kept `[~]` on unmerged `wf/*` (D4/TS/XDM k-stress + multi-k UKS Hirshfeld)
— recovered in harness wave 2 except TS k-stress.

| Queue | Specs (deleted; see git) | Shipped highlights |
| ----- | ------------------------ | ------------------ |
| cp2k-method-loop harness wave 1 (shift) | gpw-vdw-uks-kpoints, gpw-d3-kpoint-stress, gpw-d3bj-kpoint-stress, gpw-d3-atm-kpoint-stress, gpw-bomd-uks-kpoints, gpw-vdw-uks-multigrid | multi-k UKS nonlocal vdW-DF; multi-k D3/D3(BJ)/ATM stress pins; multi-k UKS BOMD dual \(P_k^\sigma\); Γ-UKS vdW multigrid |

Do not re-create closed slugs without superseding intent.

## Closed — driver registration + AIMD wrappers (2026-07-29)

Shipped earlier on `dev` (`5e688676` / `17f52238` / merge `601ac445`); specs
closed 2026-07-29 after residual AGENTS/CLAUDE teaching-sample migration.
History is the git log (specs deleted).

| Spec | One-line |
| ---- | -------- |
| driver-api-01-evaluator-registration | `set_evaluator`; no public `add` / `add_potential` |
| driver-api-02-aimd-wrappers | CarParrinello / XlBomdForce / SecondGenCp; private flags |

## Closed — CP2K harness wave 13 (2026-07-31)

Merged into `dev` (merge wave 1, nine --no-ff merges `44749b1a`..`268aaf66`);
merge gate green (scoped rebuild + `ctest -R '^test_cpu_(gpw|pw|scf|xc)_|^test_xc_'`
57/57). Specs deleted; history is the git log.

| Queue | Specs (deleted; see git) | Shipped highlights |
| ----- | ------------------------ | ------------------ |
| cp2k-method-loop harness wave 13 | gpw-d3bj-stress, gpw-mbd-kpoint-stress, gpw-d3-atm-forces, gpw-d4-atm-forces, gpw-dftu-kpoint-forces, dfpt-gga-spin-si, gpw-bomd-kpoints, gpw-multigrid-kpoints, xl-bomd-linearized-uks | D3(BJ) analytic cell stress (CP2K pin); multi-k stress folds periodic MBD@rsSCS; D3 ATM (C9) force pin + periodic image-triple/pbc anchoring fixes; analytic D4 ATM (s9) forces (dftd4 3.7.0 pins); multi-k DFT+U occupation-matrix forces; collinear-spin × pure-GGA DFPT phonons; plain multi-k BOMD P_k warm start; multi-k RKS REL_CUTOFF multigrid; zero-SCF linearized XL-BOMD on Γ-UKS |

Do not re-create closed slugs without superseding intent.

## Closed — CP2K harness wave 3 (shift) (2026-07-31)

Merged into `dev` (nine `--no-ff` merges + k-UKS+vdW fail-fast fix; HEAD
`419c1dc3`); merge gate: all nine `run_reg_*` green. Specs deleted; history
is the git log.

| Queue | Specs (deleted; see git) | Shipped highlights |
| ----- | ------------------------ | ------------------ |
| cp2k-method-loop harness wave 3 (shift) | gpw-ts-stress, gpw-xdm-stress, gpw-vdw-kpoints, gpw-vdw-uks, gpw-uks-multigrid-kpoints, cpmd-thermostat-kpoints, second-gen-cpmd-ot-uks-kpoints, xl-bomd-linearized-stress, dfpt-gga-spin-finite-q-si | TS/XDM dedicated stress pins; multi-k RKS + Γ-UKS nonlocal vdW; multi-k UKS multigrid; multi-k CPMD eNH thermostat; multi-k UKS SGCP OT; linearized XL stress; finite-q spin×GGA DFPT |

Do not re-create closed slugs without superseding intent.

## Closed — CP2K harness wave 2 (shift) (2026-07-31)

Merged into `dev` (eight `--no-ff` merges `aa897bb2`..`8b2fda00`); merge gate
green (scoped rebuild + `ctest -R '^test_cpu_(gpw|pw|scf|xc)_|^test_xc_'` 57/57).
Specs deleted (seven here; `gpw-d4-atm-stress` already deleted on its branch,
`a2c85a6b`); history is the git log.

| Queue | Specs (deleted; see git) | Shipped highlights |
| ----- | ------------------------ | ------------------ |
| cp2k-method-loop harness wave 2 (shift) | gpw-d3-stress, dfpt-gga-finite-q-si, xl-bomd-linearized-kpoints, gpw-vdw-multigrid, cpmd-uks-kpoints, second-gen-cpmd-ot-kpoints, gpw-d3-atm-stress, gpw-d4-atm-stress | D3-zero analytic cell stress (CP2K NUMERICAL difference pin); finite-q DFPT with pure PBE kernel; multi-k RKS/UKS zero-SCF linearized XL-BOMD; nonlocal vdW on the REL_CUTOFF multigrid; multi-k UKS CPMD (dual-spin Bloch \(C_k^\sigma\)); multi-k SGCP + OT partial-SCF corrector; D3 ATM (C9) + D4 ATM (\(s_9\)) cell-stress pins |

Do not re-create closed slugs without superseding intent.

## Live — esmm campaign: ES × MM connection layer (2026-07-26, 16 specs)

Operator-approved three-wave campaign (plan-mode design review + Q&A 2026-07-26):
unify the force-pipeline composition contract, redesign QM/MM around an engine
slot + `qm_z` mask column, generalize CPMD into a shared `cpu/electronic/`
propagation layer. Execution order (operator, 2026-07-26): **esmm-cpmd wave
first** (`/mol:impl-all esmm-cpmd`), then contract + qmmm waves;
esmm-contract-05 rebases its gpw/scf `=`→`+=` sites over the cpmd wave.
The libcint/C++20 sweep gate was cleared by committing it (2026-07-26).

| # | Spec | Deps | One-line |
| - | ---- | ---- | -------- |
| W1-1 | esmm-contract-01-zeroing | — | DONE 2026-07-27 (closed; see git 66664744) — Zero* unification + k/2 oracle fix |
| W1-2 | esmm-contract-02-validation-walks | — | DONE 2026-07-27 (closed; see git 17f52238 + close commit) — node-aware validation walks |
| W1-3 | esmm-contract-03-runslot-nested | — | DONE 2026-07-27 (closed; see git 3480c6ab/6ea2ea23) — run_slot_ nested dispatch + R2 CCCBDB pin |
| W1-4 | esmm-contract-04-forces-requested | W1-2 | DONE 2026-07-27 (closed; see git fc204c63 + R3 commit) — energy-only phases in the Plan + ASPC-under-MC guard + first QM-in-MC regression |
| W1-5 | esmm-contract-05-accumulate-switch | W1-1 | DONE 2026-07-27 (closed) — one accumulation contract: 5 ES evaluators `+=`, kEnergyElectronic, ASPC-fallback 2x guard, first ES+MM co-bound regression |
| W2-1 | esmm-qmmm-01-embed-forces | — | DONE 2026-07-29 — analytic embed F + MM sites + EmbeddingSite |
| W2-2 | esmm-qmmm-02-qm-engine-protocol | W2-1 | DONE 2026-07-29 — QmEngine/EmbeddingEngine; MFM×7 + AM1 + DFTB2 |
| W2-3 | esmm-qmmm-03-electrostatic-rewrite | W2-2, W1-4, W1-5 | DONE 2026-07-29 — mask + engine slot + real MM coupling |
| W2-4 | esmm-qmmm-04-mech-pol-port | W2-3 | DONE 2026-07-29 — Mechanical/Polarizable on shared machinery |
| W2-5 | esmm-qmmm-05-md-closure | W2-4 | DONE 2026-07-29 — MD conservation + roadmap ✓ |
| W2-6 | esmm-qmmm-06-api-frame-compose | W2-5 | DONE 2026-07-29 — drop links() API; mask + bond topology; AIMD composition |
| W2-7 | esmm-qmmm-07-oniom-subtractive | W2-6 | DONE 2026-07-29 — two-layer ONIOM (`QmmmOniom`); reg_qmmm_oniom_h2_ar |
| W3-1 | esmm-cpmd-01-electronic-home | — | DONE 2026-07-26 (closed; see git c40acdb3) — cpu/electronic/ shared propagators + linalg orbital_tangent extraction |
| W3-2 | esmm-cpmd-02-timestep-protocol | W3-1 | DONE 2026-07-27 (closed; see git) — TimestepProvider/Consumer; cpmd_dt deleted; cpmd×stress gate |
| W3-3 | esmm-cpmd-03-observability | W3-2 | DONE 2026-07-27 (closed; see git) — kEnergyElectronKinetic/Thermostat + CpmdConserved → Thermo drift |
| W3-4 | esmm-cpmd-04-ao-cpmd | W3-1..3 | DONE 2026-07-27 (closed; see git 2de56c65) — Gaussian AO-CPMD on MeanFieldMethod |
| W3-5 | esmm-cpmd-05-scf-xlbomd | W3-4 | DONE 2026-07-27 (closed; see git) — XL-BOMD on the SCF methods (guess ladder; ASPC-vs-explicit-guess bug fixed) |
| W3-6 | esmm-cpmd-06-single-eval | W3-3 | DONE 2026-07-27 (closed; see git) — GpwDft::overlap(); CPMD single KS eval per step |

## Ticket — design review owed (operator TODO, 2026-07-27)

- **Minimize/Optimizer taxonomy is non-orthogonal** (operator): "CG 到底
  optimizer 还是 minimizer，还是某种方法？" Three near-synonyms span the
  layers — namespace `cpu::minimizer::`, Driver preset `Minimize` (the
  loop), component kind `Optimizer` (the step-proposal role), concretes
  `CG`/`LBFGS`/`FIRE` — and "method" is already taken by electronic
  structure. Live symptom: `add(CG{})` vs `set_optimizer(CG{})` confusion
  in tests during the concurrent no-compat API migration. Route:
  /mol:discuss for a vocabulary decision (rename namespace? fold kind
  naming into the driver's? document the tri-layer as-is?); overlaps the
  known "namespaces not yet aligned to directories" debt. No mid-wave
  redesign.

## Ticket — /mol:fix owed (named 2026-07-27, esmm-cpmd-04 close-out)

- ~~**BOMD+d3bj/d4 forces are non-conservative**~~ **FIXED 2026-07-28**
  (`34479521`): the defect was in THREE entry points, not one — `forces()`,
  `mp2_forces()` and `ccsd_forces()` all reported `e_total_ += e_disp_` with a
  gradient that knew nothing about dispersion. No D3 math was copied: GPW's
  `internal/dftd3.cpp` stays the single implementation (its CN chain verified
  complete) and scf delegates; only the flag->evaluator dispatch was extracted
  to `scf/internal/dispersion.h`. Analytic-vs-FD: D3(BJ) 2.2e-11 eV/A,
  D4 9.2e-10, end-to-end total 4.3e-06 (pure O(h^2)). New regression
  `regressions/scf/bomd-d3bj-ar2-nve` (Ar2, dispersion is the only binding
  force): E_disp and its gradient pinned against offline simple-dftd3 at
  4.3e-19 Ha / 5.4e-20 Ha/Bohr, NVE drift 1.6e-05 eV vs a 9.7e-03 eV E_disp
  swing (620x). CPMD's refusal kept — but its stated reason was false and is
  rewritten: under CP, dispersion would be a force with no energy (the mirror
  defect), so lifting it means folding E_disp into the CP total + E_cons.
- ~~**D4's gradient is itself a central difference**~~ **FIXED 2026-07-29**:
  two-body (`s9=0`) forces are analytic — BJ pair + CN_d4 weight chain + EEQ
  charge-response adjoint. ATM (`s9≠0`): analytic \(F_2\) + FD of
  \(E_{\mathrm{ATM}}\) only (still conservative for \(E_2+E_{\mathrm{ATM}}\);
  no full EEQ re-solve per Cartesian). Dual-gated vs dftd4 3.7.0 `-g`.
- ~~**No end-to-end gate for MP2/CCSD + d3bj forces**~~ **FIXED 2026-07-29**:
  `mp2-d3bj-forces-h2o-sto3g` — water **dimer** STO-3G, `mp2_forces` vs FD
  of `mp2().e_tot` (max err 1.6e-6 eV/Å). `ccsd-d3bj-forces-h2o-sto3g` —
  wiring gate \(\mathbf{F}(d3)-\mathbf{F}(off)=\mathbf{F}_{\mathrm{D3}}\)
  (electronic CCSD is already bare-energy FD).
- **RKS/PBE forces are XC-grid-noisy and slow** (named 2026-07-28): ~3e-3
  eV/A noise at the default grid on Ar2, and one RKS/def2-SVP force costs 28 s
  vs 0.25 s for RHF (~2.4 s per SCF iteration on 36 AOs). Worth a /mol:fix on
  the XC grid path.
- **`regressions/gpw/gpw-d3bj-forces` exceeds its 60 s wall budget under node
  load** (72-115 s; 32 s documented when quiet) while its physics stays
  byte-identical to REFERENCE. Same class as the two OT cases above — the
  budget needs a contention-aware policy, not a physics change.
- Minor (same sweep): `reg_cpmd-water` single-threaded budget — now 54 s
  after esmm-cpmd-06's single-eval (was 62 s, under budget again);
  `test_cpu_scf_uhf_hessian` + `test_cpu_gpw_dftu` each showed one
  full-suite-context transient (3/3 green isolated) — flaky-test
  investigation. ~~`reg_cpmd-uks` is nondeterministic under threading
  (pre-existing OpenMP reduction-order noise in the Γ-UKS grid/XC path,
  ~3e-5 relative; byte-compares must run OMP_NUM_THREADS=1)~~ — **likely
  fixed by `0f78ff6d`** (`xc_grid_pade_lsd` already uses `ChunkedRange`, same
  as closed-shell `xc_grid_pade`); **gated by** unit test
  `XcGridPadeLsdEnergyIsThreadIndependent` in
  `tests/cpu/gpw/test_cpu_gpw_thread_reproducibility.cpp`. **Re-measure
  `reg_cpmd-uks` owed** if it still flakes under multi-thread.
- **Regressions run thread-unpinned; small-system cases pay catastrophic
  fork/join overhead** (named 2026-07-27, esmm-contract-04 R3): the
  `regressions` target runs binaries with no `OMP_NUM_THREADS`, and only ~3
  of ~250 registered cases pin their thread count (`set_threads(1)`;
  precedent `neb_muller_brown`, now also `mc_bias_es`). Measured on
  mc_bias_es (2 atoms, 2 AOs): 7 s CPU at 1 thread, 231 s at 8, unfinished
  after 11 min at the node default 48 — results bit-identical at every
  count, pure OpenMP fork/join waste. Every other small-system case is
  exposed. Fix options: per-case `set_threads` sweep, or the regressions
  CMake target exporting `OMP_NUM_THREADS=1` by default with per-case
  opt-out. Route /mol:fix (mechanical, wide).
- ~~**13 GPW-gated dynamics regressions do not compile at pristine HEAD**~~
  **FIXED 2026-07-28 (e30ea9a3)** — was 14, not 13 (`second-gen-cpmd-kpoints`
  was missing from the ticket). Call-site migration to the post-17f52238 API:
  order 0 is the `set_evaluator_ref` default; `cpmd_mu`/`second_gen_cp_k`/
  `_max_iter` are ctor args; `use_ot` stays a public `GPWEvaluator` member set
  before the move; `last_s2` via the wrapper's `method()`. `cmake --build`
  now exits 0 (703 targets, zero errors); all 14 PASS their drift envelopes.
  No pin or tolerance touched.
- **Two OT cases exceed the 60 s regression budget** (named 2026-07-28, same
  sweep): `second-gen-cpmd-ot` 86 s and `second-gen-cpmd-uks-ot` 95 s at the
  node's default thread count — both PASS; pre-existing case cost that was
  invisible while they did not compile. NOTE the fix is **not** `set_threads(1)`
  (opposite shape to the small-system thread trap below): single-threaded they
  take 110 s and >180 s. Real options: shrink the case (physics/spec change) or
  raise the budget for GPW/OT cases. Route /mol:discuss + /mol:fix.

## Ticket — /mol:fix owed (named 2026-07-28, repo-cleanup wave)

- ~~**DFTB3's SCC potential is not the derivative of its energy**~~
  **FIXED 2026-07-28** (`602c7d02` on the branch, merged as `68df18a8`;
  chain closed by `c0bfffef`..`148a17ed`). Euler identity now 4.4e-16, FD
  5.0e-10. The branch is merged and pruned. Original diagnosis kept below
  for the record.
  Measured on the merge candidate: `V_code` vs numerical `∂(E²+E³)/∂Δq` off by
  **0.457 eV** at Δq=(−0.6,0.3,0.3), while with `U_d=0` (pure second order) the
  residual is 4.5e-10 — the defect is isolated to the third-order block. Euler
  check at the converged Δq: `Σ Δq·V³(code) = 15.496` vs exact `3·E³ = 23.240`
  (a correct derivative satisfies it to 7e-15). The double-counting subtraction
  is also short a term: code subtracts `E²+E³`, a consistent V³ needs `E²+2E³`.
  Symptom: `|E_DFTB3 − E_DFTB2| = 46.96 eV` against a reported `E³ = 6.46 eV`.
  Its regression pin is ALSO stale (−84.5617 vs −89.0775 eV), proven to be
  100% the H-parameter upgrade — but re-pinning was refused: it would bless a
  non-variational solution as dev's reference DFTB3 number. Fix the physics
  first, then re-pin and merge the branch.
- ~~**DFTB2's charge-dependent gradient is wrong for polar systems**~~
  **FIXED 2026-07-28** — root cause was three-fold and none of it was the
  gradient formula alone: (a) the SCC never converged (undamped Picard
  limit cycle; energies were selected by `max_scc` parity), (b) the
  double-counting route used net instead of gross Mulliken population
  (a spurious term linear in dq, ~7.85 eV), (c) the analytic gamma gradient
  carried the wrong SIGN in all three assemblies. The earlier A/B that
  "ruled out the sign flip" was measuring a non-stationary solution, so it
  proved nothing. Now: Anderson/Pulay mixing (`mix_beta` public member,
  budget exhaustion throws), `E = Tr(rho H0) + E2 (+E3) (+E_spin) + E_rep`,
  and analytic-vs-FD forces at 2.5e-07 / 4.9e-09 eV/A (was 0.228 / 0.748).
  Gates tightened 1.0 -> 1e-5 eV/A and 5e-3 -> 1e-8 eV/A^3; both DFTB+
  invariant pins unmoved. Original text kept below.
  (pre-existing, now present in three functions). Every Δq=0 case is exact
  (O2 3.0e-07, H2 5e-03 eV/Å); every polar case is off (H2O mio 0.23/0.75,
  H2O 3ob **0.84/0.24** eV/Å — 8× relative on `F_H1_y`). A converged SCC-DFTB2
  gradient is stationary and must be FD-exact, so the "charge-response
  residual" framing is masking a bug; A/B rules out the cheap fixes (term
  dropped 1.020, sign flipped 2.268 eV/Å). **These gates pass only because
  their tolerance is 1.0 eV/Å** — fix, then tighten. Also: the spin gate
  cannot see it (O2 has Δq ≡ 0 by symmetry); an open-shell Δq ≠ 0 case is owed.
- ~~**D3 dispersion gaps**~~ **FIXED 2026-07-29**: forces were already
  implemented (ungated only by tests); unit gates
  `ForcesMatchCentralFDOfEDisp` + `StressIncludesD3` now on. Stress folds
  D3 pair virial (molecular) / strain-FD of E_disp (`d3_periodic=true`).
  Public `d3_periodic` (default false) wires orthorhombic lattice-sum D3.
- ~~**The mio-1-1 H–H reader exists twice**~~ **FIXED 2026-07-28**
  (`c0bfffef`): dftb2's private copy deleted, both dftb2 and dftb3 now read
  `internal/mio_hh_table.h`; `d_repulsive_dr` / `d_hopping_ss_dr` /
  `d_overlap_ss_dr` added. Proven a pure refactor — every DFTB2 number
  byte-identical across the rewire.
- **No open-shell Delta-q != 0 force gate** (named 2026-07-28): O2 is
  homonuclear, so `analytic_forces_unrestricted` carries the corrected gamma
  sign with nothing testing it — which is exactly how the sign bug survived
  in three functions. An open-shell polar case is owed.
- **3ob `with_stress` still falls back to multi-SCC strain-FD** although the
  analytic 3ob SK derivatives now exist.
- **MioSketch water charges are physically backwards** (Delta-q = +0.122 on O,
  -0.061 on each H) — a property of the invented exponential O/H sketch rows,
  not of the SCC. Real O/H mio-1-1 SK data is owed, like H-H already has.
- ~~**Seven PW/bands/DOS regressions resolve `tests/data/` relative to CWD**~~
  **FIXED 2026-07-29** (`e30c933d`) — twelve, not seven. Route: a compiled-in
  `ATV_REG_DATA_DIR` from `atv_add_regression()`, deliberately NOT
  `WORKING_DIRECTORY` (that would fix only the aggregate target and leave every
  standalone `./reg_x` broken, which is the invocation a human debugs with).
  The `ATV_SI_UPF` / `ATV_AL_UPF` / `ATV_SOURCE_DIR` env escapes were deleted
  too — an env var is a fallback by another name. Guarded by
  `tests/architecture/regression_harness_scrub.sh`.
- ~~**Regressions run thread-unpinned**~~ **FIXED 2026-07-29** (`e30c933d`):
  thread count is now DECLARED at registration,
  `atv_add_regression(<domain> <case> [THREADS <n>])`, default 1, applied via
  the environment so it also covers code that never touches a Driver (PW::run,
  GPW internals, BLAS). Measured, not guessed: 100 cases declare 8 (gpw 8x,
  scf CFMM 8x, cpmd 2x), 265 take 1 — including 27 PW/DFPT cases I first
  raised on structure and then dropped back after measuring 1.0x
  (`dfpt-spin-smearing` is 2x SLOWER at 8). Capped at 8 because
  `ot-minimizer-water` scales to 10.9 s at 8 but exceeds 200 s at 16.
- **`reg_pw-smearing-al` silently substituted a toy Hamiltonian**
  (found 2026-07-29, fixed in `e30c933d`): on a missing pseudopotential it
  called `pw.upf_path.clear()` and solved a toy local potential, printing
  `ok: ... upf=toy`. It failed only because its pin happened to catch the
  −107.76 Ry answer; a looser tolerance and it would have PASSED for a
  different Hamiltonian. Now a hard error. **Worth a sweep**: any other case
  that can quietly degrade its own physics is the same defect.
- **The `regressions` aggregate target has no `-k`** so it aborts at the first
  failure and never produces a tally (named 2026-07-29). With `-k`: 365 cases,
  5 failing, 61 min sequential. Consider making the target keep going by
  default so it reports instead of stopping.
- ~~**Two PW iron pins are genuinely wrong or genuinely broken**~~
  **RESOLVED 2026-07-29** (`e7b7b534`) — one of each, and the first was a real
  physics bug: in the collinear-UKS + smearing branch of `PW::run`,
  `ws.gvec`/`ws.n_g` were rebound to each k-point's WAVEFUNCTION G-sphere
  inside the k loop and never restored, so spin ↓ copied only the first
  `n_g(k_last)` coefficients of its `V_xc` and FFT-ed `V_eff` on the last
  k-point's (h,k,l) — **the two spin channels were solved with different
  Hamiltonians** (8.7 eV on Fe). The identical defect had been fixed in the
  sibling helper `internal::run_scf_smear_collinear` (`ea3ae031`) and never
  back-ported. Proved by an exact invariant rather than a number: at
  `multiplicity = 1` both channels are seeded identically and every operator is
  spin-symmetric, so `M = N↑ − N↓ ≡ 0` with no tolerance — measured 1.90 (Γ)
  and −0.25 (2×2×2) before, exactly 0 after. `pw-uks-kmesh-fe` was a fossil
  pin instead: it predates `c6a6273c`, which made `step_spin` a true spin-GGA
  (before that, `xc = PBE` silently ran the σ=0 LDA limit — another silent
  Hamiltonian substitution). BOTH absolute-energy freezes were **deleted, not
  re-pinned** (neither setting has a citable external value) and replaced with
  exact statements: Herzberg `M = 2`, `n_kpts = k_mesh³`, `N_occ = Z_val`,
  `S ≥ 0`, `∂F/∂T = −S`, `TS → 0`. The `dF/dT` gate discriminates: the broken
  solver gives −1.2e-3 Ry against a required < −1e-2.
- **Most PW absolute-energy pins in this tree are self-freezes** (named
  2026-07-29), including the currently-PASSING `pw-spin-polarized-o2` and
  `pw-spin-gga`. They validate nothing and go stale on every legitimate physics
  change — two of them went stale within five days. Either replace them with
  exact invariants (as the two above now are) or pin them against a citable
  external code.
- **The collinear PW convergence test measures the POST-MIX density change**
  (i.e. `β × residual`), so at `β = 0.3` it declares convergence at a residual
  3.3× looser than `tol`. Shared by both collinear branches; tightening it
  moves several other pins. Also: the collinear branches use plain linear
  mixing while only the restricted path uses Pulay — for Fe that costs ~5× the
  iterations (12 → 60).
- ~~**Four GPW unit tests are registered inside the PW guard**~~ **FIXED
  2026-07-29** — verified by running the documented command itself: exit 0,
  370/370. Original text: (named 2026-07-29, pre-existing — verified identical at the session's
  start commit): `test_cpu_gpw_forces`, `_stress`, `_uks`, `_dftu`. A build
  with `ATV_BUILD_PW=ON` and `ATV_BUILD_GPW=OFF` therefore fails to link with
  4 undefined-reference errors — and that combination is **exactly the
  copy-pasteable CPU build command in AGENTS.md and CLAUDE.md**. Fix: move the
  four registrations inside `if(ATV_BUILD_GPW)`.
- ~~**`gpw-fd-phonon-si` dual-gate fail**~~ **FIXED 2026-07-29**: root cause
  was a poorly conditioned 1-D energy curvature at \(h=10^{-3}\) Å (not broken
  analytic forces). `h_bond` → \(5\times10^{-3}\) Å; dual-gate tol → 100 cm⁻¹
  (honest residual at 8 Å/150 Ry non-eq bond). Measured after fix:
  |ω_ph − ω_1d| = **2.7 cm⁻¹** (was 71.5). Wall budget 90→110 s (host load).
- ~~**The OT and GW iterative solvers are not thread-deterministic**~~
  **FIXED 2026-07-29** (`0f78ff6d`): NINE reductions in the GPW grid kernels
  handed their summation order to the OpenMP runtime, and two of them
  (`build_compact_basis_grid`, `build_products`) were STRUCTURAL — the CSR
  layout and the product list order themselves differed run to run, which then
  fixed the summation order of everything downstream. Replaced with
  `ChunkedRange` (`cpu/device/chunked_range.h`): chunk boundaries depend on `n`
  alone, each chunk owns an accumulator, the caller folds in ascending chunk
  order; `schedule(dynamic)` stays because which thread runs a chunk no longer
  touches the arithmetic. Bitwise identical at 1/2/4/8/16 threads, gated by
  `tests/cpu/gpw/test_cpu_gpw_thread_reproducibility.cpp` (5 of 6 cases RED
  against the parent). Cost: none — 8 threads 1.40 s vs 1.47-1.70 s before.
  The three `THREADS 1` stopgaps are lifted; 6/6 consecutive passes each.
  **Γ OT aligned to multi-k 2026-07-29**: trial Löwdin on Armijo points
  (RKS + UKS Γ) and `kOtGradTol` 1e-6 → **1e-5** (match `kOtGradTolK`);
  previously gnorm plateaued ~2.7e-6 and thrash-chased a dead tolerance.
  Multi-k paths unchanged. Re-measure `ot-minimizer-water` after rebuild
  (build thrash from concurrent agents blocked a clean re-run this wave).
  Original text: `ot-minimizer-water` converges in **31 or 56 iterations
  run-to-run at 8 threads** (11 s vs 110 s, same binary, no RNG);
  `second-gen-cpmd-ot` scored 3 PASS / 1 FAIL in 4 identical runs (drift 0.245
  and 0.360 vs tol 0.10) and is bitwise reproducible at 1 thread;
  `periodic-ugw-kpoints` intermittently fails its Newton QP solve. Controls
  (`cpmd-water`, `xl-bomd-water`, `second-gen-cpmd-water`) ARE reproducible at
  8, so the OT corrector is the differentiator. Three cases pinned to
  `THREADS 1` as a stopgap; their siblings cannot afford it
  (`periodic-gw-si` 51→8 s, `ugw-uks-rpbe` 146→22 s) and stay exposed.
  IN PROGRESS 2026-07-29.
- **`gpw-d3bj-forces` is genuinely slow, not contention** — 72 s loaded, 73 s
  re-run at load 8 (correcting my earlier contention hypothesis). Genuinely
  over the 60 s budget at low load, reproduced: `gpw-uks-stress` **303 s**
  (>600 at 1 thread), `second-gen-cpmd-uks-ot` 291, `gpw-uks-revpbe` 109,
  `second-gen-cpmd-ot` 108, `gpw-uks-rpbe` 98, `gpw-uks-blyp` 96,
  `periodic-wannier-gpw` 95, `gpw-uks-forces` 67, `gpw-mbd-stress` 65,
  `gpw-fd-phonon-si` 65.
- ~~**Five genuinely failing regressions**~~ **FIXED 2026-07-28** (`fdc2f6d4`)
  — three were real library bugs, not case bugs: SETTLE was not SETTLE (it
  rebuilt the ideal triangle from the unconstrained geometry and never read
  its own reference snapshot, so it could MIRROR a molecule — a 1e-4 A
  perturbation teleported an H 1.84 A while every bond stayed exact to 12
  digits); `mc::Transaction` rollback was corrupted by a mid-trial row
  reorder (the NPT chain froze silently at trial 15 000 of 340 000 against a
  +20 keV overlap); and REMD had no thermostat at all, so Sugita-Okamoto ran
  between non-canonical ensembles and accepted 60/60. Original text:
  (rebuilt and re-run at `2b350442`, i.e. before the esmm contract wave, with
  byte-identical failure messages): `rigid_water`, `settle_tip3p` (no verdict
  line — SETTLE Frame-lookup budget), `npt_lj` ("<V> off the NPT ensemble
  prediction"), `remd_harmonic` ("accept ratio out of (0.01,0.99): 1.000"),
  `gaussian-electrode-charge` (no verdict line). Plus `python_pure_import`
  (missing `typer` in the env). Classical-MD/MC domain work, unrelated to the
  ES accumulation contract.
- ~~**Four `.claude/specs/` dirs arrived with the merged branches**~~
  **CLOSED 2026-07-28** — seven, not four (`dftb0-h2`, `dftb3-water-mio` and
  `gpw-vv10-ice` came in the same way). Each deliverable verified present and
  registered on dev (regression dir + CMake entry; gpw-vv10-ice is a negative
  reconfirm with no code, its finding recorded in the CP2K queue). All seven
  deleted per the house rule — history is the git log
  (`a82b4b73`, `757155d0`, `53ff52a0`, `d425bcf4`, `24d43574`, `bff3936d`,
  `68df18a8`).
- ~~**Citation debt: "Szabo & Ostlund Table 3.11 = −1.117506 Ha"**~~
  **CLOSED 2026-07-28** — audited every site: no REFERENCE attributes the
  digit string to S&O. `rhf_h2_linesearch` and `mc_bias_es` cite CCCBDB
  (−1.11751 Ha) as the literature value and print ours alongside;
  `lj_rhf_coupled` carries the explicit retraction. Nothing to rewrite.
- ~~**`cpu::SubDriver` phantom in CLAUDE.md / AGENTS.md**~~ **FIXED
  2026-07-28** (`6aba6114`): five sites each replaced with the real
  `DriverNode` nesting API (`outer.on(phase).add_ref(inner, order, budget)`),
  plus a deleted-overload `add(lvalue)` in the same example.


## Ticket — /mol:fix owed (named 2026-07-28, classical virial sweep)

- **`PairLJCut`'s SIMD virial was silently wrong at the shipped Release
  flags** — FIXED in `49af746a`, but recorded because the failure mode is
  worth remembering: energy exact to 5.6e-16 and forces to 1.6e-17 against a
  brute-force image sum, while `tr(sigma)` was **37 % short**. Correct at
  `-O2 -march=native`, at `-O3`, and at `-O3 -march=native
  -fno-tree-vectorize`; wrong ONLY at `-O3 -march=native`, which is what we
  ship. GCC 14.3 reassociated the six virial reduction chains because they
  sat inside a loop-invariant `if (do_virial)` guard while the unguarded
  force/energy accumulators beside them survived. Removing the branch cost
  0.4 %, not the 20-25 % the header claimed. Lesson: never put a SIMD
  reduction behind a loop-invariant branch, and never validate a virial only
  at low optimisation. The durable guard is
  `tests/cpu/potentials/test_pair_virial.cpp`, which checks
  `tr(sigma) = -3V dE/dV` and is flag-independent.
- ~~**Seven pair styles write no `kStress` at all**~~ **FIXED 2026-07-28**
  (`95722ee2`): all seven accumulate the virial unconditionally (no
  loop-invariant guard, so there is nothing for the vectorizer to reassociate).
  The sign is NOT uniform in this tree — five styles write `f -= fpair*d` and
  two write `f += fpair*d` — so each kernel was read rather than copied. Every
  style has a RED-before / GREEN-after row in
  `tests/cpu/potentials/test_pair_virial.cpp` against both an explicit pair sum
  and `tr(sigma) = -3V dE/dV`; flag independence checked by rebuilding the
  whole library at `-O2 -march=native` (bit-identical). TIP4P uses the M-site
  separation (proved sensitive: forcing the O-O separation gives a 34 % trace
  error and fails both gates). DPD tallies the CONSERVATIVE force only —
  the dissipative/random terms are thermostat forces whose instantaneous
  `r.F` diverges as `1/sqrt(dt)` while averaging to zero, so tallying them
  would feed unbounded timestep-dependent noise into every reported pressure;
  that choice is itself pinned by a test.
- ~~**DPD's conservative energy dropped the `r_c` factor**~~ **FIXED
  2026-07-28** (`95722ee2` + test widened): `U = 1/2 a w^2` instead of
  `1/2 a r_c w^2` — wrong units (`a` is a force amplitude [eV/A]) and a
  disagreement with the force the same loop applies of exactly `r_c`. Invisible
  because every DPD test and regression used `r_c = 1 A`. The unit test now
  runs at `r_c = 2.5 A` and pins force and energy together.
- ~~**The GCC-vs-UB question is unresolved**~~ **CLOSED 2026-07-29
  (diagnosis only)**: re-audited the ket-loop virial path
  (`src/cpu/potentials/pair_lj_cut.cpp` ~300–830). No concrete language-UB
  candidates — no type-punning, no uninit reads, no OOB gather, no restrict
  alias of x vs f, no volatile/asm tricks. Three independent rewrites that
  each fixed the 37 % shortfall (drop the loop-invariant branch, `#pragma
  GCC novector`, scalar-reduce per batch) is the signature of a **GCC 14.3
  reduction miscompile under `-O3 -march=native`**, not a latent OOB that all
  three would coincidentally avoid. Product warranty remains
  `tests/cpu/potentials/test_pair_virial.cpp` (flag-independent). Do **not**
  reintroduce `if (do_virial)` around the six batch FMAs; if NVE ever needs
  to skip them, specialize two kernels at `build`/`launch` rather than a
  runtime guard inside the SIMD body.

## Live — deferred PySCF engine gaps (2026-07-24)

Approved for **later** implementation (do **not** auto `/mol:impl` until
scheduled). Unblock `docs/developer/pyscf-method-queue.md` blocked rows.

| Spec | Status | Unblocks | One-line |
| ---- | ------ | -------- | -------- |
| [rsh-eri-cam-b3lyp](rsh-eri-cam-b3lyp/spec.md) | approved, deferred | `cam-b3lyp` | erf-attenuated ERI + CAM-B3LYP RKS energy foothold |
| [periodic-rhf-gamma](periodic-rhf-gamma/spec.md) | approved, deferred | `periodic-rhf-gamma` | Γ-point periodic RHF energy (lattice HF-X) foothold |

Acceptance: `rsh-eri-cam-b3lyp/acceptance.md`, `periodic-rhf-gamma/acceptance.md`.
Independent — no cross-deps. Roadmap: `docs/roadmap/dft.md` §6, `hartree-fock.md` §2.

## Closed — SCF review remediation chain (2026-07-24, all 8 done)

Fixed all 27 findings of the 2026-07-24 /mol:review of `cpu/scf` (verdict
was REQUEST CHANGES: 11 🔴 / 9 🟡 / 7 🟢). Both architecture scrubs restored
to green. Specs deleted on close; history in git (commits f9a6443a..HEAD).

| # | Spec | Deps | One-line |
| - | ---- | ---- | -------- |
| 01 | scf-review-01-linalg | — | DONE 2026-07-24 (closed; see git) — sym_power/eig_nonsymmetric/solve_pcg in cpu/linalg + OrbitalHessian::direct_rpa |
| 02 | scf-review-02-eri-hoists | 01 | DONE 2026-07-24 (closed; see git) — internal/ao2mo.h + internal/spin_orbital_eri.h + mp_denominator |
| 03 | scf-review-03-posthf-dedup | 01, 02 | DONE 2026-07-24 (closed; see git) — eom_ccsd_common (EomImds ctor) + ci_engine; build_imds scrub debt cleared |
| 04 | scf-review-04-meanfield-dedup | — | DONE 2026-07-24 (closed; see git) — FD-Hessian CRTP template + check_hessian_preconditions + GradientResult tag ctor |
| 05 | scf-review-05-api-hygiene | 01 | DONE 2026-07-24 (closed; see git) — grid ctor + throws + units::pi + .cpp moves; BOTH scrubs restored green |
| 06 | scf-review-06-scf-perf | 05 | DONE 2026-07-24 (closed; see git) — XCGridBasis cache + jk_rhf one-sweep + ERI micro-opts (~14% on fixture) |
| 07 | scf-review-07-posthf-perf | 02 | DONE 2026-07-24 (closed; see git) — block eri_dense (10x/54x) + RI-MP2 GEMM + copy elision |
| 08 | scf-review-08-ccsd-spatial-diis | 02, 03, 07 | DONE 2026-07-24 (closed; see git) — CcAmplitudeDiis + spatial CCSD (~87x, oracle parity 1e-9) |

## Live — deferred PySCF only (after harness wave 12)

| Spec | Branch / track | Status |
| ---- | -------------- | ------ |
| [periodic-rhf-gamma](periodic-rhf-gamma/) | pyscf / HF periodic | open (deferred) |
| [rsh-eri-cam-b3lyp](rsh-eri-cam-b3lyp/) | pyscf / RSH | open (deferred) |

Open CP2K specs on `dev` = **0**. Stock VV10 is **blocked**. Incomplete
`[~]` = **1** (`gpw-ts-kpoint-stress` — unmerged `wf/*` after harness wave 6
2026-08-01). Pending residual **2** (`gpw-metallic-stress`,
`gpw-bomd-smearing`; **not** queue exhausted).

## Closed — CP2K harness wave 12 (2026-07-26)

| Queue | Specs (deleted; see git) | Shipped highlights |
| ----- | ------------------------ | ------------------ |
| cp2k-method-loop harness wave 12 | xl-bomd-linearized-forces, second-gen-cpmd-kpoints, gpw-d3-atm, gpw-d4-atm, bands-spin-metallic-si, dos-spin-metallic-si | zero-SCF linearized XL; multi-k SGCP; D3/D4 ATM; spin+metallic bands/DOS |

Do not re-create closed slugs without superseding intent.

## Closed — CP2K harness wave 11 (2026-07-26)

| Queue | Specs (deleted; see git) | Shipped highlights |
| ----- | ------------------------ | ------------------ |
| cp2k-method-loop harness wave 11 | ugw-uks-rpbe, gpw-ts-forces, gpw-xdm-forces, gpw-kpoint-uks-stress-gga, cpmd-electronic-thermostat, xl-bomd-uks-kpoints, bands-metallic-si, dos-spin-si, dfpt-finite-q-spin-smearing | UGW@UKS-RPBE; TS/XDM force pins; multi-k UKS stress GGA; CPMD e-T; multi-k UKS XL; metallic bands; spin DOS; finite-q DFPT spin+smear |

Do not re-create closed slugs without superseding intent.

## Closed — CP2K harness wave 10 (2026-07-26)

| Queue | Specs (deleted; see git) | Shipped highlights |
| ----- | ------------------------ | ------------------ |
| cp2k-method-loop harness wave 10 | dfpt-spin-smearing, ot-uks-dftu-smearing, second-gen-cpmd-uks-ot, xl-bomd-kpoints, bands-spin-si, dos-metallic-si, periodic-wannier-uks, berry-spin-si, gpw-mbd-kpoint-forces | DFPT spin+smear; multi-k UKS OT+U+smear; SGCP UKS OT; multi-k XL-BOMD; spin bands; metallic DOS; spin MLWF; spin Berry; multi-k MBD forces |

Do not re-create closed slugs without superseding intent.

## Closed — CP2K harness wave 9 (2026-07-26)

| Queue | Specs (deleted; see git) | Shipped highlights |
| ----- | ------------------------ | ------------------ |
| cp2k-method-loop harness wave 9 | ot-uks-smearing, ot-uks-dftu, ot-dftu-smearing, gpw-d4-forces, gpw-mbd-forces, gpw-uks-multigrid, dfpt-finite-q-smearing, second-gen-cpmd-ot | UKS OT+smear; multi-k UKS OT+U; OT+U+smear; D4/MBD forces; UKS multigrid; finite-q DFPT smear; SGCP OT |

Do not re-create closed slugs without superseding intent.

## Closed — CP2K harness wave 8 (2026-07-26)

| Queue | Specs (deleted; see git) | Shipped highlights |
| ----- | ------------------------ | ------------------ |
| cp2k-method-loop harness wave 8 | periodic-urpa-kpoints, ot-smearing, ot-uks-kpoints, ot-dftu-kpoints, gpw-d4-stress, gpw-vdw-scf-potential, dfpt-metallic-si, dfpt-spin-incommensurate, gpw-optb88-vdw | multi-k URPA; OT+smear; multi-k UKS OT; multi-k OT+DFT+U; D4 analytic stress; nonlocal vdW SCF pot/forces; metallic DFPT; spin-incommensurate DFPT; optB88-vdW |

Do not re-create closed slugs without superseding intent.

## Closed — CP2K harness wave 7 (2026-07-26)

| Queue | Specs (deleted; see git) | Shipped highlights |
| ----- | ------------------------ | ------------------ |
| cp2k-method-loop harness wave 7 | periodic-urpa-si, periodic-ugw-kpoints, gpw-uks-smearing, gpw-hirshfeld-uks, cpmd-uks, second-gen-cpmd-uks, ot-kpoints, ot-dftu, dfpt-spin-si | URPA@UKS; multi-k UGW; GPW UKS FD/MP smear; UKS Hirshfeld; CPMD UKS; SGCP UKS; OT+k; OT+DFT+U; collinear-spin DFPT |

Do not re-create closed slugs without superseding intent.

## Closed — CP2K harness wave 6 (2026-07-26)

| Queue | Specs (deleted; see git) | Shipped highlights |
| ----- | ------------------------ | ------------------ |
| cp2k-method-loop harness wave 6 | gpw-metallic-smearing, gpw-xdm-ice, gpw-uks-rpbe, gpw-dftu-kpoint-stress, periodic-ump2-kpoints, periodic-ugw-si, gpw-hirshfeld-kpoints, dfpt-gga-si, xl-bomd-uks | GPW FD/MP smear; free-atom XDM; Γ-UKS RPBE; multi-k DFT+U stress; multi-k UMP2; UGW@UKS; multi-k Hirshfeld; DFPT-GGA; XL-BOMD UKS |

Do not re-create closed slugs without superseding intent.

## Closed — CP2K harness wave 5 (2026-07-26)

| Queue | Specs (deleted; see git) | Shipped highlights |
| ----- | ------------------------ | ------------------ |
| cp2k-method-loop harness wave 5 | gpw-dftu-kpoints, periodic-gw-finite-q, gpw-uks-revpbe, cpmd-rattle-water, periodic-ump2-si, ot-uks-o2 | multi-k PLUS_U; finite-q G0W0 \(W\); Γ-UKS revPBE; CPMD RATTLE; UMP2@UKS; OT-UKS O₂ |

Do not re-create closed slugs without superseding intent.

## Closed — CP2K harness wave 4 (2026-07-26)

| Queue | Specs (deleted; see git) | Shipped highlights |
| ----- | ------------------------ | ------------------ |
| cp2k-method-loop harness wave 4 | gpw-kpoint-uks-stress, periodic-gw-kpoints, gpw-vdw-df2-ice, gpw-rvv10-ice, gpw-mbd-periodic, pw-spin-gga, gpw-uks-blyp, periodic-rpa-finite-q, gpw-dftu-stress | multi-k UKS stress; multi-k G0W0; vdW-DF2 LMKLL; rVV10; periodic Ewald MBD; PW collinear spin-GGA; Γ-UKS BLYP; finite-q dRPA; DFT+U analytic stress |

Do not re-create closed slugs without superseding intent.

## Closed — CP2K harness wave 3 (2026-07-25)

| Queue | Specs (deleted; see git) | Shipped highlights |
| ----- | ------------------------ | ------------------ |
| cp2k-method-loop harness wave 3 | gpw-rpbe-water, gpw-kpoint-uks-forces, gpw-uks-stress, gpw-vdw-df-ice, gpw-mbd-hirshfeld, periodic-rpa-kpoints, periodic-gw-si, dfpt-incommensurate-si | RPBE GGA; multi-k UKS forces; Γ-UKS stress; vdW-DF DRSLL; Hirshfeld MBD/TS; multi-k dRPA; Γ spectral G0W0; incommensurate DFPT |

Do not re-create closed slugs without superseding intent.

## Closed — CP2K harness wave 2 (2026-07-25)

| Queue | Specs (deleted; see git) | Shipped highlights |
| ----- | ------------------------ | ------------------ |
| cp2k-method-loop harness wave 2 | gpw-kpoint-forces, gpw-revpbe-si, gpw-kpoints-uks, gpw-uks-forces, gpw-kpoint-stress, gpw-mbd-ice, periodic-mp2-kpoints, periodic-rpa-si, dfpt-finite-q-si, gpw-variable-cell-molopt | MP k-forces; revPBE GGA; k-UKS; Γ-UKS forces; k-stress; free-atom MBD; multi-k RI-MP2; Γ dRPA; finite-q DFPT X; MOLOPT CELL_OPT |

Do not re-create closed slugs without superseding intent.

MetalWalls wave 3 closed 2026-07-24 (specs deleted). **Wave-4 assign**
(2026-07-24) staffed 10 production-depth / residual map rows to
`## in_progress` (still unstarted as of **wave-5 gap-census** 2026-07-24:
MaZe, NeighborList salt/Gaussian, electrode forces, CV-SMD, aqueous EDL,
D-field, charge tools, …). Wave-5 census refilled **pending=10** (per-electrode
\(\eta\), const-charge control, MPI charge-halo, XL dipoles, QEq, PIM-NPT IL,
Voronoi TF, IL EDL box, nanoporous product, TI capacitance). No open
MD/electrochem specs on `dev` — workers write `.claude/specs/<slug>/` on
`wf/<slug>` before impl.

CP2K claude wave 1 closed 2026-07-25 (5 merged). **`xl-bomd-water` closed
2026-07-25** (resume of WIP — verified green, merged, spec deleted).


## Closed — CP2K harness wave 1 (2026-07-25)

| Queue | Specs (deleted; see git) | Shipped highlights |
| ----- | ------------------------ | ------------------ |
| cp2k-method-loop harness wave 1 | gpw-blyp-water, gpw-uks-pbe, pw-uks-fe-smearing, pw-variable-cell-si, gpw-ts-ice, phonon-dispersion-si, periodic-wannier-si, second-gen-cpmd-water, periodic-ri-mp2-si | GPW BLYP; Γ-UKS PBE O₂; bulk Fe UKS+smear; PW CELL_OPT Si; free-atom TS; phonon ω(q) IFC Fourier; periodic MV MLWF; 2nd-gen CP water; Γ RI-GPW MP2 product |

Do not re-create closed slugs without superseding intent.

## Closed — CP2K parallel wave (2026-07-25)

| Queue | Specs (deleted; see git) | Shipped |
| ----- | ------------------------ | ------- |
| cp2k-method-loop parallel | mpi-pw-scaling-4rank, gpw-analytic-stress, wannier-centers, mpi-gpw-scaling-4rank, cpmd-water | PW/GPW MPI k-RR gates; analytic GPW stress; Foster–Boys Wannier; CPMD H₂O |

## Closed — CP2K xl-bomd-water (2026-07-25)

| Queue | Spec (deleted; see git) | Shipped |
| ----- | ----------------------- | ------- |
| cp2k-method-loop | xl-bomd-water | `XlBomd` + `guess_P` + `GPWEvaluator::xl_bomd`; `reg_xl-bomd-water` NVE drift 1.2e-5 |

## Closed — CP2K capability claude wave 1 (2026-07-25)

| Queue | Specs (deleted; see git) | Shipped highlights |
| ----- | ------------------------ | ------------------ |
| cp2k-method-loop claude wave 1 | gpw-kpoints-si, gpw-variable-cell-ice, dfpt-phonon-si, berry-polarization, periodic-mp2-si | MP k-point GPW RKS; GPW variable-cell relax vs CP2K CELL_OPT; Γ-point DFPT phonons (PW); KSV Berry-phase polarization (PW); Γ-point canonical GPW MP2 foothold |

Do not re-create closed slugs without superseding intent.

## Closed — MetalWalls capability wave 3 (2026-07-24)

| Queue | Specs (deleted; see git) | Shipped highlights |
| ----- | ------------------------ | ------------------ |
| metalwalls-method-loop wave 3 | constant-force-pull-ion, oscillating-efield-md, potential-profile-observer, surface-tension-coupling, virtual-site-multi-topology, dipole-electrode-joint | FixAddForce; FixOscillatingEfield; Ψ(z) PotentialProfile; SurfaceTensionNPT; VirtualSites Weighted3+OutOfPlane3; DipoleElectrodeJoint |

Do not re-create closed slugs without superseding intent.

## Closed — MetalWalls capability wave 2 (2026-07-24)

| Queue | Specs (deleted; see git) | Shipped highlights |
| ----- | ------------------------ | ------------------ |
| metalwalls-method-loop wave 2 | const-pot-cg-large-electrode, electrode-charge-predictor, finite-field-single-electrode, pim-induced-dipole-bulk, piston-electrode-pz, thomas-fermi-electrode, fumi-tosi-nacl, steele-wall-argon, born-repulsion-foothold, capacitance-charge-fluctuation | ConstPotCG; ChargeASPC predictor; Dufils single-electrode finite-field; bulk PIM+Thole; PistonElectrode; TF metallicity; PairBorn/Fumi–Tosi; FixWallLJ1043 Steele; CapacitanceChargeFluctuation |

Do not re-create closed slugs without superseding intent.

## Closed — CP2K capability wave 2 (2026-07-24)

| Queue | Specs (deleted; see git) | Shipped highlights |
| ----- | ------------------------ | ------------------ |
| cp2k-method-loop wave 2 | gpw-stress-water, gpw-bomd-water, pw-analytic-stress-si, fd-phonon-supercell, gpw-dftu-nio, bands-dos-si, pw-uks-kmesh-fe, gpw-pbesol-si, gpw-d4-water | GPW strain-FD stress; GPW BOMD NVE; PW Nielsen–Martin stress; FD supercell phonons; Dudarev DFT+U NiO; bands/DOS Si; k-mesh UKS O₂; PBEsol GPW; periodic D4 |

Do not re-create closed slugs without superseding intent.

## Closed — CP2K capability wave 1 (2026-07-23)

| Queue | Specs (deleted; see git) | Shipped highlights |
| ----- | ------------------------ | ------------------ |
| cp2k-method-loop wave 1 | gpw-water-pbe-mt, gpw-forces-water, gpw-uks-o2-gamma, gpw-pbe-d3bj-ice, pw-lda-si, pw-pbe-gga-si, pw-smearing-al, pw-spin-polarized-o2, pw-bomd-h, ot-minimizer-water | MT water PBE pin; GPW forces; Γ-UKS O₂; GPW D3(BJ); PW LDA Si; true PBE GGA PW; Fermi/MP Al; PW spin O₂; PW BOMD H; OT water pin |

## Closed — MetalWalls capability wave 1 (2026-07-23)

| Queue | Specs (deleted; see git) | Shipped highlights |
| ----- | ------------------------ | ------------------ |
| metalwalls-method-loop wave 1 | const-pot-bo-parallel-plate, freeze-electrode-atoms, const-pot-charge-observer, charging-work-energy, const-pot-md-nvt-short, gaussian-electrode-charge, slab-2d-ewald-foothold, external-efield-md, buckingham-pair-lammps, four-site-tip4p-foothold | ConstPotBO + FixFreeze + electrode charge observer + charging work + short const-pot NVT; Gaussian electrode Coulomb; slab Ewald (`slab_factor`); FixEfield; PairBuck; TIP4P `lj/cut/tip4p/cut` |

## Closed — CPU next campaign (2026-07-23, parallel worktrees)

| Branch (deleted) | Specs | Highlights |
| ------ | ----- | ---------- |
| `feat/cpu-next-md` | parrinello-rahman-npt, nph-ensemble | PR full-cell NPT, isotropic NPH |
| `feat/cpu-next-tb-arch` | tb-sk-product, concrete-pw-method, concrete-gpw-method, concrete-nddo-method | Multi-element DFTB2; concrete PW/GPW/AM1-RM1 |
| `feat/cpu-next-msamp` | qmmm-link-atoms, polarizable-qmmm, multi-cv-metad, multi-window-wham | Link atoms, pol QM/MM, ND metaD, WHAM (molrs) |
| `feat/cpu-next-response` | full-grid-alda, analytic-tda-forces, true-nac-vectors, ri-mp2 | Full MO-product ALDA, CIS TDA forces, NAC vectors, RI-MP2 |

## Closed earlier on `dev` (2026-07-23)

| depth-next | grid-alda-tda, qmmm-electrostatic, tda-nac-skeleton, triclinic-cell-relax | ALDA diagonal, ES QM/MM, NAC strength, triclinic cell |
| open scales | alda-tddft-tda … ri-mp2-skeleton (8) | GFN0, aniso cell, mech QM/MM, metaD, multi-DPD, TDA forces, MP2 |
| method ladder | rks-tda … hamiltonian-remd-lambda (8) | RKS-TDA, DFTB2, PW-cell, DihedralCV, FENE+WCA, DPD, Yukawa, H-REMD |
| breadth-next | soft-core-coulomb … brownian-dynamics (9) | soft Coul, AngleCV, TI, joint cell, string, IRC, Rg², SETTLE, BD |
| breadth-first | FIRE … dimer (6) | soft LJ, Widom, PIMD, isotropic cell, dimer |
| architecture | backend-mirror, cuda-resurrect, cuda-pme-accuracy | layout rewrite, CUDA live, PME Madelung |

## Closed — CUDA OpenMM MD engine (2026-07-23, from origin/dev)

| cuda-openmm-md 01–05 | contract, rebuild, tile-force, multigpu, delete | CS=32 clusters, real masks, tile LJ/Coul, multi-GPU classify; Verlet default removed on CUDA |
