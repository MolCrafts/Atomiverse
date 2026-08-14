# MetalWalls method loop — handoff

Last land: **queue emptied — 2026-07-29** (residual product closed; permanent
out-of-loop blocked rows removed from active queue). Prior: real-oracle re-pin
2026-07-29; wave 5 2026-07-27 (done total **49** = 46 + 3 residual).
**pending=0, blocked=0, in_progress=0 → queue_empty for this loop.**
**incomplete_remaining=0**.


## Residual queue product land — 2026-07-29 (not a new wave)

Closed the three remaining **pending** rows without GapCensus/staff_count wave:

| slug | Regression | notes |
|------|------------|-------|
| `nanoporous-carbon-const-pot` | `reg_nanoporous-carbon-const-pot` | slit-pore C + ConstPotBO + occupancy |
| `maze-molten-salt-al` | `reg_maze-molten-salt-al` | Fumi–Tosi + Al MaZe + short NVT |
| `thermo-integration-capacitance` | `reg_thermo-integration-capacitance` | C=Q/Ψ + soft-Coul TI + ChargingWork |

**pending=0.** Former blocked (CUDA / AIM-DAIM / ML / hybrid DFT) **deleted from
the active queue** 2026-07-29 — ownership is map E11/E12 + CUDA/quantum loops,
not fake “still blocked” debt on this worklist.

## Real-oracle re-pin (2026-07-29) — MetalWalls binary now exists

**MetalWalls is built and verified**: `gitlab.com/ampere2/metalwalls` release
`c8ad175d` (2026-06-18) → `/nobackup/…/work/opt/metalwalls/mw`, GNU 14.3 +
MPICH + OpenBLAS. Rebuild via `tools/metalwalls_oracle/build_metalwalls.sh`
(self-checks against upstream `tests/maze/electrodes/LiCl-Al`; must print PASS
before any pin is taken). Waves 1–5 had pinned MW-only physics from literature
formulas **because no binary existed** — that fallback is retired.

Ten `oracle: metalwalls` regressions were re-checked against the real binary.
All ten merged into `dev`; gate green (94/94 domain ctest, all ten regressions).

| Slug | Verdict | Finding |
| ---- | ------- | ------- |
| `electrode-charge-predictor` | **atv_bug (fixed)** | `ChargeASPC::compute_coefficients_` computed Lagrange weights `{1,-3,3}` while citing Kolafa ASPC; real MW uses Kolafa K=5 `{22/7,-55/14,55/21,-22/21,5/21,-1/42}`. Fixed; discriminating observable now pinned. |
| `thomas-fermi-electrode` | **repinned** | Old pin came from `pin_gold_cap.py`, a numpy transcription of `ConstPotBO` itself — self-golden, and **6.13 % wrong**. Script deleted; ATV now 0.18 % from MW. |
| `piston-electrode-pz` | **repinned** | Old "pin" was `P_target := probe.last_P_z()` — a tautology. Now pinned to MW `electrode_forces.out`; agrees to 1.6e-8 (CODATA drift). |
| `pim-induced-dipole-bulk` | **repinned** | Thole-damped (a=0.39) number was attributed to MW, but **MW has no Thole damping** (`grep -ri thole` → 0 hits). Physics was right, attribution fabricated. Re-pinned at `thole_a=0`. |
| `dipole-electrode-joint` | not_expressible | `pin_joint.py` was also a transcription of ATV kernels. MW differs 9 % (Q) / 16 % (μ) — min-image vs Ewald. ATV kernels under a matched 3D lattice sum reproduce MW to **1.2e-8**; provenance claim corrected. |
| `maze-symplectic-charge-dynamics` | confirmed | MW matches to 1.1e-6 where both compute the same model; MW has no minimum-image mode (num_pbc ∈ {2,3}, Ewald only). |
| `const-pot-cg-large-electrode` | confirmed | 3.1e-9. |
| `per-electrode-eta` | confirmed | 8.3e-12 under a common Coulomb constant. |
| `finite-field-single-electrode` | not_expressible | MW **refuses** `external_field` + `charge_neutrality false` (well-posedness guard, `src/configuration.F90:624`). Nearest legal twin: ATV elastance +9.65 % (real-space sum vs Ewald) — recorded as a capability gap. |
| `pim-npt-polarizable-il` | not_expressible | ATV truncates the induced-dipole field, MW is Ewald-only; gap decomposed (Thole 0.9 %, cutoff 1.8×, reciprocal tail ~2.9×). ATV kernel confirmed by MW in the isolated-pair limit. |

**Binding rule added** (`.claude/notes/oracle-pin-provenance.md`, CLAUDE.md):
a pin is a number **the other code printed**; oracle scripts parse oracle
output and never reimplement ATV's algorithm; never seed a pin from ATV's own
measurement. "No binary" is no longer a valid reason for `literature` or
`perf_exempt`.

**Real perf ratios** (\(T_{\mathrm{atv}}/T_{\mathrm{MW}}\), replacing
`perf_exempt (no MW binary)`): TF electrode **4.4**, PIM bulk **0.77**,
const-pot CG **0.026**, per-electrode-η **0.09**, PIM-NPT **0.072**.

**Batch-2 re-pin CLOSED (2026-07-29)** — commit `d59e4350` + prior
`global-electroneutrality-bo` / `voronoi-volume-tf`.

| Slug | Verdict |
| ---- | ------- |
| `global-electroneutrality-bo` | **done** — MW sum \(Q=-1\); per-electrode 2e-5 |
| `voronoi-volume-tf` | **done** — real MW TF+Voronoi pins (retired 6.13% self-golden) |
| `const-pot-md-nvt-short` | **done** — soft \|Q\| vs MW pbc=3 (~3%) |
| `const-pot-il-edl-box` | **done** — empty MW pin; IL absolute not_expressible (image-sum vs Ewald) |
| `extended-lagrangian-dipoles` | **done** — stationary thole_a=0 MW pins; XL dynamics self-check |
| `const-pot-tip4p-edl` / `maze-charge-dynamics` / `electrode-atom-forces` | **provenance** — MW twins archived; foothold gates; \|Q\| model gap documented |
| `constant-d-polarization-feedback` / `displacement-dfield-md` | **literature intentional** (Zhang–Sprik / vacuum D) |

**incomplete_remaining=0** for real-oracle re-pin debt listed after wave-5.
Active metalwalls queue is empty (no pending residual, no blocked placeholders).

**Workflow order (binding):** Recover incomplete first → Census →
Assign (`incomplete ≫ pending`) → Worktrees (**spec-first**) → Merge → Harness →
Tutorial. See `docs/developer/metalwalls-method-loop.md`.

## Wave 5 land (2026-07-27, merged → `dev`)

| Slug | Component / change | Regression | oracle | ratio |
| ---- | ------------------ | ---------- | ------ | ----- |
| `per-electrode-eta` | `ConstPotBO::add_electrode(type, psi, eta)` per-group Gaussian widths, mixed \(\eta_{ij}\) | `reg_per-electrode-eta` | literature (Reed 2007 DOI 10.1063/1.2464084; Scalfi 2020 DOI 10.1063/5.0028232; MW JOSS 2020) | perf_exempt |
| `const-charge-electrode` | `ConstPotBO::add_electrode_charge` const-\(Q\) control (LAMMPS `electrode/conq` twin) | `reg_const-charge-electrode` | lammps | perf_exempt |
| `const-pot-charge-halo-mpi` | post-solve ghost-charge refresh under MPI DD (`ConstPot*` charge halo) | `reg_const-pot-charge-halo-mpi` | literature / self-consistency | perf_exempt |
| `maze-symplectic-charge-dynamics` | `ConstPotMaZe.symplectic` true SHAKE mass-zero path (Coretti) | `reg_maze-symplectic-charge-dynamics` | metalwalls (Coretti 2020 literature pin) | perf_exempt |
| `constant-d-polarization-feedback` | `FixDfield.polarization_feedback` self-consistent \(E = D - 4\pi k_e P\) (Zhang–Sprik) | `reg_constant-d-polarization-feedback` | metalwalls | perf_exempt |
| `extended-lagrangian-dipoles` | `cpu::PimXlDipole` time-reversible XL induced dipoles | `reg_extended-lagrangian-dipoles` | metalwalls (iEL/SCF literature pins — Albaugh/Niklasson; MW MaZe route cited) | perf_exempt |
| `qeq-fluctuating-charge` | `cpu::md::QEq` (LAMMPS `fix qeq/point` twin) | `reg_qeq-fluctuating-charge` | lammps | **0.45** |
| `pim-npt-polarizable-il` | bulk PIM + NPT polarizable-IL product twin | `reg_pim-npt-polarizable-il` | metalwalls (offline numpy Thole-PIM pin; MW JOSS 2020, Thole 1981) | perf_exempt |
| `voronoi-volume-tf` | `VoronoiVolume` TF site-volume companion | `reg_voronoi-volume-tf` | metalwalls | perf_exempt |
| `const-pot-il-edl-box` | planar const-pot + non-polarizable IL EDL production box | `reg_const-pot-il-edl-box` | metalwalls | perf_exempt |

All regressions registered behind `ATV_BUILD_REGRESSIONS`; every `REFERENCE.md`
states `oracle:` + `oracle_version`; ctest never launches LAMMPS/MetalWalls.

## Recover inventory (2026-07-27, post-harness wave 5)

| Source | Result |
| ------ | ------ |
| A queue `## in_progress` `[~]` | **0** (wave 5 fully merged) |
| B open `.claude/specs/` MD/electrochem | **0** (esmm-\*/dftb2-\*/periodic-rhf-gamma/rsh-eri-cam-b3lyp belong to other loops — never touch) |
| C unmerged `wf/<slug>` (this loop) | **0** (all wave-5 tips ancestors of `dev`) |
| D handoff incomplete rows | **none** |
| E half-written `regressions/{md,forcefield}/` | **0** MW slugs (method-ladder orphans unchanged, not this loop) |

**Resume set: empty.** **incomplete_remaining=0**.

## Next

**`queue_empty` for `metalwalls-method-loop`.** Nothing left to staff on the
active worklist.

Deferred **off-queue** (map / other owners — do not re-add as pending without
an explicit charter):

| former blocked slug | owner |
|---------------------|--------|
| `cuda-const-pot` | CUDA electrode track |
| `aim-daim-product` | map E11 (AIM/DAIM P3) — only if scheduled |
| `ml-pinnwall-electrode` | map E12 / external ML |
| `periodic-hybrid-electrode-dft` | CP2K / PySCF loops |

Parent stays on **`dev`**. No GapCensus wave required to “clear” these.

## Oracle

- Overlap with LAMMPS → **LAMMPS** offline pin (`tools/metalwalls_oracle/` + LAMMPS inputs)
- MW-only → **MetalWalls** offline pin (or literature when no binary)
- `ctest` never launches LAMMPS or MetalWalls
- Wave-5 oracle tags: `qeq-fluctuating-charge` + `const-charge-electrode` →
  **lammps**; `per-electrode-eta` + `const-pot-charge-halo-mpi` →
  **literature**; the other six → **metalwalls** (literature/numpy pins where
  no MW binary)

## Build (workers)

MD-only flags: `ATV_BUILD_MD=ON`, `ATV_BUILD_KSPACE=ON`, `ATV_BUILD_TESTS=ON`,
`ATV_BUILD_REGRESSIONS=ON`, **HF/PW/GPW/CUDA OFF**.

## History (compact)

- **Wave 1** (2026-07-23): 10 merged → done (const-pot BO/observer/NVT,
  charging work, Gaussian electrode, slab Ewald, E-field, Buckingham, TIP4P
  foothold, freeze). Ratios: freeze 3.98, slab 6.49, efield 0.60, tip4p 0.338;
  rest perf_exempt.
- **Wave 2** (2026-07-24): 10 merged → done (ConstPotCG, ASPC predictor,
  finite-field single electrode, bulk PIM, piston, Thomas–Fermi, Fumi–Tosi,
  Steele wall, Born, capacitance fluctuation). All perf_exempt.
- **Wave 3** (2026-07-24): 6 merged → done (force pull 0.12, oscillating
  efield 2.12, surface tension 2.88, virtual sites 0.084, \(\Psi(z)\) +
  dipole⊕electrode joint perf_exempt).
- **Wave 4** (2026-07-24 write, manual land re-verified 2026-07-25): 10 → done
  (ConstPotMaZe CG+ASPC foothold, electrode forces, global electroneutrality,
  Gaussian/Born NL bulk paths, `FixDfield` vacuum foothold, TIP4P EDL,
  \(C_{QQ}\), elastance observer, `FixMove` SMD). All perf_exempt.
- **Wave 5** (2026-07-27): table above; done total **46**.

Queue: `docs/developer/metalwalls-method-queue.md` (do not delete done rows).
Map: `.claude/notes/metalwalls-capability-map.md` (ATV marks re-audited
2026-07-27 post-wave-5).
