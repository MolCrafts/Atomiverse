# CP2K method loop — worktree → merge → dev

Abstract pattern: [`oracle-capability-loop.md`](oracle-capability-loop.md).  
Scaffold skill: `.agents/skills/create-oracle-loop/`.

## Continuous discovery (binding)

**Every wave starts with GapCensus:** re-diff CP2K GPW/PW/periodic surface +
`docs/roadmap/dft.md` vs ATV on `target_branch`, refill `## pending`, update
roadmap. Foothold ≠ capability closed. Stop `queue_empty` only when
`gaps_remaining=0` — **not** after one seed batch.

## Pipeline

```text
parent (dev)
  recover incomplete
       │
  for each wave:
    GapCensus → Assign → Worktrees → Merge? → Clearup → Harness? → Tutorial?
```

Instance = template + CONFIG/`extra_law` only
(`.grok/workflows/cp2k-method-loop.rhai`).

| Role | `isolation_worktree` | Writes |
| ---- | -------------------- | ------ |
| Recover / GapCensus / assign / merge / harness | **false** (parent) | `dev` only |
| Feature worker | **true** | private worktree only until merge |

Workers must report `worktree_path`, `branch` (`wf/<slug>`), `head_sha`.

**Recover incomplete first** (`## in_progress` `[~]`, open GPW/PW specs,
unmerged `wf/<slug>`). Spec is source of truth. Offline CP2K pins only
(`tools/cp2k_oracle/`); never live CP2K in ctest.

**Scope:** GPW / PW / periodic DFT. **Not** molecular post-HF (PySCF loop) or
classical MD electrochemistry (MetalWalls loop).

## Why not edit dev in parallel?

Parallel agents on one tree thrash shared GPW/PW headers and leave a half-broken
`dev`. Isolation = each feature lands as a branch; merge is ordered and
abortable per slug.

## Args

| Arg | Default | Meaning |
| --- | ------- | ------- |
| `staff_count` / `parallelism` | 10 | Concurrent worktrees |
| `max_waves` | 20 | Assignment rounds this shift |
| `target_branch` | `dev` | Merge destination |
| `do_perf` | true | Fair wall vs CP2K when available |
| `mode` | `exhaust` | Until gaps gone / budget |

## Launch

```text
agent_budget: 1024
args: { "mode":"exhaust", "staff_count":10, "max_waves":20, "target_branch":"dev", "do_perf":true }
```

## Queue

`docs/developer/cp2k-method-queue.md`

## Run log

- **2026-08-01 harness wave 3 (shift) (parent `dev` @ `9bd134e0`):** 4 of 7
  staffed on `dev` → done: `gpw-d4-atm-kpoint-forces` (CP2K libdftd4 multi-k
  ENERGY_FORCE ATM ΔF, ratio **0.613**), `gpw-optb88-uks-kpoints` (multi-k
  UKS OptB88+DRSLL, E_disp parity 3.77e-5 Ha, ratio **0.73**),
  `periodic-scs-mp2-kpoints` (Grimme algebraic dual on multi-k RI-GPW;
  CP2K offline aborts MP2_GPW@k — **perf_exempt** / `oracle: literature`),
  `periodic-berry-gpw` (GPW KSV Berry; Si symmetry pins; fair twin ratio
  **9.6**). Three staff stay `[~]`: `gpw-ts-kpoint-stress` @ `4afc4185`,
  `gpw-d3-atm-kpoint-forces` @ `a4168475`, `gpw-hirshfeld-forces` @
  `4f3a62c6` (feat tips; no reg on parent; do not demote). Specs closed
  (`periodic-berry-gpw`, `periodic-scs-mp2-kpoints` deleted). Residual
  pending **0** (census list emptied by assign — **not** queue exhausted).
  Roadmap: **no ○/✓ table flips**; residual prose → multi-k TS stress +
  multi-k D3 ATM forces + Hirshfeld-scaled forces (`[~]` only). Clearup
  already ran. Queue **not** exhausted. No push.

- **2026-08-01 harness wave 2 (shift) (parent `dev` @ `79bcfa14`):** 8 of 9
  staffed on `dev` → done: `gpw-d4-kpoint-stress` (CP2K libdftd4 multi-k
  NUMERICAL Δσ, ratio **0.139**), `gpw-xdm-kpoint-stress` (dual pure-Python
  strain + FD dual, **perf_exempt**), `gpw-hirshfeld-uks-kpoints` (multi-k
  UKS MBD@rsSCS E_disp, **perf_exempt**), `cpmd-uks-thermostat-kpoints`
  (multi-k UKS eNH combo, **perf_exempt**), `periodic-wannier-gpw-uks`
  (GPW UKS MLWF, ratio **9.57**), `periodic-scs-mp2-si` (Γ RI-GPW SCS
  scales, ratio **0.424**), `gpw-optb88-uks` (Γ-UKS OptB88+DRSLL, ratio
  **3.35**), `gpw-vdw-uks-kpoint-multigrid` (triple combo, ratio **0.85**),
  `gpw-optb88-kpoints` (multi-k RKS OptB88+DRSLL, ratio **0.67**). One
  staff stays `[~]`: `gpw-ts-kpoint-stress` (`wf/*` @ `65aa0e6d`; do not
  demote). Specs closed (7 deleted). Residual pending **2** (multi-k D3/D4
  ATM force pins). Roadmap: **no ○/✓ table flips**; residual prose → TS
  k-stress + ATM k-forces only. Clearup already ran. Queue **not**
  exhausted. No push.

- **2026-08-01 harness wave 1 (shift) (parent `dev` @ `424d6989`):** 6 of 10
  staffed merged → done: `gpw-vdw-uks-kpoints` (dual-spin multi-k nonlocal
  vdW-DF, CP2K NON_LOCAL+UKS+k pin, ratio **0.35**), `gpw-d3-kpoint-stress`
  (CP2K NUMERICAL Δσ multi-k, ratio **0.185**), `gpw-d3bj-kpoint-stress`
  (ratio **0.382**), `gpw-d3-atm-kpoint-stress` (C9 on−off multi-k, ratio
  **0.285**), `gpw-bomd-uks-kpoints` (dual \(P_k^\sigma\) warm start, ratio
  **1.18**), `gpw-vdw-uks-multigrid` (Γ-UKS NGRIDS+DRSLL, ratio **4.44**
  within_10x). Four staff stay `[~]` unmerged: `gpw-d4-kpoint-stress` /
  `gpw-ts-kpoint-stress` / `gpw-xdm-kpoint-stress` /
  `gpw-hirshfeld-uks-kpoints` (feat tips kept; do not demote). Specs closed
  (6 deleted). Residual pending **4** (eNH combo, GPW UKS MLWF, SCS-MP2,
  optB88 UKS). Roadmap: **no ○/✓ table flips**; residual prose already
  product for the six landings. Clearup already ran. Queue **not**
  exhausted. No push.

- **2026-08-01 bootstrap + GapCensus wave 4 (shift) (parent `dev` @
  `87ae7924`):** recover clean (incomplete=0, specs=0, wf/*=0, worktrees=0;
  wiped leftover `build/`). Prior pending residual 0 after harness wave 3 —
  refilled **11** tree residuals (multi-k UKS nonlocal vdW **impl**; multi-k
  D3/D3BJ/D3-ATM/D4/TS/XDM stress pins; multi-k UKS BOMD; Γ-UKS vdW multigrid;
  multi-k UKS Hirshfeld; multi-k UKS CPMD eNH pin). Blocked=6 unchanged. Fixed
  stale OT+U+smear residual header + roadmap residual prose. **Queue not
  exhausted** — Assign next. No push.

- **2026-07-31 harness wave 3 (shift) (parent `dev` @ `419c1dc3`):** 9/9
  staffed slugs merged → done (TS/XDM stress pins; multi-k RKS + Γ-UKS
  nonlocal vdW; multi-k UKS multigrid; multi-k CPMD eNH; multi-k UKS SGCP OT;
  linearized XL stress; finite-q spin×GGA DFPT). Post-merge fix: multi-k UKS
  + nonlocal vdW fail-fast after Γ-UKS accept. Merge gate: 9/9 `run_reg_*`
  green. Specs closed. Clearup: worktrees removed, merged `wf/*` pruned,
  builds wiped. **Queue not exhausted** — next GapCensus. No push.

(autonomous harness appends lines here)

- **2026-07-31 harness wave 2 (shift; parent `dev` @ `8b2fda00`):** 8 of 10
  staffed merged → done: `gpw-d3-stress` (resume finished — CP2K NUMERICAL
  stress-difference pin, Δσ parity 2.3e-10 Ha/Bohr³, ratio 0.55),
  `dfpt-gga-finite-q-si` (finite-q × pure PBE kernel, CP2K supercell vib
  twin, ratio 0.17), `xl-bomd-linearized-kpoints` (k-weighted complex shadow
  \(D_k^\sigma\) RKS/UKS, CP2K `.ener`-anchored trace),
  `gpw-vdw-multigrid` (nonlocal vdW on the REL_CUTOFF multigrid, CP2K
  NGRIDS 4 pin 4.0e-5 Ha, ratio 0.83; found+fixed Γ-UKS vdW silent E_c^nl
  drop → fail-fast), `cpmd-uks-kpoints` (dual-spin Bloch \(C_k^\sigma\) CP,
  oracle family lock — CP2K has no CP propagator),
  `second-gen-cpmd-ot-kpoints` (per-k OT corrector warm-started from ASPC
  natural orbitals; CP2K refuses `&OT`+`&KPOINTS`, A/B identity 3.0e-12 Ha),
  `gpw-d3-atm-stress` (C9 fold, CP2K NUMERICAL C9 on−off Δσ parity 1.1e-13,
  ratio 0.75; adds `d3_rcut_bohr`), `gpw-d4-atm-stress` (s9 strained-cell
  pins, dftd4 3.7.0, σ^ATM parity 3.7e-16 Ha/Bohr³). `gpw-ts-stress` /
  `gpw-xdm-stress` stay `[~]` (uncommitted WIP in their isolation worktrees
  at `197468ac`; do not demote). Specs closed (seven deleted here;
  `gpw-d4-atm-stress` closed on its branch `a2c85a6b`). Merge gate green on
  merged `dev` (scoped rebuild + loop-domain ctest 57/57). Roadmap: **no ○/✓
  table flips**; workers landed the residual prose with their merges;
  harness fixed the stale index GPW ships/residual line. Pending **3**;
  incomplete **2**; blocked **6**. Clearup already ran. Queue **not**
  exhausted. No push.
- **2026-07-31 harness wave 1 (shift; parent `dev` @ `8a9584f4`):** 9 of 10
  staffed merged → done: `gpw-d3bj-stress` (CP2K NUMERICAL-stress difference
  pin, ratio 0.37), `gpw-mbd-kpoint-stress` (k-summed fixed-\(P_k\) MBD fold,
  pymbd strain pins), `gpw-d3-atm-forces` (CP2K force pin; fixed periodic
  image-triple 8× overcount + pbc anchoring, ratio 3.3), `gpw-d4-atm-forces`
  (closed-form ATM triple derivatives; dftd4 3.7.0 `--grad` pins),
  `gpw-dftu-kpoint-forces` (variational ½(V̂S+SV̂) +U scatter; full-6N FD
  dual gate; 6 ATV-self +U pins re-pinned), `dfpt-gga-spin-si` (joint
  collinear-spin × PBE Sternheimer, CP2K UKS-PBE vib anchor, ratio 0.20),
  `gpw-bomd-kpoints` (previous-step \(P_k\) warm start; CP2K Quickstep AIMD
  k-mesh pin, ratio 0.064), `gpw-multigrid-kpoints` (Bloch-phased multigrid
  product collocation; CP2K pin \|ΔE\| 4.7e-7 Ha, ratio 1.6; also replaced
  the `gpw-kpoints-si` self-golden A7 pin with the genuine CP2K twin),
  `xl-bomd-linearized-uks` (zero-SCF dual-spin shadow forces; drift 1.8e-7).
  `gpw-d3-stress` stays `[~]` (`wf/gpw-d3-stress` @ `3b3872eb` — wip spec +
  CP2K NUMERICAL-stress pins committed, acceptance unfinished; do not
  demote). Specs closed (deleted at `8a9584f4`; INDEX cross-names the batch
  "harness wave 13"). Merge gate green at close (scoped rebuild + loop-domain
  ctest 57/57). Roadmap: **no ○/✓ table flips**; residual prose closed the
  nine shipped rows (dft.md §8/§10/§12, index GPW line, MD AIMD BOMD/XL
  lines). Shift Recover / GapCensus wave 15 / Assign wave 1 narratives live
  in the queue preamble (not logged here at the time). Pending **7**;
  incomplete **1**; blocked **6**. Clearup already ran (merged `wf/*` pruned;
  no isolation worktrees; no loop builds). Queue **not** exhausted. No push.
- **2026-07-26 harness wave 14 (parent `dev` @ `dcf03d6d`):** the wave was
  **interrupted during the Workers panel** and adjudicated on resume. Merged →
  done **5**: `cpmd-kpoints`, `periodic-wannier-gpw` (both complete worker tips,
  rebased and merged `--no-ff`), plus `gpw-d3-forces`, `gpw-d3bj-forces`,
  `gpw-mbd-stress` finished here. `gpw-fd-phonon-si` stays `[~]`: wiring + CP2K
  pin merged, and its **block reason was found to be wrong** — GPW analytic
  forces match energy FD to 1.7e-6 Ha/Bohr; the "271×" came from forcing Si₂
  closed-shell (OT converges on a saddle → analytic 0.1308 vs FD residual
  7.5e-2, i.e. the worker's 0.131), from an atom at (0,0,0) under
  Martyna–Tuckerman boundaries (|Σ F| = 1.98 Ha/Bohr), and from comparing an
  isolated-cluster spectrum to a `FULLY_PERIODIC T` crystal pin. Demoted **4**
  unstarted promotions back to pending (`gpw-dftu-kpoint-forces`,
  `dfpt-gga-spin-si`, `gpw-bomd-kpoints`, `gpw-multigrid-kpoints`) — worktrees
  existed, implementations did not. **Product defects fixed** (each named in its
  commit): D3-zero pair cutoff 40 Bohr → CP2K's 2 × `R_CUTOFF` (settled against
  a dedicated periodic D3-zero CP2K run; five sites unified behind
  `kD3Cp2kPairCutoffBohr`); `GpwDft::build` now throws on off-centre
  nonperiodic systems instead of returning silently wrong forces; multi-k MBD
  stress throws instead of dropping `E_disp`; four architecture scrub gates that
  were already red on `dev` repaired. New offline CP2K 2026.1 pins taken this
  wave: isolated + periodic D3, periodic D3(BJ), periodic D3-zero, Si₂-triplet
  energy/force + vib. Pending after this wave **12**; blocked **6**. Clearup ran
  (8 isolation worktrees removed, merged `wf/*` pruned, builds wiped). Queue
  **not** exhausted. No push.
- **2026-07-26 Assign wave 14 (parent `dev` @ `b06568f2`):** recover
  incomplete=**3** (resumes `cpmd-kpoints` @ `3425c7c6`,
  `periodic-wannier-gpw` @ `5c59a97f`, `gpw-fd-phonon-si` @ `7c3e1712`
  — each behind=17 / ahead=1 vs `dev`; do not demote); CP2K specs=0;
  isolation worktrees=0. Staffed **10** = 3 incomplete resumes + 7
  pending → `[~]`: resumes + `gpw-d3-forces`, `gpw-d3bj-forces`,
  `gpw-mbd-stress`, `gpw-dftu-kpoint-forces`, `dfpt-gga-spin-si`,
  `gpw-bomd-kpoints`, `gpw-multigrid-kpoints`. Held pending **8**
  (D3/MBD stress duals, finite-q GGA DFPT, linearized XL UKS/k,
  D3/D4 ATM forces). Surface notes: D3 ∥ D3BJ forces careful
  dispersion; multigrid×k free of BOMD×k; incomplete CPMD×k coordinate
  with multi-k BOMD. Next: Worktrees / workers on `wf/<slug>` (resumes
  rebase onto `dev` first).
- **2026-07-26 GapCensus wave 14 (parent `dev` @ `cee90c43`):** recover
  incomplete=**3** (wave-12 residual `[~]` preserved; do not demote);
  CP2K specs on `dev`=0; isolation worktrees=0; builds=none. Prior pending
  **9** re-verified true; **+6** residuals → pending **15**
  (`gpw-multigrid-kpoints`, `xl-bomd-linearized-uks`,
  `xl-bomd-linearized-kpoints`, `gpw-d3-stress`, `gpw-d3-atm-forces`,
  `gpw-d4-atm-forces` + wave-13 nine); in_progress **3**; blocked **6**.
  No ○/✓ table flips; §8/§10–12 + MD AIMD residual prose + index GPW
  residual line refreshed. Next: Assign prefers incomplete resumes
  (merge-ready `cpmd-kpoints` / `periodic-wannier-gpw`; force-gate
  `gpw-fd-phonon-si`) then surface-disjoint pending (D3 force ∥ D3BJ
  force/stress careful dispersion; MBD stress ∥ MBD×k stress careful
  stress tree; multigrid×k free of BOMD×k; linearized XL UKS ∥ multi-k
  careful XL evaluator; ATM forces free of two-body D3 force dual).
- **2026-07-26 harness wave 12 (parent `dev` @ `4ab58f97`):** 6 merged →
  done: `xl-bomd-linearized-forces`, `second-gen-cpmd-kpoints`, `gpw-d3-atm`,
  `gpw-d4-atm`, `bands-spin-metallic-si`, `dos-spin-metallic-si`. Incomplete
  remaining **3** (`[~] cpmd-kpoints` @ `3425c7c6`; `[~] periodic-wannier-gpw`
  @ `5c59a97f`; `[~] gpw-fd-phonon-si` @ `7c3e1712` — not demoted); pending
  residual **9** (GapCensus wave 13 list — **not** queue exhausted); blocked
  **6**. Specs closed (deleted) for all 6. Roadmap: **no new ○→✓ table
  flips** (rows already ✓ or VV10 still ○/blocked); residual prose closed
  zero-SCF linearized XL, multi-k SGCP, D3/D4 ATM, collinear spin+metallic
  bands/DOS. Clearup already ran (no isolation worktrees; no builds).
  **Queue not exhausted** — next Recover incomplete then continue. No push.
- **2026-07-26 GapCensus wave 13 (parent `dev` @ `663f4804`):** recover
  incomplete=**9** (wave-12 `[~]` preserved; isolation worktrees=**9**);
  CP2K specs on `dev`=0; prior pending residual **0**; **+9** residuals →
  pending **9** (`gpw-d3-forces`, `gpw-d3bj-forces`, `gpw-d3bj-stress`,
  `gpw-mbd-stress`, `gpw-mbd-kpoint-stress`, `gpw-dftu-kpoint-forces`,
  `dfpt-gga-spin-si`, `dfpt-gga-finite-q-si`, `gpw-bomd-kpoints`);
  in_progress **9**; blocked **6**. No ○/✓ table flips; §8/§10–12 + MD
  AIMD residual prose + index GPW residual line refreshed. Next: Assign
  prefers incomplete resumes (wave-12 merge-ready clean tips first; dirty
  WIP in place) then surface-disjoint pending (D3 force ∥ D3BJ force/stress
  careful dispersion; MBD stress ∥ MBD×k stress careful stress tree; DFT+U
  k-forces free of DFPT; DFPT GGA×spin ∥ GGA×finite-q careful DFPT tree;
  multi-k BOMD free of multi-k CPMD/SGCP incomplete surface).
- **2026-07-26 Recover (parent `dev` @ `663f4804`):** incomplete_count=**9**
  (all wave-12 `[~]`); open CP2K specs on `dev`=**0** (PySCF deferred only:
  `periodic-rhf-gamma` / `rsh-eri-cam-b3lyp`); unmerged feature `wf/*`=**9**
  (plus blocked `wf/gpw-vv10-ice`); isolation worktrees=**9**; half-regs on
  worktrees only (none on `dev`). **6 clean ahead=1** feat tips ready for
  merge/dual-gate verify: `bands-spin-metallic-si`@`ea3ae031`,
  `dos-spin-metallic-si`@`c2402fc3`, `gpw-d3-atm`@`da371950`,
  `gpw-d4-atm`@`d0371a5b`, `xl-bomd-linearized-forces`@`21d0603e`,
  `cpmd-kpoints`@`3425c7c6`. **3 dirty ahead=0** WIP (uncommitted spec+impl):
  `second-gen-cpmd-kpoints`, `periodic-wannier-gpw`, `gpw-fd-phonon-si` @
  `cbe74bdf`. Queue `## in_progress` normalized with resume tips; pending
  residual still **0**; blocked **6** (incl. stock VV10). No invent. Next:
  Assign prefers incomplete resumes (dirty WIP first or merge-ready clean
  tips) → Worktrees.
- **2026-07-26 Assign wave 12 (parent `dev` @ `cbe74bdf`):** recover
  incomplete=**0**; CP2K specs=0; unmerged feature `wf/*`=0. Staffed
  **all 9** pending → `[~]` / `## in_progress` (no resumes; no held
  pending): `bands-spin-metallic-si`, `dos-spin-metallic-si`,
  `gpw-d3-atm`, `gpw-d4-atm`, `xl-bomd-linearized-forces`,
  `cpmd-kpoints`, `second-gen-cpmd-kpoints`, `periodic-wannier-gpw`,
  `gpw-fd-phonon-si`. Surface notes: bands∥DOS share `bands_dos`;
  D3∥D4 ATM share dispersion; CPMD∥SGCP×k share AIMD k-seed; XL
  linearized free of multi-k CPMD/SGCP; GPW Wannier free of GPW FD
  phonon. Pending residual **0** after staff (not queue exhausted).
  Next: Worktrees / workers on `wf/<slug>`.
- **2026-07-26 GapCensus wave 12 (parent `dev` @ `087df536`):** recover
  incomplete=**0**; CP2K specs on `dev`=0; isolation worktrees=0;
  builds=none. Prior pending residual **0**; **+9** residuals → pending
  **9** (`bands-spin-metallic-si`, `dos-spin-metallic-si`, `gpw-d3-atm`,
  `gpw-d4-atm`, `xl-bomd-linearized-forces`, `cpmd-kpoints`,
  `second-gen-cpmd-kpoints`, `periodic-wannier-gpw`, `gpw-fd-phonon-si`);
  in_progress **0**; blocked **6**. No ○/✓ table flips; §8/§10–12 + MD
  AIMD residual prose + index GPW residual line refreshed. Next: Assign
  surface-disjoint staff (spin-metal bands ∥ DOS careful PW bands_dos
  share; D3 ATM ∥ D4 ATM careful dispersion surface; linearized XL free
  of multi-k CPMD/SGCP; GPW MLWF free of FD-phonon GPW).
- **2026-07-26 harness wave 11 (parent `dev` @ `09298298`):** 9 merged →
  done: `ugw-uks-rpbe`, `gpw-ts-forces`, `gpw-xdm-forces`,
  `gpw-kpoint-uks-stress-gga`, `cpmd-electronic-thermostat`,
  `xl-bomd-uks-kpoints`, `bands-metallic-si`, `dos-spin-si`,
  `dfpt-finite-q-spin-smearing`. Incomplete remaining **0** (no `[~]`;
  stock VV10 is **blocked**); pending residual **0** (census list emptied
  — **not** queue exhausted); blocked **6**. Specs closed (deleted) for
  all 9. Roadmap: **no new ○→✓ table flips** (rows already ✓ or VV10
  still ○/blocked); residual prose closed UGW@UKS-RPBE, dedicated TS/XDM
  force pins, multi-k UKS stress GGA expand, CPMD electronic thermostat,
  multi-k UKS XL-BOMD, metallic bands, spin DOS, finite-q DFPT
  spin+smear. Clearup already ran (no isolation worktrees; no builds).
  **Queue not exhausted** — next wave GapCensus. No push.
- **2026-07-26 GapCensus wave 11 (parent `dev` @ `96a0a7d2`):** recover
  incomplete=**1** (`[~] gpw-vv10-ice` → **blocked**, no CP2K `TYPE
  VV10`); CP2K specs on `dev`=0; isolation worktrees=0; builds=none.
  Prior pending residual **1** (`ugw-uks-rpbe`) re-verified; **+8**
  residuals → pending **9** (`ugw-uks-rpbe`, `xl-bomd-uks-kpoints`,
  `bands-metallic-si`, `dos-spin-si`, `dfpt-finite-q-spin-smearing`,
  `gpw-kpoint-uks-stress-gga`, `cpmd-electronic-thermostat`,
  `gpw-ts-forces`, `gpw-xdm-forces`); in_progress **0**; blocked **6**.
  No ○/✓ table flips; §8/§10–12 + MD AIMD residual prose + index GPW
  residual line refreshed. Next: Assign surface-disjoint staff
  (UGW-RPBE free of XL-UKS-k; bands-metallic ∥ dos-spin careful PW
  bands_dos share; DFPT finite-q spin+smear free of GPW; k-UKS stress
  GGA careful stress tree; CPMD e-T free of XL; TS forces ∥ XDM forces
  careful dispersion surface).
- **2026-07-26 harness wave 10 (parent `dev` @ `06ecaa66`):** 9 merged →
  done: `dfpt-spin-smearing`, `ot-uks-dftu-smearing`,
  `second-gen-cpmd-uks-ot`, `xl-bomd-kpoints`, `bands-spin-si`,
  `dos-metallic-si`, `periodic-wannier-uks`, `berry-spin-si`,
  `gpw-mbd-kpoint-forces`. Incomplete remaining **1** (`[~] gpw-vv10-ice`
  on unmerged `wf/gpw-vv10-ice` — not demoted); pending residual **1**
  (`ugw-uks-rpbe` — not staffed; **not** queue exhausted); blocked **5**.
  Specs closed (deleted) for all 9. Roadmap: **no new ○→✓ table flips**
  (capability rows already ✓ or VV10 still ○); residual prose closed
  DFPT spin+smear, multi-k UKS OT+U+smear, SGCP UKS OT, multi-k XL-BOMD,
  spin bands, metallic DOS, spin MLWF, spin Berry, multi-k MBD forces;
  UGW@UKS-RPBE + stock VV10 stay open. Clearup already ran (no isolation
  worktrees; no builds). **Queue not exhausted** — next wave Recover
  incomplete then GapCensus. No push.
- **2026-07-26 GapCensus wave 10 (parent `dev` @ `94c2401d`):** recover
  incomplete=**2** (`[~] gpw-vv10-ice` on unmerged `wf/gpw-vv10-ice` @
  `2f1d90e2`; `[~] dfpt-spin-smearing` on unmerged `wf/dfpt-spin-smearing`
  @ `d7576dea` — not demoted); CP2K specs on `dev`=0; isolation
  worktrees=0; builds=none. Prior pending residual **0**; **+9**
  residuals → pending **9** (`ot-uks-dftu-smearing`,
  `second-gen-cpmd-uks-ot`, `xl-bomd-kpoints`, `bands-spin-si`,
  `dos-metallic-si`, `periodic-wannier-uks`, `berry-spin-si`,
  `gpw-mbd-kpoint-forces`, `ugw-uks-rpbe`); in_progress **2**; blocked
  **5**. No ○/✓ table flips; §8/§10–12 + MD AIMD residual prose + index
  GPW residual line refreshed. Next: Assign prefers incomplete first
  (DFPT spin+smear merge gate; VV10 docs/block) then surface-disjoint
  pending (OT+U+smear careful OT tree; SGCP UKS OT ∥ XL×k free of each
  other; bands-spin ∥ dos-metallic ∥ Wannier-uks ∥ Berry-spin careful
  PW response share; MBD×k forces free of UGW-RPBE).
- **2026-07-26 harness wave 9 (parent `dev` @ `dab2b190`):** 8 merged →
  done: `ot-uks-smearing`, `ot-uks-dftu`, `ot-dftu-smearing`,
  `gpw-d4-forces`, `gpw-mbd-forces`, `gpw-uks-multigrid`,
  `dfpt-finite-q-smearing`, `second-gen-cpmd-ot`. Incomplete remaining
  **2** (`[~] gpw-vv10-ice` on unmerged `wf/gpw-vv10-ice`;
  `[~] dfpt-spin-smearing` on unmerged `wf/dfpt-spin-smearing` @
  `d7576dea` — not demoted); pending residual **0** (landed list emptied
  — **not** queue exhausted); blocked **5**. Specs closed (deleted) for
  all 8. Roadmap: **no new ○→✓ table flips** (rows already ✓ or VV10
  still ○); residual prose closed UKS OT+smear, multi-k UKS OT+DFT+U,
  OT+DFT+U+smear, D4/MBD analytic forces, UKS multigrid, finite-q DFPT
  smear, SGCP OT corrector; DFPT spin+smear + stock VV10 stay open
  incomplete. Clearup already ran (no isolation worktrees; no builds).
  **Queue not exhausted** — next wave Recover incomplete then
  GapCensus. No push.
- **2026-07-26 GapCensus wave 9 (parent `dev` @ `07dd7bf3`):** recover
  incomplete=**1** (`[~] gpw-vv10-ice` on unmerged `wf/gpw-vv10-ice` —
  not demoted); CP2K specs on `dev`=0; isolation worktrees=0;
  builds=none. Prior pending residual **0**; **+9** residuals → pending
  **9** (`ot-uks-smearing`, `ot-uks-dftu`, `ot-dftu-smearing`,
  `gpw-d4-forces`, `gpw-mbd-forces`, `gpw-uks-multigrid`,
  `dfpt-spin-smearing`, `dfpt-finite-q-smearing`, `second-gen-cpmd-ot`);
  in_progress **1**; blocked **5**. No ○/✓ table flips; §8/§10–12 + MD
  AIMD residual prose + index GPW residual line refreshed. Next: Assign
  prefers incomplete first (VV10 docs/block) then surface-disjoint
  pending (OT family careful share; D4 forces ∥ MBD forces careful
  dispersion surface; UKS multigrid free of OT; DFPT spin+smear ∥
  finite-q+smear careful DFPT tree; SGCP OT free of GPW OT smear).
- **2026-07-26 harness wave 8 (parent `dev` @ `a6752916`):** 9 merged →
  done: `periodic-urpa-kpoints`, `ot-smearing`, `ot-uks-kpoints`,
  `ot-dftu-kpoints`, `gpw-d4-stress`, `gpw-vdw-scf-potential`,
  `dfpt-metallic-si`, `dfpt-spin-incommensurate`, `gpw-optb88-vdw`.
  Incomplete remaining **1** (`[~] gpw-vv10-ice` on unmerged
  `wf/gpw-vv10-ice`); pending residual **0** (census list emptied —
  **not** queue exhausted); blocked **5**. Specs closed (deleted) for
  all 9. Roadmap: **optB88-vdW ✓**; residual prose closed multi-k URPA,
  OT+smear, multi-k UKS OT, multi-k OT+DFT+U, D4 analytic stress,
  nonlocal vdW SCF pot/forces, metallic DFPT, spin-incommensurate DFPT;
  stock VV10 stays open. Clearup already ran (no isolation worktrees;
  no builds). **Queue not exhausted** — next wave Recover incomplete
  then GapCensus. No push.
- **2026-07-26 GapCensus wave 8 (parent `dev` @ `9c57a290`):** recover
  incomplete=**1** (`[~] gpw-vv10-ice` on unmerged `wf/gpw-vv10-ice` —
  not demoted); CP2K specs on `dev`=0; isolation worktrees=0;
  builds=none. Prior pending residual **0**; **+9** residuals → pending
  **9** (`periodic-urpa-kpoints`, `ot-smearing`, `ot-uks-kpoints`,
  `ot-dftu-kpoints`, `gpw-d4-stress`, `gpw-vdw-scf-potential`,
  `dfpt-metallic-si`, `dfpt-spin-incommensurate`, `gpw-optb88-vdw`);
  in_progress **1**; blocked **5**. No ○/✓ table flips; §8/§10–12 open
  residual prose + index GPW residual line refreshed. Next: Assign
  prefers incomplete first (VV10 docs/block) then surface-disjoint
  pending (OT family careful share; URPA ∥ DFPT free; D4 stress ∥ vdW
  SCF careful dispersion surface; optB88 after vdW depth).
- **2026-07-26 harness wave 7 (parent `dev` @ `8cb47664`):** 9 merged →
  done: `periodic-urpa-si`, `periodic-ugw-kpoints`, `gpw-uks-smearing`,
  `gpw-hirshfeld-uks`, `cpmd-uks`, `second-gen-cpmd-uks`, `ot-kpoints`,
  `ot-dftu`, `dfpt-spin-si`. Incomplete remaining **1** (`[~] gpw-vv10-ice`
  on unmerged `wf/gpw-vv10-ice`); pending residual **0** (census list
  emptied — **not** queue exhausted); blocked **5**. Specs closed
  (deleted) for all 9. Roadmap: **no new ○→✓ table flips**; residual
  prose closed URPA@UKS, multi-k UGW, GPW UKS smear, UKS Hirshfeld,
  CPMD/SGCP UKS, OT+k, OT+DFT+U, collinear-spin DFPT; stock VV10 stays
  open. Clearup already ran (no isolation worktrees; no builds).
  **Queue not exhausted** — next wave Recover incomplete then GapCensus.
  No push.
- **2026-07-26 GapCensus wave 7 (parent `dev` @ `9bdeddf1`):** recover
  incomplete=**1** (`[~] gpw-vv10-ice` on unmerged `wf/gpw-vv10-ice` —
  not demoted); CP2K specs on `dev`=0; isolation worktrees=0;
  builds=none. Prior pending **1** (`periodic-urpa-si`) re-verified;
  **+8** residuals → pending **9** (`periodic-urpa-si`,
  `periodic-ugw-kpoints`, `gpw-uks-smearing`, `gpw-hirshfeld-uks`,
  `cpmd-uks`, `second-gen-cpmd-uks`, `ot-kpoints`, `ot-dftu`,
  `dfpt-spin-si`); in_progress **1**; blocked **5**. No ○/✓ table flips;
  §8/§10 residual prose + index GPW residual line + MD AIMD UKS residual
  notes refreshed. Next: Assign prefers incomplete first (VV10
  docs/block resolution) then surface-disjoint pending (URPA ∥ multi-k
  UGW careful share with GW/RPA kernels; UKS smear ∥ UKS Hirshfeld free
  of each other; CPMD UKS ∥ SGCP UKS careful AIMD surface; OT+k ∥ OT+DFT+U
  careful OT tree share; DFPT-spin free of GPW).
- **2026-07-26 harness wave 6 (parent `dev` @ `0869833a`):** 9 merged →
  done: `gpw-metallic-smearing`, `gpw-xdm-ice`, `gpw-uks-rpbe`,
  `gpw-dftu-kpoint-stress`, `periodic-ump2-kpoints`, `periodic-ugw-si`,
  `gpw-hirshfeld-kpoints`, `dfpt-gga-si`, `xl-bomd-uks`. Incomplete
  remaining **1** (`[~] gpw-vv10-ice` on unmerged `wf/gpw-vv10-ice`);
  pending residual **1** (`periodic-urpa-si` — **not** queue exhausted);
  blocked **5**. Specs closed (deleted) for all 9. Roadmap: **XDM ✓**;
  residual prose closed GPW metallic smear, Γ-UKS RPBE, multi-k DFT+U
  stress, multi-k UMP2, UGW, multi-k Hirshfeld, DFPT-GGA, XL-BOMD UKS;
  stock VV10 / URPA stay open. Clearup already ran (no isolation
  worktrees; no builds). **Queue not exhausted** — next wave Recover
  incomplete then GapCensus. No push.
- **2026-07-26 GapCensus wave 6 (parent `dev` @ `5847a35a`):** recover
  incomplete=**2** (`[~] gpw-metallic-smearing`, `[~] gpw-vv10-ice` on
  unmerged `wf/*` — not demoted); CP2K specs on `dev`=0; isolation
  worktrees=0; builds=none. Prior pending **1** (`gpw-xdm-ice`) re-
  verified; **+8** residuals → pending **9** (`gpw-xdm-ice`,
  `gpw-uks-rpbe`, `gpw-dftu-kpoint-stress`, `periodic-ump2-kpoints`,
  `periodic-ugw-si`, `periodic-urpa-si`, `gpw-hirshfeld-kpoints`,
  `dfpt-gga-si`, `xl-bomd-uks`); in_progress **2**; blocked **5**. No
  ○/✓ table flips; §3/§8/§10–12 residual prose + index GPW residual line
  refreshed. Next: Assign prefers incomplete first (metallic merge gate;
  VV10 docs/block resolution) then surface-disjoint pending (one nonlocal
  XDM; UKS RPBE free; multi-k DFT+U stress ∥ multi-k UMP2 careful share
  with RI kernels; UGW/URPA after Γ UMP2 ✓ free of closed-shell GW/RPA
  surface; Hirshfeld×k ∥ DFPT-GGA ∥ XL-BOMD UKS free).
- **2026-07-26 harness wave 5 (parent `dev` @ `2d43d1c2`):** 6 merged →
  done: `gpw-dftu-kpoints`, `periodic-gw-finite-q`, `gpw-uks-revpbe`,
  `cpmd-rattle-water`, `periodic-ump2-si`, `ot-uks-o2`. Incomplete
  remaining **2** (`[~] gpw-metallic-smearing`, `[~] gpw-vv10-ice` on
  unmerged `wf/*`); pending residual **1** (`gpw-xdm-ice` — **not**
  queue exhausted); blocked **5**. Specs closed (deleted) for all 6.
  Roadmap: no new ○→✓ table flips; residual prose closed multi-k DFT+U,
  finite-q G0W0 \(W\), Γ-UKS revPBE, CPMD electronic RATTLE, UMP2,
  OT-UKS; XDM / stock VV10 / GPW metallic smear stay open. Clearup
  already ran (no isolation worktrees; no builds). **Queue not
  exhausted** — next wave Recover incomplete then GapCensus. No push.
- **2026-07-26 GapCensus wave 5 (parent `dev` @ `4437d80f`):** recover
  clean (incomplete=0, CP2K specs=0, `wf/*`=0, builds=none). Prior pending
  residual **0** (harness emptied list — not capability-closed). **+9**
  residuals → pending **9** (`gpw-dftu-kpoints`, `periodic-gw-finite-q`,
  `gpw-uks-revpbe`, `gpw-xdm-ice`, `gpw-vv10-ice`, `cpmd-rattle-water`,
  `gpw-metallic-smearing`, `periodic-ump2-si`, `ot-uks-o2`); in_progress
  **0**; blocked **5**. No ○/✓ table flips; §10 residual prose + index
  Cell multi-k UKS line refreshed (stress productized; multi-k DFT+U open).
  Next: Assign (prefer surface-disjoint: multi-k DFT+U ∥ finite-q GW
  careful share with RPA kernels ∥ one nonlocal (XDM or VV10)/wave ∥
  UKS revPBE ∥ CPMD RATTLE free of OT-UKS / UMP2).
- **2026-07-26 harness wave 4 (parent `dev` @ `85eaa2d9`):** 9 merged →
  done: `gpw-kpoint-uks-stress`, `periodic-gw-kpoints`, `gpw-vdw-df2-ice`,
  `gpw-rvv10-ice`, `gpw-mbd-periodic`, `pw-spin-gga`, `gpw-uks-blyp`,
  `periodic-rpa-finite-q`, `gpw-dftu-stress`. Incomplete remaining **0**;
  pending residual **0** (census list emptied — **not** queue exhausted);
  blocked **5**. Specs closed (deleted) for all 9. Roadmap: vdW-DF2 ✓
  (LMKLL), rVV10 ✓; multi-k G0W0 residual closed; periodic Ewald MBD,
  PW true spin-GGA, Γ-UKS BLYP, multi-k UKS stress, finite-q inter-k
  dRPA, DFT+U analytic stress productized. Clearup already ran (no
  isolation worktrees; no builds). **Queue not exhausted** — next wave
  GapCensus. No push.
- **2026-07-26 GapCensus wave 4 (parent `dev` @ `decf917f`):** recover
  clean (incomplete=0, CP2K specs=0, `wf/*`=0, builds=none). Prior pending
  residual **0** (harness emptied list — not capability-closed). **+9**
  residuals → pending **9** (`gpw-kpoint-uks-stress`, `periodic-gw-kpoints`,
  `gpw-vdw-df2-ice`, `gpw-rvv10-ice`, `gpw-mbd-periodic`, `pw-spin-gga`,
  `gpw-uks-blyp`, `periodic-rpa-finite-q`, `gpw-dftu-stress`); in_progress
  **0**; blocked **5**. No ○/✓ table flips; §8/§10 residual prose + index
  GPW residual line refreshed. Next: Assign (prefer surface-disjoint:
  stress/DFT+U ∥ nonlocal (one VdW slug/wave) ∥ multi-k GW/RPA careful
  surface share; PW spin-GGA ∥ GPW UKS BLYP free).
- **2026-07-25 harness wave 3 (parent `dev` @ `0a642122`):** 8 merged →
  done: `gpw-rpbe-water`, `gpw-kpoint-uks-forces`, `gpw-uks-stress`,
  `gpw-vdw-df-ice`, `gpw-mbd-hirshfeld`, `periodic-rpa-kpoints`,
  `periodic-gw-si`, `dfpt-incommensurate-si`. Incomplete remaining **0**;
  pending residual **0** (census list emptied — **not** queue exhausted);
  blocked **5**. Specs closed (deleted) for all 8. Roadmap: RPBE ✓,
  vdW-DF ✓ (DRSLL), Periodic GW ✓ (Γ spectral G0W0; multi-k residual);
  Hirshfeld-scaled MBD/TS; multi-k dRPA residual closed; Γ-UKS stress +
  multi-k UKS forces + incommensurate DFPT productized (multi-k UKS stress
  residual). Clearup already ran (no isolation worktrees; no builds).
  **Queue not exhausted** — next wave GapCensus. No push.
- **2026-07-25 GapCensus wave 3 (parent `dev` @ `def040bd`):** recover
  clean (incomplete=0, CP2K specs=0, `wf/*`=0, builds=none). Prior pending
  **3** re-verified true gaps; **+5** residuals (`gpw-kpoint-uks-forces`,
  `gpw-uks-stress`, `periodic-rpa-kpoints`, `gpw-mbd-hirshfeld`,
  `dfpt-incommensurate-si`) → pending **8**; in_progress **0**; blocked
  **5**. No ○/✓ table flips; open-follow-on / forces / stress / RPA / DFPT
  prose + index Cell “strain-FD” stale line refreshed. Next: Assign
  (prefer surface-disjoint staff: RPBE alone; k-UKS forces ∥ UKS stress
  careful surface share; multi-k RPA after Γ RPA ✓ free).
- **2026-07-25 harness wave 2 (parent `dev` @ `f5739506`):** 10 merged →
  done: `gpw-kpoint-forces`, `gpw-revpbe-si`, `gpw-kpoints-uks`,
  `gpw-uks-forces`, `gpw-kpoint-stress`, `gpw-mbd-ice`,
  `periodic-mp2-kpoints`, `periodic-rpa-si`, `dfpt-finite-q-si`,
  `gpw-variable-cell-molopt`. Incomplete remaining **0**; pending residual
  **3** (`gpw-rpbe-water`, `gpw-vdw-df-ice`, `periodic-gw-si`); blocked **5**.
  Specs closed (deleted) for all 10. Roadmap: revPBE ✓, free-atom MBD ✓,
  Periodic RPA ✓ (Γ dRPA@KS), finite-q DFPT productized; multi-k MP2 residual
  closed; GPW k-UKS / k-forces / k-stress / Γ-UKS forces / MOLOPT CELL_OPT
  prose. Clearup already ran (no isolation worktrees; no builds). **Queue not
  exhausted** — next wave GapCensus. No push.
- **2026-07-25 GapCensus wave 2 (parent `dev` @ `507bb5f8`):** recover found
  unmerged `wf/gpw-kpoint-forces` (ahead 1) → `## in_progress` `[~]`. Prior
  pending (−k-forces) re-verified true gaps; **+4** follow-ons
  (`dfpt-finite-q-si`, `periodic-mp2-kpoints`, `gpw-kpoint-stress`,
  `gpw-uks-forces`) → pending **12**; in_progress **1**; blocked **5**. No
  ○/✓ table flips; stale DFPT/k-point intro prose refreshed. Next: Assign
  (prefer resume k-forces + surface-disjoint pending).
- **2026-07-25 harness wave 1 (post GapCensus staff; parent `dev` @
  `83a100df`):** 9 merged → done: `gpw-blyp-water`, `gpw-uks-pbe`,
  `pw-uks-fe-smearing`, `pw-variable-cell-si`, `gpw-ts-ice`,
  `phonon-dispersion-si`, `periodic-wannier-si`, `second-gen-cpmd-water`,
  `periodic-ri-mp2-si`. Incomplete remaining **0**; pending residual **9**;
  blocked **5**. Specs closed (deleted) for all 9. Roadmap: TS ✓, Periodic
  MP2 ✓ (Γ RI product; multi-k residual), MLWF ✓, 2nd-gen CP ✓; phonon
  IFC-Fourier productized; GPW BLYP / UKS-PBE / Fe smear / PW CELL_OPT prose.
  Clearup already ran (no isolation worktrees; no builds). **Queue not
  exhausted** — next wave GapCensus. No push.
- **2026-07-25 harness wave 1 (claude, earlier staff batch):** 5 merged into
  `dev` (head `bd39507a`) → done: `gpw-kpoints-si` (MP k-point GPW RKS),
  `gpw-variable-cell-ice` (vs CP2K CELL_OPT), `dfpt-phonon-si` (Γ DFPT, PW),
  `berry-polarization` (KSV, PW), `periodic-mp2-si` (Γ GPW MP2 **foothold** —
  later closed by `periodic-ri-mp2-si`). Incomplete then: `xl-bomd-water` WIP
  (later merged). Merge gate: cp2k domain suite green. Clearup: 71 merged
  `wf/*` pruned. Specs closed for all 5; roadmap Berry ✓, k-points prose;
  MP2 ○ at that time.
- **2026-07-25 bootstrap (parent `dev` @ `5bdae27d`):** GapCensus recover
  clean (incomplete=0, CP2K specs=0, wf/*=0, builds=none). Pending **14**
  re-verified true gaps; done **30**; blocked **5**. No roadmap ○/✓ flips.
  Ready for Assign. Suggested staff: `gpw-blyp-water`, `pw-uks-fe-smearing`,
  `pw-variable-cell-si`, `gpw-ts-ice`, `phonon-dispersion-si`,
  `periodic-ri-mp2-si`, `second-gen-cpmd-water`, `gpw-variable-cell-molopt`
  (hold revPBE/RPBE / MBD/vdW-DF / RPA/GW for surface order).
- **2026-07-25 GapCensus wave 1 (parent `dev` @ `f6838676`):** recover clean.
  Prior pending **14** re-verified; **+4** follow-on slugs (`gpw-uks-pbe`,
  `gpw-kpoints-uks`, `gpw-kpoint-forces`, `periodic-wannier-si`) → pending
  **18**; done **30**; blocked **5**. No table ○/✓ flips; fixed stale
  `docs/roadmap/index.md` GPW blurb. Ready for Assign.
