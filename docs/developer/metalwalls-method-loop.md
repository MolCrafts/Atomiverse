# MetalWalls method loop — worktree → merge → dev

Abstract pattern: [`oracle-capability-loop.md`](oracle-capability-loop.md).  
Scaffold skill: `.agents/skills/create-oracle-loop/`.

## Continuous discovery (binding)

**Every wave starts with GapCensus:** re-diff MetalWalls capability map +
LAMMPS-overlap classical MD vs ATV, refill pending, update roadmap. Foothold
≠ map row closed. **Out of scope pending:** CUDA electrode, cross-language
product, full 2D-PBC product stack, PLUMED. Stop `queue_empty` only when
`gaps_remaining=0` — not after one seed batch.

## Pipeline

```text
parent (dev)
  recover incomplete
       │
  for each wave:
    GapCensus → Assign → Worktrees → Merge? → Clearup → Harness? → Tutorial?
```

Instance = template + CONFIG/`extra_law` (LAMMPS-first, MD-only cmake, arch rules)
in `.grok/workflows/metalwalls-method-loop.rhai`.

| Role | `isolation_worktree` | Writes |
| ---- | -------------------- | ------ |
| Census / foreman / merge / harness | **false** (parent) | `dev` only |
| Feature worker | **true** | private worktree only until merge |

Workers must report `worktree_path`, `branch` (`wf/<slug>`), `head_sha`.

**Recover incomplete first** (`## in_progress` `[~]`, open MD/electrochem specs,
unmerged `wf/<slug>`). Spec is source of truth. Offline LAMMPS / MetalWalls
pins only (`tools/metalwalls_oracle/`); never live oracle binaries in ctest.

**Scope:** classical electrochemical MD + MetalWalls map **E1–E12** +
LAMMPS-overlap classical MD/FF. **Not** SCF/post-HF, GPW/PW, or CUDA OpenMM
rewrite.

**Architecture binding:** electrodes / const-pot = **Components** on
`cpu::md::MD`; never a new Driver category; never charge solve inside
`KSpace*`. Map: `.claude/notes/metalwalls-capability-map.md`.

**Oracle priority:** if the algorithm also exists in LAMMPS, offline pin + fair
bench **must** use LAMMPS. MetalWalls is oracle only for MW-only physics.
`REFERENCE.md` must state `oracle: lammps|metalwalls|literature` and
`oracle_version`.

## Why not edit dev in parallel?

Parallel agents on one tree thrash shared MD / longrange / electrode headers and
leave a half-broken `dev`. Isolation = each feature lands as a branch; merge is
ordered and abortable per slug.

## Args

| Arg | Default | Meaning |
| --- | ------- | ------- |
| `staff_count` / `parallelism` | 10 | Concurrent worktrees (wave 1 staff batch) |
| `max_waves` | 20 | Assignment rounds this shift |
| `target_branch` | `dev` | Merge destination |
| `do_perf` | false | Optional fair wall vs LAMMPS/MW |
| `mode` | `exhaust` | Until queue empty / budget |

## Launch

```text
agent_budget: 256
args: { "mode":"exhaust", "staff_count":10, "max_waves":20, "target_branch":"dev", "do_perf":false }
```

`agent_budget` = LLM-turn payroll for the shift, **not** headcount.

## Build (binding — MD-only)

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
  cmake --build build -j$(nproc)
'
```

**Forbidden:** `ATV_BUILD_HF=ON`, `ATV_BUILD_PW=ON`, `ATV_BUILD_GPW=ON`,
`ATV_BUILD_CUDA=ON`.

## Queue

`docs/developer/metalwalls-method-queue.md`

## Run log

(autonomous harness appends lines here)

- 2026-07-23 assign wave 1: const-pot-bo-parallel-plate, freeze-electrode-atoms, const-pot-charge-observer, charging-work-energy, const-pot-md-nvt-short, gaussian-electrode-charge, slab-2d-ewald-foothold, external-efield-md, buckingham-pair-lammps, four-site-tip4p-foothold → in_progress staff=10; incomplete was 0
- 2026-07-23 wave 1 merge: all ten green HEADs → dev (const-pot-bo-parallel-plate, charging-work-energy, gaussian-electrode-charge, freeze-electrode-atoms, const-pot-charge-observer, slab-2d-ewald-foothold, external-efield-md, buckingham-pair-lammps, four-site-tip4p-foothold, const-pot-md-nvt-short); blocked=[]; merge-failed=[]
- 2026-07-23 harness wave 1: queue merged→done (10); incomplete remaining=0; roadmaps ○→✓ Buckingham (CPU), constant E-field MD (CPU), TIP4P M-site virtual-site foothold (CPU); specs closed (10); ratios freeze=3.98 slab=6.49 efield=0.60 tip4p=0.338 others perf_exempt; queue_remaining=14 (pending only); handoff rewritten
- 2026-07-24 assign wave 2: const-pot-cg-large-electrode, electrode-charge-predictor, finite-field-single-electrode, pim-induced-dipole-bulk, piston-electrode-pz, thomas-fermi-electrode, fumi-tosi-nacl, steele-wall-argon, born-repulsion-foothold, capacitance-charge-fluctuation → in_progress staff=10; is_resume=false all; skipped dipole-electrode-joint (dep PIM); incomplete_before=0; pending_remaining=4; blocked=4
- 2026-07-24 wave 2 merge: all ten green HEADs → dev (const-pot-cg-large-electrode, electrode-charge-predictor, finite-field-single-electrode, pim-induced-dipole-bulk, piston-electrode-pz, thomas-fermi-electrode, fumi-tosi-nacl, steele-wall-argon, born-repulsion-foothold, capacitance-charge-fluctuation); blocked=[]; merge-failed=[]
- 2026-07-24 harness wave 2: queue merged→done (10 wave-2); incomplete remaining=0; roadmaps ○→✓ PairBorn/Fumi–Tosi, Steele wall/lj1043, bulk PIM+Thole (CPU); electrode notes CG/ASPC/finite-field single/piston/TF/capacitance; specs closed (10); all wave-2 ratios perf_exempt; queue_remaining=6 (pending only); dipole-electrode-joint unblocked; handoff rewritten
- 2026-07-24 assign wave 3: dipole-electrode-joint, potential-profile-observer, oscillating-efield-md, virtual-site-multi-topology, constant-force-pull-ion, surface-tension-coupling → in_progress staff=6 (all remaining pending; staff default 10); is_resume=false all; incomplete_before=0; pending_remaining=0; blocked=4 skipped; new_assigned=6
- 2026-07-24 wave 3 merge: all six green HEADs → dev (constant-force-pull-ion, oscillating-efield-md, potential-profile-observer, surface-tension-coupling, virtual-site-multi-topology, dipole-electrode-joint); blocked=[]; merge-failed=[]
- 2026-07-24 harness wave 3: queue merged→done (6 wave-3); incomplete remaining=0; roadmaps ○→✓ SurfaceTensionNPT / oscillating efield / FixAddForce / VirtualSites multi-topology / DipoleElectrodeJoint / Ψ(z) observer; specs closed (6); ratios force-pull=0.12 osc-efield=2.12 surface-tension=2.88 vsite=0.084 Ψ(z)+dipole-joint perf_exempt; queue_remaining=0 (pending only); handoff rewritten; classical MW staff queue exhausted
- 2026-07-24 census wave 4: gap-census on `dev` — map ATV marks refreshed (foothold ≠ ✓); pending refilled **13** production-depth/residual rows (MaZe, electrode forces, global neutrality, Gaussian/Born-Buck NL, D-field, XL-dipoles, QEq, const-pot+TIP4P EDL, CQQ, elastance cond, TI-C, CV-SMD); blocked=4 preserved; incomplete=0; gaps_remaining=13; roadmap notes production-depth (no false ○→✓)
- 2026-07-27 assign wave 5: per-electrode-eta, const-charge-electrode, const-pot-charge-halo-mpi, maze-symplectic-charge-dynamics, constant-d-polarization-feedback, extended-lagrangian-dipoles, qeq-fluctuating-charge, pim-npt-polarizable-il, voronoi-volume-tf, const-pot-il-edl-box → in_progress staff=10; incomplete_before=0; pending_remaining=3 (nanoporous-carbon-const-pot, thermo-integration-capacitance, maze-molten-salt-al); blocked=4
- 2026-07-27 wave 5 merge: all ten green HEADs → dev via merge(wf/<slug>) (per-electrode-eta, const-charge-electrode, const-pot-charge-halo-mpi, maze-symplectic-charge-dynamics, constant-d-polarization-feedback, extended-lagrangian-dipoles, qeq-fluctuating-charge, pim-npt-polarizable-il, voronoi-volume-tf, const-pot-il-edl-box); blocked=[]; merge-failed=[]
- 2026-07-27 harness wave 5: queue merged→done (10 wave-5; done=46 total); incomplete remaining=0; roadmaps ○→✓ XL induced dipoles (`PimXlDipole`), fluctuating-charge/QEq (`cpu::md::QEq`); electrode notes per-electrode η / conq const-Q control / post-solve charge-halo refresh / symplectic MaZe / constant-D feedback / VoronoiVolume TF / PIM-NPT IL / IL EDL product box; specs closed (10); ratios qeq=0.45 others perf_exempt; queue_remaining=3 (pending only); handoff rewritten
