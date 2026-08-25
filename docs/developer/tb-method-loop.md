# TB / SE method loop — worktree → merge → dev

Abstract pattern: [`oracle-capability-loop.md`](oracle-capability-loop.md).  
Scaffold skill: `.agents/skills/create-oracle-loop/`.

## Continuous discovery (binding)

**Every wave starts with GapCensus:** re-diff OpenMOPAC / DFTB+ / xtb surfaces +
`docs/roadmap/semiempirical.md` ○ vs ATV on `target_branch`, refill pending,
update roadmap. Foothold ≠ capability closed. Stop `queue_empty` only when
`gaps_remaining=0` — **not** after one seed batch and **not** because a harness
wave marked a few slugs done.

**Family-locked oracles (binding):**

| Family | Oracle |
| ------ | ------ |
| NDDO (AM1/PM*/RM1/…) | **OpenMOPAC only** |
| DFTB0/2/3 / spin / PBC / TD-DFTB | **DFTB+ only** |
| GFN-xTB | **xtb only** (not DFTB3 successor; staff late) |
| Unified API shape (no E pin) | SCINE Sparrow (architecture) |
| Evaluate/MD embed patterns | CP2K (architecture only) |

**Forbidden:** OpenMOPAC as DFTB reference; DFTB+ as PM6/PM7 reference;
treating GFN-xTB as DFTB3++.

## Pipeline

```text
parent (dev)
  recover incomplete
       │
  for each wave:
    GapCensus → Assign → Worktrees → Merge? → Clearup → Harness? → Tutorial?
```

| Role | `isolation_worktree` | Writes |
| ---- | -------------------- | ------ |
| Census / foreman / merge / harness | **false** (parent) | `dev` only |
| Feature worker | **true** | private worktree only until merge |

Workers must report `worktree_path`, `branch` (`wf/<slug>`), `head_sha`.

**Recover incomplete first** (`## in_progress` `[~]`, open SE/TB specs,
unmerged `wf/<slug>`). Spec is source of truth. Offline pins only
(`tools/tb_oracle/`); never live oracle binaries in ctest.

**Scope IN:** `cpu/semiempirical` — DFTB0/2/3 ladder, NDDO PM depth, late
GFN-xTB; static in-tree SK/NDDO tables; Evaluate + Semiempirical Component.

**Scope OUT:** Gaussian SCF (PySCF loop); GPW/PW product (CP2K loop — arch-only
here); classical electrochem MD (MetalWalls); CUDA product; GFN-FF classical;
using OpenMOPAC to pin DFTB or DFTB+ to pin PM6.

**Architecture binding:** no public `*Config`; no `make_*` / `build_*` factories;
Frame is sole I/O; parameters are members / ctor args; static tables only.

## Assign ladder (binding)

DFTB0 → DFTB2 depth (analytic F, pins, 3ob) → DFTB3 → spin-DFTB → NDDO PM
ladder → **GFN-xTB last**.

## Args

| Arg | Default | Meaning |
| --- | ------- | ------- |
| `staff_count` / `parallelism` | 3–5 | Concurrent worktrees |
| `max_waves` | 20 | Assignment rounds this shift |
| `target_branch` | `dev` | Merge destination |
| `do_perf` | false | Optional fair wall vs oracle |
| `mode` | `exhaust` | Until gaps gone / budget |

## Build (binding — SE-only)

```bash
bash -lc '
  module load buildenv-gcc/2026.03-mpich
  export OPENBLAS_ROOT="$OPENBLASDIR"
  export RUSTUP_TOOLCHAIN=stable-x86_64-unknown-linux-gnu
  export LD_LIBRARY_PATH="$LIBRARY_PATH:$LD_LIBRARY_PATH"
  cmake -B build -DCMAKE_BUILD_TYPE=Release \
        -DATV_BUILD_TESTS=ON -DATV_BUILD_REGRESSIONS=ON \
        -DATV_BUILD_SEMIEMPIRICAL=ON -DATV_BUILD_EVALUATE=ON \
        -DATV_BUILD_MD=OFF -DATV_BUILD_MC=OFF -DATV_BUILD_MINIMIZER=OFF \
        -DATV_BUILD_VIB=OFF -DATV_BUILD_PATH=OFF -DATV_BUILD_POTENTIALS=OFF \
        -DATV_BUILD_NEIGHBOR=OFF -DATV_BUILD_HF=OFF -DATV_BUILD_PW=OFF \
        -DATV_BUILD_GPW=OFF -DATV_BUILD_KSPACE=OFF -DATV_BUILD_CUDA=OFF \
        -DATV_BUILD_CLI=OFF -DBLA_VENDOR=OpenBLAS
  # SCOPED — never bare cmake --build build
  cmake --build build -j$(nproc) --target atomiverse <test_*> reg_<slug>
  ctest --test-dir build -R "^(test_...)$" --output-on-failure
  cmake --build build --target run_reg_<slug>
'
```

**Forbidden ON:** HF, PW, GPW, CUDA, MD, MC, MINIMIZER, VIB, PATH,
KSPACE, POTENTIALS, NEIGHBOR.

**Do NOT build** `reg_am1_water_bomd` / `reg_am1_water_minimize` (need MD /
MINIMIZER). Domain suite (merge gate only):
`ctest -R '^test_cpu_(am1|rm1|dftb2|xtb_gfn0|nddo_)'`. Full unfiltered ctest
never.

## Queue / handoff

- Queue: `docs/developer/tb-method-queue.md`
- Handoff: `docs/developer/tb-method-handoff.md`
- Roadmap: `docs/roadmap/semiempirical.md`
- Offline oracle: `tools/tb_oracle/README.md`

## DFTB impl template (conceptual)

```text
ParameterSet → SK → H/S → eigen → P → SCC(Hubbard, γ) → repulsive → E/F/stress
```

## Temps

`/tmp/atv-*` `/tmp/tb_oracle_*` `/tmp/mopac_*` `/tmp/dftbplus_*` `/tmp/xtb_*`

## Run log

(autonomous harness appends lines here)

- 2026-07-27 GapCensus wave-1: re-diff roadmap ○ + OpenMOPAC/DFTB+/xtb surfaces vs `cpu/semiempirical` on `dev` @ `d80fd953`; pending refilled Phase A–C (23 capability slugs); blocked=9; incomplete=0; gaps_remaining=23+9 process/capability open; foothold ≠ closed
- 2026-07-27 assign wave 1: dftb2-3ob-organic, dftb2-analytic-forces, spin-dftb2-o2 → staffed (plus other ladder tips on `wf/*` retained incomplete)
- 2026-07-27 wave 1 merge: three green HEADs → dev (`dftb2-3ob-organic`, `dftb2-analytic-forces`, `spin-dftb2-o2`); fixups for test gates / CMake registration; merge-failed=[] for those three
- 2026-07-27 harness wave 1: queue three merged→done [x]; keep other incomplete [~] (dftb0-h2, dftb2-dftbplus-energy-pin, dftb3-water-mio, dftb2-dispersion-d3, dftb2-periodic-gamma); specs closed (3 deleted); roadmap ○→✓ DFTB analytic gradient (restricted) + spin-polarized DFTB (collinear); incomplete_remaining=5; queue_remaining=15 (pending only); blocked=9; handoff written; **queue not exhausted**
- 2026-07-27 GapCensus wave-2: re-diff OpenMOPAC|DFTB+|xtb + roadmap ○ + `cpu/semiempirical` on `dev` @ `77d80486`; foothold≠closed → +5 deepen pending (`dftb2-analytic-forces-threeob`, `dftb2-spin-analytic-forces`, `dftb3-analytic-forces`, `dftb2-kpoints`, `rm1-water-mopac-energy`); pending=20; incomplete=5 (tips behind=12); blocked=9; roadmap prose refresh (no new ○→✓); `tools/tb_oracle/README.md` restored; **queue not exhausted**
- 2026-07-27 harness wave 2: after merge of `dftb2-dftbplus-energy-pin` + `dftb2-periodic-gamma` → done [x]; keep incomplete [~] (`dftb0-h2`, `dftb3-water-mio`, `dftb2-dispersion-d3`); specs closed (2 deleted); roadmap ○→✓ Periodic DFTB (Γ-point); incomplete_remaining=3; queue_remaining=20 (pending only); blocked=9; handoff rewritten; **queue not exhausted**
- 2026-07-27 GapCensus wave-3: re-diff OpenMOPAC|DFTB+|xtb + roadmap ○ + `cpu/semiempirical` on `dev`; promote three force/stress tips → `[~]`; +`dftb2-periodic-forces`; incomplete=6; pending=18; blocked=9; **queue not exhausted**
- 2026-07-27 assign wave 3 (parent `dev` @ `223dcd4c`): staff=8 — resume 6 (`dftb0-h2`, `dftb3-water-mio`, `dftb2-dispersion-d3`, `dftb2-analytic-forces-threeob`, `dftb2-spin-analytic-forces`, `dftb2-analytic-stress`) + new 2 (`dftb2-periodic-forces`, `dftb2-kpoints`); skip `dftb3-analytic-forces` (dep), `td-dftb-h2`, Phase B/C; incomplete_remaining=8; pending_remaining=16; blocked=9; **queue not exhausted**
- 2026-07-29 harness wave 3: all 8 assign-w3 staff merged → done [x] (`dftb0-h2`, `dftb3-water-mio`, `dftb2-dispersion-d3`, `dftb2-analytic-forces-threeob`, `dftb2-spin-analytic-forces`, `dftb2-analytic-stress`, `dftb2-periodic-forces`, `dftb2-kpoints`); final batch HEADs `c4369c06`/`f4c1f1fe`/`30672ccf`; specs closed (3 deleted); incomplete_remaining=0; queue_remaining=16 (pending only); blocked=9; roadmap prose: lattice-image PBC F + multi-k energy + ThreeOb analytic F shipped (table rows already ✓ for DFTB0/3/D3/stress/gradient/periodic); **queue not exhausted** — GapCensus wave 4 owns discovery
- 2026-07-29 GapCensus wave-4: re-diff OpenMOPAC|DFTB+|xtb + roadmap ○ + `cpu/semiempirical` on `dev`; incomplete=0; no TB tips; +4 Phase A deepens (multi-k F, periodic D3, ThreeOb stress, spin stress); pending=20→ staffed below; blocked=9; **queue not exhausted**
- 2026-07-29 assign wave 4 (parent `dev` @ `f8e8e432`): staff=8 all **new** — Phase A: `dftb3-analytic-forces`, `dftb2-multi-k-forces`, `dftb2-dispersion-d3-periodic`, `dftb2-analytic-stress-threeob`, `dftb2-spin-analytic-stress`, `td-dftb-h2`; Phase B seed: `am1-water-mopac-energy`, `rm1-water-mopac-energy`; skip PM3/6/7 closed-form F + Phase C GFN; incomplete_remaining=8; pending_remaining=12; blocked=9; **queue not exhausted**
- 2026-07-29 harness wave 4: **all 8** assign-w4 staff merged → done [x] (`dftb3-analytic-forces`, `dftb2-multi-k-forces`, `dftb2-dispersion-d3-periodic`, `dftb2-analytic-stress-threeob`, `dftb2-spin-analytic-stress`, `td-dftb-h2`, `am1-water-mopac-energy`, `rm1-water-mopac-energy`); 7 specs closed; 7 worktrees + branches cleared same wave; gate SE units 15/15 + SE regressions 26/26; roadmap ○→✓ DFTB3 analytic F, multi-k F/σ, periodic D3, ThreeOb/open-shell σ, TD-DFTB foothold, AM1/RM1 mopac pins; **no strain-FD product path left on DFTB2**; incomplete_remaining=0
- 2026-07-29 defects found wave 4: (1) **γ_AB dimensionally inconsistent** (Å² + eV⁻², `units::coulomb` missing; γ_AB(R→∞) 14.4× low) — **not fixed**, re-pins every polar MioSketch case, noted + slug `dftb-gamma-units-fix`; (2) self-image virial dropped from the Γ pair walk (14% σ_xx on a 2.8 Å C-chain) — **fixed**; (3) RM1 free-atom energies were AM1-derived (19 kcal/mol ΔHf error) — **fixed** with a `calpar` identity gate. Inherited red gate `test_no_factory_scrub` from the concurrent SCF/D4 session (`src/cpu/scf/dftd4.cpp build_eeq_system`), pre-dates all TB commits
- 2026-07-29 GapCensus wave-5: incomplete=0; pending=15 (+3 new: γ fix, DFTB3 periodic depth, sharper NDDO discriminator); blocked=9; **staff `dftb-gamma-units-fix` first** — every MioSketch pin is provisional until it lands; **queue not exhausted**
- 2026-08-25 GapCensus wave-6: re-diff OpenMOPAC|DFTB+|xtb + roadmap ○ + `cpu/semiempirical` on `dev` @ `cea5b9cc`; GFN0 toy deleted; GFN2 INDEX claims not on-tree; γ Elstner + DFTB3 Γ-PBC F already in ## done; +3 DFTB deepens; pending=18; incomplete=0; blocked=9; **queue not exhausted**
- 2026-08-25 assign wave 6 (parent `dev` @ `cea5b9cc`): staff=5 — `nddo-sharp-discriminator`, `pm3-organic-energy`, `nddo-closed-form-forces`, `dftb3-analytic-stress`, `dftb2-threeob-multi-k`; skip PM6/7 + Phase C GFN (ladder); incomplete_remaining=5; pending_remaining=13; blocked=9; **queue not exhausted**
- 2026-08-25 harness wave 6 (partial): `nddo-sharp-discriminator` merged → done (`591a6ba5`/`6f161e8d`); CO₂ ΔHf −81.5544 vs PARAM −81.55, r=1.17296 vs 1.173; CO −13.8686 / 1.13818; specs closed; worktree+branch cleared; remaining assign-w6 (pm3, closed-form F, DFTB3 stress, ThreeOb multi-k) not started → pending; **found:** SE-only cmake now needs `-DATV_BUILD_CLI=OFF` (CLI requires MD); incomplete=0; pending=17; blocked=9; **queue not exhausted**
- 2026-08-25 recover (parent `dev` @ `7a4e3047`): incomplete_count=**0**; `## in_progress` empty (no `[~]`); open SE/TB specs=0; TB `wf/*`=0; isolation worktrees=0; half regressions=0; GFN2 INDEX/CMake EXISTS stubs not Recover; no methods invented; pending=17; blocked=9; **queue not exhausted**
- 2026-08-25 GapCensus wave-7 (this-shift wave 1, parent `dev` @ `7a4e3047`): re-diff OpenMOPAC|DFTB+|xtb + roadmap ○ + `cpu/semiempirical`; incomplete=0; +9 foothold deepens (`dftb3-threeob`, `dftb3-open-shell`, `dftb3-dispersion-d3`, `dftb2-threeob-open-shell`, `dftb2-fermi-smearing`, `td-dftb-excited-forces`, `lc-dftb-ground`, `nddo-uhf`, `alpb-xtb`); pending=26; blocked `[!]`=5 (stale “9” corrected: Hessian/+U done, sTDA/GFN-H pending); roadmap ○ LC-DFTB + Fermi filling; **queue not exhausted**
- 2026-08-25 assign wave 7 (this-shift wave 1, parent `dev` @ `c05f6b45`, **theme=gfn**): staff=3 NEW Phase C energy — `gfn0-beyond-toy`, `gfn1-xtb-organic`, `gfn2-xtb-water`; skip DFTB/NDDO (theme) + GFN dependents (same-file); incomplete_remaining=3; pending_remaining=23; blocked=5; **queue not exhausted**
