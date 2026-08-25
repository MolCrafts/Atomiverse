# SE / tight-binding method queue

Durable worklist for workflow `tb-method-loop`. Do not delete done rows.
Census rewrites **pending** from `docs/roadmap/semiempirical.md` ○ rows that
**OpenMOPAC / DFTB+ / xtb / literature** can oracle offline; preserve
done / blocked / in_progress.

**Build rule:** SE-only cmake (`SEMIEMPIRICAL` + `EVALUATE` + TESTS/REGRESSIONS
**ON**; HF/PW/GPW/MD/MC/MINIMIZER/VIB/PATH/KSPACE/CUDA **OFF**).
See `tb-method-loop.md`.

**Oracle rule (family-locked):**

| Family | Primary pin package |
| ------ | ------------------- |
| NDDO (MNDO/AM1/PM3/RM1/PM6/PM7) | **OpenMOPAC** — never DFTB+ |
| DFTB0 / DFTB2 / DFTB3 / spin / periodic DFTB | **DFTB+** — never OpenMOPAC |
| GFN-xTB | **xtb** — not a DFTB3 successor; schedule late |
| Unified API design (no energy pin) | SCINE Sparrow (architecture) |
| Driver embed / MD / periodic patterns | CP2K (architecture only) |

`REFERENCE.md` must state `oracle: mopac|dftb+|xtb|literature` and
`oracle_version`.

**Assign priority (binding ladder):** DFTB0 → DFTB2 depth → DFTB3 → spin DFTB
→ NDDO PM ladder → **only then** GFN-xTB. Do not staff GFN2 while DFTB0/3 and
analytic DFTB forces are still open unless operator `theme=` overrides.
**Recover incomplete `[~]` before new pending** (`incomplete ≫ pending`).

## GapCensus — 2026-07-29 wave-4 (parent `dev`)

**Branch:** `dev` @ `2cf4b935` (harness wave-3 close: 8 DFTB landings).
**Live SE/TB specs:** none. **TB `wf/*` tips / isolation worktrees:** none.
**incomplete `[~]`:** 0.

### Sources re-diffed

| Source | Finding |
| ------ | ------- |
| `docs/roadmap/semiempirical.md` ○ | PM3/6/7/OM3; closed-form NDDO multipole F; SE Hessian; noncollinear / +U / SOC; multi-k F/stress; periodic D3; DFTB3 analytic F; TD-DFTB; GFN1/2/IPEA/spin/PBC/analytic G/H; GFN-FF; sTDA |
| `include/atomiverse/cpu/semiempirical/` | Public: `AM1`/`RM1` (+evaluators), `DFTB0`, `DFTB2`, `DFTB3`, `XtbGfn0`. **No** `PM*` / GFN1/2 types |
| `SemiEmpiricalMethod` | **AM1\|RM1 only** — PM tables need enum + param rows |
| `basis_data/` | mio-style sketch H/C/N/O + mio-1-1 H–H pin + 3ob-3-1 CHNO |
| `NDDOSolver::compute_forces` | **fixed-P central FD** of E (product path ✓; closed-form multipole dI/dR still ○) |
| `DFTB2` on `dev` HEAD | Restricted MioSketch analytic F+stress; collinear open-shell analytic F; ThreeOb analytic F; Γ-PBC lattice-image F; multi-k **energy only** (F/stress throw); molecular D3(BJ) (`d3`); **no** `d3_periodic` on HEAD (dirty WIP ignored) |
| `DFTB3` | Energy + third-order SCC (Euler-correct \(V=\partial E/\partial\Delta q\)); forces = **re-SCC central FD** |
| `DFTB0` | Non-SCC energy + force skeleton; mio-1-1 H–H pin |
| `xtb_gfn0` | H/C/O toy — not production GFN0 / xtb |
| Regressions on `dev` | wave-1…3 set: `dftb0-h2`, `dftb3-water-mio`, D3, ThreeOb/spin F, stress, periodic F, kpoints, pins, AM1/RM1 footholds, `h2o_gfn0` toy |
| REFERENCE hygiene | Wave DFTB regs mostly have `oracle:`; `h2_dftb2` **no** REFERENCE; `h2o_gfn0` **no** REFERENCE; AM1/RM1 lack hard mopac E pin + often lack `oracle:` line |
| `tools/tb_oracle/` on `dev` | dftbplus twins for H₂/CH4/PBC/k/D3/DFTB3 notes; **no** `inputs/mopac/` or `inputs/xtb/` trees |
| OpenMOPAC surface | MNDO / AM1 / PM3 / RM1 / PM6 / PM7 / MNDO-d — **not** OM3, **not** DFTB |
| DFTB+ surface | non-SCC, SCC-DFTB2, DFTB3, spin, PBC/k, forces/stress, TD-DFTB, +U, SOC, dispersion, … |
| xtb surface | GFN0/1/2, IPEA, analytic G; GFN-FF classical (scope-out); sTDA needs GFN GS |

### Tree vs roadmap (wave-4)

| Capability | Roadmap | Tree on `dev` HEAD | Verdict |
| ---------- | ------- | ------------------ | ------- |
| AM1 / RM1 energy + ΔHf | ✓ | ship; no hard mopac E pin | deepen → mopac pins |
| NDDO nuclear forces (product) | ✓ | fixed-P central FD | usable; **not** closed-form multipole |
| Non-SCC DFTB0 | ✓ | `DFTB0` + pin | **landed** wave-3 |
| SCC-DFTB2 H/C/N/O E | ✓ | MioSketch + ThreeOb + mio-1-1 H–H pin | energy product path |
| DFTB analytic gradient | ✓ | closed-form restricted Mio + open-shell Mio + ThreeOb | product F path; multi-k F still open |
| DFTB analytic stress | ✓ (restricted) | closed-form virial **restricted Mio only** | deepen ThreeOb + open-shell stress |
| Spin-polarized DFTB | ✓ (collinear) | MioSketch dual αβ + analytic open-shell F | foothold; noncollinear blocked |
| Periodic DFTB Γ | ✓ | lattice-sum E + lattice-image F | **landed** |
| Multi-k DFTB | ✓ energy | Monkhorst–Pack energy (restricted Mio); F/stress throw | deepen **multi-k F/stress** |
| Dispersion D3 | ✓ molecular | `DFTB2::d3` molecular + analytic F | deepen **periodic D3** lattice sum |
| DFTB3 energy | ✓ | `DFTB3` + `reg_dftb3-water-mio` | energy foothold; analytic F still ○ |
| TD-DFTB | ○ | absent | **gap** |
| GFN0-xTB | ✓ toy | sketch H/C/O | toy → Phase C |
| GFN1/2 / IPEA / spin / PBC GFN | ○ | absent | **gaps** (late) |
| PM3 / PM6 / PM7 | ○ | no types | Phase B **oracle=mopac** |
| OM3 | ○ | no type | **blocked** |
| GFN-FF / sTDA / SE Hessian (VIB) | ○ | — | **blocked** (scope or SE-only cmake) |

### Foothold ≠ closed (wave-4)

- Multi-k energy ships; **analytic multi-k forces/stress** throw → **new**
  `dftb2-multi-k-forces`.
- Molecular D3 ships; **periodic lattice-sum D3** absent on HEAD → **new**
  `dftb2-dispersion-d3-periodic` (do not treat dirty working-tree WIP as landed).
- Analytic stress is restricted MioSketch only → deepen
  `dftb2-analytic-stress-threeob`, `dftb2-spin-analytic-stress`.
- DFTB3 energy ships; forces remain re-SCC FD → keep `dftb3-analytic-forces`.
- NDDO hard OpenMOPAC E pins and closed-form multipole F still open (Phase B).
- mopac/xtb oracle input trees still empty.

### gaps_remaining (wave-4 census snapshot)

**20** pending + **0** incomplete (`[~]`) + **9** blocked (scope-out /
forbidden included) → **nonzero; continue loop**. `new_slugs_added=4`
(multi-k F, periodic D3, ThreeOb stress, spin stress). Roadmap ○/✓ tables
aligned with wave-3 landings; prose notes multi-k F + periodic D3 as next
Phase A deepens. **Staff Phase A deepen before Phase B mopac.**

## GapCensus — 2026-07-27 wave-3 (parent `dev`) — historical

**Branch:** `dev` @ `cec4f6ae` (SE product surface unchanged since harness wave-2
`ddc1ccc1`; HEAD is concurrent esmm/driver land).

### Sources re-diffed

| Source | Finding |
| ------ | ------- |
| `docs/roadmap/semiempirical.md` ○ | PM3/6/7/OM3; closed-form NDDO multipole F; Hessian; DFTB0/3/D3/stress/TD/noncollinear/+U/SOC; GFN1/2/IPEA/spin/PBC/analytic G/H; GFN-FF; sTDA |
| `include/atomiverse/cpu/semiempirical/` | Public: `AM1`/`RM1` (+evaluators), `DFTB2`, `XtbGfn0`. **No** `DFTB0`/`DFTB3`/`PM*`/GFN1/2 types on `dev` |
| `SemiEmpiricalMethod` | **AM1\|RM1 only** — PM tables still need enum + param rows |
| `basis_data/` | mio-style sketch H/C/N/O + mio-1-1 H–H pin + 3ob-3-1 CHNO |
| `NDDOSolver::compute_forces` | **fixed-P central FD** of E (product path ✓; closed-form multipole dI/dR still ○) |
| `dftb2.cpp` forces on `dev` | Restricted MioSketch: **closed-form** HF+Pulay+γ+rep; open-shell Mio + all ThreeOb: **re-SCC FD** |
| Analytic F under Γ-PBC | Energy uses lattice image sums; `analytic_forces_restricted` pair walk is **min_image only** → deepen `dftb2-periodic-forces` |
| Spin path | Collinear `multiplicity` on MioSketch; ThreeOb rejects open shell |
| `xtb_gfn0` | H/C/O toy — not production GFN0 / xtb |
| Regressions on `dev` | `am1_*`, `rm1_water_energy`, `h2_dftb2`, `h2o_dftb2_mio`, `dftb2-3ob-organic`, `dftb2-analytic-forces`, `dftb2-dftbplus-energy-pin`, `dftb2-periodic-gamma`, `spin-dftb2-o2`, `h2o_gfn0` |
| REFERENCE hygiene | Wave-1/2 DFTB regs have `oracle:`; `h2_dftb2` **no** REFERENCE; `h2o_gfn0` **no** REFERENCE; AM1/RM1 lack hard mopac E pin + often lack `oracle:` line |
| `tools/tb_oracle/` on `dev` | `ch4_3ob_dftb2`, `h2_mio_scc`, `dftb2-periodic-gamma`, force notes; **no** `inputs/mopac/` or `inputs/xtb/` trees |
| OpenMOPAC surface | MNDO / AM1 / PM3 / RM1 / PM6 / PM7 / MNDO-d — **not** OM3, **not** DFTB |
| DFTB+ surface | non-SCC, SCC-DFTB2, DFTB3, spin, PBC/k, forces/stress, TD-DFTB, +U, SOC, dispersion, … |
| xtb surface | GFN0/1/2, IPEA, analytic G; GFN-FF classical (scope-out); sTDA needs GFN GS |
| Live SE specs on `dev` | **none** |
| TB isolation worktrees | **none** (Clearup clean; only unrelated `driver-api-02` tree) |
| Unmerged TB tips | **6** with commits ahead of `dev` (3 older ladder tips behind=41; 3 force/stress tips behind=25) |

### Tree vs roadmap (wave-3)

| Capability | Roadmap | Tree on `dev` | Verdict |
| ---------- | ------- | ------------- | ------- |
| AM1 / RM1 energy + ΔHf | ✓ | ship; no hard mopac E pin | deepen → mopac pins |
| NDDO nuclear forces (product) | ✓ | fixed-P central FD | usable; **not** closed-form multipole |
| SCC-DFTB2 H/C/N/O E | ✓ | MioSketch + ThreeOb + mio-1-1 H–H pin | energy product path |
| DFTB analytic gradient | ✓ (restricted) | closed-form on **restricted MioSketch only** | deepen ThreeOb + open-shell F (tips) |
| Spin-polarized DFTB | ✓ (collinear) | MioSketch dual αβ + Mulliken m | foothold; open-shell F tip; noncollinear blocked |
| Periodic DFTB | ✓ (Γ energy) | lattice-sum H⁰/S/γ/rep on MioSketch | **landed**; k-mesh + lattice-consistent F still open |
| DFTB analytic stress | ○ | absent on `dev` | tip `wf/dftb2-analytic-stress` |
| Non-SCC DFTB0 | ○ | tip only | **incomplete** |
| DFTB3 | ○ | tip only | **incomplete** |
| Dispersion D3 | ○ | tip only | **incomplete** |
| TD-DFTB | ○ | absent | **gap** |
| GFN0-xTB | ✓ toy | sketch H/C/O | toy → Phase C |
| GFN1/2 / IPEA / spin / PBC GFN | ○ | absent | **gaps** (late) |
| PM3 / PM6 / PM7 | ○ | no types | Phase B **oracle=mopac** |
| OM3 | ○ | no type | **blocked** |
| GFN-FF / sTDA / SE Hessian (VIB) | ○ | — | **blocked** (scope or SE-only cmake) |

### Foothold ≠ closed (wave-3)

- Three force/stress pending slugs now have unmerged tips → **promote to `[~]`**
  (`dftb2-analytic-forces-threeob`, `dftb2-spin-analytic-forces`,
  `dftb2-analytic-stress`); do **not** leave them as pending while tips exist.
- Γ-PBC energy ships with `with_forces=false` pin; analytic F pair geometry is
  min_image — lattice-image forces under PBC remain a product gap → **new**
  `dftb2-periodic-forces`.
- Older ladder tips (`dftb0-h2`, `dftb3-water-mio`, `dftb2-dispersion-d3`) still
  need **Recover/rebase** (behind=41) before merge.
- NDDO hard OpenMOPAC E pins and closed-form multipole F still open (Phase B).

### gaps_remaining (wave-3 census snapshot)

**18** pending + **6** incomplete (`[~]`) + **9** blocked (scope-out /
forbidden included) → **nonzero; continue loop**. `new_slugs_added=1`
(`dftb2-periodic-forces`). Roadmap ○/✓ **unchanged** this wave (no SE product
merge onto `dev` since wave-2). **Recover incomplete ≫ staff new pending.**

## GapCensus — 2026-07-27 wave-2 (parent `dev`) — historical

**Branch:** `dev` @ `77d80486` at census time (pre later lands; see wave-3).

### Sources re-diffed

| Source | Finding |
| ------ | ------- |
| `docs/roadmap/semiempirical.md` ○ | PM3/6/7/OM3; closed-form NDDO F; Hessian; DFTB0/3/D3/PBC/stress/TD/noncollinear/+U/SOC; GFN1/2/IPEA/spin/PBC/analytic G/H; GFN-FF; sTDA |
| `include/atomiverse/cpu/semiempirical/` | Public types: `AM1`/`RM1` (+ evaluators), `DFTB2`, `XtbGfn0`. **No** `DFTB0`/`DFTB3`/`PM3`/`PM6`/`PM7`/GFN1/2 on `dev` |
| `SemiEmpiricalMethod` | Enum is **AM1\|RM1 only** — PM tables need enum + param rows |
| `basis_data/` | `dftb_mio_tables.h` mio-**style** sketch H/C/N/O; `dftb_3ob_tables.h` 3ob-3-1 organic CHNO |
| `NDDOSolver::compute_forces` | **fixed-density central FD** of E (not multipole dI/dR closed-form) |
| `dftb2.cpp` forces | Restricted MioSketch: **closed-form** HF+Pulay+γ+rep; open-shell Mio: **re-SCC FD**; ThreeOb: **re-SCC FD** |
| Spin path | Collinear `multiplicity` on MioSketch only; ThreeOb rejects open shell |
| `xtb_gfn0` | H/C/O toy onsite+hop+rep — not production GFN0 / xtb |
| Regressions on `dev` | `am1_*`, `rm1_water_energy`, `h2_dftb2`, `h2o_dftb2_mio`, `dftb2-3ob-organic`, `dftb2-analytic-forces`, `spin-dftb2-o2`, `h2o_gfn0` |
| REFERENCE hygiene | Wave-1 regs have `oracle:`; `h2_dftb2` **no** REFERENCE; `h2o_gfn0` **no** REFERENCE; AM1/RM1 lack hard mopac E pin |
| `tools/tb_oracle/` | Family README + `inputs/dftbplus/ch4_3ob_dftb2` + force note on `dev`; mopac/xtb trees empty; more twins only on `wf/*` tips |
| OpenMOPAC surface | MNDO / AM1 / PM3 / RM1 / PM6 / PM7 / MNDO-d — **not** OM3, **not** DFTB |
| DFTB+ surface | non-SCC, SCC-DFTB2, DFTB3, spin, PBC/k, forces/stress, TD-DFTB, +U, SOC, dispersion, … |
| xtb surface | GFN0/1/2, IPEA, analytic G; GFN-FF classical (scope-out); sTDA needs GFN GS |
| Live SE specs on `dev` | **none** (wave-1 + wave-2 harness specs deleted) |
| TB isolation worktrees | **none** (Clearup done); metalwalls `wf/*` worktrees unrelated |
| Unmerged TB tips | **3** still ahead of `dev` by 1–2 commits, **behind ~39** after wave-2 land (need rebase before merge) |

### Tree vs roadmap (wave-2)

| Capability | Roadmap | Tree on `dev` | Verdict |
| ---------- | ------- | ------------- | ------- |
| AM1 / RM1 energy + ΔHf | ✓ | ship; no hard mopac E pin | deepen → mopac pins |
| NDDO nuclear forces (product) | ✓ | fixed-P central FD | usable; **not** closed-form multipole |
| SCC-DFTB2 H/C/N/O E | ✓ | MioSketch + ThreeOb + **mio-1-1 H–H pin** | energy product path; H–H absolute pin **done** wave-2 |
| DFTB analytic gradient | ✓ (restricted) | closed-form on **restricted MioSketch only** | deepen ThreeOb + open-shell F |
| Spin-polarized DFTB | ✓ (collinear) | MioSketch dual αβ + Mulliken m | foothold; open-shell F still FD; noncollinear blocked |
| Production 3ob organic SK | — (prose) | `DftbSkSet::ThreeOb` + CH4 pin | landed wave-1 (DFTB2 SK, not DFTB3) |
| Non-SCC DFTB0 | ✓ | `dftb0.h`; reads the transcribed mio-1-1 H–H tables | **landed** 2026-07-28 (`757155d0`+`2c1502b6`); pin = the DFTB+ 24.1 value, licensed by SCC vanishing for homonuclear H₂ |
| DFTB3 | ✓ | `dftb3.h` + `internal/dftb3_scc.h` | **landed** 2026-07-28 (`68df18a8`..`148a17ed`); third-order potential fixed to ∂E³/∂Δq (Euler 4.4e-16), SCC now converges (Anderson mixing) |
| Dispersion D3 | ✓ | D3(BJ) applied in `DFTB2::launch` with its own analytic gradient | **landed** 2026-07-28 (`53ff52a0`); E_disp vs s-dftd3 1.5e-12 eV; absent from `kStress` (gap) |
| Periodic DFTB | ✓ (Γ) | lattice-sum H⁰/S/γ/rep on MioSketch | **landed** wave-2; k-mesh → `dftb2-kpoints` |
| DFTB analytic stress | ✓ | closed-form virial, restricted MioSketch | **landed** 2026-07-28 (`24d43574`); analytic vs strain-FD 1.1e-10 eV/Å³ |
| TD-DFTB | ○ | absent | **gap** |
| GFN0-xTB | ✓ toy | sketch H/C/O | toy → Phase C deepen |
| GFN1/2 / IPEA / spin / PBC GFN | ○ | absent | **gaps** (late) |
| PM3 / PM6 / PM7 | ○ | no types | Phase B **oracle=mopac** |
| OM3 | ○ | no type | **blocked** |
| GFN-FF / sTDA / SE Hessian (VIB) | ○ | — | **blocked** (scope or SE-only cmake) |

### Foothold ≠ closed (new deepen slugs this wave)

Wave-1 lands closed restricted Mio forces + collinear spin **energy**/mulliken
gates + 3ob energy pin — **not** product force completeness:

- ThreeOb nuclear forces remain re-SCC FD → `dftb2-analytic-forces-threeob`
- Open-shell nuclear forces remain re-SCC FD → `dftb2-spin-analytic-forces`
- DFTB3 energy tip exists but forces product still open → `dftb3-analytic-forces`
- Γ-point tip exists; DFTB+ k-mesh surface still open → `dftb2-kpoints`
- RM1 ships without OpenMOPAC hard E pin → `rm1-water-mopac-energy`

### gaps_remaining (wave-2 census snapshot; post harness wave-2 land)

**20** pending + **3** incomplete (`[~]`) + **9** blocked (includes scope-out /
forbidden-pattern rows) → **nonzero; continue loop**. Never claim queue
exhausted. Wave-2 harness closed `dftb2-dftbplus-energy-pin` +
`dftb2-periodic-gamma` → done. Next: **Recover** the three remaining `[~]`
tips (rebase onto `dev`) before staffing new Phase A pending.

## GapCensus — 2026-07-27 wave-1 (parent `dev`) — historical

**Branch:** `dev` @ `d80fd953` at census time (pre wave-1 land). Post-harness
HEAD is later; see `## done` / handoff.

### Sources re-diffed

| Source | Finding |
| ------ | ------- |
| `docs/roadmap/semiempirical.md` ○ | NDDO PM6/PM7/OM3/Hessian; DFTB ladder beyond energy foothold; full GFN product |
| `include/atomiverse/cpu/semiempirical/` | Types: `AM1`/`RM1` + evaluators, `DFTB2`, `XtbGfn0` toy; **no** `DFTB0`/`DFTB3`/`PM6`/`PM7`/GFN2 (at census) |
| `basis_data/dftb_mio_tables.h` | mio-**style** sketch (not DFTB.org mio-1-1 bit-identical); H/C/N/O only |
| `src/.../nddo_solver.cpp` `compute_forces` | **fixed-density central FD** of E (not closed-form multipole dI/dR) |
| `src/.../dftb2.cpp` `with_forces` | was re-SCC central FD; **now** closed-form restricted (post wave-1) |
| `src/.../xtb_gfn0.cpp` | H/C/O onsite+hop+rep **sketch** — not production GFN0/xtb |
| Regressions | `am1_*`, `rm1_water_energy`, `h2_dftb2`, `h2o_dftb2_mio`, + wave-1: `dftb2-3ob-organic`, `dftb2-analytic-forces`, `spin-dftb2-o2` |
| `tools/tb_oracle/` | README family matrix; offline input twins growing on unmerged tips |
| OpenMOPAC surface | MNDO / AM1 / PM3 / RM1 / PM6 / PM7 / MNDO-d — **not** OM3, **not** DFTB |
| DFTB+ surface | non-SCC, SCC-DFTB2, DFTB3, spin, PBC/k, forces/stress, TD-DFTB, +U, SOC, … |
| xtb surface | GFN0/1/2, IPEA, analytic G; GFN-FF classical (scope-out); sTDA needs GFN GS |

### Tree vs roadmap (post harness wave-1 land on `dev`)

| Capability | Roadmap | Tree | Verdict |
| ---------- | ------- | ---- | ------- |
| AM1 / RM1 energy + ΔHf | ✓ | ship; REFERENCEs lack hard mopac E pin | energy foothold; deepen → mopac pin + closed-form F |
| NDDO nuclear forces | ✓ product path | fixed-P central FD in `NDDOSolver::compute_forces` | usable F; **not** closed-form multipole → deepen slug |
| SCC-DFTB2 H/C/N/O E | ✓ | `DFTB2` + sketch SK + **3ob-3-1** static tables | energy foothold; offline DFTB+ pin hygiene still open for mio sketch |
| DFTB analytic gradient | ✓ (restricted) | closed-form HF+Pulay+γ+rep on MioSketch closed shell | **landed** wave-1; open-shell still FD; stress still ○ |
| Spin-polarized DFTB | ✓ (collinear) | `multiplicity` / dual αβ / Mulliken m; `reg_spin-dftb2-o2` | **landed** wave-1 foothold; noncollinear blocked |
| Production 3ob organic SK | — | `DftbSkSet::ThreeOb` + `dftb_3ob_tables.h` + CH4 DFTB+ pin | **landed** wave-1 (DFTB2 SK fidelity, not DFTB3 product) |
| Non-SCC DFTB0 | ✓ | `cpu::semiempirical::DFTB0` on `dev` | **landed** 2026-07-28 |
| DFTB3 / D3 / PBC / stress / TD | ○ | absent on `dev` (tips on some `wf/*`) | **gaps** |
| GFN0-xTB | ✓ toy | sketch H/C/O | toy only → Phase C |
| GFN1 / GFN2 / IPEA / spin / PBC GFN | ○ | absent | **gaps** (late) |
| PM6 / PM7 / PM3 | ○ | no types | Phase B **oracle=mopac** |
| OM3 | ○ | no type | **blocked** — not OpenMOPAC; no primary pin |
| GFN-FF / sTDA / SE Hessian (VIB) | ○ | — | **blocked** (scope or SE-only cmake) |

### Oracle / pin hygiene (process debt — not product rows)

- REFERENCEs: wave-1 lands have `oracle:` lines; older footholds may still lack hard pins.
- `reg_h2_dftb2`, `reg_h2o_gfn0` hygiene / offline pins still deepen via pending slugs.
- SK mio tables are **not** interchangeable with DFTB.org mio-1-1 for pin identity
  (3ob path pins 3ob-3-1 under pure DFTB2 SCC).

### gaps_remaining (post harness wave-1)

**15** pending + **5** incomplete (`[~]` tips) + **9** blocked (nonzero → continue loop).
Census snapshot was wave-1 pre-land; **foothold ≠ closed** for remaining ladder.
**Never claim queue exhausted** — next GapCensus owns discovery.

## Recover — 2026-07-27 (harness wave-1)

| Source | Result |
| ------ | ------ |
| Queue greens merged | `dftb2-3ob-organic`, `dftb2-analytic-forces`, `spin-dftb2-o2` → **done** |
| Unmerged TB `wf/*` with commits ahead of `dev` | **5** → keep **`[~]`** (see `## in_progress`) |
| Isolation worktrees (TB) | none (Clearup already ran) |
| Live SE/TB specs after close | none (three wave-1 specs deleted) |

**Next:** Recover/Merge incomplete tips or GapCensus+Assign remaining pending ladder.

## Recover / harness — 2026-07-27 (harness wave-2)

| Source | Result |
| ------ | ------ |
| Queue greens merged | `dftb2-dftbplus-energy-pin`, `dftb2-periodic-gamma` → **done** |
| Unmerged TB `wf/*` with commits ahead of `dev` | **3** → keep **`[~]`** (`dftb0-h2`, `dftb3-water-mio`, `dftb2-dispersion-d3`) |
| Isolation worktrees (TB) | none (Clearup already ran) |
| Live SE/TB specs after close | none (two wave-2 specs deleted) |
| Pending | **20** (unchanged this harness; GapCensus owns refill) |
| Blocked | **9** |

**Next:** Recover remaining incomplete (rebase onto `dev`) before new pending staff;
GapCensus owns discovery — **queue not exhausted**.

## Recover note — 2026-07-27 (GapCensus wave-3)

| Source | Result |
| ------ | ------ |
| SE product on `dev` | Unchanged since harness wave-2 (no new SE merge) |
| Pending rewrite | Dropped 3 staffed force/stress slugs → `## in_progress`; added `dftb2-periodic-forces` |
| Incomplete TB `wf/*` | **6** (was 3): +threeob F, +spin F, +stress |
| Isolation worktrees (TB) | none |
| Live SE/TB specs on `dev` | none |
| Roadmap ○/✓ | prose deepen only (PBC forces); no ✓ flips |

**Next:** **Recover/rebase** the six `[~]` tips onto `dev` before staffing
remaining Phase A pending; mopac twins still empty for Phase B.

## Assign — 2026-07-27 wave 3 (parent `dev` @ `223dcd4c`)

**Priority:** incomplete `[~]` ≫ open SE specs (none) ≫ pending Phase A.
**Staff = 8** (cap). Ladder: DFTB0 / DFTB2 depth / DFTB3 energy before NDDO/GFN.

| # | Slug | Kind | Action |
| - | ---- | ---- | ------ |
| 1 | `dftb0-h2` | resume | Rebase `wf/dftb0-h2` (ahead=1 behind=45) → dual gate |
| 2 | `dftb3-water-mio` | resume | Rebase `wf/dftb3-water-mio` (ahead=1 behind=45) → dual gate |
| 3 | `dftb2-dispersion-d3` | resume | Rebase `wf/dftb2-dispersion-d3` (ahead=2 behind=45) → dual gate |
| 4 | `dftb2-analytic-forces-threeob` | resume | Rebase `wf/dftb2-analytic-forces-threeob` (ahead=1 behind=29) → dual gate |
| 5 | `dftb2-spin-analytic-forces` | resume | Rebase `wf/dftb2-spin-analytic-forces` (ahead=1 behind=29) → dual gate |
| 6 | `dftb2-analytic-stress` | resume | Rebase `wf/dftb2-analytic-stress` (ahead=1 behind=29) → dual gate |
| 7 | `dftb2-periodic-forces` | **new** | Promote from pending; lattice-image F under Γ-PBC |
| 8 | `dftb2-kpoints` | **new** | Promote from pending; k-mesh beyond Γ |

**Skipped this wave:** `dftb3-analytic-forces` (needs `dftb3-water-mio` land);
`td-dftb-h2` (ground-state depth first); all Phase B NDDO / Phase C GFN
(ladder); blocked (9).

**After staff:** incomplete `[~]` = **8**; pending = **16**; blocked = **9**.
Workers: isolation worktree on nobackup; branch `wf/<slug>`; **spec-first**
`.claude/specs/<slug>/`; rebase resumes onto current `dev` before impl;
SE-only cmake; offline DFTB+ pins only. **No push.**

## Recover / harness — 2026-07-29 (harness wave 3)

| Source | Result |
| ------ | ------ |
| Queue greens merged | **all 8** assign-w3 staff → **done** (`dftb0-h2`, `dftb3-water-mio`, `dftb2-dispersion-d3`, `dftb2-analytic-forces-threeob`, `dftb2-spin-analytic-forces`, `dftb2-analytic-stress`, `dftb2-periodic-forces`, `dftb2-kpoints`) |
| Final batch on `dev` | `c4369c06` dftb3 dual-gate; `f4c1f1fe` periodic-forces; `30672ccf` kpoints |
| Unmerged TB `wf/*` | **none** (Clearup already ran; no local TB tips) |
| Isolation worktrees (TB) | none |
| Live SE/TB specs after close | **none** (3 deleted: `dftb3-water-mio`, `dftb2-periodic-forces`, `dftb2-kpoints`) |
| Incomplete | **0** |
| Pending | **16** (Phase A: `dftb3-analytic-forces`, `td-dftb-h2`; Phase B NDDO×7; Phase C GFN×7) |
| Blocked | **9** |

**Next:** GapCensus wave 4 owns discovery (foothold ≠ closed: multi-k F,
periodic D3 lattice sum, DFTB3 analytic F, TD-DFTB, Phase B mopac, …).
**Queue not exhausted.**

## Assign — 2026-07-29 wave 4 (parent `dev` @ `f8e8e432`)

**Priority:** incomplete `[~]` (0) ≫ open SE specs (none) ≫ pending Phase A
then early Phase B mopac pins. **Staff = 8** (cap). No TB `wf/*` tips / no
isolation worktrees to recover. Ladder: DFTB3 F + DFTB2 depth before remaining
NDDO PM product / GFN.

| # | Slug | Kind | Action |
| - | ---- | ---- | ------ |
| 1 | `dftb3-analytic-forces` | **new** | Closed-form DFTB3 nuclear F; replace re-SCC FD; energy foothold landed |
| 2 | `dftb2-multi-k-forces` | **new** | Analytic multi-k F (+ stress); restricted Mio first; energy MP ships |
| 3 | `dftb2-dispersion-d3-periodic` | **new** | Γ-PBC lattice-sum D3(BJ) (`d3_periodic`); not a second molecular pin |
| 4 | `dftb2-analytic-stress-threeob` | **new** | Closed-form ThreeOb cell stress; F already analytic |
| 5 | `dftb2-spin-analytic-stress` | **new** | Collinear open-shell analytic stress (MioSketch); F already analytic |
| 6 | `td-dftb-h2` | **new** | LR-TD-DFTB excitation foothold on solid SCC-DFTB2 ground state |
| 7 | `am1-water-mopac-energy` | **new** | Hard OpenMOPAC E/ΔHf pin for H2O AM1; seed `tools/tb_oracle/inputs/mopac/` |
| 8 | `rm1-water-mopac-energy` | **new** | Hard OpenMOPAC E/ΔHf pin for H2O RM1 |

**Skipped this wave:** Phase B deeper (`nddo-closed-form-forces`, PM3/6/7 —
after mopac energy pins land); all Phase C GFN (ladder); blocked (9).

**After staff:** incomplete `[~]` = **8**; pending = **12**; blocked = **9**.
Workers: isolation worktree on nobackup; branch `wf/<slug>`; **spec-first**
`.claude/specs/<slug>/`; SE-only cmake; offline pins only (DFTB+ / mopac /
literature as family-locked). **No push.**

## Recover / harness — 2026-07-29 (harness wave 4)

| Source | Result |
| ------ | ------ |
| Queue greens merged | **all 8** assign-w4 staff → **done** (`dftb3-analytic-forces`, `dftb2-multi-k-forces`, `dftb2-dispersion-d3-periodic`, `dftb2-analytic-stress-threeob`, `dftb2-spin-analytic-stress`, `td-dftb-h2`, `am1-water-mopac-energy`, `rm1-water-mopac-energy`) |
| Unmerged TB `wf/*` | **none** (all 7 worker branches merged + deleted) |
| Isolation worktrees (TB) | **none** (removed same wave, per the Clearup rule) |
| Live SE/TB specs after close | **none** (7 deleted) |
| Incomplete | **0** |
| Pending | **12** (Phase B NDDO×5, Phase C GFN×7) + **3 new** from the wave's findings |
| Blocked | **9** |

### Defects found this wave

Three real ones, all surfaced by writing the gate *before* trusting the code:

1. **`γ_AB` is dimensionally inconsistent** (`dftb2.cpp` `gamma_ab`,
   `dftb3_scc.cpp` `Dftb3Scc::gamma`) — `r²` [Å²] added to `τ²` [eV⁻²], the
   Coulomb constant missing. γ_AA = U_A is accidentally right, which is why it
   survived; γ_AB(R→∞) → 1/R instead of k_e/R, **14.4× low**. Invisible because
   the only real-mio-1-1 system in tree is H₂ (Δq ≡ 0). **Not fixed** — it
   re-pins every polar MioSketch case. `.claude/notes/dftb-vs-dftbplus-cleanup.md`,
   new slug `dftb-gamma-units-fix` (Phase A head).
2. **Self-image virial dropped** — the Γ pair virial walked `a<b` only, so
   A–(A+T) image bonds (zero force, nonzero virial) were missing: **14% σ_xx
   error** on a 2.8 Å C-chain cell. **Fixed** in `dftb2-multi-k-forces`, carried
   onto the spin-resolved assembly at merge, gated by
   `PeriodicAnalyticStressMatchesStrainFdCChain`.
3. **RM1 free-atom energies were AM1-derived** — `eisol`/`aq` did not satisfy
   MOPAC's `calpar` identity, giving ΔHf(H₂O) = −38.82 vs −57.86 kcal/mol
   (19 kcal/mol, for a model whose published AUE is 5.0). **Fixed** in
   `rm1-water-mopac-energy` with a `calpar` identity gate over all rows.

### Inherited red gate (not TB)

`test_no_factory_scrub` is **red on `dev`** from `src/cpu/scf/dftd4.cpp`
`build_eeq_system` (a free `build_*` in `cpu/scf`, which admits none),
introduced by the concurrent SCF/D4 session at `3ee28243` — already red at
`bfe049f6`, before any TB commit, and no TB commit touches `src/cpu/scf/`.
Named here per the iron law; belongs to that loop's owner to fix.

## GapCensus — 2026-07-29 wave-5 (parent `dev`)

**Branch:** `dev` @ harness-wave-4 close. **Live SE/TB specs:** none.
**TB `wf/*` tips / isolation worktrees:** none. **incomplete `[~]`:** 0.

### Tree vs roadmap (wave-5)

| Capability | Roadmap | Tree on `dev` HEAD | Verdict |
| ---------- | ------- | ------------------ | ------- |
| DFTB2 analytic F | ✓ | restricted + open shell + ThreeOb + Γ-PBC images + **multi-k** | **closed** for MioSketch/3ob |
| DFTB2 analytic stress | ✓ | restricted + open shell + ThreeOb + multi-k + self-image virial; **no strain-FD left** | **closed** for MioSketch/3ob |
| DFTB3 | ✓ E + **analytic F** | molecular min-image only | deepen: **no PBC / k / stress / open shell / ThreeOb / D3 on DFTB3** |
| Dispersion D3 | ✓ molecular + **periodic** | `d3` + `d3_periodic` + `d3_cutoff`; F and σ gated | orthorhombic only; no ATM |
| TD-DFTB | ✓ foothold | `DFTB2::td_casida` energies + \(f\) | no excited-state F, no LC |
| Multi-k | ✓ E + F + σ | restricted + **collinear open shell** Mio | ThreeOb multi-k throw; no Fermi smearing |
| AM1 / RM1 | ✓ + **mopac pins** | absolute ΔHf pins; RM1 params repaired | closed-form multipole F still ○ |
| PM3 / PM6 / PM7 | ○ | no types | Phase B **oracle=mopac** |
| GFN1/2 / IPEA / PBC / spin | ○ | `XtbGfn0` toy only | Phase C **oracle=xtb** |
| **γ_AB units** | — | **defective** (see above) | **blocks pin fidelity of every polar MioSketch row** |

### gaps_remaining (wave-5 census snapshot)

**15** pending + **0** incomplete + **9** blocked → **nonzero; continue loop.**
`new_slugs_added=3` (γ fix, DFTB3 periodic depth, sharper NDDO discriminator).
**Staff `dftb-gamma-units-fix` first** — every MioSketch pin below it is
provisional until it lands.

## GapCensus — 2026-08-25 wave-6 (parent `dev`)

**Branch:** `dev` @ `cea5b9cc`. **Live SE/TB spec files:** none (INDEX still
lists a GFN2-CHON chain as "code-complete"; those spec paths and
`src/cpu/semiempirical/gfn2.cpp` are **absent** — a stale INDEX, not Recover).
**TB `wf/*` tips / isolation worktrees:** none. **incomplete `[~]`:** 0.
Parent working tree is dirty with concurrent non-TB work; census reads the
committed SE surface at HEAD.

### Sources re-diffed

| Source | Finding |
| ------ | ------- |
| `docs/roadmap/semiempirical.md` ○ | PM3/6/7/OM3; closed-form NDDO F; GFN1/2/IPEA/spin/PBC/analytic G/H; GFN-FF; sTDA; noncollinear / SOC |
| `include/atomiverse/cpu/semiempirical/` | Public: `AM1`/`RM1` (+evaluators), `DFTB0`, `DFTB2`, `DFTB3`, `TdDftbResult`. **No** `PM*` / `XtbGfn0` / `GFN1` / `GFN2` |
| `SemiEmpiricalMethod` | **AM1\|RM1 only** |
| `XtbGfn0` / `reg_h2o_gfn0` | **deleted** `bc14dabd` (hygiene; INDEX GFN2-05 said "toy GFN0 deleted; TB pause") |
| CMake | `if(EXISTS gfn2.cpp)` / `test_gfn2_params.cpp` stubs — sources missing |
| `NDDOSolver::compute_forces` | still **fixed-P central FD** |
| `DFTB2` | analytic F+σ on Mio (restricted, collinear, multi-k) and ThreeOb; ThreeOb **multi-k throws**; D3 molecular + orthorhombic periodic; Elstner γ |
| `DFTB3` | energy + analytic F + **Γ-PBC lattice-image F**; no k-mesh, no `with_stress`, no open shell, no ThreeOb, no D3 |
| `tools/tb_oracle/` | mopac AM1/RM1 water twins; dftbplus twins; **xtb 6.7.1 twins on disk** (`inputs/xtb/{h2o,ch4,nh3,ch3oh}_gfn2` + off-eq grad) but `*.out` is gitignored |
| OpenMOPAC surface | MNDO / AM1 / PM3 / RM1 / PM6 / PM7 / MNDO-d — **not** OM3, **not** DFTB |
| DFTB+ surface | DFTB0/2/3, spin, PBC/k, F/σ, TD-DFTB, +U, SOC, D3, … |
| xtb surface | GFN0/1/2, IPEA, analytic G; GFN-FF classical (scope-out) |

### Tree vs roadmap (wave-6)

| Capability | Roadmap | Tree on `dev` HEAD | Verdict |
| ---------- | ------- | ------------------ | ------- |
| AM1 / RM1 + mopac H₂O pins | ✓ | ship | deepen: CO₂/CO discriminator still ○ |
| NDDO closed-form F | ○ | fixed-P FD | **gap** |
| PM3 / PM6 / PM7 | ○ | no types | Phase B **oracle=mopac** |
| DFTB0 / DFTB2 E+F+σ / Γ / multi-k | ✓ | product | ThreeOb multi-k still throws |
| DFTB3 E+F + Γ-PBC F | ✓ | molecular + Γ F | deepen: **no k / stress / open shell / ThreeOb / D3** |
| γ_AB Elstner | ✓ | `internal/dftb_gamma.h` | **closed** (wave-5 head landed) |
| TD-DFTB | ✓ foothold | Casida + \(f\) | no excited-state F, no LC |
| GFN0-xTB | was ✓ toy | **absent** (deleted) | flip ○; Phase C first row |
| GFN1/2 | ○ | absent (INDEX claims are not on-tree) | **gaps** (late; ladder = NDDO first) |

### Foothold ≠ closed (wave-6)

- DFTB3 Γ-PBC F ships; **cell stress / k-mesh** still missing → new
  `dftb3-analytic-stress`, `dftb3-kpoints`.
- ThreeOb energy/F/σ ship; **multi-k throws** → new `dftb2-threeob-multi-k`.
- Water mopac pins ship; RM1 ΔHf(H₂O) is 0.06 kcal/mol from experiment —
  **does not discriminate the method** → keep `nddo-sharp-discriminator`
  (OpenMOPAC accuracy table: RM1 CO₂ error **+12.50** kcal/mol on
  ΔHf_exp = −94.05 → RM1 = **−81.55**).
- GFN2 CHON campaign never merged; do **not** Recover from missing specs.
  Staff GFN only after NDDO PM or `theme=gfn\|xtb`.

### gaps_remaining (wave-6 census snapshot)

**18** pending + **0** incomplete + **9** blocked → **nonzero; continue loop.**
`new_slugs_added=3` (`dftb3-analytic-stress`, `dftb3-kpoints`,
`dftb2-threeob-multi-k`). Roadmap: GFN0 ✓toy → ○ (deleted). **Queue not
exhausted.** Ladder this wave: **NDDO PM** (DFTB0/2/3 product + γ fix landed).

## Assign — 2026-08-25 wave 6 (parent `dev` @ `cea5b9cc`)

**Priority:** incomplete `[~]` (0) ≫ open SE specs (none) ≫ pending Phase B
NDDO then remaining DFTB3/ThreeOb deepens. **Staff = 5** (user cap 3–5).
No TB `wf/*` / no isolation worktrees to recover.

| # | Slug | Kind | Action |
| - | ---- | ---- | ------ |
| 1 | `nddo-sharp-discriminator` | **new** | RM1 CO₂ ΔHf vs OpenMOPAC accuracy table (−81.55 vs exp −94.05) |
| 2 | `pm3-organic-energy` | **new** | PM3 tables + `SemiEmpiricalMethod::PM3` + mopac pin |
| 3 | `nddo-closed-form-forces` | **new** | closed-form NDDO multipole dI/dR; replace fixed-P FD |
| 4 | `dftb3-analytic-stress` | **new** | DFTB3 closed-form cell virial (Γ); DFTB2 already has it |
| 5 | `dftb2-threeob-multi-k` | **new** | lift ThreeOb multi-k throw; Bloch F/σ like MioSketch |

**Skipped this wave:** `pm6-*` / `pm7-*` (after PM3 table pipeline);
`dftb3-kpoints` (after DFTB3 stress); all Phase C GFN (ladder; no `theme=gfn`);
blocked (9).

**After staff:** incomplete `[~]` = **5**; pending = **13**; blocked = **9**.
**Harness wave 6 (partial):** `nddo-sharp-discriminator` merged → done
(`6f161e8d`); remaining four staffed slugs were **not started** (no `wf/*`
tips) → return to pending. After harness: incomplete=0; pending=17;
blocked=9. **Queue not exhausted.**

## Recover — 2026-08-25 (parent `dev` @ `7a4e3047`)

| Source | Result |
| ------ | ------ |
| Queue `[~]` / `## in_progress` | **0** (already empty; leave empty) |
| Open SE/TB specs | **none** (live `.claude/specs/` is component-plugin, other loop) |
| Unmerged TB `wf/*` | **none** (no local/remote `wf/` tips) |
| Isolation worktrees | **none** (`git worktree list` = parent; `atv-wf/` empty) |
| Half regressions | **none** (no pending-slug dirs; every `regressions/semiempirical/*` has `reg_*.cpp` + `REFERENCE.md`) |
| GFN2 INDEX / `if(EXISTS gfn2.cpp)` | **not Recover** — never-landed campaign; Phase C pending only |
| Incomplete | **0** — pick ≤5 → **none**; do not invent methods |

**Found (not a slug):** parent tree has uncommitted SE comment/path edits deleting duplicate `dftb_mio_hh_sk.h` / `mio_hh_table.h` while HEAD still ships both copies — concurrent dirt, not `[~]`.

**Next:** GapCensus/Assign remaining pending (**17**). Do not promote pending to `[~]` from this recover.

## GapCensus — 2026-08-25 wave-7 (parent `dev`; this-shift wave 1)

**Branch:** `dev` @ `7a4e3047`. **Live SE/TB spec files:** none (INDEX GFN2-CHON
is stale — sources never landed). **TB `wf/*` tips / isolation worktrees:**
none. **incomplete `[~]`:** 0. Census reads the committed SE surface at HEAD;
parent working tree is dirty with concurrent non-TB work (ignored).

### Sources re-diffed

| Source | Finding |
| ------ | ------- |
| `docs/roadmap/semiempirical.md` ○ | PM3/6/7/OM3; closed-form NDDO F; GFN0/1/2/IPEA/spin/PBC/analytic G/H; GFN-FF; sTDA; noncollinear / SOC; DFTB3 k/σ/open-shell/3ob/D3; ThreeOb multi-k; TD excited F; LC-TD |
| `include/atomiverse/cpu/semiempirical/` | Public: `AM1`/`RM1` (+evaluators), `DFTB0`, `DFTB2`, `DFTB3`, `TdDftbResult`. **No** `PM*` / `MNDO` / `XtbGfn0` / `GFN1` / `GFN2` / LC-DFTB types |
| `SemiEmpiricalMethod` | **AM1\|RM1 only** |
| `NDDOSolver::run` | **closed-shell RHF only** — `n_alpha != n_beta` throws; OpenMOPAC UHF is a gap |
| `NDDOSolver::compute_forces` | still **fixed-P central FD** |
| `DFTB2` | analytic F+σ on Mio (restricted, collinear, multi-k) and ThreeOb Γ; ThreeOb **multi-k throws** and **open-shell/charge throws**; D3 molecular + orthorhombic periodic; Elstner γ; integer filling only (no Fermi); `td_casida` energies+\(f\), no excited F |
| `DFTB3` | energy + analytic F + **Γ-PBC lattice-image F**; **no** `k_mesh`, `with_stress`, multiplicity, `DftbSkSet::ThreeOb`, D3. Hubbard \(U^{\mathrm{d}}\) is a mio-order **sketch** |
| `dftb2.h` comments | **stale:** still claim multi-k open shell throws — `scc_energy_unrestricted_kmesh` ships (`dftb2-multi-k-open-shell` done) |
| `tools/tb_oracle/` | mopac AM1/RM1 water + RM1 CO₂/CO; dftbplus twins incl. `h2o_mio_scc`; xtb 6.7.1 GFN2 twins on disk (`*.out` gitignored) |
| OpenMOPAC surface | MNDO / AM1 / PM3 / RM1 / PM6 / PM7 / MNDO-d / PM6-D3H4 / UHF / COSMO — **not** OM3, **not** DFTB |
| DFTB+ surface | DFTB0/2/3, spin, PBC/k, F/σ, Fermi filling, TD-DFTB + excited F, LC-DFTB, +U, SOC, D3, 3ob-3-1 as **DFTB3** SK, … |
| xtb surface | GFN0/1/2, IPEA, analytic G, ALPB/GBSA solvent, PBC, spin; GFN-FF classical (scope-out) |
| CMake | `if(EXISTS gfn2.cpp)` / `test_gfn2_params.cpp` stubs — sources missing |

### Tree vs roadmap (wave-7)

| Capability | Roadmap | Tree on `dev` HEAD | Verdict |
| ---------- | ------- | ------------------ | ------- |
| AM1 / RM1 + mopac H₂O + CO₂/CO | ✓ | ship | deepen: closed-form F, UHF, PM* |
| NDDO closed-form F | ○ | fixed-P FD | **gap** |
| NDDO UHF | — | RHF-only throw | **gap** (OpenMOPAC UHF) |
| PM3 / PM6 / PM7 | ○ | no types | Phase B **oracle=mopac** |
| DFTB0 / DFTB2 E+F+σ / Γ / multi-k / open-shell k | ✓ | product | ThreeOb multi-k + ThreeOb open shell still throw; no Fermi |
| DFTB3 E+F + Γ-PBC F | ✓ | molecular + Γ F | deepen: **no k / stress / open shell / ThreeOb / D3** |
| γ_AB Elstner | ✓ | `internal/dftb_gamma.h` | **closed** |
| TD-DFTB | ✓ foothold | Casida + \(f\) | no excited-state F, no LC |
| LC-DFTB | (new ○) | absent | **gap** (DFTB+ range-separated) |
| Fermi filling | (new ○) | integer occ only | **gap** (DFTB+ `Filling`) |
| GFN0/1/2 / IPEA / PBC / spin / ALPB | ○ | absent | Phase C **oracle=xtb** |

### Foothold ≠ closed (wave-7)

- DFTB3 Γ-PBC F ships; **cell stress / k-mesh / collinear spin / 3ob SK / D3** still missing → keep `dftb3-analytic-stress`, `dftb3-kpoints`; **new** `dftb3-threeob`, `dftb3-open-shell`, `dftb3-dispersion-d3`.
- ThreeOb energy/F/σ ship; **multi-k throws** and **open-shell/charge throws** → keep `dftb2-threeob-multi-k`; **new** `dftb2-threeob-open-shell`.
- Multi-k energy/F/σ ship with **integer filling** → **new** `dftb2-fermi-smearing` (DFTB+ `Filling { Fermi {…} }`).
- TD-DFTB Casida +\(f\) is a foothold; DFTB+ has excited-state forces and LC-TD → **new** `td-dftb-excited-forces`, `lc-dftb-ground`.
- AM1/RM1 RHF + mopac pins ship; **UHF throws** → **new** `nddo-uhf`.
- xtb 6.7.1 ALPB/GBSA is a first-class solvent surface, not a GFN Hamiltonian → **new** `alpb-xtb` (after GFN2 energy).
- GFN2 CHON campaign never merged; do **not** Recover from missing specs. Staff GFN only after NDDO PM or `theme=gfn\|xtb`.
- **Not pending:** noncollinear / SOC / OM3 / +U deepen / GFN-FF / mopac-as-dftb-oracle (blocked or permanent). CUDA SE is this loop's scope-out.

### gaps_remaining (wave-7 census snapshot)

**26** pending + **0** incomplete + **5** blocked/permanent `[!]` → **nonzero; continue loop.**
`new_slugs_added=9`. Blocked count **corrected** from the stale “9”:
`se-analytic-hessian` and `dftb-plus-u` are **done**; `stda-xtb` /
`gfn-xtb-analytic-hessian` live under Phase C pending. Remaining `[!]`:
3 permanent + 2 science. Roadmap: add ○ LC-DFTB + Fermi filling; prose
fixes (open-shell strain-FD gone; mopac H₂O/CO₂ pins landed). **Queue not
exhausted.** Ladder this wave: **NDDO PM** then remaining DFTB3/ThreeOb
deepens; skip LC and Phase C GFN unless `theme=`.

## Assign — 2026-08-25 wave 7 (this-shift wave 1, parent `dev` @ `c05f6b45`)

**Priority:** incomplete `[~]` (0) ≫ open SE specs (none; component-plugin is
another loop) ≫ pending. **Staff = 3** (cap 5). Operator **`theme=gfn`**
overrides the NDDO-first ladder for **NEW** slugs only. No TB `wf/*` / no
isolation worktrees to recover. Re-diff at HEAD (census commit only since
wave-7) — SE product surface unchanged: still no `PM*` / `XtbGfn0` / `GFN1` /
`GFN2` types; `if(EXISTS gfn2.cpp)` stub still empty; xtb 6.7.1 GFN2 twins on
disk (`h2o`/`ch4`/`nh3`/`ch3oh` + off-eq grad).

Staff **Phase C energy footholds** in published order
(`gfn0-beyond-toy` → `gfn1` / `gfn2-water`). Skip same-file dependents this
wave (`gfn2-xtb-forces`, `alpb-xtb`, Hessian, sTDA, IPEA, PBC, spin) so three
workers do not collide on one Hamiltonian. Native static tables — do **not**
wrap `libxtb` (CMake `ATV_XTB_ROOT` is leftover from the never-landed CHON
campaign).

| # | Slug | Kind | Action |
| - | ---- | ---- | ------ |
| 1 | `gfn0-beyond-toy` | **new** | Production GFN0 (toy type deleted `bc14dabd`); multi-element E vs xtb `--gfn 0` |
| 2 | `gfn1-xtb-organic` | **new** | GFN1-xTB energy on small organic vs xtb `--gfn 1` |
| 3 | `gfn2-xtb-water` | **new** | Production GFN2-xTB energy; pin from on-disk xtb 6.7.1 `h2o_gfn2` (−5.070383095219 Eh) |

**Skipped this wave:** Phase A DFTB deepens + Phase B NDDO (`theme=gfn`);
`gfn2-xtb-forces` / `alpb-xtb` / `ipea-xtb-small` / `periodic-gfn-xtb` /
`spin-gfn-xtb` / `stda-xtb` / `gfn-xtb-analytic-hessian` (need a landed GFN
GS); blocked (5). Two staff slots left empty rather than parallelize
dependents onto `gfn2.cpp`.

**After staff:** incomplete `[~]` = **3**; pending = **23**; blocked = **5**.
Workers: isolation worktree on nobackup; branch `wf/<slug>`; **spec-first**
`.claude/specs/<slug>/`; SE-only cmake (incl. CLI=OFF); offline **xtb** pins
only. **No push.**

## Recover / harness — 2026-08-25 (harness wave 7 / this-shift wave 1)

| Source | Result |
| ------ | ------ |
| Queue greens merged | `gfn0-beyond-toy` (`07f5c188` / `0219fa95`), `gfn2-xtb-water` (`dee2e8ce` / `982027ea`) → **done** |
| Unmerged TB `wf/*` | **1** → keep **`[~]`** `gfn1-xtb-organic` @ `30510fc0` |
| Isolation worktrees (TB) | none (Clearup already ran) |
| Live SE/TB specs after close | **none** of these two (deleted); gfn1 spec lives on the unmerged tip |
| Incomplete | **1** |
| Pending | **23** (unchanged this harness; GapCensus owns refill) |
| Blocked | **5** |
| Roadmap | ○→✓ GFN0-xTB + GFN2-xTB energy (H/C/N/O); GFN1 stays ○ |

**Next:** Recover `gfn1-xtb-organic` before new pending. Foothold ≠ closed
(GFN F / ALPB / IPEA / PBC / spin / sTDA / Hessian). **Queue not exhausted.**

## GapCensus — 2026-08-25 wave-8 (this-shift wave 2, parent `dev`)

**Branch:** `dev` @ `446172b4` (harness wave-7 close: GFN0 + GFN2 energy).
**Live SE/TB spec files on `dev`:** none (gfn1 spec lives only on the unmerged
tip). **TB isolation worktrees:** none. **incomplete `[~]`:** 1
(`gfn1-xtb-organic`, `wf/gfn1-xtb-organic` @ `30510fc0`, ahead=1 behind=5).
Census reads the committed SE surface at HEAD; parent working tree is dirty
with concurrent non-TB work (ignored).

### Sources re-diffed

| Source | Finding |
| ------ | ------- |
| `docs/roadmap/semiempirical.md` ○ | PM3/6/7/OM3; closed-form NDDO F; NDDO UHF; GFN1/IPEA/spin/PBC/analytic G/H; GFN-FF; sTDA; ALPB; noncollinear / SOC; DFTB3 k/σ/open-shell/3ob/D3; ThreeOb multi-k / open shell; TD excited F; LC-DFTB; Fermi filling |
| `include/atomiverse/cpu/semiempirical/` | Public: `AM1`/`RM1` (+evaluators), `DFTB0`, `DFTB2`, `DFTB3`, `GFN0`, `GFN2`, `TdDftbResult`. **No** `PM*` / `MNDO` / `GFN1` / LC-DFTB types |
| `SemiEmpiricalMethod` | **AM1\|RM1 only** |
| `NDDOSolver::run` | **closed-shell RHF only** — `n_alpha != n_beta` throws |
| `NDDOSolver::compute_forces` | still **fixed-P central FD**; no COSMO |
| `DFTB2` | analytic F+σ on Mio (restricted, collinear, multi-k incl. open-shell k) and ThreeOb Γ; ThreeOb **multi-k throws** and **open-shell/charge throws**; D3 molecular + orthorhombic periodic; Elstner γ; integer filling only; `td_casida` energies+\(f\), no excited F; `u_eff` energy + **re-SCC FD** F (no analytic ∂E_U/∂R; stress/k throw) |
| `DFTB3` | energy + analytic F + Γ-PBC F; **no** `k_mesh`, `with_stress`, multiplicity, `DftbSkSet::ThreeOb`, D3 |
| `GFN0` | H/C/N/O energy vs xtb 6.7.1 methanol; `with_forces` = **central FD of E** (not xtb analytic G); odd-electron throws; no PBC/spin/solvent |
| `GFN2` | H/C/N/O energy vs xtb 6.7.1 H₂O (residual **1.45 meV** — missing quadrupole AES); `with_forces` **throws**; no PBC/spin/ALPB |
| `tools/tb_oracle/` | mopac AM1/RM1 water + RM1 CO₂/CO; dftbplus twins; xtb 6.7.1 `ch3oh_gfn0` + `h2o_gfn2` (+ extra GFN2 `*.out` gitignored). **No** GFN1 twin on `dev` (lives on `wf/gfn1-xtb-organic`). **No** PM3/6/7 mopac twins |
| OpenMOPAC surface | MNDO / AM1 / PM3 / RM1 / PM6 / PM7 / MNDO-d / PM6-D3H4 / UHF / **COSMO** — **not** OM3, **not** DFTB |
| DFTB+ surface | DFTB0/2/3, spin, PBC/k, F/σ, Fermi filling, TD-DFTB + excited F, LC-DFTB, +U, SOC, D3, **D4**, 3ob-3-1 as **DFTB3** SK, implicit solvent |
| xtb surface | GFN0/1/2, IPEA, analytic G, ALPB/GBSA, PBC, spin; GFN-FF classical (scope-out). g-xTB is a different binary — **not** xtb 6.7.1 |

### Tree vs roadmap (wave-8)

| Capability | Roadmap | Tree on `dev` HEAD | Verdict |
| ---------- | ------- | ------------------ | ------- |
| AM1 / RM1 + mopac H₂O + CO₂/CO | ✓ | ship | deepen: closed-form F, UHF, PM*, COSMO |
| NDDO closed-form F | ○ | fixed-P FD | **gap** |
| NDDO UHF | — | RHF-only throw | **gap** (OpenMOPAC UHF) |
| NDDO COSMO | (new ○) | absent | **gap** (OpenMOPAC COSMO) |
| PM3 / PM6 / PM7 | ○ | no types | Phase B **oracle=mopac** |
| DFTB0 / DFTB2 E+F+σ / Γ / multi-k / open-shell k | ✓ | product | ThreeOb multi-k + ThreeOb open shell still throw; no Fermi |
| DFTB3 E+F + Γ-PBC F | ✓ | molecular + Γ F | deepen: **no k / stress / open shell / ThreeOb / D3** |
| DFTB-D3 | ✓ | molecular + orthorhombic PBC | DFTB+ **D4** still absent |
| DFTB+U | ✓ energy | `u_eff` + FD F | deepen: **analytic F** |
| γ_AB Elstner | ✓ | `internal/dftb_gamma.h` | **closed** |
| TD-DFTB | ✓ foothold | Casida + \(f\) | no excited-state F, no LC |
| LC-DFTB / Fermi filling | ○ | absent / integer occ | **gaps** |
| GFN0-xTB | ✓ energy CHNO | `GFN0` + FD F | deepen: analytic G, beyond CHNO |
| GFN2-xTB | ✓ energy CHNO | `GFN2`; F throw; 1.45 meV AES hole | deepen: quadrupole AES, analytic F, beyond CHNO |
| GFN1 / IPEA / PBC / spin / ALPB | ○ | GFN1 = `[~]` only | Phase C **oracle=xtb** |

### Foothold ≠ closed (wave-8)

- GFN0/GFN2 **energy** shipped this shift — **not** the GFN product. GFN0 F is
  central FD; GFN2 F throws → keep `gfn2-xtb-forces`; **new** `gfn0-xtb-forces`.
- GFN2 vs xtb residual is the missing **quadrupole AES** (REFERENCE Gaps) →
  **new** `gfn2-quadrupole-aes`.
- Static tables are H/C/N/O only; xtb is all-element → **new**
  `gfn2-beyond-chno` (GFN0 same shape after GFN2).
- OpenMOPAC **COSMO** is first-class; AM1/RM1 have no continuum solvent →
  **new** `nddo-cosmo`.
- DFTB+U energy ships; F is re-SCC FD (design note: remaining depth is
  ordinary pending) → **new** `dftb-plus-u-forces`.
- DFTB+ **D4** (and MBD) is a first-class `Dispersion` path; ATV DFTB is
  D3(BJ) only → **new** `dftb-dispersion-d4`.
- Keep DFTB3 / ThreeOb / Fermi / TD-excited / LC / PM* / GFN dependents.
- **Not pending:** noncollinear / SOC / OM3 / GFN-FF / mopac-as-dftb-oracle
  (blocked or permanent). CUDA SE is this loop's scope-out. g-xTB is not
  xtb 6.7.1. DFTB+ GBSA/COSMO solvent noted, not opened (Fermi/LC/D4 first).

### gaps_remaining (wave-8 census snapshot)

**29** pending + **1** incomplete (`[~]`) + **5** blocked/permanent `[!]` →
**nonzero; continue loop.** `new_slugs_added=6`. Roadmap: add ○ NDDO COSMO +
DFTB-D4; GFN0/GFN2 stay ✓ energy (CHNO foothold); GFN1 stays ○.
**Queue not exhausted.** Ladder: **Recover `gfn1-xtb-organic` first**; without
`theme=gfn` staff Phase B NDDO / DFTB3 deepens, not GFN dependents.

## Assign — 2026-08-25 wave 8 (this-shift wave 2, parent `dev` @ `dd8b1459`)

**Priority:** incomplete `[~]` (1) ≫ open SE specs (none; component-plugin is
another loop) ≫ pending. **Staff = 4** (cap 5). Operator **`theme=gfn`**
overrides the NDDO-first ladder for **NEW** slugs only. Census refresh at
HEAD: SE product surface unchanged since wave-8 (`GFN0`/`GFN2` energy CHNO;
no `GFN1`/`PM*` on `dev`). **TB `wf/*`:** `wf/gfn1-xtb-organic` @ `30510fc0`
(ahead=1 behind=6). Isolation worktrees: none.

xtb 6.7.1 banner also lists **ddCOSMO / CPCM-X** solvent (Stahn 2023) — **new**
pending `cpcm-x-xtb` (after `alpb-xtb`; not staffed). Not a GFN Hamiltonian.

Staff **Recover GFN1** then Phase C deepens on **distinct files**:
GFN0 analytic G (`gfn0.cpp`), GFN2 quadrupole AES (`gfn2.cpp`), IPEA
(own type). Skip same-file GFN2 dependents (`gfn2-xtb-forces`, `alpb-xtb`,
beyond-CHNO) so AES can close the 1.45 meV H₂O hole **before** an xtb force
pin. Skip Hessian (needs forces + VIB=ON), sTDA (would hook GFN2 internals),
PBC/spin (would touch GFN0/2). Native static tables — do **not** wrap
`libxtb`. One staff slot left empty rather than collide on `gfn2.cpp`.

| # | Slug | Kind | Action |
| - | ---- | ---- | ------ |
| 1 | `gfn1-xtb-organic` | resume | Rebase `wf/gfn1-xtb-organic` (ahead=1 behind=6) onto `dev`; dual-gate CH4 vs xtb `--gfn 1` (−4.274232224875 Eh). Spec lives on the tip. |
| 2 | `gfn0-xtb-forces` | **new** | Replace GFN0 central-FD F with analytic G vs xtb `--gfn 0`; energy already ships |
| 3 | `gfn2-quadrupole-aes` | **new** | GFN2 quadrupole AES (closes 1.45 meV H₂O residual; unblocks later xtb F pin) |
| 4 | `ipea-xtb-small` | **new** | IPEA-xTB energy foothold vs xtb; own Hamiltonian / tables (after GFN2 energy) |

**Skipped this wave:** Phase A DFTB + Phase B NDDO (`theme=gfn`);
`gfn2-xtb-forces` / `alpb-xtb` / `gfn2-beyond-chno` (same-file as AES);
`gfn-xtb-analytic-hessian` / `stda-xtb` / `periodic-gfn-xtb` / `spin-gfn-xtb`
(need AES/F or would collide); `cpcm-x-xtb` (after ALPB); blocked (5).

**After staff:** incomplete `[~]` = **4**; pending = **27**; blocked = **5**.
Workers: isolation worktree on nobackup; branch `wf/<slug>`; **spec-first**
`.claude/specs/<slug>/`; resume rebases the existing spec; SE-only cmake
(incl. CLI=OFF); offline **xtb** pins only. **No push.**

## Recover / harness — 2026-08-25 (harness wave 8 / this-shift wave 2)

| Source | Result |
| ------ | ------ |
| Queue greens merged | `gfn1-xtb-organic` (`d214f365` / `924d1327`), `gfn0-xtb-forces` (`e870f86d` / `a420b3d6`), `gfn2-quadrupole-aes` (`a48f0ee3` / `e6a06026`) → **done** |
| Unmerged TB `wf/*` | **1** → keep **`[~]`** `ipea-xtb-small` @ `01275c77` (ahead=1 behind=7) |
| Isolation worktrees (TB) | none (Clearup already ran) |
| Live SE/TB specs after close | **none** of these three (deleted); ipea spec lives on the unmerged tip |
| Incomplete | **1** |
| Pending | **27** (unchanged this harness; GapCensus owns refill) |
| Blocked | **5** |
| Roadmap | GFN1-xTB already ✓ on merge; family GFN analytic gradient stays ○ (GFN0 G ships; GFN1/2 F throw); IPEA stays ○ |

**Next:** Recover `ipea-xtb-small` before new pending. Foothold ≠ closed
(GFN2 F / ALPB / CPCM-X / PBC / spin / sTDA / Hessian / beyond-CHNO).
**Queue not exhausted.**

## GapCensus — 2026-08-25 wave-9 (this-shift wave 3, parent `dev`)

**Branch:** `dev` @ `c5a0a93c` (harness wave-8 close: GFN1 energy + GFN0
analytic G + GFN2 quadrupole AES). **Live SE/TB spec files on `dev`:** none
(ipea spec lives only on the unmerged tip). **TB isolation worktrees:** none.
**incomplete `[~]`:** 1 (`ipea-xtb-small`, `wf/ipea-xtb-small` @ `01275c77`,
ahead=1 behind=9). Census reads the committed SE surface at HEAD.

### Sources re-diffed

| Source | Finding |
| ------ | ------- |
| `docs/roadmap/semiempirical.md` ○ | PM3/6/7/OM3; closed-form NDDO F; NDDO UHF; NDDO COSMO; GFN1/2 F; IPEA/spin/PBC/analytic G/H; GFN-FF; sTDA; ALPB; CPCM-X; noncollinear / SOC; DFTB3 k/σ/open-shell/3ob/D3; ThreeOb multi-k / open shell; TD excited F; LC-DFTB; Fermi filling; DFTB-D4 |
| `include/atomiverse/cpu/semiempirical/` | Public: `AM1`/`RM1` (+evaluators), `DFTB0`, `DFTB2`, `DFTB3`, `GFN0`, `GFN1`, `GFN2`, `TdDftbResult`. **No** `PM*` / `MNDO` / `IPEA` / LC-DFTB types on `dev` |
| `SemiEmpiricalMethod` | **AM1\|RM1 only** |
| `NDDOSolver::run` | **closed-shell RHF only** — `n_alpha != n_beta` throws (`nddo_solver.cpp:108`) |
| `NDDOSolver::compute_forces` | still **fixed-P central FD**; no COSMO; AM1 under PBC **unwraps** an isolated fragment (not OpenMOPAC Γ-PBC) |
| `DFTB2` | analytic F+σ on Mio (restricted, collinear, multi-k incl. open-shell k) and ThreeOb Γ **molecular cluster** (no lattice images); ThreeOb **multi-k throws** and **open-shell/charge throws**; D3 molecular + orthorhombic periodic; Elstner γ; integer filling only; `td_casida` energies+\(f\), no excited F; `u_eff` energy + **re-SCC FD** F (stress/k throw); **no GBSA/COSMO** |
| `DFTB3` | energy + analytic F + Γ-PBC F; **no** `k_mesh`, `with_stress`, multiplicity, `DftbSkSet::ThreeOb`, D3 |
| `GFN0` | H/C/N/O energy + **analytic G** vs xtb 6.7.1; odd-electron throws; D4 `periodic=false`; no PBC/spin/solvent |
| `GFN1` | H/C/N/O energy vs xtb 6.7.1 CH4 (**10.2 meV** residual — D3/intcut/mixer); `with_forces` **throws**; closed shell only |
| `GFN2` | H/C/N/O energy + quadrupole AES vs xtb 6.7.1 H₂O (26 µeV); `with_forces` **throws**; no PBC/spin/ALPB |
| `tools/tb_oracle/` | mopac AM1/RM1 water + RM1 CO₂/CO; dftbplus twins; xtb 6.7.1 GFN0 methanol + off-eq G, GFN1 CH4, GFN2 H₂O (+ extra GFN2 `*.out` gitignored). **No** IPEA twin on `dev` (lives on `wf/ipea-xtb-small`). **No** PM3/6/7 mopac twins. **No** GFN1 `--grad` twin |
| OpenMOPAC surface | MNDO / AM1 / PM3 / RM1 / PM6 / PM7 / MNDO-d / PM6-D3H4 / UHF / COSMO / **Γ-PBC 1D–3D** — **not** OM3, **not** DFTB |
| DFTB+ surface | DFTB0/2/3, spin, PBC/k, F/σ, Fermi filling, TD-DFTB + excited F, LC-DFTB, +U, SOC, D3, D4, MBD/TS, 3ob-3-1 as **DFTB3** SK, **implicit solvent (GBSA/…)** |
| xtb surface | GFN0/1/2, IPEA, analytic G, `--hess` (numerical H), ALPB/GBSA, `--cosmo` / CPCM-X, PBC, `--uhf` spin; GFN-FF classical (scope-out); PTB ≠ GFN (not opened). g-xTB is a different binary — **not** xtb 6.7.1 |

### Tree vs roadmap (wave-9)

| Capability | Roadmap | Tree on `dev` HEAD | Verdict |
| ---------- | ------- | ------------------ | ------- |
| AM1 / RM1 + mopac H₂O + CO₂/CO | ✓ | ship | deepen: closed-form F, UHF, PM*, COSMO, **Γ-PBC** |
| NDDO closed-form F | ○ | fixed-P FD | **gap** |
| NDDO UHF | — | RHF-only throw | **gap** (OpenMOPAC UHF) |
| NDDO COSMO | ○ | absent | **gap** |
| NDDO Γ-PBC | (new ○) | molecular unwrap under wrap | **gap** (OpenMOPAC Γ 1D–3D) |
| PM3 / PM6 / PM7 | ○ | no types | Phase B **oracle=mopac** |
| DFTB0 / DFTB2 E+F+σ / Γ / multi-k / open-shell k | ✓ | product | ThreeOb multi-k + ThreeOb open shell still throw; ThreeOb is **cluster** (no lattice images); no Fermi |
| DFTB3 E+F + Γ-PBC F | ✓ | molecular + Γ F | deepen: **no k / stress / open shell / ThreeOb / D3** |
| DFTB-D3 | ✓ | molecular + orthorhombic PBC | DFTB+ **D4** still absent |
| DFTB+U | ✓ energy | `u_eff` + FD F | deepen: **analytic F** |
| DFTB implicit solvent | (new ○) | absent | **gap** (DFTB+ GBSA) |
| γ_AB Elstner | ✓ | `internal/dftb_gamma.h` | **closed** |
| TD-DFTB | ✓ foothold | Casida + \(f\) | no excited-state F, no LC |
| LC-DFTB / Fermi filling | ○ | absent / integer occ | **gaps** |
| GFN0-xTB | ✓ energy + G CHNO | `GFN0` analytic G | deepen: Hessian, beyond CHNO, odd-e / PBC / solvent |
| GFN1-xTB | ✓ energy CHNO | `GFN1`; F throw; 10.2 meV | deepen: **analytic G** |
| GFN2-xTB | ✓ energy CHNO | `GFN2` + quadrupole AES; F throw | deepen: analytic F, beyond CHNO |
| GFN family analytic G | ○ | GFN0 G ships; GFN1/2 throw | **keep ○** |
| IPEA / PBC / spin / ALPB / CPCM-X / Hessian / sTDA | ○ | IPEA = `[~]` only | Phase C **oracle=xtb** |

### Foothold ≠ closed (wave-9)

- GFN1 **energy** shipped this shift — **not** GFN1 product. `with_forces`
  throws → **new** `gfn1-xtb-forces` (CH4 10.2 meV residual is the same kernel
  hole the force pin will force tight; no separate residual slug).
- GFN0 analytic G shipped — family Hessian still waits on GFN2 F → **new**
  `gfn0-xtb-analytic-hessian` (`cpu::vib::Hessian` + VIB=ON exception, same
  shape as `se-analytic-hessian`).
- OpenMOPAC **Γ-PBC** (1D/2D/3D) is first-class; ATV AM1 unwraps an isolated
  fragment under wrap → **new** `nddo-periodic-gamma`.
- DFTB+ **implicit solvent** (GBSA / related) is first-class; ATV DFTB is
  vacuum-only → **new** `dftb-gbsa` (after Fermi/LC/D4, not a GFN ALPB twin).
- Keep DFTB3 / ThreeOb / Fermi / TD-excited / LC / +U F / D4 / PM* / GFN2 F /
  ALPB / CPCM-X / PBC / spin / sTDA / beyond-CHNO.
- **Not pending:** noncollinear / SOC / OM3 / GFN-FF / mopac-as-dftb-oracle
  (blocked or permanent). CUDA SE is this loop's scope-out. PTB and g-xTB are
  not GFN-xTB / not xtb 6.7.1. DFTB+ MBD/TS after D4. PM6-D3H4 after PM6.
  REKS / NEGF / Ehrenfest / ChIMES / MOZYME stay unopened (not method-ladder).

### Found (not a slug)

`dftb2.h` still claims multi-k open shell throws (lines 53, 102, 180);
`scc_energy_unrestricted_kmesh` ships (`dftb2-multi-k-open-shell` done).
Stale comment — named, not a capability gap.

### gaps_remaining (wave-9 census snapshot)

**31** pending + **1** incomplete (`[~]`) + **5** blocked/permanent `[!]` →
**nonzero; continue loop.** `new_slugs_added=4`. Roadmap: add ○ NDDO Γ-PBC +
DFTB GBSA; GFN family analytic G stays ○ (GFN0 G ships; GFN1/2 throw).
**Queue not exhausted.** Ladder: **Recover `ipea-xtb-small` first**; without
`theme=gfn` staff Phase B NDDO / DFTB3 deepens, not GFN dependents.

## Assign — 2026-08-25 wave 9 (this-shift wave 3, parent `dev` @ `7bc52489`)

**Priority:** incomplete `[~]` (1) ≫ open SE specs (none; component-plugin is
another loop) ≫ pending. **Staff = 4** (cap 5). Operator **`theme=gfn`**
overrides the NDDO-first ladder for **NEW** slugs only. Census refresh at
HEAD: SE product surface unchanged since wave-9 (`GFN0` G + `GFN1`/`GFN2`
energy CHNO; no `IPEA` / `PM*` on `dev`). **TB `wf/*`:** `wf/ipea-xtb-small`
@ `01275c77` (ahead=1 behind=10). Isolation worktrees: none. No new pending
from this assign-time re-diff.

Staff **Recover IPEA** then Phase C deepens on **distinct files**:
GFN1 analytic G (`gfn1.cpp`), GFN2 analytic G (`gfn2.cpp`; AES is done),
GFN0 FD Hessian (`cpu::vib::Hessian`, VIB=ON exception — GFN0 G already
ships). Skip same-file GFN2 dependents (`alpb-xtb`, `gfn2-beyond-chno`,
sTDA) so F can land before solvent / extra-Z tables. Skip family Hessian
(needs GFN1/2 F), PBC/spin (would touch GFN0/1/2). Native static tables —
do **not** wrap `libxtb`. One staff slot left empty rather than collide
on `gfn2.cpp`.

| # | Slug | Kind | Action |
| - | ---- | ---- | ------ |
| 1 | `ipea-xtb-small` | resume | Rebase `wf/ipea-xtb-small` (ahead=1 behind=10) onto `dev`; dual-gate CH4 vs xtb IPEA-xTB (−4.670476737294 Eh). Spec lives on the tip. |
| 2 | `gfn1-xtb-forces` | **new** | Replace GFN1 `with_forces` throw with analytic G vs xtb `--gfn 1 --grad`; energy already ships (CH4 10.2 meV residual is the same kernel hole) |
| 3 | `gfn2-xtb-forces` | **new** | Replace GFN2 `with_forces` throw with analytic G vs xtb `--gfn 2 --grad`; on-disk `h2o_gfn2_offeq_grad` (−5.058625058504 Eh, ‖∇E‖=0.10186 Eh/a0) |
| 4 | `gfn0-xtb-analytic-hessian` | **new** | GFN0 FD Hessian via `cpu::vib::Hessian` (analytic G already ships); VIB=ON exception like `se-analytic-hessian` |

**Skipped this wave:** Phase A DFTB + Phase B NDDO (`theme=gfn`);
`alpb-xtb` / `gfn2-beyond-chno` / `stda-xtb` (same-file as GFN2 F);
`gfn-xtb-analytic-hessian` / `periodic-gfn-xtb` / `spin-gfn-xtb`
(need GFN1/2 F or would collide); `cpcm-x-xtb` (after ALPB); blocked (5).

**After staff:** incomplete `[~]` = **4**; pending = **28**; blocked = **5**.
Workers: isolation worktree on nobackup; branch `wf/<slug>`; **spec-first**
`.claude/specs/<slug>/`; resume rebases the existing spec; SE-only cmake
(incl. CLI=OFF; `gfn0-xtb-analytic-hessian` may add `ATV_BUILD_VIB=ON` only);
offline **xtb** pins only. **No push.**

## Recover / harness — 2026-08-25 (harness wave 9 / this-shift wave 3)

| Source | Result |
| ------ | ------ |
| Queue greens merged | `gfn1-xtb-forces` (`276c7353` / `53fe282b`), `gfn2-xtb-forces` (`b8e86743` / `37b9a008`), `gfn0-xtb-analytic-hessian` (`d4582d31` / `30a6be4c`) → **done** |
| Unmerged TB `wf/*` | **1** → keep **`[~]`** `ipea-xtb-small` @ `054f6c8e` (ahead=2 behind=6) |
| Isolation worktrees (TB) | none (Clearup already ran) |
| Live SE/TB specs after close | **none** of these three (deleted); ipea spec lives on the unmerged tip |
| Incomplete | **1** |
| Pending | **28** (unchanged this harness; GapCensus owns refill) |
| Blocked | **5** |
| Roadmap | ○→✓ **GFN-xTB analytic gradient** (GFN0/1/2 CHNO \(F=-\nabla E\)). Family Hessian stays ○ (GFN0 FD H ships; GFN1/2 H is `gfn-xtb-analytic-hessian`). IPEA stays ○ |

**Found (not a slug):** GFN2 AES is still a **post-SCC energy** (not in the Fock). Stretch \(F_{O,x}\) matches xtb; angular \(F_{H1,y}\) is the AES-in-Fock hole (`reg_gfn2-xtb-forces` Gaps). Next GapCensus may open `gfn2-aes-fock`.

**Next:** Recover `ipea-xtb-small` before new pending. Foothold ≠ closed
(ALPB / CPCM-X / PBC / spin / sTDA / family Hessian / beyond-CHNO / AES-in-Fock).
**Queue not exhausted.**

## GapCensus — 2026-08-25 wave-10 (this-shift wave 4, parent `dev`)

**Branch:** `dev` @ `68de136b` (harness wave-9 close: GFN1/2 analytic G +
GFN0 FD Hessian). **Live SE/TB spec files on `dev`:** none (ipea spec lives
only on the unmerged tip). **TB isolation worktrees:** none. **incomplete
`[~]`:** 1 (`ipea-xtb-small`, `wf/ipea-xtb-small` @ `054f6c8e`, ahead=2
behind=8). Census reads the committed SE surface at HEAD.

### Sources re-diffed

| Source | Finding |
| ------ | ------- |
| `docs/roadmap/semiempirical.md` ○ | PM3/6/7/OM3; closed-form NDDO F; NDDO UHF; NDDO COSMO; NDDO Γ-PBC; IPEA/spin/PBC GFN; family Hessian; GFN-FF; sTDA; ALPB; CPCM-X; noncollinear / SOC; DFTB3 k/σ/open-shell/3ob/D3; ThreeOb multi-k / open shell; TD excited F; LC-DFTB; Fermi filling; DFTB-D4; DFTB GBSA |
| `include/atomiverse/cpu/semiempirical/` | Public: `AM1`/`RM1` (+evaluators), `DFTB0`, `DFTB2`, `DFTB3`, `GFN0`, `GFN1`, `GFN2`, `TdDftbResult`. **No** `PM*` / `MNDO` / `IPEA` / LC-DFTB types on `dev` |
| `SemiEmpiricalMethod` | **AM1\|RM1 only** |
| `NDDOSolver::run` | **closed-shell RHF only** — `n_alpha != n_beta` throws (`nddo_solver.cpp:108`) |
| `NDDOSolver::compute_forces` | still **fixed-P central FD**; no COSMO; AM1 under PBC **unwraps** an isolated fragment (not OpenMOPAC Γ-PBC) |
| `DFTB2` | analytic F+σ on Mio (restricted, collinear, **multi-k open shell ships** via `scc_energy_unrestricted_kmesh`) and ThreeOb Γ **molecular cluster** (throw: "no lattice sum at all"); ThreeOb **multi-k throws** and **open-shell/charge throws**; D3 molecular + orthorhombic periodic; Elstner γ; integer filling only; `td_casida` energies+\(f\), no excited F; `u_eff` energy + **re-SCC FD** F (stress/k throw); **no GBSA** |
| `DFTB3` | energy + analytic F + Γ-PBC F; **no** `k_mesh`, `with_stress`, multiplicity, `DftbSkSet::ThreeOb`, D3 |
| `GFN0` | H/C/N/O energy + analytic G + FD Hessian vs xtb 6.7.1; odd-electron throws; D4 `periodic=false`; no PBC/spin/solvent |
| `GFN1` | H/C/N/O energy + **analytic G** vs xtb 6.7.1 (CH4 10.2 meV; off-eq water \(F_{O,x}\) 1.87e-3 eV/Å); closed shell only |
| `GFN2` | H/C/N/O energy + quadrupole AES + **analytic G** vs xtb 6.7.1; AES still **post-SCC** (angular \(F_{H1,y}\) Fock hole); no PBC/spin/ALPB |
| `tools/tb_oracle/` | mopac AM1/RM1 water + RM1 CO₂/CO; dftbplus twins; xtb 6.7.1 GFN0 methanol + off-eq G + Hessian, GFN1 CH4 + off-eq G, GFN2 H₂O + AES + off-eq G. **No** IPEA twin on `dev` (lives on `wf/ipea-xtb-small`). **No** PM3/6/7 mopac twins |
| OpenMOPAC surface | MNDO / AM1 / PM3 / RM1 / PM6 / PM7 / MNDO-d / PM6-D3H4 / UHF / COSMO / Γ-PBC 1D–3D — **not** OM3, **not** DFTB |
| DFTB+ surface | DFTB0/2/3, spin, PBC/k, F/σ, Fermi filling, TD-DFTB + excited F + NAC, LC-DFTB, +U, SOC, D3, D4, MBD/TS, 3ob-3-1 as **DFTB3** SK, implicit solvent, xTB Hamiltonian (**family-locked to xtb, not a DFTB pin**). Unopened: REKS, Δ-DFTB, Ehrenfest, NEGF, ChIMES, helical, f-electrons, pSIC |
| xtb surface | GFN0/1/2, IPEA, analytic G, `--hess`, ALPB/GBSA, `--cosmo` / CPCM-X, PBC, `--uhf` spin; GFN-FF classical (scope-out); PTB ≠ GFN; g-xTB is a **different binary** — **not** xtb 6.7.1 |

### Tree vs roadmap (wave-10)

| Capability | Roadmap | Tree on `dev` HEAD | Verdict |
| ---------- | ------- | ------------------ | ------- |
| AM1 / RM1 + mopac H₂O + CO₂/CO | ✓ | ship | deepen: closed-form F, UHF, PM*, COSMO, Γ-PBC |
| NDDO closed-form F | ○ | fixed-P FD | **gap** |
| NDDO UHF | — | RHF-only throw | **gap** (OpenMOPAC UHF) |
| NDDO COSMO / Γ-PBC | ○ | unwrap-under-wrap only | **gaps** |
| PM3 / PM6 / PM7 | ○ | no types | Phase B **oracle=mopac** |
| DFTB0 / DFTB2 E+F+σ / Γ / multi-k / open-shell k | ✓ | Mio product | ThreeOb is **cluster** (no lattice images); ThreeOb multi-k + open shell still throw; no Fermi |
| DFTB3 E+F + Γ-PBC F | ✓ | molecular + Γ F | deepen: **no k / stress / open shell / ThreeOb / D3** |
| DFTB-D3 | ✓ | molecular + orthorhombic PBC | DFTB+ **D4** still absent |
| DFTB+U | ✓ energy | `u_eff` + FD F | deepen: **analytic F** |
| DFTB implicit solvent / LC / Fermi / TD-exc F | ○ | absent / integer occ / Casida \(E,f\) | **gaps** |
| γ_AB Elstner | ✓ | `internal/dftb_gamma.h` | **closed** |
| GFN0-xTB | ✓ energy + G + FD H CHNO | `GFN0` | deepen: beyond CHNO, odd-e / PBC / solvent |
| GFN1-xTB | ✓ energy + G CHNO | `GFN1` | deepen: Hessian, kernel residual is not a missing term |
| GFN2-xTB | ✓ energy + G CHNO | `GFN2` + AES energy + frozen-CAMM G | deepen: **AES-in-Fock**, beyond CHNO |
| GFN family analytic G | ✓ | GFN0/1/2 CHNO \(F=-\nabla E\) | **landed** wave-9; AES Fock hole is a GFN2 deepen |
| GFN family Hessian | ○ | GFN0 FD H ships; GFN1/2 H absent | **keep ○** (`gfn-xtb-analytic-hessian`; GFN1/2 F now exist) |
| IPEA / PBC / spin / ALPB / CPCM-X / sTDA | ○ | IPEA = `[~]` only | Phase C **oracle=xtb** |

### Foothold ≠ closed (wave-10)

- GFN2 **analytic G** shipped this shift — **not** GFN2 product. AES is still a
  post-SCC energy (angular \(F_{H1,y}\) vs xtb is the Fock hole;
  `reg_gfn2-xtb-forces` Gaps) → **new** `gfn2-aes-fock`.
- ThreeOb energy/F/σ ship as a **molecular cluster**; the multi-k throw names
  "no lattice sum at all" → **new** `dftb2-threeob-periodic` (Γ lattice-image
  walk like MioSketch) **before** `dftb2-threeob-multi-k`.
- GFN1/2 F now exist → keep `gfn-xtb-analytic-hessian` (VIB=ON exception);
  no extra family-Hessian slug.
- Keep DFTB3 / ThreeOb open-shell / Fermi / TD-excited / LC / +U F / D4 / GBSA
  / PM* / ALPB / CPCM-X / PBC GFN / spin GFN / sTDA / beyond-CHNO.
- **Not pending:** noncollinear / SOC / OM3 / GFN-FF / mopac-as-dftb-oracle
  (blocked or permanent). CUDA SE is this loop's scope-out. PTB and g-xTB are
  not GFN-xTB / not xtb 6.7.1. DFTB+ xTB Hamiltonian is **family-locked to
  xtb**, never a DFTB+ energy pin. DFTB+ MBD/TS after D4. PM6-D3H4 after PM6.
  MNDO-d after PM6 (d-orbitals). REKS / Δ-DFTB / NEGF / Ehrenfest / ChIMES /
  MOZYME / NAC / helical / f-electrons / pSIC stay unopened (not method-ladder).

### Found (not a slug)

`dftb2.h` still claims multi-k open shell throws (lines 53, 102, 180);
`scc_energy_unrestricted_kmesh` ships (`dftb2-multi-k-open-shell` done).
Stale comment — named again, not a capability gap.

### gaps_remaining (wave-10 census snapshot)

**30** pending + **1** incomplete (`[~]`) + **5** blocked/permanent `[!]` →
**nonzero; continue loop.** `new_slugs_added=2`. Roadmap: no ○→✓ (GFN
analytic G already ✓ on harness wave-9); prose notes AES-in-Fock + ThreeOb
cluster PBC. Family Hessian stays ○. **Queue not exhausted.** Ladder:
**Recover `ipea-xtb-small` first**; without `theme=gfn` staff Phase B NDDO /
DFTB3/ThreeOb deepens, not GFN dependents.

## pending

### Phase A — DFTB ladder (oracle = DFTB+)

Product DFTB0/2/3 + γ Elstner + Γ-PBC F landed. Remaining are **deepens**.
**Prefer Recover of `## in_progress` over new pending.**

- [ ] `dftb3-analytic-stress` | DFTB3 closed-form cell stress (pair virial, Γ); DFTB2 already has no strain-FD path | dftb3 | C-chain / H2O cell | σ vs strain-FD | **oracle=dftb+** (formula)
- [ ] `dftb3-kpoints` | DFTB3 Monkhorst–Pack energy (+ F later) | dftb3 | C-chain | E | **oracle=dftb+**; after Γ stress
- [ ] `dftb2-threeob-periodic` | ThreeOb is a molecular cluster (no lattice images); multi-k throw names "no lattice sum at all". Wire Γ lattice-image H⁰/S/γ/rep/F like MioSketch | dftb2 | C-chain 3ob | E/F | **oracle=dftb+**; **before** `dftb2-threeob-multi-k`
- [ ] `dftb2-threeob-multi-k` | ThreeOb currently throws on `k_mesh≠{1,1,1}` (no Bloch H0(k)/S(k)) | dftb2 | CH4 / C-chain 3ob | E/F | **oracle=dftb+**; after `dftb2-threeob-periodic`
- [ ] `dftb3-threeob` | Wire `DftbSkSet::ThreeOb` (3ob-3-1 CHNO) into `DFTB3`; transcribe Hubbard \(U^{\mathrm{d}}\) — 3ob is a DFTB3 set, not a DFTB2 one | dftb3 | H2O / CH4 | E/F | **oracle=dftb+**
- [ ] `dftb3-open-shell` | Collinear spin-DFTB3 (`multiplicity`); DFTB2 already has dual α/β | dftb3 | O2 triplet | E / m | **oracle=dftb+**
- [ ] `dftb3-dispersion-d3` | Opt-in D3(BJ) on `DFTB3` (molecular first; periodic later). DFTB2 already has `d3` / `d3_periodic` | dftb3 | water dimer | E/F | **oracle=dftb+** / s-dftd3
- [ ] `dftb2-threeob-open-shell` | ThreeOb currently throws on `multiplicity!=1` or `charge!=0` | dftb2 | open-shell CHNO | E/F | **oracle=dftb+**; after `dftb2-threeob-periodic`
- [ ] `dftb2-fermi-smearing` | DFTB+ `Filling { Fermi {…} }` — multi-k is integer occupation only | dftb2 | metallic C-chain / small cell | E / n(εF) | **oracle=dftb+**
- [ ] `td-dftb-excited-forces` | Casida nuclear forces on `DFTB2::td_casida` (energies+\(f\) already ship) | dftb2/td | H2 / small | F | **oracle=dftb+** (formula / FD)
- [ ] `lc-dftb-ground` | Range-separated LC-DFTB energy foothold (Yukawa ω; DFTB+ `RangeSeparated`) | dftb2/lc | small organic | E | **oracle=dftb+**; staff after DFTB3 deepens, before GFN
- [ ] `dftb-plus-u-forces` | Analytic ∂E_U/∂R on `DFTB2::u_eff` (energy + re-SCC FD F already ship; stress/k still throw) | dftb2 | O water | F | **oracle=dftb+**; design `.claude/notes/dftb-plus-u.md`
- [ ] `dftb-dispersion-d4` | DFTB+ `Dispersion { DftD4 {…} }` on DFTB2 (ATV DFTB is D3(BJ) only; GFN2's built-in D4 is a different Hamiltonian) | dftb2 | water dimer / small | E/F | **oracle=dftb+**
- [ ] `dftb-gbsa` | DFTB+ implicit solvent (GBSA / related Poisson–Born) on DFTB2; ATV DFTB is vacuum-only | dftb2 | H2O / small | E | **oracle=dftb+**; after Fermi/LC/D4; **not** xtb ALPB / OpenMOPAC COSMO

### Phase B — NDDO product depth (oracle = OpenMOPAC)

AM1/RM1 energy footholds + water + CO₂/CO mopac pins ship. Forces are still
fixed-density FD; SCF is **RHF only**; no continuum solvent; no Γ-PBC (AM1
unwraps an isolated fragment under wrap). Do **not** use DFTB+ for these rows.

- [ ] `nddo-closed-form-forces` | Closed-form multipole / integral-derivative NDDO nuclear forces (replace fixed-P FD) | nddo | H2O off-eq | F vs mopac + FD | **oracle=mopac**
- [ ] `nddo-uhf` | Unrestricted NDDO SCF (`n_alpha != n_beta` currently throws RHF-only) | nddo | O2 / radical | E | **oracle=mopac**
- [ ] `pm3-organic-energy` | PM3 (or MNDO) on NDDO ladder for table pipeline | nddo | small organic | E | **oracle=mopac**; bridge MNDO→PM6; extend `SemiEmpiricalMethod`
- [ ] `pm6-water-energy` | PM6 energy vs OpenMOPAC | nddo/pm6 | H2O | E | **oracle=mopac**
- [ ] `pm6-water-forces` | PM6 forces vs OpenMOPAC / FD | nddo/pm6 | H2O off-eq | F | **oracle=mopac**
- [ ] `pm7-water-energy` | PM7 energy vs OpenMOPAC | nddo/pm7 | H2O | E | **oracle=mopac**; after PM6 foothold
- [ ] `nddo-cosmo` | OpenMOPAC COSMO continuum solvent on AM1/RM1 (family-locked NDDO solvent; not xtb ALPB) | nddo | H2O / small | E | **oracle=mopac**
- [ ] `nddo-periodic-gamma` | OpenMOPAC Γ-point PBC (1D/2D/3D); ATV AM1 currently unwraps an isolated fragment under wrap | nddo | small cell | E | **oracle=mopac**; after closed-form F / PM3 tables

### Phase C — GFN-xTB (oracle = xtb) — product ladder open

xTB is **not** DFTB3++. **GFN0** energy + analytic G + FD Hessian, **GFN1**
energy + analytic G, and **GFN2** energy + quadrupole AES + analytic G ship
on `dev` (H/C/N/O; `reg_gfn0-beyond-toy`, `reg_gfn0-xtb-forces`,
`reg_gfn0-xtb-analytic-hessian`, `reg_gfn1-xtb-organic`, `reg_gfn1-xtb-forces`,
`reg_gfn2-xtb-water`, `reg_gfn2-quadrupole-aes`, `reg_gfn2-xtb-forces`;
oracle=xtb 6.7.1). IPEA remains `[~]` (`wf/ipea-xtb-small` @ `054f6c8e`).
Toy `XtbGfn0` was deleted (`bc14dabd`). **Staff order:** IPEA recover →
AES-in-Fock / ALPB / beyond-CHNO / family Hessian / sTDA / PBC / spin.
GFN1/2 analytic G landed; family Hessian is GFN1/2 (`gfn-xtb-analytic-hessian`).

- [ ] `gfn2-aes-fock` | Put GFN2 AES (charge–dipole/quadrupole + AXC) in the SCC Fock, not post-SCC energy. Angular \(F_{H1,y}\) vs xtb is the Fock hole (`reg_gfn2-xtb-forces` Gaps) | xtb | off-eq H2O | F | **oracle=xtb**; CHNO F already ships
- [ ] `gfn2-beyond-chno` | GFN2 static tables beyond H/C/N/O (xtb is all-element; GFN0/GFN1 same shape after) | xtb | small hetero | E | **oracle=xtb**; after CHNO F
- [ ] `alpb-xtb` | xtb ALPB/GBSA implicit solvent on production GFN2 | xtb | H2O / small organic | E | **oracle=xtb**; after `gfn2-xtb-water`
- [ ] `cpcm-x-xtb` | xtb ddCOSMO / CPCM-X continuum solvent on production GFN2 (Stahn 2023; banner on xtb 6.7.1) | xtb | H2O / small | E | **oracle=xtb**; after `alpb-xtb`
- [ ] `gfn-xtb-analytic-hessian` | GFN1/2 FD Hessian via `cpu::vib::Hessian` (product GFN1/2 forces now exist; GFN0 H already ships) | xtb/vib | small | ω | **oracle=xtb**; VIB=ON exception like `se-analytic-hessian`
- [ ] `stda-xtb` | sTDA-xTB excitations on production GFN ground state | xtb | small organic | ω_exc | **oracle=xtb|literature**; after `gfn2-xtb-water`
- [ ] `periodic-gfn-xtb` | Periodic GFN-xTB Γ / PBC energy | xtb | small cell | E | **oracle=xtb**
- [ ] `spin-gfn-xtb` | Spin-polarized GFN-xTB | xtb | open shell | E / spin | **oracle=xtb**; after GFN2

## in_progress

- [~] `ipea-xtb-small` | IPEA-xTB energy foothold | xtb | CH4 r=1.09 Å | E | **oracle=xtb**; `wf/ipea-xtb-small` @ `054f6c8e` (ahead=2 behind=8)

## permanent rules / scope-out (never pending)

Not work items — see `.claude/notes/tb-blocked-capabilities.md`.

- [!] `mopac-as-dftb-oracle` | **forbidden pattern** — never pin DFTB to OpenMOPAC
- [!] `gfn-ff` | **scope-out** — classical GFN-FF → MD / force-field surface, not this loop
- [!] `om3-nddo` | **permanent** — OM3 not in OpenMOPAC primary surface; no family-locked pin without expanding the oracle set (Thiel OMx). Not a missing ATV gate.

## blocked (science predecessors — multi-wave)

- [!] `noncollinear-spin-dftb` | collinear product exists; noncollinear needs spinor 2×2 density (new layout, not a flag)
- [!] `spin-orbit-dftb` | SOC/SK design; after noncollinear

## done

_(footholds + wave-1/2/3/4 product depth on `dev`)_

- [x] `gfn1-xtb-forces` | Analytic GFN1 G (HF+Pulay, CN/shell-poly, \(\partial\gamma/\partial R\), rep, D3); off-eq water vs xtb 6.7.1 `--gfn 1 --grad` | `reg_gfn1-xtb-forces` | **oracle=xtb**
- [x] `gfn2-xtb-forces` | Analytic GFN2 G (HF+Pulay, IES \(\partial\gamma/\partial R\), rep, D4, AES kernel at frozen CAMM); off-eq water vs xtb 6.7.1 `--gfn 2 --grad` (−5.058625058504 Eh). AES still post-SCC (angular \(F\) hole) | `reg_gfn2-xtb-forces` | **oracle=xtb**
- [x] `gfn0-xtb-analytic-hessian` | GFN0 FD Hessian via `cpu::vib::Hessian` (analytic G already ships); water ω vs xtb 6.7.1 `--gfn 0 --hess` | `reg_gfn0-xtb-analytic-hessian` | **oracle=xtb**; VIB=ON exception
- [x] `gfn0-xtb-forces` | Analytic GFN0 G (HF+Pulay, EEQ/CN, repulsion, D4, SRB); off-eq water vs xtb 6.7.1 `--gfn 0 --grad` | `reg_gfn0-xtb-forces` | **oracle=xtb**
- [x] `gfn1-xtb-organic` | GFN1-xTB organic energy (H/C/N/O); CH4 vs xtb 6.7.1 `--gfn 1` (−4.274232224875 Eh; 10.2 meV) | `reg_gfn1-xtb-organic` | **oracle=xtb**
- [x] `gfn2-quadrupole-aes` | GFN2 charge–quadrupole AES + qkernel AXC; H₂O vs xtb 6.7.1 −5.070383095219 Eh (dE ≈ −26 µeV) | `reg_gfn2-quadrupole-aes` | **oracle=xtb**
- [x] `gfn0-beyond-toy` | Production GFN0-xTB energy (H/C/N/O); methanol vs xtb 6.7.1 −7.597234498685 Eh | `reg_gfn0-beyond-toy` | **oracle=xtb**
- [x] `gfn2-xtb-water` | Production GFN2-xTB energy (H/C/N/O); H₂O vs xtb 6.7.1 −5.070383095219 Eh (forces throw; AES residual closed by `gfn2-quadrupole-aes`) | `reg_gfn2-xtb-water` | **oracle=xtb**
- [x] `nddo-sharp-discriminator` | RM1 CO₂/CO ΔHf vs PARAM survey (−81.55 / −13.87 kcal/mol; r=1.173 / 1.138 Å); water is 0.06 from exp, these are 12.5 | `reg_nddo-sharp-discriminator` | **oracle=mopac**
- [x] `se-analytic-hessian` | `cpu::vib::Hessian` + DFTB2 forces on H₂; symmetric H, linear ω≈4500 cm⁻¹ | `reg_se-analytic-hessian` | **oracle=literature** (structural); needs `ATV_BUILD_VIB=ON`
- [x] `dftb-plus-u` | DFTB2 Dudarev equal-filling \(E_U\) on O water; formula identity; forces throw | `reg_dftb-plus-u` | design `.claude/notes/dftb-plus-u.md`
- [x] `dftb-gamma-units-fix` / Elstner γ | eV·Å units then full DFTB+ Elstner `expGamma` (`internal/dftb_gamma.h`); H₂O/CH₄ DFTB+ pins closed; equal-U ∂γ/∂U for DFTB3 F | 2026-07-29 | note: `.claude/notes/dftb-vs-dftbplus-cleanup.md`
- [x] `dftb3-periodic-forces` | DFTB3 Γ-PBC lattice-image energy + analytic F | `reg_dftb3-periodic-forces` | **oracle=dftb+** (formula / FD identity)
- [x] `td-dftb-oscillator-strengths` | Casida length-form \(f\) + \(\boldsymbol{\mu}\) on `DFTB2::td_casida` (triplets dark) | unit + `reg_td-dftb-h2` | deepen of `td-dftb-h2`
- [x] `dftb2-multi-k-open-shell` | Collinear multi-k dual α/β H(k) + dual-channel analytic F | unit `KMeshOpenShell*` | deepen of multi-k
- [x] `rm1-water-mopac-energy` | Absolute OpenMOPAC RM1 ΔHf(H₂O) pin at the RM1-optimized geometry; **fixed RM1 free-atom energies** (`eisol`/`aq` were AM1-derived — 19 kcal/mol error) + `calpar` identity unit gate | `reg_rm1-water-mopac-energy` | **oracle=mopac** (OpenMOPAC PARAM survey + Scientia Plena 2016)
- [x] `am1-water-mopac-energy` | First hard OpenMOPAC pin on the NDDO surface: AM1 ΔHf / E / charges at the AM1-optimized geometry; seeds `tools/tb_oracle/inputs/mopac/` | `reg_am1-water-mopac-energy` | **oracle=mopac** (MOPAC 7.00 transcript; Dewar 1985)
- [x] `td-dftb-h2` | Linear-response TD-DFTB (Casida) foothold — `DFTB2::td_casida`, singlet + triplet energies, molecular restricted MioSketch | `reg_td-dftb-h2` | **oracle=literature** (Niehaus 2001 formalism; **no absolute pin exists** — structural gates only)
- [x] `dftb2-spin-analytic-stress` | Collinear open-shell closed-form cell stress; `analytic_forces_restricted`/`_unrestricted` folded into one spin-resolved `analytic_forces_stress` | `reg_dftb2-spin-analytic-stress` | **oracle=dftb+** (formula)
- [x] `dftb2-analytic-stress-threeob` | Closed-form ThreeOb cell stress; strain-FD product path deleted | `reg_dftb2-analytic-stress-threeob` | **oracle=dftb+** (formula)
- [x] `dftb2-dispersion-d3-periodic` | Γ-PBC lattice-image D3(BJ) (`d3_periodic`) + `d3_cutoff` member — a lattice sum is undefined without its truncation, so it was unpinnable against any engine | `reg_dftb2-dispersion-d3-periodic` | **oracle=literature** (offline s-dftd3 1.2.1, periodic `.gen`)
- [x] `dftb2-multi-k-forces` | Bloch-sum analytic multi-k F **and** stress (restricted Mio); **fixed the Γ pair virial**, which dropped self-image A–(A+T) terms (14% σ_xx error on a periodic cell) | `reg_dftb2-multi-k-forces` | **oracle=dftb+** (formula; F + strain-FD identity)
- [x] `dftb3-analytic-forces` | Closed-form DFTB3 nuclear gradient incl. ∂Γ/∂R; re-SCC FD deleted; the mio SK value+derivative kernel unified on `internal::MioSketchPair` (was three copies) | `reg_dftb3-analytic-forces` | **oracle=dftb+** (Gaus 2011 formula + h² FD identity)

- [x] `am1-water-mopac-energy` | First **absolute OpenMOPAC pin** on the NDDO surface: AM1 H₂O total energy / ΔHf / Mulliken charges + optimum-stationarity, at the AM1-optimized geometry from the same transcript | `reg_am1-water-mopac-energy` | **oracle=mopac** (MOPAC 7.00 archived transcript; `tools/tb_oracle/inputs/mopac/am1-water/` — first mopac twin)
- [x] `dftb2-kpoints` | Monkhorst–Pack multi-k SCC-DFTB2 beyond Γ (MioSketch energy; multi-k F/stress throw) | `reg_dftb2-kpoints` | **oracle=dftb+**
- [x] `dftb2-periodic-forces` | Lattice-image-consistent Γ-PBC nuclear F (match H⁰/S/γ/rep energy lattice sums) | `reg_dftb2-periodic-forces` | **oracle=literature** (+ family dftb+)
- [x] `dftb3-water-mio` | SCC-DFTB3 energy (third-order; \(E^{(3)}\) + exact \(\partial E^{(3)}/\partial\Delta q\); Anderson mix) | `reg_dftb3-water-mio` | **oracle=dftb+**
- [x] `dftb0-h2` | Non-SCC DFTB0 energy + force skeleton; mio-1-1 H–H pin vs DFTB+ 24.1 | `reg_dftb0-h2` | **oracle=dftb+**
- [x] `dftb2-dispersion-d3` | Opt-in Grimme D3(BJ) molecular dispersion + analytic F (`DFTB2::d3`) | `reg_dftb2-dispersion-d3` | **oracle=literature** (s-dftd3)
- [x] `dftb2-analytic-forces-threeob` | Closed-form nuclear F on `DftbSkSet::ThreeOb` | `reg_dftb2-analytic-forces-threeob` | **oracle=dftb+** (formula)
- [x] `dftb2-spin-analytic-forces` | Closed-form collinear open-shell nuclear F (MioSketch) | `reg_dftb2-spin-analytic-forces` | **oracle=dftb+** (formula)
- [x] `dftb2-analytic-stress` | Restricted MioSketch analytic cell stress (pair virial); open-shell/ThreeOb strain-FD | `reg_dftb2-analytic-stress` | **oracle=dftb+** (formula)
- [x] `dftb2-dftbplus-energy-pin` | Offline DFTB+ 24.1 H₂ total-energy pin on mio-1-1 H–H static SK (`dftb_mio_sk_tables.h`) | `reg_dftb2-dftbplus-energy-pin` | **oracle=dftb+**
- [x] `dftb2-periodic-gamma` | Γ-point periodic SCC-DFTB2 lattice-sum energy (H⁰/S, γ, repulsive) | `reg_dftb2-periodic-gamma` | **oracle=literature** (+ family dftb+)
- [x] `dftb2-3ob-organic` | Static 3ob-3-1 organic CHNO SK (`DftbSkSet::ThreeOb`) + CH4 DFTB+ 24.1 pin | `reg_dftb2-3ob-organic` | **oracle=dftb+**
- [x] `dftb2-analytic-forces` | Closed-form DFTB2 nuclear gradient (HF+Pulay+SCC γ+repulsive); replace re-SCC FD on restricted MioSketch | `reg_dftb2-analytic-forces` | **oracle=dftb+** (formula / module boundary)
- [x] `spin-dftb2-o2` | Collinear unrestricted spin-DFTB2; O₂ triplet Mulliken m | `reg_spin-dftb2-o2` | **oracle=literature** (+ family dftb+)
- [x] `h2_dftb2` / `h2o_dftb2_mio` | SCC-DFTB2 H/C/N/O static SK sketch + forces path | absolute H–H pin closed via `dftb2-dftbplus-energy-pin`
- [x] `h2o_gfn0` | GFN0-xTB **toy** | not product GFN0/1/2
- [x] `am1_*` / `rm1_water_energy` | NDDO AM1/RM1 energy + fixed-P FD forces | further NDDO → OpenMOPAC (PM6/PM7, mopac E pin, closed-form F)
