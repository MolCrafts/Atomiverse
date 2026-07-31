# CP2K capability loop — handoff

## Harness wave 2 (shift) 2026-07-31 (parent `dev` @ `8b2fda00`)

| Field | Value |
| ----- | ----- |
| Branch | `dev` (tracks `origin/dev`, ahead of origin by merges) |
| Incomplete / `in_progress` | **2** (`[~] gpw-ts-stress`, `[~] gpw-xdm-stress` — uncommitted WIP in their isolation worktrees at the assign parent `197468ac`; no `wf/*` commits ahead) |
| Open CP2K specs on `dev` | **0** (PySCF deferred only: `periodic-rhf-gamma`, `rsh-eri-cam-b3lyp`) |
| Specs closed this harness | **8** (seven deleted here; `gpw-d4-atm-stress` deleted on its branch, `a2c85a6b`) |
| Unmerged staff `wf/*` | **2** incomplete kept (`wf/gpw-ts-stress`, `wf/gpw-xdm-stress` — branch tips at the assign parent; the WIP is uncommitted in their worktrees) |
| Isolation worktrees | **2** kept for the incomplete staff (dirty WIP — do not remove) |
| Leftover builds | merge-gate `build/` wiped at exit |
| Pending gaps | **3** (held assign wave-2 list — not exhausted) |
| Blocked | **6** (VV10 + hybrid / GAPW / PAW / CUDA-GPW / meta-GGA) |
| Done this wave | **8** landings (eight `--no-ff` merges `aa897bb2`..`8b2fda00`) |
| Merge gate | scoped rebuild + `ctest -R '^test_cpu_(gpw\|pw\|scf\|xc)_\|^test_xc_'` **57/57** green on merged `dev` |
| Oracle | `tools/cp2k_oracle/` offline pins only |
| Loop docs | queue / loop / handoff / roadmap residual prose **synced** |

### Wave staff → done

| Slug | Domain | Regression | ratio \(T_{\mathrm{atv}}/T_{\mathrm{cp2k}}\) |
| ---- | ------ | ---------- | --------------------------------------------- |
| `gpw-d3-stress` (resume) | gpw | `reg_gpw-d3-stress` | **0.55** (ATV 18.5 s / CP2K NUMERICAL twin 33.6 s) |
| `dfpt-gga-finite-q-si` | response | `reg_dfpt-gga-finite-q-si` | **0.17** (CP2K 288.7 s / ATV 49.8 s) |
| `xl-bomd-linearized-kpoints` | dynamics | `reg_xl-bomd-linearized-kpoints` | **perf_exempt** (NVE envelope; CP2K `.ener`-anchored trace) |
| `gpw-vdw-multigrid` | gpw | `reg_gpw-vdw-multigrid` | **0.83** (ATV 20.6 s / CP2K NGRIDS 4 24.7 s) |
| `cpmd-uks-kpoints` | dynamics | `reg_cpmd-uks-kpoints` | **perf_exempt** (CP2K has no fictitious-mass CP propagator) |
| `second-gen-cpmd-ot-kpoints` | dynamics | `reg_second-gen-cpmd-ot-kpoints` | **perf_exempt** (CP2K refuses `&OT` + `&KPOINTS`) |
| `gpw-d3-atm-stress` | gpw | `reg_gpw-d3-atm-stress` | **0.75** (ATV 40.8 s / CP2K 54.5 s) |
| `gpw-d4-atm-stress` | gpw | `reg_gpw-d4-atm-stress` | **perf_exempt** (no CP2K DFTD4 build) |

### Incomplete kept `[~]` (do not demote)

| Slug | Branch | Tip | Note |
| ---- | ------ | --- | ---- |
| `gpw-ts-stress` | `wf/gpw-ts-stress` | `197468ac` (= assign parent) | uncommitted WIP in worktree `wf_b2c72c75-58a-34`: spec + `regressions/gpw/gpw-ts-stress` + oracle inputs scaffolded, TS force-pin files touched; acceptance unfinished |
| `gpw-xdm-stress` | `wf/gpw-xdm-stress` | `197468ac` (= assign parent) | uncommitted WIP in worktree `wf_b2c72c75-58a-35`: spec + `regressions/gpw/gpw-xdm-stress` + oracle inputs scaffolded; acceptance unfinished |

### Roadmap ○→✓ this harness

| Row | Doc | Note |
| --- | --- | ---- |
| *(none)* | — | capability table rows already ✓ (dispersion / DFPT / AIMD / multigrid) |
| Residual prose | `dft.md` §8/§10/§12 + `molecular-dynamics.md` AIMD | landed **with the worker merges**: D3-zero stress, finite-q DFPT GGA, multi-k linearized XL, vdW multigrid, multi-k UKS CPMD, multi-k SGCP OT, D3/D4 ATM stress |
| Index GPW line | `index.md` | harness fixed the stale ships/residual line (vdW multigrid + multi-k SGCP OT + D3/D4 ATM stress were missing from ships and still listed pending) |
| Still open incomplete | TS/XDM stress pins | `[~]` WIP worktrees; stock VV10 stays ○ (blocked) |
| Still open pending | vdW × k-points; multi-k UKS multigrid; multi-k CPMD electronic thermostat | held wave-2 list |

### Pending residual

**Pending = 3** (held wave-2 list). **Incomplete = 2** (`gpw-ts-stress`,
`gpw-xdm-stress` — dirty WIP worktrees). Next wave starts with **Recover
incomplete** (finish/commit the two WIP worktrees on their `wf/*` branches)
then GapCensus. Blocked = 6 unchanged.

Do **not** staff PySCF deferred specs. No push from parent.
**Queue not exhausted** — GapCensus owns discovery; incomplete ≫ pending.

---

## Harness wave 1 (shift) 2026-07-31 (parent `dev` @ `8a9584f4`) — historical

| Field | Value |
| ----- | ----- |
| Branch | `dev` (tracks `origin/dev`, ahead of origin by merges) |
| Incomplete / `in_progress` | **1** (`[~] gpw-d3-stress` on `wf/gpw-d3-stress` @ `3b3872eb`) |
| Open CP2K specs on `dev` | **0** (PySCF deferred only: `periodic-rhf-gamma`, `rsh-eri-cam-b3lyp`) |
| Specs closed this harness | **9** (deleted at `8a9584f4`; INDEX cross-names the batch "harness wave 13") |
| Unmerged staff `wf/*` | **1** incomplete kept (`wf/gpw-d3-stress`) |
| Isolation worktrees | **0** (clearup already ran; main tree only) |
| Leftover builds | **none** in loop scope (concurrent CUDA-RHF campaign artifacts left untouched) |
| Pending gaps | **7** (held wave-1 list from GapCensus wave 15 — not exhausted) |
| Blocked | **6** (VV10 + hybrid / GAPW / PAW / CUDA-GPW / meta-GGA) |
| Done this wave | **9** landings (nine `--no-ff` merges `44749b1a`..`268aaf66`) |
| Merge gate | scoped rebuild + `ctest -R '^test_cpu_(gpw\|pw\|scf\|xc)_\|^test_xc_'` 57/57 green |
| Oracle | `tools/cp2k_oracle/` offline pins only |
| Loop docs | queue / loop / handoff / roadmap residual prose **synced** |

### Wave staff → done

| Slug | Domain | Regression | ratio \(T_{\mathrm{atv}}/T_{\mathrm{cp2k}}\) |
| ---- | ------ | ---------- | --------------------------------------------- |
| `gpw-d3bj-stress` | gpw | `reg_gpw-d3bj-stress` | **0.37** (CP2K 83.8 s / ATV 31.1 s) |
| `gpw-mbd-kpoint-stress` | gpw | `reg_gpw-mbd-kpoint-stress` | **perf_exempt** (CP2K stock has no MBD) |
| `gpw-d3-atm-forces` | gpw | `reg_gpw-d3-atm-forces` | **3.3** (ATV 34.5 s / CP2K 10.5 s) |
| `gpw-d4-atm-forces` | gpw | `reg_gpw-d4-atm-forces` | **perf_exempt** (no CP2K DFTD4 build) |
| `gpw-dftu-kpoint-forces` | gpw | `reg_gpw-dftu-kpoint-forces` | — (no wall recorded; full-6N FD dual gate 6.4e-6 Ha/Bohr) |
| `dfpt-gga-spin-si` | response | `reg_dfpt-gga-spin-si` | **0.20** (CP2K 221.8 s / ATV 44.7 s) |
| `gpw-bomd-kpoints` | dynamics | `reg_gpw-bomd-kpoints` | **0.064** (ATV 2.14 s / CP2K 33.4 s) |
| `gpw-multigrid-kpoints` | gpw | `reg_gpw-multigrid-kpoints` | **1.6** (wall 55 s) |
| `xl-bomd-linearized-uks` | dynamics | `reg_xl-bomd-linearized-uks` | **perf_exempt** (no cp2k binary at pin host) |

### Incomplete kept `[~]` (do not demote)

| Slug | Branch | Tip | Note |
| ---- | ------ | --- | ---- |
| `gpw-d3-stress` | `wf/gpw-d3-stress` | `3b3872eb` | wip spec + CP2K NUMERICAL-stress pins + D3 stress gates committed; merge-base `03c16eef` (behind the nine wave merges); acceptance unfinished |

### Roadmap ○→✓ this harness

| Row | Doc | Note |
| --- | --- | ---- |
| *(none)* | — | capability table rows already ✓ (DFT+U / dispersion / GGA / AIMD / multigrid) |
| Residual prose | `dft.md` §8/§10/§12 + `index.md` GPW + `molecular-dynamics.md` AIMD | closed D3(BJ) stress; multi-k MBD stress; D3/D4 ATM forces; multi-k DFT+U forces; joint DFPT GGA×spin; multi-k BOMD; multi-k multigrid; Γ-UKS linearized XL |
| Pin provenance | `gpw-kpoints-si` | ATV-frozen A7 self-golden pin replaced by genuine CP2K 2026.1 twin (`966d9b8a`) |
| Still open incomplete | D3-zero stress | `[~] gpw-d3-stress`; stock VV10 stays ○ (blocked) |
| Still open pending | TS/XDM stress; DFPT GGA×finite-q; vdW multigrid; linearized XL multi-k; multi-k UKS CPMD; multi-k SGCP OT | held wave-1 list |

### Pending residual

**Pending = 7** (held wave-1 list). **Incomplete = 1** (`gpw-d3-stress`).
Next wave starts with **Recover incomplete** (prefer resume/merge of
`wf/gpw-d3-stress`) then GapCensus. Blocked = 6 unchanged.

Do **not** staff PySCF deferred specs. No push from parent.
**Queue not exhausted** — GapCensus owns discovery; incomplete ≫ pending.

---

## Harness wave 12 2026-07-26 (parent `dev` @ `4ab58f97`) — historical

| Field | Value |
| ----- | ----- |
| Branch | `dev` (tracks `origin/dev`, ahead of origin by merges) |
| Incomplete / `in_progress` | **3** (`[~] cpmd-kpoints`, `[~] periodic-wannier-gpw`, `[~] gpw-fd-phonon-si`) |
| Open CP2K specs on `dev` | **0** (PySCF deferred only: `periodic-rhf-gamma`, `rsh-eri-cam-b3lyp`) |
| Specs closed this harness | **6** (deleted on close) |
| Unmerged staff `wf/*` | **3** incomplete kept (clearup already ran — no isolation worktrees) |
| Isolation worktrees | **0** (clearup already ran; main tree only) |
| Leftover builds | **none** |
| Pending gaps | **9** residual from GapCensus wave 13 (not exhausted) |
| Blocked | **6** (VV10 + hybrid / GAPW / PAW / CUDA-GPW / meta-GGA) |
| Done this wave | **6** landings (partial close of assign wave 12 staff) |
| Oracle | `tools/cp2k_oracle/` offline pins only |
| Loop docs | queue / loop / handoff / roadmap residual prose **synced** |

### Wave staff → done

| Slug | Domain | Regression | ratio \(T_{\mathrm{atv}}/T_{\mathrm{cp2k}}\) |
| ---- | ------ | ---------- | --------------------------------------------- |
| `xl-bomd-linearized-forces` | dynamics | `reg_xl-bomd-linearized-forces` | **perf_exempt** (NVE drift) |
| `second-gen-cpmd-kpoints` | dynamics | `reg_second-gen-cpmd-kpoints` | **perf_exempt** (NVE drift) |
| `gpw-d3-atm` | gpw | `reg_gpw-d3-atm` | **perf_exempt** (E_disp pin) |
| `gpw-d4-atm` | gpw | `reg_gpw-d4-atm` | **perf_exempt** (E_disp pin) |
| `bands-spin-metallic-si` | pw | `reg_bands-spin-metallic-si` | **perf_exempt** |
| `dos-spin-metallic-si` | pw | `reg_dos-spin-metallic-si` | **perf_exempt** |

### Incomplete kept `[~]` (do not demote)

| Slug | Branch | Tip | Note |
| ---- | ------ | --- | ---- |
| `cpmd-kpoints` | `wf/cpmd-kpoints` | `3425c7c6` | ahead=1 behind=15; multi-k Bloch \(C_k\) feat unmerged |
| `periodic-wannier-gpw` | `wf/periodic-wannier-gpw` | `5c59a97f` | ahead=1 behind=15; GPW MLWF feat unmerged |
| `gpw-fd-phonon-si` | `wf/gpw-fd-phonon-si` | `7c3e1712` | ahead=1 behind=15; Phonon+GPW wiring; tip notes blocked on Si forces |

### Roadmap ○→✓ this harness

| Row | Doc | Note |
| --- | --- | ---- |
| *(none)* | — | XL / SGCP / bands-DOS / dispersion capability rows already ✓ |
| Residual prose | `dft.md` §8/§10–12 + `index.md` + MD AIMD | closed zero-SCF linearized XL; multi-k SGCP; D3/D4 ATM; spin+metallic bands/DOS |
| Still open incomplete | multi-k CPMD; GPW MLWF; GPW FD phonon | keep `[~]`; stock VV10 stays ○ (blocked) |
| Still open pending | wave-13 D3/D3BJ forces+stress; MBD stress; multi-k DFT+U forces; DFPT GGA combos; multi-k BOMD | GapCensus wave 13 list |

### Pending residual

**Pending = 9** (GapCensus wave 13). **Incomplete = 3**. Next wave starts with
**Recover incomplete** (prefer merge/resume of the three `[~]` tips) then
continue. Blocked = 6 unchanged.

Do **not** staff PySCF deferred specs. No push from parent.
**Queue not exhausted** — GapCensus owns discovery; incomplete ≫ pending.

---

## Harness wave 11 2026-07-26 (parent `dev` @ `09298298`) — historical

| Field | Value |
| ----- | ----- |
| Branch | `dev` (tracks `origin/dev`, ahead of origin by merges) |
| Incomplete / `in_progress` | **0** (no residual `[~]`; stock VV10 is **blocked**) |
| Open CP2K specs on `dev` | **0** (PySCF deferred only) |
| Specs closed this harness | **9** (deleted on close) |
| Pending residual | **0** then refilled by GapCensus wave 12 |
| Done this wave | **9** landings |

### Wave staff → done (historical)

| Slug | Domain | ratio |
| ---- | ------ | ----- |
| `ugw-uks-rpbe` | gpw | **perf_exempt** |
| `gpw-ts-forces` | gpw | **perf_exempt** |
| `gpw-xdm-forces` | gpw | **perf_exempt** |
| `gpw-kpoint-uks-stress-gga` | gpw | **perf_exempt** |
| `cpmd-electronic-thermostat` | dynamics | **perf_exempt** |
| `xl-bomd-uks-kpoints` | dynamics | **perf_exempt** |
| `bands-metallic-si` | pw | **perf_exempt** |
| `dos-spin-si` | pw | **perf_exempt** |
| `dfpt-finite-q-spin-smearing` | response | **perf_exempt** |

---

## Harness wave 10 2026-07-26 (parent `dev` @ `06ecaa66`) — historical

| Field | Value |
| ----- | ----- |
| Branch | `dev` (tracks `origin/dev`, ahead of origin by merges) |
| Incomplete / `in_progress` | **1** (`[~] gpw-vv10-ice`) |
| Open CP2K specs on `dev` | **0** (PySCF deferred only) |
| Specs closed this harness | **9** (deleted on close) |
| Pending residual | **1** (`ugw-uks-rpbe`) then refilled by GapCensus wave 11 |
| Done this wave | **9** landings |

### Wave staff → done (historical)

| Slug | Domain | ratio |
| ---- | ------ | ----- |
| `dfpt-spin-smearing` | response | **perf_exempt** |
| `ot-uks-dftu-smearing` | gpw | **perf_exempt** |
| `second-gen-cpmd-uks-ot` | dynamics | **perf_exempt** |
| `xl-bomd-kpoints` | dynamics | **perf_exempt** |
| `bands-spin-si` | pw | **perf_exempt** |
| `dos-metallic-si` | pw | **perf_exempt** |
| `periodic-wannier-uks` | response | **perf_exempt** |
| `berry-spin-si` | response | **perf_exempt** |
| `gpw-mbd-kpoint-forces` | gpw | **perf_exempt** |

---

## Harness wave 9 2026-07-26 (parent `dev` @ `dab2b190`) — historical

| Field | Value |
| ----- | ----- |
| Branch | `dev` (tracks `origin/dev`, ahead of origin by merges) |
| Incomplete / `in_progress` | **2** (`[~] gpw-vv10-ice`, `[~] dfpt-spin-smearing`) |
| Open CP2K specs on `dev` | **0** (PySCF deferred only) |
| Specs closed this harness | **8** (deleted on close) |
| Pending residual | **0** then refilled by GapCensus wave 10 |
| Done this wave | **8** landings |

### Wave staff → done (historical)

| Slug | Domain | ratio |
| ---- | ------ | ----- |
| `ot-uks-smearing` | gpw | **~0.12** |
| `ot-uks-dftu` | gpw | **perf_exempt** |
| `ot-dftu-smearing` | gpw | **perf_exempt** |
| `gpw-d4-forces` | gpw | **perf_exempt** |
| `gpw-mbd-forces` | gpw | **perf_exempt** |
| `gpw-uks-multigrid` | gpw | **~1.54** |
| `dfpt-finite-q-smearing` | response | **perf_exempt** |
| `second-gen-cpmd-ot` | dynamics | **perf_exempt** |

---

## Harness wave 8 2026-07-26 (parent `dev` @ `a6752916`) — historical

| Field | Value |
| ----- | ----- |
| Branch | `dev` (tracks `origin/dev`, ahead of origin by merges) |
| Incomplete / `in_progress` | **1** (`[~] gpw-vv10-ice`) |
| Open CP2K specs on `dev` | **0** (PySCF deferred only: `periodic-rhf-gamma`, `rsh-eri-cam-b3lyp`) |
| Specs closed this harness | **9** (deleted on close) |
| Unmerged staff `wf/*` | **1** incomplete kept: `wf/gpw-vv10-ice` |
| Isolation worktrees | **0** (clearup already ran; main tree only) |
| Leftover builds | **none** |
| Pending gaps | **0** residual from prior census list (not exhausted — GapCensus owns refill) |
| Blocked | **5** (hybrid / GAPW / PAW / CUDA-GPW / meta-GGA) |
| Done this wave | **9** landings (see table) |
| Oracle | `tools/cp2k_oracle/` offline pins only |
| Loop docs | queue / loop / handoff / roadmap residual prose **synced** |

### Wave staff → done (historical)

| Slug | Domain | Regression | ratio \(T_{\mathrm{atv}}/T_{\mathrm{cp2k}}\) |
| ---- | ------ | ---------- | --------------------------------------------- |
| `periodic-urpa-kpoints` | gpw | `reg_periodic-urpa-kpoints` | **perf_exempt** |
| `ot-smearing` | gpw | `reg_ot-smearing` | **~0.20** |
| `ot-uks-kpoints` | gpw | `reg_ot-uks-kpoints` | **perf_exempt** |
| `ot-dftu-kpoints` | gpw | `reg_ot-dftu-kpoints` | **perf_exempt** |
| `gpw-d4-stress` | gpw | `reg_gpw-d4-stress` | **perf_exempt** |
| `gpw-vdw-scf-potential` | gpw | `reg_gpw-vdw-scf-potential` | **perf_exempt** |
| `dfpt-metallic-si` | response | `reg_dfpt-metallic-si` | **perf_exempt** |
| `dfpt-spin-incommensurate` | response | `reg_dfpt-spin-incommensurate` | **perf_exempt** |
| `gpw-optb88-vdw` | gpw | `reg_gpw-optb88-vdw` | **perf_exempt** (E_disp / XC combo) |

### Incomplete kept `[~]` (do not demote)

| Slug | Branch | Tip | Note |
| ---- | ------ | --- | ---- |
| `gpw-vv10-ice` | `wf/gpw-vv10-ice` | `d15cd804` | CP2K 2026.1 has no stock `TYPE VV10` (≠ rVV10); incomplete |

### Roadmap ○→✓ this harness (historical)

| Row | Doc | Note |
| --- | --- | ---- |
| optB88-vdW | `dft.md` §8 | exchange + DRSLL nonlocal (`reg_gpw-optb88-vdw`) |
| Residual prose | `dft.md` §8/§10–12 + `index.md` | multi-k URPA; OT+smear; multi-k UKS OT; multi-k OT+DFT+U; D4 stress; vdW SCF pot/forces; metallic DFPT; spin-incommensurate DFPT |
| Still open | stock VV10 | `[~] gpw-vv10-ice` (incomplete staff) |

### Pending residual

**Pending = 0** residual from GapCensus wave 8 staff list (emptied — **not**
queue exhausted). Then GapCensus wave 9 refilled 9 pending. **Incomplete = 1**
`[~]` (`gpw-vv10-ice`). Blocked = 5 unchanged.

Do **not** staff PySCF deferred specs. No push from parent.
**Queue not exhausted** — GapCensus owns next discovery.

---

## Harness wave 7 2026-07-26 (parent `dev` @ `8cb47664`) — historical

| Field | Value |
| ----- | ----- |
| Branch | `dev` (tracks `origin/dev`, ahead of origin by merges) |
| Incomplete / `in_progress` | **1** (`[~] gpw-vv10-ice`) |
| Open CP2K specs on `dev` | **0** (PySCF deferred only) |
| Specs closed this harness | **9** (deleted on close) |
| Pending residual | **0** then refilled by GapCensus wave 8 |
| Done this wave | **9** landings |

### Wave staff → done (historical)

| Slug | Domain | ratio |
| ---- | ------ | ----- |
| `periodic-urpa-si` | gpw | **perf_exempt** |
| `periodic-ugw-kpoints` | gpw | **perf_exempt** |
| `gpw-uks-smearing` | gpw | **~0.017** |
| `gpw-hirshfeld-uks` | gpw | **perf_exempt** (E_disp pin) |
| `cpmd-uks` | dynamics | **perf_exempt** (NVE drift) |
| `second-gen-cpmd-uks` | dynamics | **perf_exempt** (NVE drift) |
| `ot-kpoints` | gpw | **perf_exempt** |
| `ot-dftu` | gpw | **~0.26** |
| `dfpt-spin-si` | response | **perf_exempt** |

---

## Harness wave 6 2026-07-26 (parent `dev` @ `0869833a`) — historical

| Field | Value |
| ----- | ----- |
| Branch | `dev` (tracks `origin/dev`, ahead of origin by merges) |
| Incomplete / `in_progress` | **1** (`[~] gpw-vv10-ice`) |
| Open CP2K specs on `dev` | **0** (PySCF deferred only) |
| Specs closed this harness | **9** (deleted on close) |
| Pending residual | **1** (`periodic-urpa-si`) then refilled by GapCensus wave 7 |
| Done this wave | **9** landings |

### Wave staff → done (historical)

| Slug | Domain | ratio |
| ---- | ------ | ----- |
| `gpw-metallic-smearing` | gpw | **~0.011** |
| `gpw-xdm-ice` | gpw | **perf_exempt** |
| `gpw-uks-rpbe` | gpw | **~2.70** |
| `gpw-dftu-kpoint-stress` | gpw | **perf_exempt** |
| `periodic-ump2-kpoints` | gpw | **perf_exempt** |
| `periodic-ugw-si` | gpw | **perf_exempt** |
| `gpw-hirshfeld-kpoints` | gpw | **perf_exempt** |
| `dfpt-gga-si` | response | **perf_exempt** |
| `xl-bomd-uks` | dynamics | **perf_exempt** |

---

## Harness wave 5 2026-07-26 (parent `dev` @ `2d43d1c2`) — historical

| Field | Value |
| ----- | ----- |
| Branch | `dev` (tracks `origin/dev`, ahead of origin by merges) |
| Incomplete / `in_progress` | **2** (`[~] gpw-metallic-smearing`, `[~] gpw-vv10-ice`) |
| Open CP2K specs on `dev` | **0** (PySCF deferred only) |
| Specs closed this harness | **6** (deleted on close) |
| Pending residual | **1** (`gpw-xdm-ice`) then refilled by GapCensus wave 6 |
| Done this wave | **6** landings |

### Wave staff → done (historical)

| Slug | Domain | ratio |
| ---- | ------ | ----- |
| `gpw-dftu-kpoints` | gpw | **perf_exempt** |
| `periodic-gw-finite-q` | gpw | **perf_exempt** |
| `gpw-uks-revpbe` | gpw | **~3.22** |
| `cpmd-rattle-water` | dynamics | **perf_exempt** |
| `periodic-ump2-si` | gpw | **perf_exempt** |
| `ot-uks-o2` | gpw | **~8.50** |

---

## Harness wave 4 2026-07-26 (parent `dev` @ `85eaa2d9`) — historical

| Field | Value |
| ----- | ----- |
| Branch | `dev` (tracks `origin/dev`, ahead of origin by merges) |
| Incomplete / `in_progress` | **0** (no residual `[~]`) |
| Open CP2K specs | **0** (PySCF deferred only: `periodic-rhf-gamma`, `rsh-eri-cam-b3lyp`) |
| Specs closed this harness | **9** (deleted on close) |
| Unmerged staff `wf/*` | **0** for the 9 wave slugs (clearup already ran) |
| Isolation worktrees | **0** (main tree only) |
| Leftover builds | **none** |
| Pending gaps | **0** residual from prior census list (not exhausted — GapCensus owns refill) |
| Blocked | **5** (hybrid / GAPW / PAW / CUDA-GPW / meta-GGA) |
| Done this wave | **9** landings (see table) |
| Oracle | `tools/cp2k_oracle/` offline pins only |
| Loop docs | queue / loop / handoff / roadmap **synced** |

### Wave staff → done

| Slug | Domain | Regression | ratio \(T_{\mathrm{atv}}/T_{\mathrm{cp2k}}\) |
| ---- | ------ | ---------- | --------------------------------------------- |
| `gpw-kpoint-uks-stress` | gpw | `reg_gpw-kpoint-uks-stress` | **perf_exempt** |
| `periodic-gw-kpoints` | gpw | `reg_periodic-gw-kpoints` | **perf_exempt** |
| `gpw-vdw-df2-ice` | gpw | `reg_gpw-vdw-df2-ice` | **perf_exempt** (E_disp pin) |
| `gpw-rvv10-ice` | gpw | `reg_gpw-rvv10-ice` | **perf_exempt** (E_disp pin) |
| `gpw-mbd-periodic` | gpw | `reg_gpw-mbd-periodic` | **perf_exempt** (E_disp pin) |
| `pw-spin-gga` | pw | `reg_pw-spin-gga` | **perf_exempt** |
| `gpw-uks-blyp` | gpw | `reg_gpw-uks-blyp` | **~2.48** |
| `periodic-rpa-finite-q` | gpw | `reg_periodic-rpa-finite-q` | **perf_exempt** |
| `gpw-dftu-stress` | gpw | `reg_gpw-dftu-stress` | **perf_exempt** |

### Roadmap ○→✓ this harness

| Row | Doc | Note |
| --- | --- | ---- |
| vdW-DF2 | `dft.md` §8 | LMKLL nonlocal on GPW (`reg_gpw-vdw-df2-ice`) |
| rVV10 | `dft.md` §8 | nonlocal rVV10 on GPW (`reg_gpw-rvv10-ice`); VV10 stock still ○ |
| Multi-k G0W0 residual | `dft.md` §10 prose | closed (`reg_periodic-gw-kpoints`) |
| Periodic Ewald MBD | `dft.md` §8/§10 prose | productized (`reg_gpw-mbd-periodic`) |
| PW true collinear spin-GGA | `dft.md` §10 prose | productized (`reg_pw-spin-gga`) |
| Γ-UKS BLYP / multi-k UKS stress / finite-q dRPA / DFT+U stress | prose + index | productized |

### Pending residual

**Pending = 0** residual from GapCensus wave 4 staff list (emptied — **not**
queue exhausted). Next wave starts with **GapCensus** (tree/roadmap ○ vs
`dev`). Blocked = 5 unchanged.

Do **not** staff PySCF deferred specs. No push from parent.
**Queue not exhausted** — GapCensus owns next discovery.

---

## Harness wave 3 2026-07-25 (parent `dev` @ `0a642122`) — historical

| Field | Value |
| ----- | ----- |
| Branch | `dev` (tracks `origin/dev`, ahead of origin by merges) |
| Incomplete / `in_progress` | **0** (no residual `[~]`) |
| Open CP2K specs | **0** (PySCF deferred only: `periodic-rhf-gamma`, `rsh-eri-cam-b3lyp`) |
| Specs closed this harness | **8** (deleted on close) |
| Unmerged staff `wf/*` | **0** for the 8 wave slugs (clearup already ran) |
| Isolation worktrees | **0** (main tree only) |
