# CP2K capability loop — handoff

## Harness wave 4 (shift) 2026-08-01 (parent `dev` @ `6f1218b3`)

| Field | Value |
| ----- | ----- |
| Branch | `dev` (ahead of `origin/dev`; **no push**) |
| Incomplete / `in_progress` | **1** (`[~] gpw-ts-kpoint-stress`) |
| Open CP2K specs on `dev` | **0** (PySCF deferred only: `periodic-rhf-gamma`, `rsh-eri-cam-b3lyp`) |
| Specs closed this harness | **8** (deleted: wave-4 staff batch; `gpw-d3bj-kpoint-forces` had no live spec on parent) |
| Unmerged staff `wf/*` | **1** incomplete kept (`wf/gpw-ts-kpoint-stress` @ `2d0977c8`; do not prune) |
| Isolation worktrees | **0** (clearup already ran) |
| Leftover builds | none in loop scope |
| Pending gaps | **2** residual (not exhausted — GapCensus owns discovery) |
| Blocked | **6** (VV10 + hybrid / GAPW / PAW / CUDA-GPW / meta-GGA) |
| Done this wave | **9** landings |
| Oracle | CP2K 2026.1 source build `…/cp2k/install` (`git:5e54ba2`); offline pins only; D4 multi-k uses private libdftd4 rebuild; Hirshfeld TS/MBD forces `perf_exempt` (no stock CP2K twin) |
| Loop docs | queue / loop / handoff / specs INDEX / residual prose **synced** |

### Wave staff → done

| Slug | Domain | Regression | ratio \(T_{\mathrm{atv}}/T_{\mathrm{cp2k}}\) |
| ---- | ------ | ---------- | --------------------------------------------- |
| `gpw-d3-atm-kpoint-forces` | gpw | `reg_gpw-d3-atm-kpoint-forces` | **9.22** (13.28 / 1.44; CP2K multi-k ENERGY_FORCE C9) |
| `gpw-hirshfeld-forces` | gpw | `reg_gpw-hirshfeld-forces` | **perf_exempt** (no CP2K stock TS/MBD) |
| `gpw-d3-kpoint-forces` | gpw | `reg_gpw-d3-kpoint-forces` | **4.353** (16.284 / 3.741) |
| `gpw-d3bj-kpoint-forces` | gpw | `reg_gpw-d3bj-kpoint-forces` | **3.544** (13.933 / 3.931) |
| `gpw-d4-kpoint-forces` | gpw | `reg_gpw-d4-kpoint-forces` | **2.426** (9.987 / 4.117; libdftd4) |
| `gpw-d4-atm-kpoint-stress` | gpw | `reg_gpw-d4-atm-kpoint-stress` | **0.077** (10.58 / 137.945; libdftd4 NUMERICAL) |
| `periodic-berry-gpw-uks` | response | `reg_periodic-berry-gpw-uks` | **0.214** (21.09 / 98.68) |
| `periodic-scs-ump2-si` | gpw | `reg_periodic-scs-ump2-si` | **4.12** (15.79 / 3.83) |
| `dfpt-gga-spin-smearing` | response | `reg_dfpt-gga-spin-smearing` | **0.21** (48.3 / 232.5 serial) |

### Incomplete kept `[~]` (do not demote)

| Slug | Branch | Tip | Note |
| ---- | ------ | --- | ---- |
| `gpw-ts-kpoint-stress` | `wf/gpw-ts-kpoint-stress` | `2d0977c8` | feat tip; no reg dir on parent |

### Roadmap ○→✓ this harness

| Row | Doc | Note |
| --- | --- | ---- |
| *(none)* | — | capability table rows already ✓ (dispersion / DFPT / Berry / MP2) |
| Residual prose | `dft.md` intro + §8/§10/§11 + `index.md` GPW | open residuals → multi-k **TS stress** (`[~]` only) + multi-k **TS/XDM force** pins pending |

### Pending residual

**Pending = 2** (`gpw-ts-kpoint-forces`, `gpw-xdm-kpoint-forces`).
**Incomplete = 1** (multi-k free-atom TS stress — unmerged `wf/*`). Next wave
starts with **Recover incomplete** (merge or finish `gpw-ts-kpoint-stress`)
then GapCensus. Blocked = 6 unchanged.

Do **not** staff PySCF deferred specs. No push from parent.
**Queue not exhausted** — GapCensus owns discovery; incomplete ≫ pending.

---

## Harness wave 3 (shift) 2026-08-01 (parent `dev` @ `9bd134e0`) — historical

| Field | Value |
| ----- | ----- |
| Branch | `dev` (ahead of `origin/dev`; **no push**) |
| Incomplete / `in_progress` | **3** (`[~] gpw-ts-kpoint-stress`, `[~] gpw-d3-atm-kpoint-forces`, `[~] gpw-hirshfeld-forces`) |
| Open CP2K specs on `dev` | **0** (PySCF deferred only: `periodic-rhf-gamma`, `rsh-eri-cam-b3lyp`) |
| Specs closed this harness | **2** (deleted: `periodic-berry-gpw`, `periodic-scs-mp2-kpoints`; d4-atm-k / optb88-uks-k had no live specs on parent) |
| Unmerged staff `wf/*` | **3** incomplete kept (tips ahead of `dev`; do not prune) |
| Isolation worktrees | **0** (clearup already ran) |
| Leftover builds | none in loop scope |
| Pending gaps | **0** residual (not exhausted — GapCensus owns discovery) |
| Blocked | **6** (VV10 + hybrid / GAPW / PAW / CUDA-GPW / meta-GGA) |
| Done this wave | **4** landings |
| Oracle | CP2K 2026.1 source build `…/cp2k/install` (`git:5e54ba2`); offline pins only; D4 multi-k ATM uses private libdftd4 rebuild; multi-k SCS-MP2 is `oracle: literature` (CP2K aborts MP2_GPW@k) |
| Loop docs | queue / loop / handoff / specs INDEX / residual prose **synced** |

### Wave staff → done

| Slug | Domain | Regression | ratio \(T_{\mathrm{atv}}/T_{\mathrm{cp2k}}\) |
| ---- | ------ | ---------- | --------------------------------------------- |
| `gpw-d4-atm-kpoint-forces` | gpw | `reg_gpw-d4-atm-kpoint-forces` | **0.613** (9.682 / 15.806; CP2K libdftd4 multi-k ENERGY_FORCE) |
| `gpw-optb88-uks-kpoints` | gpw | `reg_gpw-optb88-uks-kpoints` | **0.73** (G2 9.8 / CP2K 13.29 OMP=8) |
| `periodic-scs-mp2-kpoints` | gpw | `reg_periodic-scs-mp2-kpoints` | **perf_exempt** (CP2K no multi-k MP2_GPW; algebraic Grimme dual) |
| `periodic-berry-gpw` | response | `reg_periodic-berry-gpw` | **9.6** (11.3 / 1.18; within_10x) |

### Incomplete kept `[~]` (do not demote)

| Slug | Branch | Tip | Note |
| ---- | ------ | --- | ---- |
| `gpw-ts-kpoint-stress` | `wf/gpw-ts-kpoint-stress` | `4afc4185` | feat tip; no reg dir on parent |
| `gpw-d3-atm-kpoint-forces` | `wf/gpw-d3-atm-kpoint-forces` | `a4168475` | feat tip + OpenMP k-forces; no reg dir on parent |
| `gpw-hirshfeld-forces` | `wf/gpw-hirshfeld-forces` | `4f3a62c6` | feat tip (volume-scaled force/stress fold); no reg dir on parent |

### Roadmap ○→✓ this harness

| Row | Doc | Note |
| --- | --- | ---- |
| *(none)* | — | capability table rows already ✓ (optB88 / SCS-MP2 / Berry / ATM forces) |
| Residual prose | `dft.md` intro + §8/§10/§11 + `index.md` GPW | open residuals → multi-k **TS stress** + multi-k D3 **ATM forces** + Hirshfeld-scaled TS/MBD **forces** (`[~]` only) |

### Pending residual

**Pending = 0** (assign emptied the census list; **not** queue exhausted).
**Incomplete = 3** (unmerged `wf/*`). Next wave starts with **Recover
incomplete** (merge or finish the three `[~]` tips) then GapCensus. Blocked
= 6 unchanged.

Do **not** staff PySCF deferred specs. No push from parent.
**Queue not exhausted** — GapCensus owns discovery; incomplete ≫ pending.

---

## Harness wave 2 (shift) 2026-08-01 (parent `dev` @ `79bcfa14`) — historical

| Field | Value |
| ----- | ----- |
| Branch | `dev` (ahead of `origin/dev`; **no push**) |
| Incomplete / `in_progress` | **1** (`[~] gpw-ts-kpoint-stress`) |
| Open CP2K specs on `dev` | **0** (PySCF deferred only: `periodic-rhf-gamma`, `rsh-eri-cam-b3lyp`) |
| Specs closed this harness | **7** (deleted: wave-2 staff batch; optB88-kpoints / hirshfeld had no live specs) |
| Unmerged staff `wf/*` | **1** incomplete kept (`wf/gpw-ts-kpoint-stress` @ `65aa0e6d`; do not prune) |
| Isolation worktrees | **0** (clearup already ran) |
| Leftover builds | none in loop scope |
| Pending gaps | **2** residual (not exhausted — GapCensus owns discovery) |
| Blocked | **6** (VV10 + hybrid / GAPW / PAW / CUDA-GPW / meta-GGA) |
| Done this wave | **9** landings (staff list; 8 new rows under harness wave 2 + prior worker-wave 2 pair folded in) |
| Oracle | CP2K 2026.1 source build `…/cp2k/install` (`git:5e54ba2`); offline pins only; D4 twin uses private libdftd4 rebuild |
| Loop docs | queue / loop / handoff / specs INDEX / residual prose **synced** |

### Wave staff → done

| Slug | Domain | Regression | ratio \(T_{\mathrm{atv}}/T_{\mathrm{cp2k}}\) |
| ---- | ------ | ---------- | --------------------------------------------- |
| `gpw-d4-kpoint-stress` | gpw | `reg_gpw-d4-kpoint-stress` | **0.139** (13.11 / 94.525; CP2K libdftd4 NUMERICAL Δσ) |
| `gpw-xdm-kpoint-stress` | gpw | `reg_gpw-xdm-kpoint-stress` | **perf_exempt** (no CP2K XDM) |
| `gpw-hirshfeld-uks-kpoints` | gpw | `reg_gpw-hirshfeld-uks-kpoints` | **perf_exempt** (no CP2K TS/MBD) |
| `cpmd-uks-thermostat-kpoints` | dynamics | `reg_cpmd-uks-thermostat-kpoints` | **perf_exempt** (no CP fictitious-mass twin) |
| `periodic-wannier-gpw-uks` | response | `reg_periodic-wannier-gpw-uks` | **9.57** (within_10x) |
| `periodic-scs-mp2-si` | gpw | `reg_periodic-scs-mp2-si` | **0.424** |
| `gpw-optb88-uks` | gpw | `reg_gpw-optb88-uks` | **3.35** |
| `gpw-vdw-uks-kpoint-multigrid` | gpw | `reg_gpw-vdw-uks-kpoint-multigrid` | **0.85** |
| `gpw-optb88-kpoints` | gpw | `reg_gpw-optb88-kpoints` | **0.67** |

### Incomplete kept `[~]` (do not demote)

| Slug | Branch | Tip | Note |
| ---- | ------ | --- | ---- |
| `gpw-ts-kpoint-stress` | `wf/gpw-ts-kpoint-stress` | `65aa0e6d` | feat tip ahead of `dev`; no `regressions/gpw/gpw-ts-kpoint-stress` on parent |

### Roadmap ○→✓ this harness

| Row | Doc | Note |
| --- | --- | ---- |
| *(none)* | — | capability table rows already ✓ (dispersion / vdW / AIMD / MLWF / MP2 / optB88) |
| Residual prose | `dft.md` intro + §8/§10/§11 + `index.md` GPW | refresh open residuals → multi-k **TS stress** `[~]` + multi-k D3/D4 **ATM forces** pending only |

### Pending residual

**Pending = 2** (`gpw-d3-atm-kpoint-forces`, `gpw-d4-atm-kpoint-forces`).
**Incomplete = 1** (multi-k free-atom TS stress — unmerged `wf/*`). Next wave
starts with **Recover incomplete** (merge or finish `gpw-ts-kpoint-stress`)
then GapCensus. Blocked = 6 unchanged.

Do **not** staff PySCF deferred specs. No push from parent.
**Queue not exhausted** — GapCensus owns discovery; incomplete ≫ pending.

---

## Harness wave 1 (shift) 2026-08-01 (parent `dev` @ `424d6989`)

| Field | Value |
| ----- | ----- |
| Branch | `dev` (ahead of `origin/dev`; **no push**) |
| Incomplete / `in_progress` | **4** (`[~] gpw-d4-kpoint-stress`, `[~] gpw-ts-kpoint-stress`, `[~] gpw-xdm-kpoint-stress`, `[~] gpw-hirshfeld-uks-kpoints`) |
| Open CP2K specs on `dev` | **0** (PySCF deferred only: `periodic-rhf-gamma`, `rsh-eri-cam-b3lyp`) |
| Specs closed this harness | **6** (deleted: wave-1 staff batch) |
| Unmerged staff `wf/*` | **4** incomplete kept (tips ahead of `dev`; do not prune) |
| Isolation worktrees | **0** (clearup already ran) |
| Leftover builds | none in loop scope |
| Pending gaps | **4** residual (not exhausted — GapCensus owns discovery) |
| Blocked | **6** (VV10 + hybrid / GAPW / PAW / CUDA-GPW / meta-GGA) |
| Done this wave | **6** landings |
| Oracle | CP2K 2026.1 source build `…/cp2k/install` (`git:5e54ba2`); offline pins only |
| Loop docs | queue / loop / handoff / specs INDEX / residual prose **synced** |

### Wave staff → done

| Slug | Domain | Regression | ratio \(T_{\mathrm{atv}}/T_{\mathrm{cp2k}}\) |
| ---- | ------ | ---------- | --------------------------------------------- |
| `gpw-vdw-uks-kpoints` | gpw | `reg_gpw-vdw-uks-kpoints` | **0.35** (G2 twin vs CP2K 60.2 s) |
| `gpw-d3-kpoint-stress` | gpw | `reg_gpw-d3-kpoint-stress` | **0.185** (21.84 / 118.34) |
| `gpw-d3bj-kpoint-stress` | gpw | `reg_gpw-d3bj-kpoint-stress` | **0.382** (23.0 / 60.111) |
| `gpw-d3-atm-kpoint-stress` | gpw | `reg_gpw-d3-atm-kpoint-stress` | **0.285** (5.34 / 18.74) |
| `gpw-bomd-uks-kpoints` | dynamics | `reg_gpw-bomd-uks-kpoints` | **1.18** |
| `gpw-vdw-uks-multigrid` | gpw | `reg_gpw-vdw-uks-multigrid` | **4.44** (within_10x; SC RPS FFT) |

### Incomplete kept `[~]` (do not demote)

| Slug | Branch | Tip | Note |
| ---- | ------ | --- | ---- |
| `gpw-d4-kpoint-stress` | `wf/gpw-d4-kpoint-stress` | `95a51e49` | feat + reg + REFERENCE (dftd4 strained-cell; perf_exempt); unmerged |
| `gpw-ts-kpoint-stress` | `wf/gpw-ts-kpoint-stress` | `1a4a5311` | feat + reg + REFERENCE (libMBD; perf_exempt); unmerged |
| `gpw-xdm-kpoint-stress` | `wf/gpw-xdm-kpoint-stress` | `e73e18d7` | feat + reg + REFERENCE (pure-Python; perf_exempt); unmerged |
| `gpw-hirshfeld-uks-kpoints` | `wf/gpw-hirshfeld-uks-kpoints` | `bc981d93` | feat + reg + REFERENCE (perf_exempt; no CP2K TS/MBD); unmerged |

### Roadmap ○→✓ this harness

| Row | Doc | Note |
| --- | --- | ---- |
| *(none)* | — | capability table rows already ✓ (dispersion / vdW / AIMD / multigrid) |
| Residual prose | `dft.md` §8/§10 + `index.md` GPW + MD AIMD | green wave-5 pins already product on `dev` (`424d6989`); residual list now holds D4/TS/XDM k-stress + UKS Hirshfeld (incomplete) + eNH/MLWF/SCS/optB88 pending |

### Pending residual

**Pending = 4** (`cpmd-uks-thermostat-kpoints`, `periodic-wannier-gpw-uks`,
`periodic-scs-mp2-si`, `gpw-optb88-uks`). **Incomplete = 4** (k-path
D4/TS/XDM stress + multi-k UKS Hirshfeld — unmerged `wf/*`). Next wave
starts with **Recover incomplete** (merge or finish the four `[~]` tips)
then GapCensus. Blocked = 6 unchanged.

Do **not** staff PySCF deferred specs. No push from parent.
**Queue not exhausted** — GapCensus owns discovery; incomplete ≫ pending.

---

## Bootstrap + GapCensus wave 4 (shift) 2026-08-01 (parent `dev` @ `87ae7924`) — historical

| Field | Value |
| ----- | ----- |
| Branch | `dev` (ahead of `origin/dev` by harness/docs commits; **no push**) |
| Incomplete / `in_progress` | **0** |
| Open CP2K specs on `dev` | **0** (PySCF deferred only: `periodic-rhf-gamma`, `rsh-eri-cam-b3lyp`) |
| Unmerged staff `wf/*` | **0** |
| Isolation worktrees | **0** |
| Leftover builds | wiped (`build/` ~10 G) |
| Pending gaps | **11** (GapCensus wave 4 refill — not exhausted) |
| Blocked | **6** (VV10 + hybrid / GAPW / PAW / CUDA-GPW / meta-GGA) |
| Oracle | CP2K 2026.1 source build `…/cp2k/install` (`git:5e54ba2`); offline pins only |
| Loop docs | queue / loop / handoff / roadmap residual prose **synced** |

### Recover

Clean. Wave-3 staff (9) all merged and reg dirs present. No open GPW/PW
specs, no `wf/*`, no isolation worktrees. Strict fair-twin pin law + source
build notes already on `dev` (`87ae7924`).

### GapCensus (pending refill)

Tree-declared residuals after harness wave 3 emptied the list (foothold ≠
closed):

| Slug | Kind | Twin |
| ---- | ---- | ---- |
| `gpw-vdw-uks-kpoints` | **impl** (fail-fast) | CP2K NON_LOCAL + UKS + k |
| `gpw-d3-kpoint-stress` | pin (fold product) | CP2K NUMERICAL Δσ + k |
| `gpw-d3bj-kpoint-stress` | pin | CP2K NUMERICAL BJ + k |
| `gpw-d3-atm-kpoint-stress` | pin | CP2K C9 on−off + k |
| `gpw-d4-kpoint-stress` | pin | dftd4 strained-cell |
| `gpw-ts-kpoint-stress` | pin | libMBD strain |
| `gpw-xdm-kpoint-stress` | pin | pure-Python XDM strain |
| `gpw-bomd-uks-kpoints` | product pin | CP2K UKS AIMD k |
| `gpw-vdw-uks-multigrid` | product pin | CP2K NGRIDS+DRSLL+UKS |
| `gpw-hirshfeld-uks-kpoints` | product pin | libMBD / literature |
| `cpmd-uks-thermostat-kpoints` | product pin | literature eNH envelope |

**Not staffed yet** — next step is **Assign** (prefer surface-disjoint:
one multi-k stress pin family vs `gpw-vdw-uks-kpoints` impl; BOMD UKS free
of stress trees). Incomplete ≫ pending if any `[~]` reappears.

### Found / fixed

- Stale `gpw_solver.h` claim that multi-k UKS OT+DFT+U+smear was still a
  residual (product: `reg_ot-uks-dftu-smearing`) — header comment fixed.
- Roadmap residual prose still listed wave-3 open items — refreshed.

### Next

**Assign wave** from the 11 pending (staff ≤10; no PySCF deferred specs).
No invent methods. No push.

---

## Harness wave 3 (shift) 2026-07-31 (parent `dev` @ `419c1dc3`)

| Field | Value |
| ----- | ----- |
| Branch | `dev` (ahead of `origin/dev`) |
| Incomplete / `in_progress` | **0** |
| Open CP2K specs on `dev` | **0** (PySCF deferred only: `periodic-rhf-gamma`, `rsh-eri-cam-b3lyp`) |
| Specs closed this harness | **9** (wave-3 staff batch) |
| Unmerged staff `wf/*` | **0** after clearup prune |
| Isolation worktrees | **0** after clearup |
| Leftover builds | wiped |
| Pending gaps | **0 residual** (not capability-closed — GapCensus owns discovery) |
| Blocked | **6** |
| Done this wave | **9** landings |
| Merge gate | all 9 `run_reg_*` green on merged `dev` |
| Oracle | `tools/cp2k_oracle/` offline pins only |
| Loop docs | queue / loop / handoff / specs INDEX **synced** |

### Wave staff → done

| Slug | Domain | Regression | ratio |
| ---- | ------ | ---------- | ----- |
| `gpw-ts-stress` | gpw | `reg_gpw-ts-stress` | **perf_exempt** (libMBD; no CP2K free-atom TS) |
| `gpw-xdm-stress` | gpw | `reg_gpw-xdm-stress` | **perf_exempt** (dual pure-Python; no CP2K XDM) |
| `gpw-vdw-kpoints` | gpw | `reg_gpw-vdw-kpoints` | **0.74** |
| `gpw-vdw-uks` | gpw | `reg_gpw-vdw-uks` | **~1.23** |
| `gpw-uks-multigrid-kpoints` | gpw | `reg_gpw-uks-multigrid-kpoints` | **perf_exempt** (H₂ twin ratio ~14) |
| `cpmd-thermostat-kpoints` | dynamics | `reg_cpmd-thermostat-kpoints` | **perf_exempt** (no CP fictitious-mass twin) |
| `second-gen-cpmd-ot-uks-kpoints` | dynamics | `reg_second-gen-cpmd-ot-uks-kpoints` | **perf_exempt** (CP2K OT Γ-only) |
| `xl-bomd-linearized-stress` | dynamics | `reg_xl-bomd-linearized-stress` | **2.4** |
| `dfpt-gga-spin-finite-q-si` | response | `reg_dfpt-gga-spin-finite-q-si` | **0.19** |

### Next

GapCensus on `dev` — do not invent methods; refill `## pending` from roadmap ○ with offline CP2K twins. Recover first if any `[~]` reappears.

---

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
