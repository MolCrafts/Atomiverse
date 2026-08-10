# ES CPU + CUDA optimization loop

Instance of the [oracle capability loop](oracle-capability-loop.md) for
**dual-track** electronic-structure work:

| Track | Goal | Oracle / gate |
| --- | --- | --- |
| **cpu** | Cut wall on landed GPW/PW/SCF | Real CP2K fair twins + pin provenance |
| **cuda** | Corresponding CUDA for landed ES | Same-settings CPU twin; device-only; GH200 logs |

Script: `.grok/workflows/es-cpu-cuda-opt.rhai`  
Queue: [es-cpu-cuda-queue.md](es-cpu-cuda-queue.md)  
Handoff: [es-cpu-cuda-handoff.md](es-cpu-cuda-handoff.md)

## Binding phase order

```text
GapCensus → Assign → Worktrees → Merge? → Clearup → Harness? → Tutorial?
```

- **GapCensus every wave** — re-diff baselines, notes, roadmap, and tree; foothold ≠ closed.
- **Research-then-impl** — workers name lever (file/function/metric) before coding.
- **Clearup always** — drop worktrees, prune fully-merged `wf/*`, wipe every `build*`.
- **Disk law** — no multi-GB leftovers; re-scan empty before claiming wipe.

## Not this loop

- New electronic methods → `cp2k-method-loop` / `pyscf-method-loop`
- Classical MD CUDA → OpenMM track
- New hybrid/GAPW/PAW/meta-GGA product (method loops); re-opening MG soft-tols

## Notes

- Order: `.claude/notes/es-optimization-order.md`
- Walls: `.claude/notes/gpw-perf-baseline.md`
- CUDA RHF: `.claude/notes/cuda-rhf-device-only.md`

## Run log

(autonomous harness appends lines here)

- **2026-08-10 GapCensus+Assign wave 9 (parent `dev` @ `8b517bb1`):** Re-diff
  post wave-8 land (`4f78d716` + chore). Fair walls held MG MAIN **2.33×**;
  **cpu_pending=0**. Product depth on base: collocate + MG + UKS + device
  forces + PBE energy + O₂ UKS gate. Staff **3** P2: `product-h2o-gate`
  (GH200 **1128194**), `stress` (**1128195**), `nondefault-stream`
  (**1128196**). **pending_count≈68**; **in_progress=3**;
  **gaps_remaining≈71**. Queue **not** exhausted. No push.


- **2026-08-10 harness wave 7 (parent `dev` @ `f679807c`):** 2 of staffed
  → done [x]: `cuda-gpw-multigrid-run` (feat `2d780cff` / `e201a6cb`; GH200
  **1126433** \|dE\|=**1.587e-7** Ha H₂ Γ RKS PADE ngrids=4; prior
  **1125894**; **perf_exempt**) + `cuda-gpw-uks-gamma` (feat `f679807c` /
  `254ff67c`; GH200 **1126430** \|ΔE\|=**4.760e-6** Ha high-spin H₂ UKS
  PADE ⟨S²⟩=2.0; prior **1125905**; **perf_exempt**). Keep other `[~]`:
  `cuda-gpw-device-forces` tip `9ec138dc` (GH200 **1126434** max\|ΔF\|=
  **2.321e-6** Ha/Bohr). Specs closed (deleted
  `.claude/specs/cuda-gpw-multigrid-run/`,
  `.claude/specs/cuda-gpw-uks-gamma/`). Roadmap prose: product MG + Γ UKS
  energy landed (table full GPW remains ◐ — forces / multi-k / stress /
  DIIS depth open — no ○→✓ for incomplete product). **cpu_pending=0**;
  **cuda_pending=51**; **in_progress=1**; **gaps_remaining≈53**. Clearup
  already ran. Queue **not** exhausted (GapCensus owns discovery). No push.

- **2026-08-10 Assign wave 7 (parent `dev` @ `9f5a76ed`):** Priority `[~]`
  first — capacity **3/3** already filled by wave-6 P1 tips not on base.
  **Resume** (no new staff): `cuda-gpw-device-forces` (`wf/…` `0f70ad52`,
  GH200 **1125926**), `cuda-gpw-multigrid-run` (`b9236f17`, **1125894**),
  `cuda-gpw-uks-gamma` (`6deea27e`, **1125905**). **new_assigned=0**;
  **incomplete_remaining=0** (tip green claims — Merge owns land). Held
  pending P1 `pbe-product-gate` + P2/P3 + wave-7 joint rows. Theme for NEW
  only N/A this wave. Queue **not** exhausted. No push.

- **2026-08-10 GapCensus wave 7 (parent `dev` @ `d83de986`):** Re-diff
  tree/baseline/notes/roadmap vs base (tips not merged). Fair walls **held**
  MG MAIN **2.33×**; **cpu_pending=0**. Base product hybrid held (device
  collocate mid-SCF; forces CPU-backed; PADE H₂ dual-gate only). Three
  `[~]` P1 tips remain (`device-forces` **1125926**, `multigrid-run`
  **1125894**, `uks-gamma` **1125905**). Autonomous discovery opens **15**
  deeper rows (GPW MG/UKS/PBE F·σ joint paths + path×PBE gates; PW
  closed-shell multi-k F/σ + spin-kmesh-PBE/F/σ) → **pending_count=51**,
  **in_progress=3**, **gaps_remaining≈55**. Roadmap full GPW CUDA stays
  **◐** (no ○→✓). Queue **not** exhausted. No push.

- **2026-08-10 harness wave 5 (parent `dev` @ `69296751`):** 1 of staffed
  → done [x]: `cuda-gpw-device-collocate` (device product collocate/integrate
  mid-SCF via `KsChainDevice::run`; merge `69296751` / feat `430a2268`;
  GH200 **1125709** \|ΔE\|=**1.617e-7** Ha max\|ΔF\|=**2.137e-6** Ha/Bohr
  H₂ Γ RKS PADE; prior **1124643** / **1123960** / **1122743**;
  **perf_exempt**). Spec closed (deleted
  `.claude/specs/cuda-gpw-device-collocate/`). No other `[~]`. Roadmap prose:
  device mid-SCF collocate landed (table full GPW remains ◐ — forces / MG /
  UKS / multi-k / stress / DIIS depth open — no ○→✓ for incomplete product).
  **cpu_pending=0**; **cuda_pending=29**; **gaps_remaining≈30**. Clearup
  already ran. Queue **not** exhausted (GapCensus owns discovery). No push.

- **2026-08-10 harness wave 4 (parent `dev` @ `118d3003`):** 1 of staffed
  → done [x]: `cuda-pw-spin-pbe` (collinear spin-GGA PBE energy on device;
  merge `118d3003` / feat `7949e29d`; GH200 **1124645** \|ΔE\|=**1.14e-13** Ha
  O₂ both −28.7585210635 Ha mag=2; prior **1123961** / **1122874**;
  **perf_exempt**). Spec closed (deleted `.claude/specs/cuda-pw-spin-pbe/`).
  Keep other `[~]`: `cuda-gpw-device-collocate` tip `6a564562` (GH200
  **1124643**). Roadmap prose: collinear spin-PBE energy landed (table PBE ✓
  held; spin F/σ + GPW product still open — no ○→✓ for incomplete depth).
  **cpu_pending=0**; **cuda_pending=23**; **gaps_remaining≈24**. Clearup
  already ran. Queue **not** exhausted (GapCensus owns discovery). No push.

- **2026-08-10 harness wave 3 (parent `dev` @ `18663d13`):** 1 of staffed
  → done [x]: `cuda-pw-pbe-stress` (closed-shell PBE Nielsen–Martin dual-gate;
  merge `18663d13` / feat `b0523de0`; GH200 re-verify **1123959**
  max‖Δσ‖=**5.585e-12** eV/Å³, E both −13.90403280 Ry; prior **1122705** /
  **1122143**; **perf_exempt**). Spec closed (deleted
  `.claude/specs/cuda-pw-pbe-stress/`). Keep other `[~]`:
  `cuda-gpw-device-collocate`, `cuda-pw-spin-pbe`. Roadmap prose: closed-shell
  PW stress LSDA+PBE landed (table remains ◐ — spin/smear/GPW open).
  **cpu_pending=0**; **cuda_pending=20**; **gaps_remaining≈21**. Clearup
  already ran. Queue **not** exhausted (GapCensus owns discovery). No push.

- **2026-08-10 harness wave 1 (parent `dev` @ `948d0d86`):** 1 of staffed
  → done [x]: `cuda-pw-pbe-device` (device closed-shell PBE-GGA for
  `cuda::pw::PW`; merge `3f901034`; GH200 job **1048320** units 4/4 PASS +
  dual-gate \(E_\mathrm{cpu}=E_\mathrm{cuda}=-14.84955549\) Ry,
  \(\lvert\Delta\rvert=1.421\times10^{-14}\) Ry; roadmap PBE CUDA ✓ already
  on branch — no ○→✓ this wave). Spec closed (deleted
  `.claude/specs/cuda-pw-pbe-device/`). Keep other `[~]`:
  `cuda-pw-large-cell-measure`, `cuda-pw-nondefault-stream`. Pending **4**
  (`cuda-pw-forces`, `cuda-gpw-ks-chain`, `cpu-simd-2center-setup`,
  `cuda-rhf-nondefault-stream`). Blocked **3** (`global-xtensor-xsimd`,
  `cuda-gpw-full-product`, `mg-main-structural`). Clearup already ran. Queue
  **not** exhausted (GapCensus owns discovery). No push.

- **2026-08-10 harness wave 2 (parent `dev` @ `8280ed89`):** land
  `cuda-pw-nondefault-stream` (merge `8280ed89`; GH200 **1049681** — 9/9
  ctest, MAX_EIG_DIFF 5.274e-15 Ha, |E_cuda−E_cpu|=1.243e-14 Ry on
  `pw-cuda-lda-si` + `reg_cuda-pw-nondefault-stream`; wipe-shield
  `_pw_stream_work`). Also close bookkeeping for `cuda-pw-large-cell-measure`
  (already merge `948d0d86`). CPU fair walls **unchanged** (worst fair ratio
  **2.33×** MG MAIN). Specs closed. Pending: `cuda-pw-forces` P0,
  `cuda-gpw-ks-chain` P1, spin/smear P2, simd/RHF-stream P3. Clearup this
  wave. Queue **not** exhausted. No push.

- **2026-08-10 GapCensus (parent `dev` @ `0316e7d0`):** Bootstrap + re-diff
  tree (`cuda/gpw` = internal only; no public `GpwDft`), notes (freeze green
  `0147fe02`), arch-campaign residual closed (`arch-pw-driver-paths` /
  `arch-mg-main-structural` merged), roadmap DFT CUDA GPW ○, fair walls
  authority **2.33×** MG MAIN. Queue refilled: **cuda_pending=5**
  (`cuda-gpw-full-product` P0 + four PW spin-depth rows); **cpu_pending=0**.
  Spec-lite `.claude/specs/cuda-gpw-full-product/`. Worktrees/`wf/*`/`build*`
  empty. Staff next: product only. Queue **not** exhausted. No push.

- **2026-08-10 GapCensus re-diff (parent `dev` @ `8122eb0e`):** Re-diff
  tree/baseline/notes/roadmap + `pw_solver` reject matrix. Product gap +
  freeze green **held**; **cpu_pending=0**; fair wall **2.33×** MG MAIN
  blocked. Autonomous discovery opens **3** closed-shell PW depth rows
  (`cuda-pw-pbe-stress`, `cuda-pw-smear-forces`, `cuda-pw-smear-stress`)
  missed by spin-only staffing → **pending_count=8**, **gaps_remaining≈10**.
  Roadmap prose: freeze staffable (no ○→✓). Worktrees/`build*` empty. Staff
  next: product P0. Queue **not** exhausted. No push.


- **2026-08-10 GapCensus bootstrap (parent `dev` @ `a6049f4f`):** Re-diff tree
  (`cuda::gpw::GpwDft` public first slice **on base**; host collocate/integrate +
  device `KsChainDevice`; forces CPU-backed; no CUDA GPW stress), PW reject
  matrix (spin∧PBE energy; spin/smear forces/stress), notes (freeze green;
  fair walls **2.33×** MG MAIN held), roadmap DFT CUDA GPW columns still ○ for
  full RKS/UKS product depth. **cpu_pending=0**; **cuda_pending=13**;
  **pending_count=13**; **gaps_remaining≈14** (no new slug invention — depth
  rows already cover hybrid compose + PW spin/PBE residual). Spec
  `cuda-gpw-full-product` closed (acceptance_met=true; deleted). Worktrees /
  `wf/*` / `build*` empty. Staff next: P1 GPW device-collocate/forces +
  multigrid-run/uks-gamma + PW pbe-stress/spin-pbe. Queue **not** exhausted.
  No push.
