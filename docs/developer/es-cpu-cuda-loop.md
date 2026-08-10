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
- Full `cuda-gpw-full-product` before architecture freeze

## Notes

- Order: `.claude/notes/es-optimization-order.md`
- Walls: `.claude/notes/gpw-perf-baseline.md`
- CUDA RHF: `.claude/notes/cuda-rhf-device-only.md`

## Run log

(autonomous harness appends lines here)

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

