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

