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
