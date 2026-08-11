# ES CPU + CUDA opt — handoff

Last operator-facing snapshot for `es-cpu-cuda-opt`. Rewrite in place each harness
wave; do not append history.

## Status

- **Parent:** `dev` @ `79d617ca` (land `cuda-pw-spin-stress`)
- **Queue:** [es-cpu-cuda-queue.md](es-cpu-cuda-queue.md)
- **Script:** `.grok/workflows/es-cpu-cuda-opt.rhai`
- **Last action:** **Harness wave 3** — 1 of 3 staffed → done [x]:
  `cuda-pw-spin-stress` (merge `79d617ca` / feat `841ac78f`; GH200
  **1135756** / prior **1133857** max‖Δσ‖=**1.435e-11** eV/Å³ O₂ triplet Γ
  collinear LSDA spin cell stress; **perf_exempt**). Keep other `[~]`:
  `cuda-gpw-kpoint-h-residency` (`181a81c3`, GH200 **1133843**
  \|dE\|=**3.442e-14**) + `cuda-gpw-multigrid-si-gate` (Assign wave 3 NEW).
  Spec closed (deleted `.claude/specs/cuda-pw-spin-stress/`); dropped
  `job-cuda-pw-spin-stress.sh`. Roadmap prose: collinear LSDA spin stress
  landed (table CUDA PW remains **◐** — spin-PBE F/σ + smear-PBE + multi-k
  F/σ + spin×kmesh×smear open — no ○→✓).
  **pending_count=137**; **in_progress=2**; **cpu_pending=0**;
  **cuda_pending=139**; **gaps_remaining=140**; **new_slugs_added=0** this
  harness. Free staff held: device-payload / uks-payload / device-cold-start /
  multi-k energy/payload/cold-start/P / **chi/orth/vnl/h0** / device-h-assemble /
  smear-stress-al2 / kmesh-smear-forces. Clearup already ran. Queue **not**
  exhausted. No push.

## How to run

```text
workflow name=es-cpu-cuda-opt
args: {
  "mode": "exhaust",
  "staff_count": 3,
  "max_waves": 6,
  "target_branch": "dev",
  "theme": "High-value residual: cuda-gpw-kpoint-h-residency OR cuda-gpw-multigrid-si-gate OR kpoint-device-chi/orth/vnl/h0 OR device-payload; max 3 NEW; forbid dual-gate matrix explosion; no soft tols; GH200 green"
}
agent_budget: 384+
```

## Open residual (high value)

| Track | Residual |
| --- | --- |
| **cuda GPW [~]** | `kpoint-h-residency` (staffed; unmerged) + `multigrid-si-gate` (staffed NEW) |
| **cuda GPW free** | device-payload / uks-payload / device-cold-start; multi-k energy/payload/cold-start/P; **kpoint-device-chi** / **orth** / **vnl** / **h0-residency**; device-h-assemble |
| **cuda PW free** | smear-stress-al2-gate; kmesh-smear-forces; smear-PBE; spin-PBE F/σ; multi-k F/σ; spin×kmesh×smear |
| **cpu** | MG MAIN **2.33×** pin-unsafe blocked |

**Contract:** [.claude/notes/cuda-gpw-public-shape.md](../../.claude/notes/cuda-gpw-public-shape.md)
