# ES CPU + CUDA opt — handoff

Last operator-facing snapshot for `es-cpu-cuda-opt`. Rewrite in place each harness
wave; do not append history.

## Status

- **Parent:** `dev` @ GapCensus wave 16 queue tip (tree ≡ opt-4 land `689f772b`)
- **Queue:** [es-cpu-cuda-queue.md](es-cpu-cuda-queue.md)
- **Script:** `.grok/workflows/es-cpu-cuda-opt.rhai`
- **Last action:** **GapCensus wave 16** — re-diff fair walls (held MG
  **2.33×**; cpu_pending=0) + CUDA product tree vs `dev` + unmerged Assign
  wave 1 tips. **new_slugs_added=2**: `cuda-gpw-kpoint-device-chi` (host
  `build_kpoint_basis_grid` + χ H2D residual after device collocate/integrate)
  + `cuda-gpw-kpoint-device-orth` (host Löwdin from D2H \(S_k\) after device
  STVnl tip). **pending_count=136**; **in_progress=3** (held `[~]`:
  `spin-stress` / `kpoint-h-residency` / `kpoint-device-stvnl`);
  **cpu_pending=0**; **gaps_remaining=140**. Free staff held: device-payload /
  uks-payload / device-cold-start / multi-k energy/payload/cold-start/P /
  **chi/orth** / device-h-assemble / smear-stress-al2 / kmesh-smear-forces /
  multigrid-si-gate. Roadmap ○/✓ unchanged (no capability closed). No push.

## How to run

```text
workflow name=es-cpu-cuda-opt
args: {
  "mode": "exhaust",
  "staff_count": 3,
  "max_waves": 6,
  "target_branch": "dev",
  "theme": "High-value residual ONLY after smear-stress+kpoint-h-assemble: cuda-pw-spin-stress OR cuda-gpw-kpoint-device-stvnl OR kpoint-h-residency OR multigrid-si-gate; max 3 NEW; forbid dual-gate matrix explosion; no soft tols; GH200 green"
}
agent_budget: 384+
```

## Open residual (high value)

| Track | Residual |
| --- | --- |
| **cuda GPW [~]** | `kpoint-h-residency` + `kpoint-device-stvnl` (staffed; unmerged) |
| **cuda PW [~]** | `spin-stress` (staffed; unmerged) |
| **cuda GPW free** | device-payload / uks-payload / device-cold-start; multi-k energy/payload/cold-start/P; **kpoint-device-chi** / **kpoint-device-orth** (wave 16); device-h-assemble; multigrid-si-gate held |
| **cuda PW free** | smear-stress-al2-gate; kmesh-smear-forces; smear-PBE; multi-k F/σ |
| **cpu** | MG MAIN **2.33×** pin-unsafe blocked |

**Contract:** [.claude/notes/cuda-gpw-public-shape.md](../../.claude/notes/cuda-gpw-public-shape.md)
