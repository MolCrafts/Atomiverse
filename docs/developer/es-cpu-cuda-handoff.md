# ES CPU + CUDA opt — handoff

Rewrite in place each harness wave.

## Status

- **Parent:** `dev` (post opt-4/5 partial lands)
- **Workflow `es-cpu-cuda-opt-5`:** **PAUSED** for CUDA iron-law纠偏
- **Script:** `.grok/workflows/es-cpu-cuda-opt.rhai` — **updated** with:
  1. mandatory CUDA kernel skills
  2. **CUDA-only / no mid-path DH** iron law (`.claude/notes/cuda-only-no-dh.md`)
- **Do not resume opt-5** on the old immutable script. Launch a **new** run after
  residency P0 slugs land (or `script_path` of an edited projection).

## Iron laws (CUDA)

1. **No mid-SCF Device↔Host scientific exchange** — setup H2D once, payload D2H once.
2. **CUDA kernel skills required** before editing kernels (see workflow `extra_law`).
3. Prefer residency fix over dual-gate matrix.

## Staff order now (纠偏)

| P | Slug | Goal |
| --- | --- | --- |
| **P0** | `cuda-gpw-device-h-assemble` | Kill Γ RKS mid-iter `download_V_host` + host \(H\); residual/DIIS on device \(H\) |
| **P0** | `cuda-gpw-uks-device-h` | Same for UKS dual spin (no dual \(V\) D2H / \(P\) re-upload) |
| **P0** | `cuda-gpw-kpoint-p-residency` | No host \(P_k\) ping-pong mid multi-k SCF |
| **P1** | force buffer reuse / device residual nrm | Kill force full-grid re-H2D; DIIS max-norm on device |
| **P1** | `multigrid-si-energy-path` | Si MG correctness before any MG claim |
| **P2** | sphere collocate/integrate | Kernel rewrite after residency |

## How to re-run loop (after P0)

```text
workflow name=es-cpu-cuda-opt
args: {
  "mode": "exhaust",
  "staff_count": 2,
  "max_waves": 4,
  "target_branch": "dev",
  "theme": "CUDA iron law: residency only — device-h-assemble / uks-device-h / kpoint-p-residency; load CUDA kernel skills; zero mid-path DH; no dual-gate spam"
}
agent_budget: 256
```

## Contract

- [.claude/notes/cuda-only-no-dh.md](../../.claude/notes/cuda-only-no-dh.md)
- [.claude/notes/cuda-gpw-public-shape.md](../../.claude/notes/cuda-gpw-public-shape.md)
- [.claude/notes/cuda-rhf-device-only.md](../../.claude/notes/cuda-rhf-device-only.md) (skill table)
