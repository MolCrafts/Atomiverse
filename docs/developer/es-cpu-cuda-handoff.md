# ES CPU + CUDA opt — handoff

Last operator-facing snapshot for `es-cpu-cuda-opt`. Rewrite in place each harness
wave; do not append history.

## Status

- **Parent:** `dev` @ `5319a282` (harness wave 6 after `cuda-gpw-kpoint-device-diis`)
- **Queue:** [es-cpu-cuda-queue.md](es-cpu-cuda-queue.md)
- **Script:** `.grok/workflows/es-cpu-cuda-opt.rhai`
- **Last action:** Harness wave 6 — **1 of 3** staffed → done [x]:
  - [x] `cuda-gpw-kpoint-device-diis` merge `5319a282` / feat `f3f55dc2` /
    re-verify `8511a1fb` — GH200 **1133110** \|dE\|=**3.419e-14** Ha
    (prior tip **1132761**; H₂ multi-k RKS PADE device KDiis+zheevd;
    **perf_exempt**)
  - [~] `cuda-pw-smear-stress` tip `4c9a7b62` / GH200 **1132266** /
    re-verify **1133111** (tip-green not base)
  - [~] `cuda-gpw-kpoint-h-assemble` tip `5b64229b` / **1132528**
    (tip-green not base)
  - Spec closed: deleted `.claude/specs/cuda-gpw-kpoint-device-diis/`
- **Counts:** pending=133; in_progress=2; cpu_pending=0; cuda_pending=135;
  gaps_remaining=136 (incl. blocked `mg-main-structural`). Queue **not**
  exhausted.
- **Roadmap:** table CUDA GPW remains **◐** (no ○→✓ — H assemble / residency /
  STVnl / Si-water multi-k F·σ depth open). Prose: multi-k device DIIS landed.

## How to run

```text
workflow name=es-cpu-cuda-opt
args: {
  "mode": "exhaust",
  "staff_count": 3,
  "max_waves": 6,
  "target_branch": "dev",
  "theme": "High-value residual only: multi-k device H assemble / H residency / energy / payload / STVnl or device-product-residency / device-h-assemble / smear-stress / spin-stress; no dual-gate matrix explosion; no soft tols"
}
agent_budget: 384+
```

## Open residual (high value)

| Track | Residual |
| --- | --- |
| **cuda GPW (staffed now)** | multi-k **`kpoint-h-assemble`** tip-green (Merge) |
| **cuda GPW (next after)** | `kpoint-h-residency` / `kpoint-device-energy` / `kpoint-payload-device` / `kpoint-p-residency` / `kpoint-device-stvnl`; Si/water multi-k gates; device-h-assemble; uks-h-assemble; water σ; device-stress; MG/UKS F/σ |
| **cuda PW (staffed now)** | **`smear-stress`** tip-green GH200 **1132266** / **1133111** (Merge) |
| **cuda PW (next after)** | `smear-stress-al2-gate`; smear-PBE; spin-stress; multi-k PBE F/σ; spin+smear; kmesh-smear F/σ |
| **cpu** | MG MAIN **2.33×** pin-unsafe blocked |

**Contract:** [.claude/notes/cuda-gpw-public-shape.md](../../.claude/notes/cuda-gpw-public-shape.md)
