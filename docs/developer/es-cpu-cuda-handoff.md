# ES CPU + CUDA opt — handoff

Last operator-facing snapshot for `es-cpu-cuda-opt`. Rewrite in place each harness
wave; do not append history.

## Status

- **Parent:** `dev` @ `1733535f` (post harness wave 4: kpoint-device-collocate + integrate)
- **Queue:** [es-cpu-cuda-queue.md](es-cpu-cuda-queue.md)
- **Script:** `.grok/workflows/es-cpu-cuda-opt.rhai`
- **Last action:** Harness wave 4 — land multi-k device spine half:
  - [x] `cuda-gpw-kpoint-device-collocate` — multi-k mid-SCF χ^k density
    collocate device-side; merge `355bcc42` / feat `47c2c16e`; GH200
    **1130716** \|dE\|=**3.353e-14** Ha (prior **1130429**); **perf_exempt**
  - [x] `cuda-gpw-kpoint-device-integrate` — multi-k mid-SCF V→H_k integrate
    device-side; merge `1733535f` / feat `a73c2281` / docs `d98c2093`; GH200
    **1130717** \|dE\|=**3.397e-14** Ha; **perf_exempt**
  - [~] none remaining
- **Counts:** pending=123; in_progress=0; cpu_pending=0; cuda_pending=123;
  gaps_remaining=124 (incl. blocked `mg-main-structural`). Queue **not**
  exhausted.

## How to run

```text
workflow name=es-cpu-cuda-opt
args: {
  "mode": "exhaust",
  "staff_count": 3,
  "max_waves": 6,
  "target_branch": "dev",
  "theme": "High-value residual only: multi-k device DIIS/STVnl or device-product-residency / device-h-assemble / smear-stress / spin-stress; no dual-gate matrix explosion; no soft tols"
}
agent_budget: 384+
```

## Open residual (high value)

| Track | Residual |
| --- | --- |
| **cuda GPW (staffed done)** | multi-k device **collocate + integrate** **landed** (H₂ k-class) |
| **cuda GPW (next)** | multi-k device **DIIS/STVnl**; Si/water multi-k gates; device-product-residency; device-h-assemble; device-stvnl (Γ); water σ; device-stress; MG/UKS F/σ |
| **cuda PW (staffed done)** | restricted metallic-smear **forces** **landed** |
| **cuda PW (next)** | smear-stress; smear-PBE; spin-stress; multi-k PBE F/σ; spin+smear; kmesh-smear F/σ |
| **cpu** | MG MAIN **2.33×** pin-unsafe blocked |

**Contract:** [.claude/notes/cuda-gpw-public-shape.md](../../.claude/notes/cuda-gpw-public-shape.md)
