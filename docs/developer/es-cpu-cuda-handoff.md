# ES CPU + CUDA opt — handoff

Last operator-facing snapshot for `es-cpu-cuda-opt`. Rewrite in place each harness
wave; do not append history.

## Status

- **Parent:** `dev` @ `118d3003` (landed `cuda-pw-spin-pbe`; queue harness bookkeeping)
- **Queue:** [es-cpu-cuda-queue.md](es-cpu-cuda-queue.md)
- **Script:** `.grok/workflows/es-cpu-cuda-opt.rhai`
- **Last action:** Harness wave 4 2026-08-10 — mark **`cuda-pw-spin-pbe`**
  done [x] (feat `7949e29d` / land `118d3003`; GH200 **1124645**
  \|ΔE\|=**1.14e-13** Ha O₂ spin-PBE). Keep other [~]:
  `cuda-gpw-device-collocate` tip `6a564562` (GH200 **1124643**). Spec closed
  (deleted). **cpu_pending=0**; **cuda_pending=23**; **gaps_remaining≈24**.
  Fair wall MG MAIN **2.33×** still blocked pin-unsafe. Clearup already ran;
  worktrees empty; `wf/cuda-gpw-device-collocate` remains open tip (ahead 5 /
  behind 6 — rebase before merge).

## How to run

```text
workflow name=es-cpu-cuda-opt
args: {
  "mode": "exhaust",
  "staff_count": 2,
  "max_waves": 8,
  "target_branch": "dev",
  "theme": "merge device-collocate; then GPW product depth (forces, multigrid-run, uks-gamma, pbe-product-gate) + PW spin F/σ; GH200 green; no soft tols"
}
agent_budget: 384+
```

**Staff order (P1 first):**
1. **merge** `cuda-gpw-device-collocate` (tip GH200-green; rebase onto `dev`)
2. `cuda-gpw-device-forces` + `cuda-gpw-multigrid-run` + `cuda-gpw-uks-gamma` + `cuda-gpw-pbe-product-gate`
3. P2: kpoints / stress / smear-forces / spin-forces / spin-stress / spin-smear / nondefault-stream / diis-eigen / h2o-gate
4. P3 after parents: spin-pbe-forces / spin-pbe-stress / uks-forces / uks-pbe-gate / multigrid-forces / kpoint-forces / spin-smear F/σ

Workers only `wf/escuda-<slug>` or `wf/<slug>` off current `dev`.

## Disk

Wipe `build` / `build-*` under parent **and** every worktree every wave. CUDA
jobs must use wipe-shield names — never leave multi-GB trees. Re-scan empty
before exit.

## Open residual

| Track | Residual | Action |
| --- | --- | --- |
| **cuda GPW** | host collocate + host integrate; CPU-backed forces; no stress | merge collocate tip then staff P1 depth |
| **cuda GPW** | multigrid / UKS / multi-k reject | staff P1–P2 |
| **cuda PW** | spin-PBE energy **landed**; spin/smear F/σ + spin-smear energy open | staff P2–P3 depth |
| **cpu** | MG MAIN **2.33×** | blocked pin-unsafe / not soft-tol |

**Contract:** [.claude/notes/cuda-gpw-public-shape.md](../../.claude/notes/cuda-gpw-public-shape.md)
