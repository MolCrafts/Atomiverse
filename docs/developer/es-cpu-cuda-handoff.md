# ES CPU + CUDA opt — handoff

Rewrite in place each harness wave.

## Status

- **Workflow:** `es-cpu-cuda-opt-6` **complete** (~6h36m). Script
  `.grok/workflows/es-cpu-cuda-opt.rhai` (iron laws + kernel skills +
  residency staff order + mid-path DH gate).
- **Law:** [.claude/notes/cuda-only-no-dh.md](../../.claude/notes/cuda-only-no-dh.md)
- **HEAD:** `783e9a55` (wave-6 green tips merged; harness sync next).

## Landed on `dev` this loop (high-signal)

| Item | Merge / evidence |
| --- | --- |
| residency P0 (Γ RKS/UKS/multi-k) | `5a5076d0` + force stream `61b91971` / GH200 **1158949** |
| product-si-gate | `a158848f` / **1167099** |
| product-collocate-pbc + SG device KS | `cfb246d1` / **1173564** |
| multigrid-si-energy-path | `88d8ec59` / **1180883** |
| device-payload (Γ RKS final eigen) | `d476ee7a` / **1182615** |
| **multigrid-device-ks** | `3730ada7` / tip **1186912** \|dE\|=**8.652e-06** (device monom MG default) |
| **uks-payload-device** | `f2be952a` / tip **1186414** \|dE\|=**4.163e-09** max\|deps\|=**1.481e-06** |

## Clearup

- Residual `wf/*` branches: **none**
- Worktrees: **none** (main tree only)
- Stale uncommitted Assign-wave-6 queue draft discarded (superseded by tip-green + merge)

## Residual (do not soft-land)

| Slug | Status |
| --- | --- |
| `multigrid-si-gate` | free staff — production dual-gate after monom on base |
| `device-cold-start` | open (Γ densify still host) |
| multi-k χ / orth / vnl / h0 / payload / cold-start | open |
| force free-fn AO Pulay / force-handoff | open (stream landed) |
| stress free-fn | full host |
| MG MAIN fair-wall **2.33×** | blocked pin-unsafe (`mg-main-structural`) |
| **cpu_pending=0** | no promote from gpw-perf-queue |

## Stated hybrid residual

- Γ RKS/UKS mid-SCF: device H + DIIS; MG mid-SCF: **device monom** default
  (after multigrid-device-ks).
- Final payload: RKS + UKS **device eigen** (boundary C/ε/H D2H).
- Still hybrid: host Γ STVnl setup, cold-start densify, multi-k payload /
  χ/orth/vnl/h0, force AO Pulay.

## Launch next

```text
workflow name=es-cpu-cuda-opt
args: {
  "mode": "exhaust",
  "staff_count": 2,
  "max_waves": 4,
  "do_perf": false,
  "target_branch": "dev",
  "theme": "multigrid-si-gate dual-gate; device-cold-start; kpoint payload/chi/orth/vnl/h0; force AO-Pulay"
}
agent_budget: 256
```
