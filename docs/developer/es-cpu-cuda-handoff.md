# ES CPU + CUDA opt — handoff

Rewrite in place each harness wave.

## Status

- **Workflow:** `es-cpu-cuda-opt` script
  `.grok/workflows/es-cpu-cuda-opt.rhai` — iron laws + kernel skills +
  residency staff order + acceptance gate (grep mid-path DH).
- **Law:** [.claude/notes/cuda-only-no-dh.md](../../.claude/notes/cuda-only-no-dh.md)
- **Harness wave 2** on `dev` @ `cfb246d1`: pending **133** / in_progress **0**
  / gaps **134**; **cpu_pending=0**; **new_slugs_added=0**.
- **Residency P0:** Γ RKS / UKS / multi-k mid-SCF landed (`5a5076d0` + force
  stream `61b91971`). GH200 residency dual-gates **1158949** — queue marks
  `device-h-assemble` / `uks-h-assemble` / `uks-p-residency` /
  `kpoint-p-residency` / `kpoint-device-energy` / `force-stream` **done**.
- **Si product gate:** **landed** on `dev` (`a158848f`). GH200 **1167099**
  Si Γ ngrids=1 \|dE\|=**2.130e-08** Ha; post-collocate-pbc hold **1173564**
  \|dE\|=**3.504e-05** Ha.
- **PBC collocate + SG device KS:** **landed** merge `cfb246d1` / feat
  `5648608f`. GH200 **1173564** / **1177100**.
- **Staffed:** none (`[~]` empty).

## Landed this harness

| Item | Evidence |
| --- | --- |
| `cuda-gpw-product-collocate-pbc` | merge `cfb246d1` / feat `5648608f`; sphere-window PBC collocate/integrate + SG pack→`KsChain`→scatter; GH200 **1173564** edge rel=**5.2e-14** H₂ \|dE\|=**1.252e-08** product-si \|dE\|=**3.504e-05**; **1177100** full-product hold; skills: choose-launch-configuration, optimize-global-memory-access, handle-boundary-conditions, av-perf |
| prior product-si + residency P0 | `a158848f` / **1167099**; `5a5076d0` / `61b91971` / **1158949** |

## Residual (do not soft-land)

| Slug | Status |
| --- | --- |
| `multigrid-si-energy-path` | **unblocked** — staff next (PBC collocate + H0/KB on base) |
| `multigrid-si-gate` | **blocked** on parent energy-path |
| `device-payload` / `uks-payload-device` / `device-cold-start` | open (Γ host re-diag / cold densify) |
| multi-k χ / orth / vnl / h0 / payload / cold-start | open |
| force free-fn | host AO Pulay residual (P1); stream **landed** |
| stress free-fn | full host |

## Stated hybrid residual

Single-grid `cuda::gpw::GpwDft` mid-SCF is **device** product KS with
sphere-window PBC collocate/integrate (iron-law P0 SG closed). CompactBasisGrid
retained **force-only** / payload boundary. Remaining hybrid: host Γ STVnl,
final payload host re-diag, cold-start densify, multi-k χ·orth·vnl·h0,
host AO Pulay forces, full-host stress.

## Launch

```text
workflow name=es-cpu-cuda-opt
args: {
  "mode": "exhaust",
  "staff_count": 2,
  "max_waves": 4,
  "do_perf": false,
  "target_branch": "dev",
  "theme": "multigrid-si-energy-path (Si ngrids>1 after PBC collocate); then multigrid-si-gate / device-payload"
}
agent_budget: 256
```
