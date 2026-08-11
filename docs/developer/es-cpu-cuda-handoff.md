# ES CPU + CUDA opt — handoff

Rewrite in place each harness wave.

## Status

- **Workflow:** `es-cpu-cuda-opt` script
  `.grok/workflows/es-cpu-cuda-opt.rhai` — iron laws + kernel skills +
  residency staff order + acceptance gate (grep mid-path DH).
- **Law:** [.claude/notes/cuda-only-no-dh.md](../../.claude/notes/cuda-only-no-dh.md)
- **Residency P0:** Γ RKS / UKS / multi-k mid-SCF landed (`5a5076d0` + force
  stream `61b91971`). GH200 residency dual-gates **1158949**.
- **Si product gate:** **landed** on `dev` (`a158848f`, tip `a30c675d`).
  GH200 **1167099** Si Γ ngrids=1 \|dE\|=**2.130e-08** Ha; H2/water hold.

## Landed this serial clearup

| Item | Evidence |
| --- | --- |
| `cuda-gpw-product-si-gate` | merge `a158848f`; KB Vnl via `add_nonlocal_kb_k`@Γ + SG host **CompactBasisGrid** KS twin (CPU ngrids=1 path) |
| Worktrees / `wf/cuda-gpw-*` residual tips | **none** — product-si merged+deleted; multigrid-si-{energy-path,gate} **dropped** (blocked) |

## Residual (do not soft-land)

| Slug | Status |
| --- | --- |
| `product-collocate-pbc` | **open** — restore device continuous product collocate with collint PBC wrap; remove SG mid-SCF host cbg hybrid (iron-law debt) |
| `multigrid-si-energy-path` | **blocked** on product-collocate-pbc (+ re-gate after H0/KB on base). Prior research tip deleted; first-iter charge ~0.5 on continuous MG without wrap |
| `multigrid-si-gate` | **blocked** parent energy-path |
| force free-fn | host AO Pulay residual (P1) |

## Stated hybrid (product-si)

Single-grid `cuda::gpw::GpwDft` mid-SCF currently uses **host** `collocate_density` /
`ks_grid_build_rks` / `integrate_potential` on CompactBasisGrid (CPU twin), with
P/V boundary copies per iter. Device DIIS/eigen + \(H=H_0+V\) retained.
**Not** CUDA-only mid-path — staff `product-collocate-pbc` next; do not open more
energy dual-gates that rely on this hybrid.

## Launch

```text
workflow name=es-cpu-cuda-opt
args: {
  "mode": "exhaust",
  "staff_count": 2,
  "max_waves": 4,
  "do_perf": false,
  "target_branch": "dev",
  "theme": "product-collocate-pbc (device continuous + PBC wrap); then multigrid-si-energy-path; drop SG host cbg hybrid"
}
agent_budget: 256
```
