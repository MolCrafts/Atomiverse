# ES CPU + CUDA opt — handoff

Rewrite in place each harness wave.

## Status

- **Workflow:** `es-cpu-cuda-opt` script
  `.grok/workflows/es-cpu-cuda-opt.rhai` — iron laws + kernel skills +
  residency staff order + acceptance gate (grep mid-path DH).
- **Law:** [.claude/notes/cuda-only-no-dh.md](../../.claude/notes/cuda-only-no-dh.md)
- **Harness wave 4** on `dev` @ `88d8ec59` (after GapCensus wave 4 @
  `213d794d` / `5216ecda`): pending **132** / in_progress **1** / gaps
  **134**; **cpu_pending=0**; **new_slugs_added=0** this harness.
- **Residency P0:** Γ RKS / UKS / multi-k mid-SCF landed (`5a5076d0` + force
  stream `61b91971`). GH200 residency dual-gates **1158949** — queue marks
  `device-h-assemble` / `uks-h-assemble` / `uks-p-residency` /
  `kpoint-p-residency` / `kpoint-device-energy` / `force-stream` **done**.
- **Si product gate:** **landed** on `dev` (`a158848f`). GH200 **1167099**
  Si Γ ngrids=1 \|dE\|=**2.130e-08** Ha; post-collocate-pbc hold **1173564**
  \|dE\|=**3.504e-05** Ha.
- **PBC collocate + SG device KS:** **landed** merge `cfb246d1` / feat
  `5648608f`. GH200 **1173564** / **1177100**.
- **Si MG energy path:** **landed** merge `88d8ec59` / feat `c99f4622` + fix
  `c2084efe`. GH200 **1180883** \|dE\|=**2.130e-08** Ha (host collint
  multi-level mid-SCF + device H/DIIS). Iron-law mid-path P/V →
  `multigrid-device-ks`.
- **Staffed `[~]` (remain):** `cuda-gpw-device-payload` only (tip **1178026**
  unmerged). Free next: **`multigrid-device-ks`** (iron-law monom restore) /
  `multigrid-si-gate` (unblocked) / payload-family residual.

## Landed this harness

| Item | Evidence |
| --- | --- |
| `cuda-gpw-multigrid-si-energy-path` | merge `88d8ec59` / feat `c99f4622` / fix `c2084efe` / re-verify `60c77321`; host collint multi-level (`multigrid_collocate_rho` + `ks_grid_build_rks` + `multigrid_integrate_veff`) + device H/DIIS; GH200 **1180883** \|dE\|=**2.130e-08** Ha (prior **1180409**); device monom residual **1180025** \|dE\|=**4.3e-4** → child **`multigrid-device-ks`**; **perf_exempt** |
| prior collocate-pbc + product-si + residency P0 | `cfb246d1` / **1173564**; `a158848f` / **1167099**; `5a5076d0` / `61b91971` / **1158949** |

## Residual (do not soft-land)

| Slug | Status |
| --- | --- |
| `device-payload` | **`[~]`** tip **1178026** unmerged — boundary payload eigen |
| `multigrid-device-ks` | **open** — restore device monom MG mid-path (iron-law; **1180025** monom fail) |
| `multigrid-si-gate` | **unblocked** after energy-path on base (staff free) |
| `uks-payload-device` / `device-cold-start` | open (Γ UKS payload / cold densify) |
| multi-k χ / orth / vnl / h0 / payload / cold-start | open |
| force free-fn | host AO Pulay residual (P1); stream **landed** |
| stress free-fn | full host |

## Stated hybrid residual

Single-grid `cuda::gpw::GpwDft` mid-SCF is **device** product KS with
sphere-window PBC collocate/integrate (iron-law P0 SG closed). Si MG default
on base is **host collint multi-level** mid-SCF (energy dual-gate green) until
`multigrid-device-ks` restores device pack→`KsMultiGridChainDevice`→scatter.
CompactBasisGrid retained **force-only** / payload boundary. Remaining hybrid:
host Γ STVnl, final payload host re-diag (tip only for device payload),
cold-start densify, multi-k χ·orth·vnl·h0, host AO Pulay forces, full-host
stress.

## Launch

```text
workflow name=es-cpu-cuda-opt
args: {
  "mode": "exhaust",
  "staff_count": 2,
  "max_waves": 4,
  "do_perf": false,
  "target_branch": "dev",
  "theme": "merge device-payload tip; staff multigrid-device-ks (mid-path DH monom restore) + multigrid-si-gate"
}
agent_budget: 256
```
