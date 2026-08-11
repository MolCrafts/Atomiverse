# ES CPU + CUDA opt — handoff

Rewrite in place each harness wave.

## Status

- **Workflow:** `es-cpu-cuda-opt` script
  `.grok/workflows/es-cpu-cuda-opt.rhai` — iron laws + kernel skills +
  residency staff order + acceptance gate (grep mid-path DH).
- **Law:** [.claude/notes/cuda-only-no-dh.md](../../.claude/notes/cuda-only-no-dh.md)
- **Harness wave 5** on `dev` @ `d476ee7a` (after Assign wave 5 @
  `d3c6c5dc` / GapCensus @ `6595ef11`): pending **131** / in_progress **1** /
  gaps **133**; **cpu_pending=0**; **new_slugs_added=0** this harness.
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
  **`multigrid-device-ks` staffed `[~]`**.
- **Γ RKS device final payload:** **landed** merge `d476ee7a` / feat
  `d4a00181` + GH200 **1182615** / **1180932**. Boundary D2H only
  (`eigen_from_device`); no host re-diag.
- **Staffed `[~]` (remain):** `cuda-gpw-multigrid-device-ks` only (iron-law
  monom restore). Free next: `multigrid-si-gate` (unblocked) /
  `uks-payload-device` / `device-cold-start` / force handoff.

## Landed this harness

| Item | Evidence |
| --- | --- |
| `cuda-gpw-device-payload` | merge `d476ee7a` / feat `d4a00181` / re-verify `61c2dfd3`; `AoDiisEigenDevice::eigen_from_device` of assembled H; GH200 **1182615** \|dE\|=**1.252e-08** max\|deps\|=**1.888e-06** Ha (prior **1180932** / **1178026**); mid-path scientific matrix DH **zero**; skills: av-perf + write-numerically-stable-kernel / optimize-global-memory-access; **perf_exempt** |
| prior si-energy-path + collocate-pbc + product-si + residency P0 | `88d8ec59` / **1180883**; `cfb246d1` / **1173564**; `a158848f` / **1167099**; `5a5076d0` / `61b91971` / **1158949** |

## Residual (do not soft-land)

| Slug | Status |
| --- | --- |
| `multigrid-device-ks` | **`[~]`** Assign wave 5 — restore device monom MG mid-path (iron-law; **1180025** monom fail) |
| `multigrid-si-gate` | **unblocked** after energy-path on base (staff free after monom) |
| `uks-payload-device` / `device-cold-start` | open (Γ UKS payload / cold densify) |
| multi-k χ / orth / vnl / h0 / payload / cold-start | open |
| force free-fn | host AO Pulay residual (P1); stream **landed** |
| stress free-fn | full host |

## Stated hybrid residual

Single-grid `cuda::gpw::GpwDft` mid-SCF is **device** product KS with
sphere-window PBC collocate/integrate (iron-law P0 SG closed). Γ RKS final
payload is **device eigen** (boundary C/ε/H D2H). Si MG default on base is
**host collint multi-level** mid-SCF (energy dual-gate green) until
`multigrid-device-ks` restores device pack→`KsMultiGridChainDevice`→scatter.
CompactBasisGrid retained **force-only**. Remaining hybrid: host Γ STVnl,
UKS/multi-k final payload host re-diag, cold-start densify, multi-k
χ·orth·vnl·h0, host AO Pulay forces, full-host stress.

## Launch

```text
workflow name=es-cpu-cuda-opt
args: {
  "mode": "exhaust",
  "staff_count": 2,
  "max_waves": 4,
  "do_perf": false,
  "target_branch": "dev",
  "theme": "Harness wave 5 residual: staff multigrid-device-ks monom restore (zero mid-path matrix DH; GH200; no dual-gate invent); free uks-payload / cold-start / multigrid-si-gate after monom"
}
agent_budget: 256
```
