# ES CPU + CUDA opt — handoff

Rewrite in place each harness wave.

## Status

- **Workflow:** `es-cpu-cuda-opt-5` **cancelled**. Do not resume old script.
- **Script:** `.grok/workflows/es-cpu-cuda-opt.rhai` — iron laws + skill gate +
  residency staff order + acceptance gate (grep mid-path DH).
- **Law:** [.claude/notes/cuda-only-no-dh.md](../../.claude/notes/cuda-only-no-dh.md)
- **Code:** Γ RKS / UKS / multi-k mid-SCF residency P0 landed (`5a5076d0` +
  force stream `61b91971`). GH200 residency dual-gates **1158949**.

## Tip-green (unmerged)

| Slug | Tip | Evidence |
| --- | --- | --- |
| `cuda-gpw-product-si-gate` | `wf/cuda-gpw-product-si-gate` @ `85d4ce33` | GH200 **1167099** \|dE\|=**2.130e-08** Ha Si Γ ngrids=1; KB Vnl@Γ + host cbg KS twin |

## P0 / staff residual

| Slug | Status |
| --- | --- |
| Γ RKS `device-h-assemble` | landed |
| UKS device H + pack from `d_P` | landed (`5a5076d0`) |
| multi-k collocate from `d_P` | landed (`5a5076d0`) |
| force free-fn non-default stream | landed (`61b91971`); host AO Pulay residual |
| GH200 residency retest | **GREEN job 1158949** |
| **`cuda-gpw-product-si-gate`** | **tip-green** GH200 **1167099** — merge candidate |
| Si MG energy path | open — rebase energy-path tip on product-si (same H0/KB + cbg/collint) |
| `product-collocate-pbc` | open — restore device continuous collocate after host twin |

## Launch

```text
workflow name=es-cpu-cuda-opt
args: {
  "mode": "exhaust",
  "staff_count": 2,
  "max_waves": 4,
  "do_perf": false,
  "target_branch": "dev",
  "theme": "Merge product-si-gate; Si MG energy-path after rebase; product-collocate-pbc for device continuous"
}
agent_budget: 256
```
