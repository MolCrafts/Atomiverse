# ES CPU + CUDA opt — handoff

Rewrite in place each harness wave.

## Status

- **Workflow:** `es-cpu-cuda-opt-5` **cancelled**. Do not resume old script.
- **Script:** `.grok/workflows/es-cpu-cuda-opt.rhai` — iron laws + skill gate +
  residency staff order + acceptance gate (grep mid-path DH).
- **Law:** [.claude/notes/cuda-only-no-dh.md](../../.claude/notes/cuda-only-no-dh.md)
- **Code:** Γ RKS mid-SCF no V_AO DH (`af1ddf75` + follow-ons). UKS / multi-k
  P-residency / forces still residual until landed.

## Launch after residency P0

```text
workflow name=es-cpu-cuda-opt
args: {
  "mode": "exhaust",
  "staff_count": 2,
  "max_waves": 4,
  "do_perf": false,
  "target_branch": "dev",
  "theme": "CUDA iron law only: finish mid-path DH kill (uks-device-h, kpoint-p-residency); load CUDA kernel skills; zero scientific mid-path DH; no dual-gate spam"
}
agent_budget: 256
```

## P0 checklist

| Slug | Status |
| --- | --- |
| Γ RKS `device-h-assemble` | landed |
| UKS device H + pack from `d_P` | landed (`5a5076d0`) |
| multi-k collocate from `d_P` (no P ping-pong) | landed (`5a5076d0`) |
| force free-fn non-default stream | landed (`61b91971`); host AO Pulay residual |
| GH200 retest | **GREEN job 1158949** — `RESIDENCY_DUAL_GATES_PASS` @ `856e7448` |
| Si MG energy path | open (correctness) |

### GH200 1158949 evidence (2026-08-11)

| Gate | Result |
| --- | --- |
| `reg_cuda-gpw-full-product` | PASS \|dE\|=**1.616e-07** max\|dF\|=**2.787e-06** |
| `reg_cuda-gpw-device-forces` | PASS \|dE\|=**1.616e-07** max\|dF\|=**3.068e-06** |
| `reg_cuda-gpw-uks-o2-gate` | PASS \|dE\|=**3.520e-06** |
| `reg_cuda-gpw-kpoints` | PASS \|dE\|=**3.086e-14** nk=2 |
| ctest `test_cuda_gpw_(full_product\|diis_eigen\|kpoints\|forces)` | PASS (log) |
