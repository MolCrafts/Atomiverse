# Architecture campaign — handoff

Rewrite in place each harness wave.

## Status

- **Parent:** `dev`
- **Wave-1 done:** e2e-in-tests delete (78 sources)
- **Wave-2 done:** `arch-gpw-driver-node` — GpwDft is `cpu::Driver<gpw::Phase>`;
  Γ RKS DIIS driven by stages; residual multi-k/UKS/OT path migration staffed as
  `arch-gpw-driver-paths`
- **Next staff:** `arch-gpw-driver-paths` + `arch-pw-driver-node` (P0);
  `arch-e2e-tests-wave2` (P2 independent)

## Run

```text
workflow name=arch-campaign
args: { "mode": "exhaust", "staff_count": 4, "max_waves": 8, "target_branch": "dev" }
agent_budget: 256+
```

## Disk

Always wipe builds; re-scan empty before next GapCensus.
