# Architecture campaign — handoff

Rewrite in place each harness wave.

## Status

- **Parent:** `dev`
- **Wave-1 done:** e2e-in-tests delete (78 sources) — see
  [e2e-in-tests-campaign.md](e2e-in-tests-campaign.md)
- **Next staff:** `arch-gpw-driver-node` (P0)

## Run

```text
workflow name=arch-campaign
args: { "mode": "exhaust", "staff_count": 4, "max_waves": 8, "target_branch": "dev" }
agent_budget: 256+
```

## Disk

Always wipe builds; re-scan empty before next GapCensus.
