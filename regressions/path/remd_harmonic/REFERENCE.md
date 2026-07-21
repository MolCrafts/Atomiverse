# Reference — REMD Metropolis ladder (LJ dimer)

## Physics

Four temperatures, two LJ atoms, nested MD + Metropolis exchange

```
P_acc = min(1, exp[(β_i − β_j)(U_i − U_j)])
```

(Hukushima & Nemoto 1996; Sugita & Okamoto 1999). On accept, configurations
swap and velocities rescale by √(T_i/T_j) so each slot keeps its thermostat
target.

## Checks

- REMD constructs N independent MD programs (no shared NeighborList instance).
- Even/odd adjacent exchange schedule produces ≥ 50 attempts for 40 rounds × 4
  replicas.
- Accept ratio in (0.01, 0.99); seed-7 is deterministic across two runs.
- Collapse returns a 2-atom Frame (lowest-T slot) with finite coordinates.

## Deferred

Per-slot temperature-marginal histograms vs analytic harmonic-oscillator
energies need a one-particle external spring evaluator and longer sampling;
tracked under cpu-breadth-05 ac-009.
