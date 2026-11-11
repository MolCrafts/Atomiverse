# Sampling and Free Energy

Tutorial twin of the [Sampling roadmap](../roadmap/sampling.md).

Monte Carlo and enhanced sampling share the same Frame currency as MD, but
the loop topology differs: MC proposes a trial and accepts or rejects; bias
methods reshape the effective free-energy surface along collective variables.

## What problem this solves

Rare events and free-energy differences are not reachable by plain NVT MD on
laboratory timescales. Sampling methods either change the ensemble (GCMC,
REMD) or add a history-dependent or windowed bias (umbrella, metadynamics)
and reconstruct the unbiased free energy afterwards (WHAM, TI).

## Roadmap spine

| Topic | Roadmap |
| ----- | ------- |
| NVT / NPT / GCMC | [Sampling §1](../roadmap/sampling.md#1-monte-carlo) |
| Umbrella, metaD, REMD | [Sampling §2](../roadmap/sampling.md#2-enhanced-sampling) |
| TI, WHAM, Widom | [Sampling §3](../roadmap/sampling.md#3-free-energy) |

Related pedagogy: [Statistical ensembles](ch11-ensembles.md) (thermodynamic
contexts for NVT/NPT) and [classical MD](ch10-classical-md.md).

## Practice pointers

- MC presets: `cpu::mc::MC`, `NptMc`, `Gcmc` — outcome policy + mover, single
  rank.
- Bias: `DistanceCV` / `AngleCV` / `DihedralCV` + `HarmonicBias` or
  `Metadynamics` / `Metadynamics1D`.
- Regression reports: [method catalog](methods/catalog.md) (search `umbrella`,
  `metad`, `wham`, `gcmc`, `remd`).
