# Sampling and Free Energy

Tutorial: [Sampling and free energy](../tutorial/sampling.md).

Monte Carlo: three CPU presets over one trial loop (`MC`, `NptMc`, `Gcmc`) —
outcome policy + mover, no category driver. Enhanced sampling is a textbook
suite, not a PLUMED replacement. Legend: [Roadmap index](index.md).

## 1. Monte Carlo

| Capability | CPU | CUDA | Notes |
| ---------- | :-: | :--: | ----- |
| NVT Metropolis | ✓ | ○ | Single rank; O(N) trial energy |
| NPT-MC | ✓ | ○ | |
| GCMC (μVT) | ✓ | ○ | Single insertable species |
| Gibbs / CBMC / Wang–Landau / … | ○ | ○ | |

## 2. Enhanced sampling

| Capability | CPU | CUDA | Notes |
| ---------- | :-: | :--: | ----- |
| Umbrella (Distance / Angle) | ✓ | ○ | `HarmonicBias` |
| Umbrella (Dihedral) | ◐ | ○ | `HarmonicBias`; no external pin, self-pinned case removed |
| Well-tempered metaD (1D) | ◐ | ○ | `Metadynamics1D`; no external pin, self-pinned case removed |
| Multi-CV well-tempered metaD | ✓ | ○ | `Metadynamics` |
| T-REMD | ✓ | ○ | |
| H-REMD (λ ladder) | ✓ | ○ | |
| Multi-walker / OPES / ABF / REST2 | ○ | ○ | |

## 3. Free energy

| Capability | CPU | CUDA | Notes |
| ---------- | :-: | :--: | ----- |
| Soft-core λ TI (`dU/dλ`) | ✓ | ○ | Primitive, not multi-window campaign |
| FEP (same soft-core path) | ✓ | ○ | |
| WHAM (multi-window) | ○ | ○ | Needs a molpy WHAM solver; umbrella regression carries a smoke WHAM only |
| Widom insertion | ✓ | ○ | Observer / snapshot estimator |
| MBAR / absolute solvation / BAR campaigns | ○ | ○ | |
