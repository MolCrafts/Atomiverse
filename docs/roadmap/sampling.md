# Sampling and Free Energy

Tutorial: [Sampling and free energy](../tutorial/sampling.md).

The CPU `MC` driver uses an equilibrium outcome policy and a mover.
The `KMC` driver derives from `MC` and composes event and solver components.
Enhanced sampling is a textbook
suite, not a PLUMED replacement. Legend: [Roadmap index](index.md).

## 1. Monte Carlo

| Capability | CPU | CUDA | Notes |
| ---------- | :-: | :--: | ----- |
| NVT Metropolis | ✓ | ○ | Single rank; O(N) trial energy |
| NPT-MC | ✓ | ○ | |
| GCMC (μVT) | ✓ | ○ | Single insertable species |
| Continuous-time kMC | ✓ | ○ | Direct scheduler, user-supplied rates, discrete-state model |
| Weighted mover composition | ✓ | ○ | One fixed-weight draw per trial, including nested selectors |
| Local polymer rotations, pivot, reptation | ✓ | ○ | Tag-selected paths, ordinary reptation uses linear homopolymers |
| CBMC / fixed-endpoint CBMC | ◐ | ○ | Flexible branch-free growth paths and full-energy candidate scores |
| ConRot / parallel rotation | ◐ | ○ | Reversible local closure branch with Jacobian correction |
| Intra-chain / inter-chain concerted rotation for semicrystalline interphases | ○ | ○ | Requires a defined topology and interphase ensemble |
| Generalized reptation | ◐ | ○ | Bond-only linear homopolymers with variable chain lengths |
| Atom identity exchange | ◐ | ○ | Fixed-composition site swaps between unbonded particles |
| Gibbs / Wang–Landau | ○ | ○ | |

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
