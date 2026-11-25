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

- `cpu::mc::MC` runs single-rank equilibrium sampling with `set_nvt`,
  `set_npt` or `set_gcmc` and a mover. `cpu::mc::KMC` inherits
  the run lifecycle and uses `set_event` and `set_solver` for continuous-time sampling.
- Bias: `DistanceCV` / `AngleCV` / `DihedralCV` + `HarmonicBias` or
  `Metadynamics` / `Metadynamics1D`.
- Regression reports: [method catalog](methods/catalog.md) (search `umbrella`,
  `metad`, `wham`, `gcmc`, `remd`).

## Mixing polymer moves

Pass one `MoveSelector` to `set_mover`. Each `add(mover, weight)` gives a
relative draw weight, and each trial picks exactly one child. A selector can
contain another selector. A rejected trial consumes its draw.

```python
import atomiverse as av

mc = av.MC(seed=23)
mc.set_nvt(300)
mc.set_units("metal")
mc.set_mover(
    av.MoveSelector()
    .add(av.TorsionMove(paths=[(23, 24)], max_angle=0.4), weight=4)
    .add(av.CBMCMove(paths=[(24, 25, 26)], trials=6, bond_sigma=1.4, bias_beta=15), weight=1)
)
# Attach the same classical evaluators and neighbor settings used for this
# system in MD, then call mc.run(frame, n_trials).
```

The integer paths are ordered atom tags following existing bonds. The first
tag of a directed bond stays fixed. The remainder of the frame supplies the
environment for candidate scoring. Angles are in radians. `bond_sigma` uses
the active length unit and `bias_beta` is an inverse energy for candidate
selection. The ensemble temperature controls the final Metropolis test.

| Mover | Selection and applicability |
| ----- | --------------------------- |
| `TorsionMove` | Directed bond. Rotate the fragment on the second tag's side. An alternate path across the cut makes the draw ineligible. |
| `EndMerRotationMove` | Three consecutive tags. Rotate the terminal bead around the first bond. |
| `LibrationMove`, `DimerFlipMove` | Three or four consecutive tags. Rotate the interior about the line through the exterior anchors. The affected bond angles must be flexible. |
| `PivotMove` | Directed bond with an isotropic rotation axis through the first tag. The angles at the pivot must be flexible. |
| `ReptationMove` | Whole linear homopolymer paths. Shift coordinates and draw a new end bond from a Cartesian Gaussian. |
| `CBMCMove` | First tag fixed, remaining branch-free flexible segment grown toward a chain end. |
| `ExtendedCBMCMove` | First and last tags fixed. Regrow the flexible branch-free interior using conditional Gaussian draws. |
| `ConRotMove` | Eight consecutive tags with a branch-free moving interior. Change one driver torsion, solve five dependent torsions, and hold the two tags at each end fixed. |
| `ParallelRotationMove` | Six consecutive tags. Change one driver torsion and solve three dependent torsions. Translate the outgoing tail while preserving its orientation. |
| `GeneralizedReptationMove` | Transfer an end bead between linear homopolymers with identical bead and bond parameters. Use it for a bond-only model that permits chain lengths to fluctuate. |
| `AtomIdentityExchangeMove` | Fixed list of pairs of unbonded atom tags. Swap their spatial sites at fixed composition. |
| `VolumeMove` | Isotropic box rescaling for NPT. Use Cartesian particles with compatible molecular geometry. |

Both CBMC movers grow the segment one bead at a time. Each candidate uses
full-frame energy with a provisional completion of the ungrown tail. The
forward and reverse pools share the same auxiliary completion rule. The
reverse pool contains the old bead. The reference Gaussian densities and
the Rosenbluth selection probabilities enter the proposal ratio once.
Use energy evaluators whose scores are reproducible functions of the trial
state. A draw with no valid reverse candidate is rejected.

ConRot and parallel rotation use local deterministic closure followed by
a reverse solve to verify the same branch. The acceptance correction is
the ratio of inverse closure Jacobians. The move is rejected if closure
is singular, fails in either direction or changes the nearest periodic
image of an affected bond.

Topology-changing moves require a target ensemble over both coordinates and
bond graphs. `GeneralizedReptationMove` samples the Boltzmann weight of the
specified Hamiltonian over allowed labeled bond graphs. It conserves the
number of chains and beads, and retains a minimum of two beads per chain.
Combine it with components whose applicability survives the bond change.
