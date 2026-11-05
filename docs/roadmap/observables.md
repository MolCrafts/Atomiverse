# Observables and Analysis

Tutorial: [Trajectory analysis](../tutorial/ch13-observables.md).

Census of `src/cpu/observe/`. Observers **compute**; Recorders write I/O —
never both. Trajectory math lives in **molpy**; C++ streams frames in.
Legend: [Roadmap index](index.md).

CUDA counterparts: Mulliken, RDF, stress only.

| Observable | Where | Anchoring |
| ---------- | ----- | --------- |
| RDF g(r) | `RDF` (+ CUDA); math in molpy | **No ATV e2e regression** — molpy-side only |
| MSD / VACF | AsyncObservers | Unit tests; numerics molpy-side |
| Trajectory-metric accumulators (MSD / VACF / RDF, streaming only) | `trajectory_metrics.h` PIMPL | Unit tests; numerics molpy-side |
| Electron density grid | `DensityGrid` | Via cube Recorder unit test |
| Species density profile | `SpeciesDensityProfile` | Unit + EDL box regression |
| Multipole / dipole | `MultipoleObserver` | PySCF-pinned |
| Virial / kinetic stress | `Virial` | Indirect via NPT/NPH regressions |
| Mulliken | CPU + CUDA | MOPAC / GPW DFT+U |
| Löwdin / MK / CHELPG | CPU | PySCF `h2o_charges_pyscf` |
| Widom μ_ex | `Widom` | Analytic HS free volume |
| Voronoi volume | `VoronoiVolume` | MetalWalls |
| Ψ(z) profile | `PotentialProfile` | Analytic planar Poisson |
| Electrode Q(t) | `ElectrodeChargeObserver` | LAMMPS |
| Capacitance ⟨δQ²⟩ | `CapacitanceChargeFluctuation` | Limmer closed form |
| C_QQ(τ) / elastance | electrode observers | **No standing regression** — owe external pin |
| Classical polarization P | `ClassicalPolarization` | MetalWalls finite-field |

## Claim gap

Every observable owes **one analytic pin** of its own. Worst offenders: RDF
(ideal-gas g(r)=1 or FCC peak), MSD (free-flight / Einstein), VACF
(Langevin exponential). molpy’s tests validate math, not this engine’s
streaming path (cadence, unwrap, masks).
