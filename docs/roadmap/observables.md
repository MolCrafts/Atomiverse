# Observables and Analysis

Tutorial: [Trajectory analysis](../tutorial/ch13-observables.md).

Census of `src/cpu/observe/`. Observers **compute**; Recorders write I/O —
never both. Trajectory math lives in **molrs**; C++ streams frames in.
Legend: [Roadmap index](index.md).

CUDA counterparts: Mulliken, RDF, stress only.

| Observable | Where | Anchoring |
| ---------- | ----- | --------- |
| RDF g(r) | `RDF` (+ CUDA); math in molrs | **No ATV e2e regression** — molrs-side only |
| MSD / VACF | AsyncObservers | Unit tests; numerics molrs-side |
| Trajectory-metric accumulators (MSD / VACF / RDF, streaming only) | `trajectory_metrics.h` PIMPL | Unit tests; numerics molrs-side |
| Electron density grid | `DensityGrid` | Via cube Recorder unit test |
| Species density profile | `SpeciesDensityProfile` | Unit + EDL box regression |
| Multipole / dipole | `MultipoleObserver` | PySCF-pinned |
| Virial / kinetic stress | `Virial` | Indirect via NPT/NPH regressions |
| Mulliken | CPU + CUDA | MOPAC / GPW DFT+U |
| Löwdin / MK / CHELPG | CPU | PySCF `h2o_charges_pyscf` |
| Widom μ_ex | `Widom` | Analytic HS free volume |
| WHAM | `Wham` → molrs | Analytic double-well |
| Voronoi volume | `VoronoiVolume` | MetalWalls |
| Ψ(z) profile | `PotentialProfile` | Analytic planar Poisson |
| Electrode Q(t) | `ElectrodeChargeObserver` | LAMMPS |
| Capacitance ⟨δQ²⟩ | `CapacitanceChargeFluctuation` | Limmer closed form |
| C_QQ(τ) / elastance | electrode observers | **No standing regression** — owe external pin |
| Classical polarization P | `ClassicalPolarization` | MetalWalls finite-field |

## Claim gap

Every observable owes **one analytic pin** of its own. Worst offenders: RDF
(ideal-gas g(r)=1 or FCC peak), MSD (free-flight / Einstein), VACF
(Langevin exponential). molrs’s tests validate math, not this engine’s
streaming path (cadence, unwrap, masks).
