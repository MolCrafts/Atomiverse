# Observables and Analysis

Legend and caveats: [Roadmap index](index.md).

This is the census of `src/cpu/observe/` — everything that computes a derived
quantity from a Frame or a trajectory. Observers **compute**; Recorders
(`cpu/record/`) write I/O; the two never mix. Per the molrs boundary, **no
trajectory-analysis math lives in C++**: the RDF/MSD/VACF observers stream
positions or velocities into molrs accumulators, and the batch compute objects
in `trajectory_metrics.h` are PIMPL wrappers over molrs `compute::*`. CUDA has
exactly three counterparts (`cuda/observe/mulliken.cuh`, `rdf.cuh`,
`stress.cuh`); everything else is CPU-only.

Anchoring is the honest column here. "Pinned" means a `regressions/` case
checks the number against an external code or a closed form; "unit tests
only" means the ATV-side plumbing is tested but no end-to-end number is
gated; "molrs-side" means the numerical math is covered only by molrs's own
test suite, not by anything in this repo.

| Observable | Where it lives | Anchoring |
| ---------- | -------------- | --------- |
| RDF g(r) | `RDF` (`cpu/observe/rdf.h`, AsyncObserver; math in molrs; CUDA `rdf.cuh`) | **No regression.** No CPU unit test of the observer; CUDA has a molrs-bridge unit test (`tests/cuda/observe/test_rdf_molrs.cu`). g(r) numerics covered molrs-side only. |
| g(r) curve emitter | `rdf_write_curve` (`cpu/observe/rdf_curve.h`) | I/O only — the single emitter of the g(r) file format. |
| MSD | `MSD` (`cpu/observe/msd.h`, AsyncObserver) | **No regression.** Unit test only (`test_msd.cpp`); numerics molrs-side. |
| VACF | `VACF` (`cpu/observe/vacf.h`, AsyncObserver) | **No regression.** Unit test only (`test_vacf.cpp`); numerics molrs-side. |
| Batch trajectory metrics | `MSD` / `VACF` / `RDF` / `EinsteinDiffusion` + streaming accumulators (`cpu/observe/trajectory_metrics.h`, PIMPL over molrs) | **No ATV-side numerical coverage** — molrs's own tests are the only gate. |
| Electron density grid | `eval_density_grid` / `DensityGrid` (`cpu/observe/density.h`) | No regression; exercised through the cube Recorder (`test_density_cube.cpp`). |
| Species density profile | `SpeciesDensityProfile` (`cpu/observe/species_density_profile.h`) | Unit test + exercised inside the MetalWalls-pinned EDL box `regressions/md/const-pot-il-edl-box`. |
| Multipole / dipole | `MultipoleObserver` (`cpu/observe/multipole.h`) | PySCF-pinned: `regressions/scf/h2o_rhf_dipole`, `h2o_embedding_field`. |
| Virial / kinetic stress | `Virial` (`cpu/observe/virial_stress.h`) | Indirect: the pressure path is load-bearing in the NPT/NPH cell regressions (`regressions/md/npt_pme_nacl` analytic rock salt, `pr_npt_lj`, `nph_lj`); no standalone stress pin. |
| Mulliken charges / spin | `mulliken_charge` / `mulliken_spin` (`cpu/observe/mulliken.h`; CUDA `mulliken.cuh`) | MOPAC-pinned via `regressions/semiempirical/am1-water-mopac-energy`; Mulliken occupations also load-bearing in the CP2K-pinned GPW DFT+U cases. |
| Löwdin charges / spin | `lowdin_charge` / `lowdin_spin` (`cpu/observe/lowdin.h`) | PySCF-pinned: `regressions/scf/h2o_charges_pyscf` (≤ 1e-6 e). |
| Merz–Kollman ESP charges | `MKCharges` (`cpu/observe/esp_charges.h`) | PySCF-pinned: `regressions/scf/h2o_charges_pyscf` (≤ 1e-3 e). |
| CHELPG charges | `ChelpgCharges` (`cpu/observe/esp_charges.h`) | PySCF-pinned: `regressions/scf/h2o_charges_pyscf` (≤ 1e-3 e). |
| Widom insertion μ_ex | `Widom` (`cpu/observe/widom.h`) | Analytic pin: hard-sphere free volume, `regressions/mc/widom_hs`. |
| WHAM bridge | `Wham` (`cpu/observe/wham.h`; solver in molrs) | Analytic pin: double-well reconstruction, `regressions/sampling/wham_double_well`. |
| Voronoi volume | `VoronoiVolume` (`cpu/observe/voronoi_volume.h`) | MetalWalls-pinned: `regressions/md/voronoi-volume-tf`. |
| Slab potential profile Ψ(z) | `PotentialProfile` (`cpu/observe/potential_profile.h`) | Analytic planar-Poisson pin: `regressions/md/potential-profile-observer`; also in `const-pot-il-edl-box`. |
| Electrode charge Q(t) | `ElectrodeChargeObserver` (`cpu/observe/electrode_charge.h`) | LAMMPS-pinned: `regressions/md/const-pot-charge-observer`. |
| Capacitance from charge fluctuations | `CapacitanceChargeFluctuation` (`cpu/observe/capacitance_charge_fluctuation.h`) | Literature closed form (Limmer JCP 2013): `regressions/md/capacitance-charge-fluctuation`. |
| Charge–charge autocorrelation C_QQ(τ) | `ChargeChargeAutocorr` (`cpu/observe/charge_autocorr.h`) | **Self-consistency pins only** (`regressions/md/charge-charge-autocorr`) — owes a real external number. |
| Elastance cond(A) / empty capacitance | `ElastanceObserver` (`cpu/observe/elastance_observer.h`) | **Self-consistency pins only** (`regressions/md/elastance-condition-observer`) — owes a real external number. |
| Classical polarization density P | `ClassicalPolarization` (`cpu/observe/classical_polarization.h`) | MetalWalls-pinned via `regressions/md/finite-field-single-electrode`. |

`ConstantCountGate` (`cpu/observe/constant_count_gate.h`) is shared observer
infrastructure (snapshot atom-count latch), not an observable.

## Claim gap

Every observable owes **one analytic pin** of its own — a closed-form case
that gates the number, not the plumbing. The trajectory trio is the worst
offender: RDF has no end-to-end case at all (an ideal-gas g(r) = 1 or an FCC
lattice peak-position case would do), MSD owes a free-flight
MSD(t) = ⟨v²⟩t² or Einstein-diffusion Langevin pin, VACF an exponential-decay
Langevin pin. Relying on molrs's own tests is a dependency statement, not a
validation of this engine's streaming path (snapshot cadence, unwrapping,
species masks). The same standard retires the two self-consistency-pinned
electrode observers above.
