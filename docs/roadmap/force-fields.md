# Force Fields

Legend and caveats: [Roadmap index](index.md).

This is the thinnest area of the engine relative to its infrastructure. The
MD machinery around the potentials (neighbor lists, domain decomposition, halo
exchange, PME, special-bonds exclusions, virial accumulation) is mature and
LAMMPS-shaped; the *catalogue of styles plugged into it* is small — one real
pair style, three bonded styles, no impropers, no many-body potentials.

The Lennard–Jones 12-6 pair style is the only ⚡ in the roadmap: it is
SIMD-vectorized (xsimd, `-march=native`) on CPU, has a device kernel on CUDA,
and both are benchmarked against LAMMPS/Kokkos in
`/* benchmarks removed; re-add later */ lj/performance/`.

## 1. Classical Bonded Potentials

| Method                             | CPU | CUDA |
| ---------------------------------- | :-: | :--: |
| Harmonic bond potential            |  ✓  |   ✓  |
| Quartic bond potential             |  ○  |   ○  |
| Morse bond potential               |  ○  |   ○  |
| FENE bond potential                |  ○  |   ○  |
| Tabulated bond potential           |  ○  |   ○  |
| Harmonic angle potential           |  ✓  |   ✓  |
| Cosine angle potential             |  ○  |   ○  |
| Restricted-bending angle potential |  ○  |   ○  |
| Urey–Bradley potential             |  ○  |   ○  |
| Tabulated angle potential          |  ○  |   ○  |
| Periodic Fourier torsion           |  ✓  |   ○  |
| OPLS torsion                       |  ○  |   ○  |
| Ryckaert–Bellemans torsion         |  ✓  |   ✓  |
| CHARMM torsion                     |  ✓  |   ○  |
| Harmonic improper torsion          |  ✓  |   ○  |
| Periodic improper torsion          |  ○  |   ○  |
| CVFF improper torsion              |  ○  |   ○  |
| CMAP correction                    |  ○  |   ○  |
| Bond–bond cross term               |  ○  |   ○  |
| Bond–angle cross term              |  ○  |   ○  |
| Angle–angle cross term             |  ○  |   ○  |
| Bond–torsion cross term            |  ○  |   ○  |
| Class-II bonded potential          |  ○  |   ○  |
| Tabulated bonded potential         |  ○  |   ○  |

Two torsion styles share the same topology and geometry helpers
(`dihedral_geom`):

- `DihedralMultiHarmonic` — LAMMPS `dihedral_style multi/harmonic`:
  \(V = \sum_n C_n \cos^n\phi\) (Ryckaert–Bellemans), CUDA mirror present.
- `DihedralCharmm` — LAMMPS `dihedral_style charmm` **without** optional 1–4
  pair weights: \(V = K(1 + \cos(n\phi - \delta))\). CPU only for now; 1–4
  scaling remains on pair styles via `special_bonds`. Regression
  `regressions/md/dihedral_charmm` pins analytic energies at \(\phi=\pi/2\).

Neither is an OPLS parameterization table.

**Improper topology** is first-class (schema columns `num_improper` /
`improper_atom` / `improper_type`, independent of dihedrals).
`ImproperHarmonic` implements LAMMPS `improper_style harmonic`
\(V = K(\chi-\chi_0)^2\) using the shared `dihedral_geom` helpers
(`regressions/md/improper_harmonic`). Periodic / CVFF impropers remain ○.

Bonded forces are cross-validated against LAMMPS in
`/* benchmarks removed; re-add later */ pe/validation/` (multi/harmonic path).

## 2. Classical Nonbonded Potentials

| Method                              | CPU | CUDA |
| ----------------------------------- | :-: | :--: |
| Lennard–Jones 12-6 potential        |  ⚡  |   ⚡  |
| Mie n-m potential                   |  ○  |   ○  |
| Buckingham potential                |  ○  |   ○  |
| Morse nonbonded potential           |  ✓  |   ○  |
| Tabulated pair potential            |  ○  |   ○  |
| Coulomb potential                   |  ✓  |   ○  |
| Reaction-field electrostatics       |  ○  |   ○  |
| Damped-shifted-force electrostatics |  ○  |   ○  |
| Wolf summation                      |  ○  |   ○  |
| Direct Ewald summation              |  ✓  |   ✓  |
| Particle Mesh Ewald                 |  ✓  |   ✓  |
| Smooth Particle Mesh Ewald          |  ✓  |   ✓  |
| Lennard–Jones PME                   |  ○  |   ○  |
| Fast multipole electrostatics       |  ○  |   ○  |
| Multilevel summation method         |  ○  |   ○  |
| Soft-core Lennard–Jones potential   |  ○  |   ○  |
| Soft-core Coulomb potential         |  ○  |   ○  |

Notes:

- **LJ 12-6** is energy-shifted, supports a **per-type ε/σ table** with
  Lorentz–Berthelot mixing (and a single-ε fast path), honours LAMMPS
  `special_bonds` **scales** (0 = drop; Amber/GAFF 1–4 uses 0.5), and
  optionally accumulates the per-pair virial. See `set_special_bonds(0,0,0.5)`.
- **Morse** exists as a demo style: O(N²) all-pairs, deliberately opting out of
  the neighbor list. Correct, but not a production style — and CPU-only.
- **Cut-off Coulomb** is `PairCoulCut` (CPU; special_bonds scales). **CoulLong**
  (real-space `erfc(αr)/r`) also applies special-bonds scales on the real-space
  loop; reciprocal/self terms are unscaled (LAMMPS convention). GAFF path
  regression: `regressions/forcefield/gaff_peo_fragment`.
- The PME implementation is Essmann **smooth** PME (B-spline charge spreading →
  FFT → influence function), so the "Particle Mesh Ewald" and "Smooth Particle
  Mesh Ewald" rows are the same code. Despite an in-code reference to LAMMPS
  `kspace_style pppm`, this is SPME, not PPPM.
- Reciprocal space is **single-GPU only** on CUDA; the CPU PME replicates the
  grid across MPI ranks rather than distributing the FFT.

## 3. Polarizable Force Fields

| Method                              | CPU | CUDA |
| ----------------------------------- | :-: | :--: |
| Drude oscillator model              |  ○  |   ○  |
| Self-consistent induced dipoles     |  ○  |   ○  |
| Extended-Lagrangian induced dipoles |  ○  |   ○  |
| Thole-damped polarization           |  ○  |   ○  |
| Atomic-multipole electrostatics     |  ○  |   ○  |
| AMOEBA electrostatics               |  ○  |   ○  |
| AMOEBA polarization                 |  ○  |   ○  |
| Fluctuating-charge model            |  ○  |   ○  |
| QEq polarization                    |  ○  |   ○  |
| Charge-equilibration dynamics       |  ○  |   ○  |

## 4. Many-Body and Reactive Potentials

| Method                         | CPU | CUDA |
| ------------------------------ | :-: | :--: |
| Embedded-atom method           |  ✓  |   ○  |
| Modified embedded-atom method  |  ○  |   ○  |
| Finnis–Sinclair potential      |  ○  |   ○  |
| Stillinger–Weber potential     |  ✓  |   ○  |
| Tersoff potential              |  ○  |   ○  |
| AIREBO                         |  ○  |   ○  |
| AIREBO-M                       |  ○  |   ○  |
| EDIP                           |  ○  |   ○  |
| COMB                           |  ○  |   ○  |
| ReaxFF                         |  ○  |   ○  |
| Reactive charge equilibration  |  ○  |   ○  |
| Generic bond-order potential   |  ○  |   ○  |
| Three-body tabulated potential |  ○  |   ○  |
| Many-body tabulated potential  |  ○  |   ○  |

**EAM** (analytic Sutton-Chen, `cpu::EAM`) and **Stillinger-Weber** (`cpu::StillingerWeber`)
are implemented on CPU, single-element, and run **under MPI**; both are FD-force-gated
and carry literature regressions (`regressions/md/eam_cu`, `regressions/md/sw_si`). See
[Chapter 16](../tutorial/ch16-many-body-potentials.md).

A many-body force needs a derived per-atom quantity (EAM's embedding density) at *ghost*
atoms, and the engine has exactly one forward halo (`pre_neighbor`) and one reverse halo
(`post_force`, forces only) — no mid-force communication slot to publish it in. Rather
than add one, EAM declares `halo_cutoff() = 2 × cutoff`: with a ghost-of-ghost shell,
every ghost's density completes **locally** and nothing has to be communicated mid-force.
Comm traded for halo width, which is the standard trade.

Alloy cross-terms, MEAM, Tersoff and the reactive potentials remain ○ — for those,
the quantum-mechanical route (GPW, plane waves) is still the only option.

## 5. Coarse-Grained Methods

| Method                                     | CPU | CUDA |
| ------------------------------------------ | :-: | :--: |
| MARTINI force field                        |  ○  |   ○  |
| SDK coarse-grained force field             |  ○  |   ○  |
| Dissipative particle dynamics              |  ○  |   ○  |
| Multiparticle collision dynamics           |  ○  |   ○  |
| Brownian dynamics                          |  ○  |   ○  |
| Generic tabulated coarse-grained potential |  ○  |   ○  |
| Iterative Boltzmann inversion potential    |  ○  |   ○  |
| Force-matching coarse-grained potential    |  ○  |   ○  |
| Relative-entropy coarse graining           |  ○  |   ○  |

## 6. Machine-Learning Potentials

| Method                                     | CPU | CUDA |
| ------------------------------------------ | :-: | :--: |
| Behler–Parrinello neural-network potential |  ○  |   ○  |
| ANI potential                              |  ○  |   ○  |
| DimeNet++                                  |  ○  |   ○  |
| PaiNN                                      |  ○  |   ○  |
| GemNet                                     |  ○  |   ○  |
| NequIP                                     |  ○  |   ○  |
| Allegro                                    |  ○  |   ○  |
| MACE                                       |  ○  |   ○  |
| Deep Potential / DeepMD                    |  ○  |   ○  |
| SOAP-GAP                                   |  ○  |   ○  |
| Atomic cluster expansion                   |  ○  |   ○  |
| Moment tensor potential                    |  ○  |   ○  |
| SNAP                                       |  ○  |   ○  |
| Long-range electrostatic ML potential      |  ○  |   ○  |
| Charge-aware ML potential                  |  ○  |   ○  |
| Spin-dependent ML potential                |  ○  |   ○  |
| Delta-learning potential                   |  ○  |   ○  |
| Ensemble ML potential                      |  ○  |   ○  |

**The LibTorch backend is not a machine-learning potential**, and this table
should not be read as "the torch backend is unfinished ML". `ATV_BUILD_TORCH`
builds a *differentiable classical* force field: LJ and harmonic bond as
`autograd::Function` operators with analytic backward passes w.r.t. positions
**and** the force-field parameters (ε, σ, K, r₀). It is a parameter-fitting
surface — there is no network architecture, no descriptor, no integrator, and
no driver on that backend.
