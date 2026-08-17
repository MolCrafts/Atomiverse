# Force Fields

Legend and caveats: [Roadmap index](index.md).

The MD machinery around the potentials (neighbor lists, domain decomposition,
halo exchange, PME, special-bonds exclusions, virial accumulation) is mature
and LAMMPS-shaped. The *catalogue of styles* is now GAFF-capable on CPU
(LJ multi-type, CoulCut/Long, harmonic bond/angle, CHARMM + multi/harmonic
dihedrals, harmonic improper) plus EAM/SW many-body — still thin relative to
LAMMPS, but no longer "one pair style and nothing else".

The Lennard–Jones 12-6 pair style is the only ⚡ in the roadmap: SIMD on CPU
(xsimd, `-march=native`) and the CUDA OpenMM tile engine. Standing bench:
`benchmarks/lj/` + `tools/cuda_nonbonded_style_bench`. GH200 **1332540**:
`96.997 Matom*step/s` (N=10648, 80 steps). LAMMPS twin deck:
`tools/bench_lj_vs_lammps/in.lj_nve.kk`. Keep ⚡ only while a same-node
LAMMPS Kokkos ratio still supports it.
**Kokkos backend: ○** (planned portable path; no in-tree Kokkos port yet).

## 1. Classical Bonded Potentials

| Method                             | CPU | CUDA | Kokkos |
| ---------------------------------- | :-: | :--: | :---: |
| Harmonic bond potential            |  ✓  |   ✓  |   ○  |
| Quartic bond potential             |  ○  |   ○  |   ○  |
| Morse bond potential               |  ○  |   ○  |   ○  |
| FENE bond potential                |  ✓  |   ○  |   ○  |
| Tabulated bond potential           |  ○  |   ○  |   ○  |
| Harmonic angle potential           |  ✓  |   ✓  |   ○  |
| Cosine angle potential             |  ○  |   ○  |   ○  |
| Restricted-bending angle potential |  ○  |   ○  |   ○  |
| Urey–Bradley potential             |  ○  |   ○  |   ○  |
| Tabulated angle potential          |  ○  |   ○  |   ○  |
| Periodic Fourier torsion           |  ✓  |   ○  |   ○  |
| OPLS torsion                       |  ○  |   ○  |   ○  |
| Ryckaert–Bellemans torsion         |  ✓  |   ✓  |   ○  |
| CHARMM torsion                     |  ✓  |   ○  |   ○  |
| Harmonic improper torsion          |  ✓  |   ○  |   ○  |
| Periodic improper torsion          |  ○  |   ○  |   ○  |
| CVFF improper torsion              |  ○  |   ○  |   ○  |
| CMAP correction                    |  ○  |   ○  |   ○  |
| Bond–bond cross term               |  ○  |   ○  |   ○  |
| Bond–angle cross term              |  ○  |   ○  |   ○  |
| Angle–angle cross term             |  ○  |   ○  |   ○  |
| Bond–torsion cross term            |  ○  |   ○  |   ○  |
| Class-II bonded potential          |  ○  |   ○  |   ○  |
| Tabulated bonded potential         |  ○  |   ○  |   ○  |

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

| Method                              | CPU | CUDA | Kokkos |
| ----------------------------------- | :-: | :--: | :---: |
| Lennard–Jones 12-6 potential        |  ⚡  |   ⚡  |   ○  |
| Mie n-m potential                   |  ○  |   ○  |   ○  |
| Buckingham potential                |  ✓  |   ○  |   ○  |
| Morse nonbonded potential           |  ✓  |   ○  |   ○  |
| Tabulated pair potential            |  ○  |   ○  |   ○  |
| Coulomb potential                   |  ✓  |   ○  |   ○  |
| Reaction-field electrostatics       |  ○  |   ○  |   ○  |
| Damped-shifted-force electrostatics |  ○  |   ○  |   ○  |
| Wolf summation                      |  ○  |   ○  |   ○  |
| Direct Ewald summation              |  ✓  |   ✓  |   ○  |
| Particle Mesh Ewald                 |  ✓  |   ✓  |   ○  |
| Smooth Particle Mesh Ewald          |  ✓  |   ✓  |   ○  |
| Lennard–Jones PME                   |  ○  |   ○  |   ○  |
| Fast multipole electrostatics       |  ○  |   ○  |   ○  |
| Multilevel summation method         |  ○  |   ○  |   ○  |
| Soft-core Lennard–Jones potential   |  ✓  |   ○  |   ○  |
| Soft-core Coulomb potential         |  ✓  |   ○  |   ○  |

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
  grid across MPI ranks rather than distributing the FFT. CUDA PME energy
  Madelung residual (`cuda-pme-accuracy`) closed 2026-07-23.
- Open-boundary **FMM tree math** (`FmmTree`) is validated against direct
  Coulomb sums (`reg_fmm_point_charges`) and powers CFMM-J on the SCF path;
  it is **not** a periodic MD `KSpace` style, so the FMM row above stays ○.
- **Soft-core LJ** (`cpu::PairLJSoft`) is Beutler/GROMACS **one-leg appearance**
  form with public `lambda` (default 1 = hard shifted LJ parity). Not LAMMPS
  `lj/cut/soft`. Soft-core Coulomb ships as `cpu::PairCoulSoft` (same Beutler
  family). Regressions: `lj_soft_dimer`, `coul_soft_dimer`.
- **Buckingham** (`cpu::PairBuck`) is LAMMPS `pair_style buck`
  \(A e^{-r/\rho}-C/r^{6}\) with optional energy shift; CPU dimer pin
  `reg_buckingham-pair-lammps`.
- **Born–Mayer–Huggins** (`cpu::PairBorn`) is LAMMPS `pair_style born`
  \(A e^{(\sigma-r)/\rho}-C/r^{6}+D/r^{8}\) with energy shift; single-type dimer
  pin `reg_born-repulsion-foothold`. **Multi-type Fumi–Tosi** uses the same
  `PairBorn` type table + `PairCoulCut` (LAMMPS `hybrid/overlay born coul/cut`);
  pin `reg_fumi-tosi-nacl`. **NeighborList-backed bulk Born path** ships
  (`PairBorn` prefers `ctx.md.pair_list` when published;
  `reg_born-buck-neighborlist-bulk`). Same NeighborList preference on
  `PairGaussianCoul` (`reg_gaussian-coul-neighborlist`).
- **Steele 10-4-3 wall** ships as `cpu::md::FixWallLJ1043` (LAMMPS
  `fix wall/lj1043`, not `wall/lj93`); pin `reg_steele-wall-argon`.
- **FENE** (`BondFene`) + **WCA** (`PairWCA`) and **Yukawa** (`PairYukawa`) and
  **DPD** (`PairDpd`) are CPU textbook styles from the method ladder
  (`reg_fene_wca_chain`, `reg_yukawa_dimer`, `reg_dpd_groot_warren`).
- **Direct Ewald** supports Yeh–Berkowitz slab correction via public
  `KSpaceEwald::slab_factor` (LAMMPS `kspace_modify slab`); pin
  `reg_slab-2d-ewald-foothold`. True 2D-only Ewald (no 3D+slab) remains
  **out of MetalWalls-loop product scope** (full 2D-PBC stack).

## 3. Polarizable Force Fields

| Method                              | CPU | CUDA | Kokkos |
| ----------------------------------- | :-: | :--: | :---: |
| Drude oscillator model              |  ○  |   ○  |   ○  |
| Self-consistent induced dipoles     |  ✓  |   ○  |   ○  |
| Extended-Lagrangian induced dipoles |  ✓  |   ○  |   ○  |
| Thole-damped polarization           |  ✓  |   ○  |   ○  |
| Atomic-multipole electrostatics     |  ○  |   ○  |   ○  |
| AMOEBA electrostatics               |  ○  |   ○  |   ○  |
| AMOEBA polarization                 |  ○  |   ○  |   ○  |
| Fluctuating-charge model            |  ✓  |   ○  |   ○  |
| QEq polarization                    |  ✓  |   ○  |   ○  |
| Charge-equilibration dynamics       |  ○  |   ○  |   ○  |

**Bulk PIM** (`cpu::PimInducedDipole`) is MetalWalls-style self-consistent
induced dipoles with Thole exponential damping and optional Tang–Toennies
(`tt_b`); permanent charge–charge is separate (`PairCoulCut` / PME). Pin
`reg_pim-induced-dipole-bulk`. **Electrode-coupled dipole ⊕ \(Q\) joint** ships
as `cpu::md::DipoleElectrodeJoint` (alternating ConstPotBO \(Q\) + PIM \(\mu\)
until joint tol; `reg_dipole-electrode-joint`, oracle=metalwalls).
**Extended-Lagrangian induced dipoles** ship as `cpu::PimXlDipole`
(time-reversible XL propagation of the PIM dipoles;
`reg_extended-lagrangian-dipoles`, oracle=metalwalls / iEL-SCF literature).
**QEq / fluctuating-charge** ships as `cpu::md::QEq` (LAMMPS `fix qeq/point`
twin; `reg_qeq-fluctuating-charge`, oracle=lammps, fair-bench ratio 0.45).
The **PIM-NPT polarizable IL product twin** landed as
`reg_pim-npt-polarizable-il` (oracle=metalwalls offline Thole-PIM pin).
Charge-equilibration *dynamics* (extended-Lagrangian charge propagation,
LAMMPS `qeq/dynamic`-class) and Drude / AMOEBA remain ○.

## 4. Many-Body and Reactive Potentials

| Method                         | CPU | CUDA | Kokkos |
| ------------------------------ | :-: | :--: | :---: |
| Embedded-atom method           |  ✓  |   ○  |   ○  |
| Modified embedded-atom method  |  ○  |   ○  |   ○  |
| Finnis–Sinclair potential      |  ○  |   ○  |   ○  |
| Stillinger–Weber potential     |  ✓  |   ○  |   ○  |
| Tersoff potential              |  ○  |   ○  |   ○  |
| AIREBO                         |  ○  |   ○  |   ○  |
| AIREBO-M                       |  ○  |   ○  |   ○  |
| EDIP                           |  ○  |   ○  |   ○  |
| COMB                           |  ○  |   ○  |   ○  |
| ReaxFF                         |  ○  |   ○  |   ○  |
| Reactive charge equilibration  |  ○  |   ○  |   ○  |
| Generic bond-order potential   |  ○  |   ○  |   ○  |
| Three-body tabulated potential |  ○  |   ○  |   ○  |
| Many-body tabulated potential  |  ○  |   ○  |   ○  |

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

| Method                                     | CPU | CUDA | Kokkos |
| ------------------------------------------ | :-: | :--: | :---: |
| MARTINI force field                        |  ○  |   ○  |   ○  |
| SDK coarse-grained force field             |  ○  |   ○  |   ○  |
| Dissipative particle dynamics              |  ✓  |   ○  |   ○  |
| Multiparticle collision dynamics           |  ○  |   ○  |   ○  |
| Brownian dynamics                          |  ○  |   ○  |   ○  |
| Generic tabulated coarse-grained potential |  ○  |   ○  |   ○  |
| Iterative Boltzmann inversion potential    |  ○  |   ○  |   ○  |
| Force-matching coarse-grained potential    |  ○  |   ○  |   ○  |
| Relative-entropy coarse graining           |  ○  |   ○  |   ○  |

## 6. Machine-Learning Potentials

| Method                                     | CPU | CUDA | Kokkos |
| ------------------------------------------ | :-: | :--: | :---: |
| Behler–Parrinello neural-network potential |  ○  |   ○  |   ○  |
| ANI potential                              |  ○  |   ○  |   ○  |
| DimeNet++                                  |  ○  |   ○  |   ○  |
| PaiNN                                      |  ○  |   ○  |   ○  |
| GemNet                                     |  ○  |   ○  |   ○  |
| NequIP                                     |  ○  |   ○  |   ○  |
| Allegro                                    |  ○  |   ○  |   ○  |
| MACE                                       |  ○  |   ○  |   ○  |
| Deep Potential / DeepMD                    |  ○  |   ○  |   ○  |
| SOAP-GAP                                   |  ○  |   ○  |   ○  |
| Atomic cluster expansion                   |  ○  |   ○  |   ○  |
| Moment tensor potential                    |  ○  |   ○  |   ○  |
| SNAP                                       |  ○  |   ○  |   ○  |
| Long-range electrostatic ML potential      |  ○  |   ○  |   ○  |
| Charge-aware ML potential                  |  ○  |   ○  |   ○  |
| Spin-dependent ML potential                |  ○  |   ○  |   ○  |
| Delta-learning potential                   |  ○  |   ○  |   ○  |
| Ensemble ML potential                      |  ○  |   ○  |   ○  |

This table is **true ML / reactive many-body potentials** only. Classical pair
and bonded styles live in sections 1–2; they are not “unfinished ML.”
