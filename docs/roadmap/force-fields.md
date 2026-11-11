# Force Fields

Tutorial: [Many-body potentials](../tutorial/ch16-many-body-potentials.md) · [classical MD](../tutorial/ch10-classical-md.md).

MD machinery (neighbor lists, DD, halo, PME, specials, virial) is
LAMMPS-shaped. CPU supports GAFF plus EAM/SW; CUDA supports the harmonic
bond/angle and polynomial-torsion GAFF subset on one GPU, plus the
Kremer–Grest FENE bond and cosine angle. Legend:
[Roadmap index](index.md).

**⚡** — only LJ 12-6 (CPU xsimd + CUDA OpenMM tile engine vs LAMMPS/Kokkos).

A molpy GAFF/GAFF2 force field runs from Python through
`atomiverse.forcefield.compile` — one molpy style to one evaluator: `lj/cut` →
per-type `PairLJCut`, `coul/cut` → `PairCoulCut` (special weights, stated
Coulomb constant), harmonic bond/angle, periodic / charmm torsions as
`DihedralCharmm` rows, `ImproperCvff` / `ImproperHarmonic`; CPU. Ewald styles
are not translated.

Molecular GAFF/GAFF2 parameterization has a separate
[scientific benchmark](https://github.com/MolCrafts/Atomiverse/blob/dev/benchmarks/forcefield/README.md), using native
Atomiverse AM1 charges and molpy typing against live AmberTools. Its scope is
fixed-geometry closed-shell molecules and parameter-level comparisons; these
checks do not qualify geometry optimization, force evaluation or dynamics.

<!-- benchmark-evidence:start -->

No source-qualified parameterization campaign has been recorded here. Local
accuracy and timing results remain provisional until the declared coverage and
build provenance pass the scientific benchmark policy.

<!-- benchmark-evidence:end -->

Per-atom identity is two Frame columns, one meaning each (LAMMPS semantics):
the force-field type `kAtom_type` (molpy `atoms.type_id`, LAMMPS data
`Atoms` type) keys every per-type table, and the species
`kAtom_atomic_number` (molpy `atoms.atomic_number`, Z, 0 = no element) keys
every element lookup. Coarse-grained models carry types only; no table keys
on Z. See `docs/developer/frame-architecture.md` §1.

## 1. Bonded

| Capability | CPU | CUDA | Notes |
| ---------- | :-: | :--: | ----- |
| Harmonic bond / angle | ✓ | ✓ | |
| FENE bond | ◐ | ◐ | One law for both backends (`core/fene.h`). No external pin; self-pinned chain case removed. CUDA `BondFene` (Python CUDA MD/Evaluate) unit test: energy, forces and virial vs analytic and CPU, overstretch and untyped-bond faults |
| Cosine angle (Kremer–Grest bending) | ◐ | ◐ | `AngleCosine`, LAMMPS `angle_style cosine` port, \(K(1+\cos\theta)\), shared `core/angle_cosine.h`. Unit tests: CPU energy/FD forces/virial at 60–150°, CUDA vs analytic energy and CPU forces/virial. No external pin. CUDA MD angle slot is singular (one angle style per run). Wire ID 37; Python `AngleCosine` on CPU and CUDA |
| Multi/harmonic torsion (RB) | ✓ | ✓ | `DihedralMultiHarmonic` |
| CHARMM torsion | ✓ | ○ | No optional 1–4 pair weights in style |
| Fourier torsion (Amber/GAFF multi-term) | ✓ | ○ | `DihedralFourier`; LAMMPS `dihedral_style fourier` pin |
| Harmonic improper | ✓ | ○ | First-class topology |
| CVFF improper (Amber/GAFF) | ✓ | ○ | `ImproperCvff`; LAMMPS `improper_style cvff` pin; Python `potentials.improper.Cvff` (local runs) |
| OPLS / CMAP / Class-II cross | ○ | ○ | |

Per-term energies: every force-field style adds its energy to the total and
to its LAMMPS thermo term on both backends — `energy_pair` (pair styles,
LJ tail, Ewald/PME/PPPM self, reciprocal and background), `energy_bond`,
`energy_angle`, `energy_dihedral`, `energy_improper` — so `cuda::to_cpu` and
the recorders report the CPU breakdown (CUDA `cuda::EnergyTerm`,
`tests/cuda/potentials/test_energy_terms.cu`). Not on the `cuda::MultiGpu`
stack, which reduces the total only.

## 2. Nonbonded

| Capability | CPU | CUDA | Notes |
| ---------- | :-: | :--: | ----- |
| LJ 12-6 | ⚡ | ⚡ | ⚡ = the single-type path. Per-type table on both backends (`PairLJCut(std::vector<LJTypeCoeff>, cutoff)`): LAMMPS `init_one` completion in `core/lj_type_table.h` (geometric ε, arithmetic σ and cutoff, explicit rows override, per-pair cutoffs, shifted at each pair's cutoff, type without diagonal = no LJ site); CPU scalar loop, CUDA device-table tile; evidence is analytic unit tests plus the LAMMPS single point `regressions/forcefield/cg-electrolyte`. `set_type_coeff` (add/override one row) and `set_tail_correction` (LAMMPS `pair_modify tail yes`, each pair at its own cutoff; CUDA: counts at build, volume read on the device, single GPU) on both backends. Special bonds: fractional 1-2/1-3/1-4 scales on both backends (CUDA: weights ≠ 1 leave the tiles, a sparse exception pass applies w · (LJ − offset + r⁻⁴) inside the pair cutoffs; fractional weights need a single GPU). Kokkos ◐: `ATV::kokkos::PairLJCut` on `Kokkos::Serial` only — not multi-type, not special_bonds, not an MD step |
| r⁻⁴ attraction (force-shifted) | ◐ | ◐ | ATV-native term `set_r4_attraction(std::vector<R4TypeCoeff>)` on `PairLJCut` (per-type) and `PairLJCutCoulLong`: U = −S[(σ/r)⁴ − (σ/r_c)⁴ + 4(σ/r_c)⁴(r/r_c − 1)], same table walk as LJ, scaled by the LJ special weight, energy + virial. Analytic unit tests; no LAMMPS style exists, but `regressions/forcefield/cg-electrolyte` evaluates the same closed form through LAMMPS `pair_style table` (agreement limited by the table's linear interpolation, 3e-10 at 2·10⁵ points, 4e-12 at 2·10⁶) |
| Buckingham / Morse / Born (Fumi–Tosi) | ✓ | ○ | |
| WCA / Yukawa / DPD / FENE+WCA | ◐ | ○ | No external pin; self-pinned cases removed |
| Soft-core LJ / Coulomb | ✓ | ○ | Beutler; alchemical λ |
| CoulCut / CoulLong | ✓ | ◐ | special_bonds on real space. CUDA `CoulLong`: real space only, no exclusion / special_bonds correction |
| Fused LJ + screened Coulomb | ✓ | ✓ | `PairLJCutCoulLong(std::vector<LJTypeCoeff>, alpha, cutoff)`, same API and defaults on both backends: typed `init_one` table with per-pair LJ cutoffs (Coulomb at the global cutoff), `set_special_bonds` LJ/Coulomb weights (default Amber), `set_energy_shift` (default on), r⁻⁴ term, `set_type_coeff`, LJ tail correction (`set_tail_correction`, per-pair cutoffs; CUDA reads the volume on the device). erfc(αr) is analytic on both backends (CPU: vectorized fdlibm erfc, `cpu/potentials/ewald_erfc.h`, xsimd; CUDA: erfcx·e^{−x²}); no tabulated mode. CUDA: per-atom Lorentz–Berthelot tile when the table is pure LB at one cutoff (OpenMM-benchmarked path), device-table tile otherwise; pairs whose nearest graph shell has weights ≠ (1, 1) go to a sparse exception pass with the LAMMPS cutoffs (LJ inside the pair's LJ cutoff, the Ewald exclusion correction inside the Coulomb cutoff only, like the CPU neighbor list); single GPU. LAMMPS single point (CG polymer electrolyte, typed LJ + r⁻⁴ + PPPM + FENE + cosine): `regressions/forcefield/cg-electrolyte`. Task wire ID 36 (CPU launcher, tail flag on the wire) and Python on both backends |
| Direct Ewald / PME / SPME | ✓ | ✓ | SPME = Essmann; neutralizing background energy and virial on both backends. CUDA PME and Ewald: single GPU, fixed atom set and mesh / mode set, changing cell (NPT/deform) with six-component virial |
| PPPM (Hockney–Eastwood, ik) | ✓ | ◐ | `KSpacePPPM`, LAMMPS `compute_gf_ik(_triclinic)`; general triclinic cell, tables follow the cell revision, six-component virial. Not a task-wire component. CPU: rock-salt Madelung pin (`regressions/longrange/pppm`), unit and MPI parity tests. CUDA `KSpacePPPM` (Python CUDA MD) shares the PME cuFFT path (its own spread / gather stencil); CUDA unit test: energy, forces and six-component virial vs direct Ewald (1e-12) on a 12-ion orthorhombic and triclinic cell, neutral and net-charged. Both backends spread and gather on LAMMPS's centred stencil (`particle_map` / `make_rho` / `fieldforce_ik`, `core/pppm_ik.h`; CPU brick ghosts per `Grid3d::ghost_grid`), choose the mesh by LAMMPS `factorable` (2, 3, 5) and default to order 6; an explicit mesh (`kspace_modify mesh`) and `KSpacePPPM::from_accuracy` (LAMMPS `PPPM::init`: `set_grid_global`, `estimate_ik_error`, `adjust_gewald`, `final_accuracy`, ported once in `core/pppm_accuracy.h`) on both backends and in Python (`KSpacePPPM.from_accuracy(frame, accuracy, cutoff, order=6, alpha=None, mesh=None)`); chosen α / mesh / estimate equal LAMMPS's printout (`tests/core/test_pppm_accuracy.cpp`). The r2c/c2r ik field, energy and virial fold each mode with its conjugate partner (`pppm_half_mode`), so even meshes reproduce LAMMPS's full complex `poisson_ik` (the Nyquist-index derivative cancels as in its real part); the triclinic alias bound uses LAMMPS `lamda2xT`'s |tilt|. CPU brick ghosts include LAMMPS `set_distance(0.5·skin)` from the neighbor skin the Runtime publishes (`Context::md.neighbor_skin`). Unit tests against a literal LAMMPS `make_rho` / `poisson_ik` / `fieldforce_ik` reference (energy, forces, six-component virial) for orders 3–7 on odd and even meshes on both backends, MPI brick-vs-replicated parity for orders 3–7 and for an owned atom 2 cells past its cut. LAMMPS single point `regressions/forcefield/cg-electrolyte` (CPU and CUDA, orders 5, 6, 7): k-space equal to LAMMPS `pppm` up to its truncated 4π = 12.5663706 (6e-9 energy, 1.1e-9 force and virial) at every order |
| Gaussian-charge Coulomb | ✓ | ○ | |
| Steele 10-4-3 wall | ✓ | ○ | `FixWallLJ1043` |
| Periodic FMM as MD KSpace | ○ | ○ | Open-boundary FMM math ✓ (powers CFMM-J); not MD drop-in |
| Reaction-field / Wolf / LJ-PME | ○ | ○ | |

## 3. Polarizable

| Capability | CPU | CUDA | Notes |
| ---------- | :-: | :--: | ----- |
| PIM induced dipoles (SCF) | ✓ | ○ | Thole; MetalWalls |
| Extended-Lagrangian dipoles | ✓ | ○ | |
| QEq fluctuating charge | ✓ | ○ | LAMMPS `fix qeq/point` |
| Drude / AMOEBA | ○ | ○ | |

## 4. Many-body / reactive

| Capability | CPU | CUDA | Notes |
| ---------- | :-: | :--: | ----- |
| EAM | ✓ | ○ | MPI; ghost-of-ghost halo |
| Stillinger–Weber | ✓ | ○ | MPI |
| Tersoff / MEAM / ReaxFF / AIREBO | ○ | ○ | |

## 5. Coarse-grained / ML

| Capability | CPU | CUDA | Notes |
| ---------- | :-: | :--: | ----- |
| DPD | ◐ | ○ | Multi-type \(a_{ij}\). No external pin; self-pinned Groot–Warren / mixing cases removed |
| MARTINI / SDK / IBI / … | ○ | ○ | |
| ML potentials (ANI, MACE, DeepMD, …) | ○ | ○ | Entire catalogue ○ |
