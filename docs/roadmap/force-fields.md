# Force Fields

Tutorial: [Many-body potentials](../tutorial/ch16-many-body-potentials.md) · [classical MD](../tutorial/ch10-classical-md.md).

MD machinery (neighbor lists, DD, halo, PME, specials, virial) is
LAMMPS-shaped. CPU supports GAFF plus EAM/SW; CUDA supports the harmonic
bond/angle and polynomial-torsion GAFF subset on one GPU. Legend:
[Roadmap index](index.md).

**⚡** — only LJ 12-6 (CPU xsimd + CUDA OpenMM tile engine vs LAMMPS/Kokkos).

Molecular GAFF/GAFF2 parameterization has a separate
[scientific benchmark](../../benchmarks/forcefield/README.md), using native
Atomiverse AM1 charges and molpy typing against live AmberTools. Its scope is
fixed-geometry closed-shell molecules and parameter-level comparisons; these
checks do not qualify geometry optimization, force evaluation or dynamics.

<!-- benchmark-evidence:start -->

No source-qualified parameterization campaign has been recorded here. Local
accuracy and timing results remain provisional until the declared coverage and
build provenance pass the scientific benchmark policy.

<!-- benchmark-evidence:end -->

## 1. Bonded

| Capability | CPU | CUDA | Notes |
| ---------- | :-: | :--: | ----- |
| Harmonic bond / angle | ✓ | ✓ | |
| FENE bond | ◐ | ○ | No external pin; self-pinned chain case removed |
| Multi/harmonic torsion (RB) | ✓ | ✓ | `DihedralMultiHarmonic` |
| CHARMM torsion | ✓ | ○ | No optional 1–4 pair weights in style |
| Fourier torsion (Amber/GAFF multi-term) | ✓ | ○ | `DihedralFourier`; LAMMPS `dihedral_style fourier` pin |
| Harmonic improper | ✓ | ○ | First-class topology |
| CVFF improper (Amber/GAFF) | ✓ | ○ | `ImproperCvff`; LAMMPS `improper_style cvff` pin |
| OPLS / CMAP / Class-II cross | ○ | ○ | |

## 2. Nonbonded

| Capability | CPU | CUDA | Notes |
| ---------- | :-: | :--: | ----- |
| LJ 12-6 | ⚡ | ⚡ | CPU: multi-type + special_bonds scales. CUDA: single type, special_bonds only with equal 1-2/1-3/1-4 weights. Kokkos ◐: `ATV::kokkos::PairLJCut` on `Kokkos::Serial` only — not multi-type, not special_bonds, not an MD step |
| Buckingham / Morse / Born (Fumi–Tosi) | ✓ | ○ | |
| WCA / Yukawa / DPD / FENE+WCA | ◐ | ○ | No external pin; self-pinned cases removed |
| Soft-core LJ / Coulomb | ✓ | ○ | Beutler; alchemical λ |
| CoulCut / CoulLong | ✓ | ◐ | special_bonds on real space. CUDA `CoulLong`: real space only, no exclusion / special_bonds correction |
| Fused LJ + screened Coulomb | ✓ | ✓ | CUDA `PairLJCutCoulLong`: Lorentz–Berthelot, Amber graph exceptions, periodic sparse reciprocal corrections; single GPU |
| Direct Ewald / PME / SPME | ✓ | ✓ | SPME = Essmann; neutralizing background energy and virial on both backends. CUDA PME and Ewald: single GPU, fixed atom set and mesh / mode set, changing cell (NPT/deform) with six-component virial |
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
