# Force Fields

Tutorial: [Many-body potentials](../tutorial/ch16-many-body-potentials.md) · [classical MD](../tutorial/ch10-classical-md.md).

MD machinery (neighbor lists, DD, halo, PME, specials, virial) is
LAMMPS-shaped. Catalogue is GAFF-capable on CPU plus EAM/SW. Legend:
[Roadmap index](index.md).

**⚡** — only LJ 12-6 (CPU xsimd + CUDA OpenMM tile engine vs LAMMPS/Kokkos).

## 1. Bonded

| Capability | CPU | CUDA | Notes |
| ---------- | :-: | :--: | ----- |
| Harmonic bond / angle | ✓ | ✓ | |
| FENE bond | ✓ | ○ | |
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
| WCA / Yukawa / DPD / FENE+WCA | ✓ | ○ | |
| Soft-core LJ / Coulomb | ✓ | ○ | Beutler; alchemical λ |
| CoulCut / CoulLong | ✓ | ◐ | special_bonds on real space. CUDA `CoulLong`: real space only, no exclusion / special_bonds correction |
| Direct Ewald / PME / SPME | ✓ | ✓ | SPME = Essmann; CUDA reciprocal single-GPU |
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
| DPD | ✓ | ○ | Multi-type \(a_{ij}\) |
| MARTINI / SDK / IBI / … | ○ | ○ | |
| ML potentials (ANI, MACE, DeepMD, …) | ○ | ○ | Entire catalogue ○ |
