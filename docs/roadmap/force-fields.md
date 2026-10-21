# Force Fields

Tutorial: [Many-body potentials](../tutorial/ch16-many-body-potentials.md) · [classical MD](../tutorial/ch10-classical-md.md).

MD machinery (neighbor lists, DD, halo, PME, specials, virial) is
LAMMPS-shaped. CPU supports GAFF plus EAM/SW; CUDA supports the harmonic
bond/angle and polynomial-torsion GAFF subset on one GPU. Legend:
[Roadmap index](index.md).

**⚡** — only LJ 12-6 (CPU xsimd + CUDA OpenMM tile engine vs LAMMPS/Kokkos).

## 1. Bonded

| Capability | CPU | CUDA | Notes |
| ---------- | :-: | :--: | ----- |
| Harmonic bond / angle | ✓ | ✓ | |
| FENE bond | ✓ | ○ | |
| Multi/harmonic torsion (RB) | ✓ | ✓ | `DihedralMultiHarmonic` |
| CHARMM torsion | ✓ | ○ | No optional 1–4 pair weights in style |
| Harmonic improper | ✓ | ○ | First-class topology |
| OPLS / CVFF / CMAP / Class-II cross | ○ | ○ | |

## 2. Nonbonded

| Capability | CPU | CUDA | Notes |
| ---------- | :-: | :--: | ----- |
| LJ 12-6 | ⚡ | ⚡ | Multi-type + special_bonds scales. Kokkos ◐: `ATV::kokkos::PairLJCut` on `Kokkos::Serial` only — not multi-type, not special_bonds, not an MD step |
| Buckingham / Morse / Born (Fumi–Tosi) | ✓ | ○ | |
| WCA / Yukawa / DPD / FENE+WCA | ✓ | ○ | |
| Soft-core LJ / Coulomb | ✓ | ○ | Beutler; alchemical λ |
| CoulCut / CoulLong | ✓ | ○ | special_bonds on real space |
| Fused LJ + screened Coulomb | ✓ | ✓ | CUDA `PairLJCutCoulLong`: Lorentz–Berthelot, Amber graph exceptions, periodic sparse reciprocal corrections; single GPU |
| Direct Ewald / PME / SPME | ✓ | ✓ | SPME = Essmann; CUDA PME: single GPU, fixed cell and atom set |
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
