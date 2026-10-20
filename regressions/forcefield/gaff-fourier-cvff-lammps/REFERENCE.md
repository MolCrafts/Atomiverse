# Reference — GAFF2 fourier torsions + cvff impropers (LAMMPS offline pin)

## Oracle

```text
oracle: lammps
oracle_version: LAMMPS (30 Sep 2026 - Development - a0e2bd6)
packages: MOLECULE EXTRA-MOLECULE KSPACE RIGID OPENMP
```

Twin input: `tools/lammps_oracle/inputs/gaff_fourier_cvff/in.torsion`, run
in `tests/data/gaff_fourier_cvff/` (it reads `system.data` + `torsion.ff`
there and writes `lammps_torsion_forces.dump`, the file this regression
reads). Output line pinned in the source:

```text
Step PotEng E_dihed E_impro
0    3.6701632327249165e+01   3.6499054964912958e+01   2.0257736233620535e-01
```

**ctest never launches LAMMPS.**

## System

One N-methylacetamide, one ethylene carbonate and one toluene (37 atoms,
65 dihedrals, 9 impropers) in a 30 Å periodic box, GAFF2 + AM1-BCC from
AmberTools25, converted by `benchmarks/gaff/prmtop_to_lammps.py` (molrs).
Pair, bond and angle styles are `zero` on the LAMMPS side; Atomiverse
registers only `DihedralFourier` and `ImproperCvff`.

## Forms

- `dihedral_style fourier`: \(E = \sum_m K_m [1 + \cos(n_m\phi - d_m)]\)
- `improper_style cvff`: \(E = K[1 + d\cos(n\phi)]\), Amber atom order
  (central atom third)

Gate: energies within 1e-10 kcal/mol and every force component within
1e-10 kcal/mol/Å of the LAMMPS dump. Units convert with
`cpu::kKcalPerMolInEv`.
