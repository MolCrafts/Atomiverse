# forcefield/gaff

One case; every section below is run by `atv_regressions --case forcefield/gaff` and keeps its own oracle record.


## gaff_peo_fragment — GAFF-like charged chain single-point

### What is pinned

A hand-built 8-atom C–C–O–C fragment with:

- multi-type LJ (Lorentz–Berthelot) from GAFF-like ε/σ (kcal/mol → eV via `lammps_real.h`)
- cut Coulomb with **Amber specials** `(0, 0, 0.5)`
- harmonic bond/angle + single-term CHARMM dihedral (Fourier m=1 map)

No PME in this case (cut Coulomb only) so the oracle is algorithmically exact
within ATV; the gate is **finite energy + bonded contribution present + |E|<50 eV**.

### Settings

| Item | Value |
|------|--------|
| cutoff | 10 Å (LJ and Coulomb) |
| special_bonds | 0, 0, 0.5 (lj and coul) |
| box | cubic 30 Å |
| units | Atomiverse eV / Å |

### How to regenerate a tight pin

1. Run `reg_gaff_peo_fragment` and record printed `E`, `pair`, `bond`, `angle`, `dihed`.
2. Optionally cross-check the same geometry in LAMMPS (`pair_style lj/cut/coul/cut`,
   `special_bonds amber`, same coeffs in real units) and convert kcal/mol → eV with
   `1/23.060547830619026`.
3. Tighten the `|E - E_pin|` band in the source when a stable pin is available.

### SPME note

Production GAFF uses `lj/cut/coul/long` + PPPM. This regression deliberately uses
cut Coulomb so CI does not depend on FFT grid choices. CoulLong+PME specials are
covered by unit tests (`test_coul_long_special`) and `reg_npt_pme_nacl` (ions).


## Reference — GAFF2 fourier torsions + cvff impropers (LAMMPS offline pin)

### Oracle

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

### System

One N-methylacetamide, one ethylene carbonate and one toluene (37 atoms,
65 dihedrals, 9 impropers) in a 30 Å periodic box, GAFF2 + AM1-BCC from
AmberTools25, converted by `benchmarks/compare/gaff-liquid/prmtop_to_lammps.py` (molpy).
Pair, bond and angle styles are `zero` on the LAMMPS side; Atomiverse
registers only `DihedralFourier` and `ImproperCvff`.

### Forms

- `dihedral_style fourier`: \(E = \sum_m K_m [1 + \cos(n_m\phi - d_m)]\)
- `improper_style cvff`: \(E = K[1 + d\cos(n\phi)]\), Amber atom order
  (central atom third)

Gate: energies within 1e-10 kcal/mol and every force component within
1e-10 kcal/mol/Å of the LAMMPS dump. Units convert with
`cpu::kKcalPerMolInEv`.


## CUDA GAFF neutral two-chain fragment

oracle: openmm

Force-field form: J. Wang, R. M. Wolf, J. W. Caldwell, P. A. Kollman and
D. A. Case, “Development and testing of a general amber force field”,
*J. Comput. Chem.* **25** (2004), 1157–1174,
[doi:10.1002/jcc.20035](https://doi.org/10.1002/jcc.20035).

`reference.txt` is the energy [eV] followed by all eight Cartesian force
vectors [eV/Å] printed by OpenMM 8.6.1, CUDA double, on Arrhenius GH200.
It is an engine-generated pin, not a number inferred from Atomiverse.
The exact oracle input is `benchmarks/compare/gaff-peo-openmm/fragment.data` and
`fragment.ff`. Reproduce inside a GPU allocation:

```sh
"$OPENMM_ENV/bin/python" benchmarks/compare/gaff-peo-openmm/compare.py \
  --data benchmarks/compare/gaff-peo-openmm/fragment.data \
  --ff benchmarks/compare/gaff-peo-openmm/fragment.ff --atv "$BUILD/bin/bench_gaff" \
  --out logs/gaff-fragment-pin --engine openmm --platform CUDA \
  --precision double --steps 1 --warmup 1
```

The first line of `logs/gaff-fragment-pin/openmm.txt` is the potential
energy. Copy that raw file to `reference.txt` only after actually running the
oracle. The standalone consumer uses the same explicitly listed positions,
charges, diagonal LJ parameters, harmonic bonds/angles, two Fourier torsion
terms, and Amber graph-distance specials. PME alpha is 0.35 Å⁻¹, mesh 24³,
order 5, cutoff 9 Å; LJ is unshifted without a dispersion correction.

This covers normal intermolecular nonbonded pairs, all three exception
distances, partial tiles, and a neutral PME system. Force and energy absolute
tolerances are 5e-5 in double and 5e-4 in mixed (eV/Å and eV respectively).
No trajectory or performance assertion is registered with ctest.


## Reference — CHARMM periodic dihedral energy

### Form

\[
V(\phi) = K\bigl(1 + \cos(n\phi - \delta)\bigr)
\]

with \(K\) [eV], multiplicity \(n \ge 1\), phase \(\delta\) [rad]. This is the
standard CHARMM / AMBER torsion term and matches LAMMPS `dihedral_style charmm`
**without** the optional 1–4 nonbonded weight (special-bond scaling stays on
pair styles). See MacKerell et al., *J. Phys. Chem. B* **102**, 3586 (1998)
(CHARMM22 torsion form) and LAMMPS `dihedral_style charmm` documentation.

### Geometry

Four atoms with orthogonal bond vectors:

| atom | position [Å] |
|------|----------------|
| 1    | (0, 0, 0)      |
| 2    | (1, 0, 0)      |
| 3    | (1, 1, 0)      |
| 4    | (1, 1, 1)      |

Planes n1 = b1×b2 and n2 = b2×b3 are orthogonal, so \(\phi = \pi/2\) exactly
(\(\cos\phi = 0\), \(\sin\phi = +1\)).

### Pins

| \(K\) [eV] | \(n\) | \(\delta\) | \(V\) [eV] | why |
|------------|-------|------------|------------|-----|
| 0.5        | 1     | 0          | 0.5        | \(1+\cos(\pi/2)=1\) |
| 0.5        | 2     | 0          | 0.0        | \(1+\cos(\pi)=0\) |
| 0.5        | 1     | \(\pi\)    | 0.5        | \(1+\cos(-\pi/2)=1\) |

Tolerance: \(10^{-10}\) eV (analytic identity, double precision).

Forces are checked by unit FD in `tests/cpu/md/test_pe_fd_force.cpp`, not here.


## Reference — harmonic improper energy

### Form

\[
V(\chi) = K\,(\chi - \chi_0)^2
\]

with \(K\) [eV/rad²] and equilibrium improper angle \(\chi_0\) [rad]. This is
LAMMPS `improper_style harmonic` (see LAMMPS documentation; functional form also
in CHARMM out-of-plane restraints, MacKerell et al., *J. Phys. Chem. B* **102**,
3586 (1998)).

\(\chi\) is the torsion angle between the two planes of the ordered 4-tuple
(same plane-normal geometry as a proper dihedral).

### Geometry

Four atoms with orthogonal bond vectors (owner = atom 2):

| atom | position [Å] |
|------|----------------|
| 1    | (0, 0, 0)      |
| 2    | (1, 0, 0)      |
| 3    | (1, 1, 0)      |
| 4    | (1, 1, 1)      |

⇒ \(\chi = \pi/2\) exactly.

### Pins

| \(K\) [eV/rad²] | \(\chi_0\) | \(V\) [eV] | why |
|-----------------|------------|------------|-----|
| 0.4             | \(\pi/2\)  | 0          | at equilibrium |
| 0.4             | 0          | \(0.4\,(\pi/2)^2\) | analytic |

Tolerance: \(10^{-10}\) eV.
