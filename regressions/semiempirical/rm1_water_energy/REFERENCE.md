# Reference — RM1 water total energy

## Method

RM1 (Recife Model 1) is a reparameterization of AM1 for H, C, N, O, P, S, F,
Cl, Br, and I (Rocha, Freire, Simas, Stewart, *J. Comput. Chem.* **27**, 1101
(2006)). This build ports the MOPAC `parameters_for_RM1_C.F90` electronic
parameters (Apache-2.0) for H/C/N/O/P/S into `AM1Parameter` tables and runs the
shared `NDDOSolver` RHF loop.

## System

Gas-phase water, experimental-like geometry (Å):

| Atom | x | y | z |
|------|---|---|---|
| O | 0 | 0 | 0 |
| H | 0.9572 | 0 | 0 |
| H | −0.239987 | 0.926627 | 0 |

Charge 0, singlet. Box 20 Å cubic (isolation / unwrap only).

## Pin

1. SCF converges.
2. Total energy is finite [eV].
3. `|E_RM1 − E_AM1| > 0.05 eV` on the same geometry (parameter tables differ).
4. MOPAC-style ΔHf is finite [kcal/mol].

A hard literature energy pin for this geometry is not required for this link;
the acceptance criterion is a method-discriminating, literature-cited method
identity plus convergence.
