# Reference — RM1 water total energy

```text
oracle: literature
oracle_version: Rocha, Freire, Simas & Stewart, J. Comput. Chem. 27, 1101 (2006)
family_oracle: mopac
```

**Scope:** this case is deliberately **method-discriminating only** — it checks
that RM1 converges and is not AM1. The absolute OpenMOPAC pin for RM1 lives in
`regressions/semiempirical/rm1-water-mopac-energy/` (ΔHf at the RM1 stationary
geometry against the OpenMOPAC PARAM survey tables).

## Method

RM1 (Recife Model 1) is a reparameterization of AM1 for H, C, N, O, P, S, F,
Cl, Br, and I (Rocha, Freire, Simas, Stewart, *J. Comput. Chem.* **27**, 1101
(2006)). This build ports the MOPAC `parameters_for_RM1_C.F90` electronic
parameters (Apache-2.0) for H/C/N/O/P/S into `AM1Parameter` tables and runs the
shared `NDDOSolver` RHF loop. The derived columns (`eisol`, `dd`, `qq`, `am`,
`ad`, `aq`) are recomputed from MOPAC `calpar.F90` on each row's own parameters
— gated by `tests/cpu/semiempirical/test_nddo_calpar_derived.cpp`.

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
