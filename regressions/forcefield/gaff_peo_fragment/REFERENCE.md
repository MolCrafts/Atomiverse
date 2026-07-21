# gaff_peo_fragment — GAFF-like charged chain single-point

## What is pinned

A hand-built 8-atom C–C–O–C fragment with:

- multi-type LJ (Lorentz–Berthelot) from GAFF-like ε/σ (kcal/mol → eV via `lammps_real.h`)
- cut Coulomb with **Amber specials** `(0, 0, 0.5)`
- harmonic bond/angle + single-term CHARMM dihedral (Fourier m=1 map)

No PME in this case (cut Coulomb only) so the oracle is algorithmically exact
within ATV; the gate is **finite energy + bonded contribution present + |E|<50 eV**.

## Settings

| Item | Value |
|------|--------|
| cutoff | 10 Å (LJ and Coulomb) |
| special_bonds | 0, 0, 0.5 (lj and coul) |
| box | cubic 30 Å |
| units | Atomiverse eV / Å |

## How to regenerate a tight pin

1. Run `reg_gaff_peo_fragment` and record printed `E`, `pair`, `bond`, `angle`, `dihed`.
2. Optionally cross-check the same geometry in LAMMPS (`pair_style lj/cut/coul/cut`,
   `special_bonds amber`, same coeffs in real units) and convert kcal/mol → eV with
   `1/23.060547830619026`.
3. Tighten the `|E - E_pin|` band in the source when a stable pin is available.

## SPME note

Production GAFF uses `lj/cut/coul/long` + PPPM. This regression deliberately uses
cut Coulomb so CI does not depend on FFT grid choices. CoulLong+PME specials are
covered by unit tests (`test_coul_long_special`) and `reg_npt_pme_nacl` (ions).
