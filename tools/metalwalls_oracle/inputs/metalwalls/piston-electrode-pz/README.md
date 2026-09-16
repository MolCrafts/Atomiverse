# Pin notes — `piston-electrode-pz` (real-binary re-pin, 2026-07-29)

**oracle:** metalwalls — MW-only rigid electrode piston, no LAMMPS twin
(LAMMPS `fix wall/piston` is SHOCK-package shock-wave kinematics, not an
electrode rigid body with \(F_{\mathrm{ext}} = \pm P A\)).

**oracle_version:** MetalWalls git **c8ad175d** (release branch, 2026-06-18),
binary `<work>/opt/metalwalls/mw`, run as `mpirun -n 1 mw`.

Supersedes the wave-1–5 note, which pinned this slug from the JOSS paper
(Marin-Laflèche et al., JOSS **5**, 2373 (2020), DOI 10.21105/joss.02373)
because no binary existed. That pin was self-consistent — the target
pressure was whatever ATV itself measured — so it could not have caught a
disagreement.

## What is in this directory

| File | Role |
| --- | --- |
| `gen_inputs.py` | emits `runtime.inpt` + `data.inpt` from the ATV regression's own geometry constants; also prints an independent analytic min-image LJ \(F_z\) |
| `runtime.inpt`, `data.inpt` | the generated MW twin (24 atoms, 2D PBC, Bohr/Hartree) |
| `electrode_forces.ref` | MW's `electrode_forces.out`, kept verbatim as the pin's provenance (renamed: `*.out` is gitignored) |

## Procedure

1. `python3 gen_inputs.py`
2. copy `runtime.inpt` + `data.inpt` into a scratch dir and run
   `mpirun -n 1 mw` there (login shell, `module load
   buildenv-gcc/2026.03-mpich`, `LD_LIBRARY_PATH` mirrored from
   `LIBRARY_PATH` — never pipe `module load` into a subshell).
3. read `electrode_forces.out`, `# step 0`, **second** data line
   (electrode `Wup` = the piston), **column 15** = `fz_lennard_jones`.
4. convert Ha/Bohr → eV/Å and divide by \(A = L_xL_y = 100\) Å².

## Result

```text
fz_lennard_jones (Wup, step 0) = +1.609547904622151 Ha/Bohr
                               =  82.7662809943017 eV/Å
P_z = F_z / A                  =   8.27662809943017e-01 eV/Å³
```

Independent analytic min-image LJ sum: 82.76628232744 eV/Å — 1.6e-8
relative, i.e. the CODATA drift only.

## Two things this re-pin changed

1. **Fixture defect.** The old fixture used \(r_c = 6\) Å in a 10 Å box.
   ATV's single-rank `CellList` is minimum-image, so \(r_c > L/2\) silently
   drops images (3.7e-4 error vs the true lattice sum); MW rejects such an
   input outright. The cutoff is now 4.5 Å (skin 0.3), where min-image and
   the full lattice sum agree to machine precision.
2. **The pin is now an external number.** `reg_piston-electrode-pz.cpp`
   hard-codes \(P_z^{\mathrm{MW}} = 8.27662809943017\times10^{-1}\) eV/Å³
   at 1e-6 relative tolerance, in addition to the equilibration gate.

## MW-side accommodations (both neutral for the pinned number)

- MW's piston lives only on a constant-potential electrode, and its CG
  computes `alpha = dp/dAd` (`src/cg.f90`) → \(0/0\) NaN when the RHS
  vanishes, so a chargeless electrode aborts with *"CG failed to
  converge"*. The twin therefore applies \(\Delta\psi = 0.5\) a.u.; the LJ
  column is a function of geometry only and is unaffected. Consequently the
  `pressure` column (= `fz_total/area`) is contaminated by induced-charge
  Coulomb and must **not** be read here.
- MW does not compute electrode–electrode LJ
  (`src/lennard_jones_2DPBC.inc` does ion–ion and ion–electrode only).
  Irrelevant: the piston's four atoms are coplanar (no \(z\) force) and the
  plate–plate gap (6 Å) exceeds \(r_c\).

## No fair wall-clock twin

MW's piston requires a constant-potential electrode, so every MW step also
runs a CG charge solve plus a 2D Ewald sum that the chargeless ATV
regression has no counterpart for; and MW cannot propagate this fixture's
compressed start at all (NVE → *"Temperature is NaN or meaningless"* at
step ~12; Nose-Hoover chain → *"Velocity thermostating failed to
converge"*). `perf_exempt: true`, reason recorded in REFERENCE.md.

## Numbers live in

`regressions/md/piston-electrode-pz/REFERENCE.md`
