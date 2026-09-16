# Pin notes — constant-d-polarization-feedback

Queue row: oracle=metalwalls (Zhang–Sprik constant-D). Wave 5.

## Oracle decision (binding priority rule)

- **LAMMPS twin?** No. Checked `lmp -h` style listing on the PATH binary
  (LAMMPS 30 Mar 2026 Development, `patch_30Mar2026-1074-g7f680de296`):
  it ships `fix efield` / `efield/tip` (constant-E, no polarization
  feedback) but **no constant-D ensemble** — zero matches for `dfield`;
  no fix applies E = D − 4πP with itinerant-polarization feedback.
  Not a LAMMPS-overlap feature.
- **MetalWalls binary?** Not installed on this machine (per loop contract).
- ⇒ **Literature pin** with full citation + `perf_exempt` (no fair twin
  binary to wall-clock against).

## Literature pinned

- C. Zhang, M. Sprik, Phys. Rev. B 93, 144201 (2016) — constant-D electric
  enthalpy F(D) = E_PBC + V (D − 4πP)² / 8π; Maxwell field E = D − 4πP.
- S. J. Cox, M. Sprik, J. Chem. Phys. 151, 064506 (2019) — finite-field
  formalism for bulk electrolyte solutions; itinerant polarization from
  unwrapped charge positions.
- M. Stengel, N. A. Spaldin, D. Vanderbilt, Nat. Phys. 5, 304 (2009) —
  origin of the constant-D functional.

The pins are the closed-form expressions of that functional for a point
±1 e pair (separation d, cubic V), evaluated in the project unit system
(k_e = 14.399645 eV·Å/e², force_to_accel = 1/103.6427):

- P_z = q d / V, E_z = D_z − 4π k_e P_z, U_D = V E_z²/(8π k_e),
  F_z = q E_z.
- At D_z = 0 the pair sits in the itinerant-polarization harmonic well
  U = 2π k_e q² d²/V → oscillation ω = sqrt(4π k_e q²/(μV) ·
  force_to_accel); this frequency is the pinned dynamic observable
  (period 106.329 fs for d0 = 2 Å, m = 1 amu, V = 1000 Å³).

Derivation script (reproducible): `inputs/literature/constant_d_pins.py`.
Numbers land in `regressions/md/constant-d-polarization-feedback/`
(REFERENCE.md + hard-coded asserts). ctest never runs any oracle.

## perf

`perf_exempt = true` — no constant-D implementation exists in an installed
oracle binary; a wall-clock ratio would compare different physics.
