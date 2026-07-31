# Offline pin — GPW DFT-D4 + ATM (`gpw-d4-atm`)

## Geometry

Same periodic water dimer as `gpw-d4-water` (`POSCAR` here is a copy).

## Hard pins (dftd4 3.7.0)

```bash
export PATH=/software/sse2/el9_epyc9005/easybuild/prefix/software/dftd4/3.7.0-gfbf-2025b/bin:$PATH
export LD_LIBRARY_PATH=/software/sse2/el9_epyc9005/easybuild/prefix/software/dftd4/3.7.0-gfbf-2025b/lib:$LD_LIBRARY_PATH
# + FlexiBLAS / GCCcore from module load buildenv-gcc/2026.03-mpich

dftd4 -f pbe --json pin_periodic_atm.json POSCAR
# s9=1 (default pbe-D4-ATM)
# Dispersion energy: -9.9554475834240E-04 Eh

dftd4 -f pbe --mbdscale 0.0 --json pin_periodic_s9_0.json POSCAR
# two-body only: -9.9458842745847E-04 Eh

# Gradient pins (gpw-d4-atm-forces, 2026-07-31):
dftd4 -f pbe --grad grad_periodic_atm.txt --json pin_periodic_atm_grad.json POSCAR
# E = -9.9554475834240310E-04 Eh + 6x3 gradient (POSCAR order O O H H H H)
dftd4 -f pbe --grad grad_isolated_atm.txt --json pin_isolated_atm_grad.json water_dimer.xyz
# E = -9.8819741533958229E-04 Eh + gradient (ATV order O H H O H H)
```

Arrhenius **CP2K 2026.1** lacks DFTD4 — no QS ENERGY twin on that host.

## Strained-cell pins (`gpw-d4-atm-stress`, 2026-07-31)

For each axis a ∈ {x,y,z}: scale lattice row a **and** the Cartesian
a-components of every atom by (1 ± h), h = 2×10⁻³ — exactly the
`scale_axis` fixed-fractional-coordinate convention of the analytic stress
path (`gpw_stress`). Twelve runs (dftd4 3.7.0), each in its own directory
with the strained file named `POSCAR` (dftd4 detects format by filename):

```bash
# per strained POSCAR (example: x, +h):
dftd4 -f pbe --json pin_strain_x_plus_atm.json POSCAR          # s9=1
dftd4 -f pbe --mbdscale 0.0 --json pin_strain_x_plus_2b.json POSCAR  # two-body
```

Printed dispersion energies [Eh]:

| Strain | s9=1 (`pin_strain_*_atm.json`) | two-body (`pin_strain_*_2b.json`) |
| ------ | ------------------------------ | --------------------------------- |
| x, +h | -9.9375767649976510e-04 | -9.9281871131614977e-04 |
| x, −h | -9.9733212469993015e-04 | -9.9635829706912161e-04 |
| y, +h | -9.9553758220206315e-04 | -9.9456763760422703e-04 |
| y, −h | -9.9555200940363935e-04 | -9.9460945924979903e-04 |
| z, +h | -9.9558989265599194e-04 | -9.9463925734105045e-04 |
| z, −h | -9.9549659214095867e-04 | -9.9453454831018983e-04 |

These are the dedicated pins behind
`regressions/gpw/gpw-d4-atm-stress/` (dispersion cell stress
σ_aa = −[E(+h) − E(−h)]/(2hΩ) and its ATM-only part are FD over these
printed numbers; the ATM stress contribution is ~10⁻¹⁰ Ha/Bohr³ — far below
the GPW analytic-vs-FD dual-gate residual, so only these energy pins gate it).

## Files

| File | Content |
| ---- | ------- |
| `POSCAR` | VASP-format water dimer (12 Å cubic; order O O H H H H) |
| `water_dimer.xyz` | same nuclei, isolated, ATV order O H H O H H |
| `pin_periodic_atm.json` | dftd4 3.7.0 energy with s9=1 |
| `pin_periodic_s9_0.json` | two-body context |
| `pin_isolated_atm.json` | isolated s9=1 energy |
| `pin_periodic_atm_grad.json` | s9=1 energy + gradient (periodic) |
| `pin_isolated_atm_grad.json` | s9=1 energy + gradient (isolated) |
| `pin_strain_{x,y,z}_{plus,minus}_atm.json` | s9=1 energies at ±2e-3 strained cells |
| `pin_strain_{x,y,z}_{plus,minus}_2b.json` | two-body energies at the same strained cells |
