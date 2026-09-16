# MetalWalls twin — global-electroneutrality-bo

Geometry matches `regressions/md/global-electroneutrality-bo`:
two single-site Gaussian electrodes (η = 2 Å⁻¹) at z = 5 / 15 Å,
Ψ = 0 / 1 V, one +1 e point ion at z = 10 Å, box 10×10×20 Å,
`charge_neutrality true global`, `num_pbc 2`, matrix inversion.

## Run

```bash
module load buildenv-gcc/2026.03-mpich
export LD_LIBRARY_PATH="$LIBRARY_PATH:$LD_LIBRARY_PATH"
MW=/nobackup/proj/disk/teoroo/personal/jicli594/work/opt/metalwalls/mw
rm -rf /tmp/mw-run-global-en && mkdir -p /tmp/mw-run-global-en
cp data.inpt runtime.inpt /tmp/mw-run-global-en/
cd /tmp/mw-run-global-en && mpirun -n 1 "$MW"
cat total_charges.out
```

## Pin (MetalWalls c8ad175d, 2026-07-29)

From `total_charges.out` step 0:

| electrode | Q [e] |
|-----------|-------|
| E1 (Ψ=0)  | −5.189485789245821e-1 |
| E2 (Ψ=1 V)| −4.810514210754179e-1 |
| **sum**   | **−1.0** (= −q_ion) |

Discriminating observable for this slug: global electroneutrality
constraint `sum_e Q_e = −q_electrolyte`. Per-electrode partition is a
secondary check; ATV uses a truncated xy-image elastance (not MW's 2D
Ewald), so individual Q only agree at large `xy_image_range` (~1e-5 at 1280).
