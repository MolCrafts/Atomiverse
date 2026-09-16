# gpw-d4-forces — offline pin notes

Reuses geometry / POSCAR from `../gpw-d4-water/`.

## Hard pins (dftd4 3.7.0)

```bash
export PATH=/software/sse2/el9_epyc9005/easybuild/prefix/software/dftd4/3.7.0-gfbf-2025b/bin:$PATH
export LD_LIBRARY_PATH=/software/sse2/el9_epyc9005/easybuild/prefix/software/dftd4/3.7.0-gfbf-2025b/lib:$LD_LIBRARY_PATH
cd tools/cp2k_oracle/inputs/gpw-d4-water
dftd4 -f pbe --mbdscale 0.0 --grad pin_grad.txt --json pin.json POSCAR
```

- **E_disp** = −9.9458842745847e−4 Ha
- **Gradient** printed as dE/dR [Ha/Bohr]; ATV force = −gradient
- POSCAR atom order is **O O H H H H**; ATV regressions use **O H H O H H**

Arrhenius **CP2K 2026.1** lacks DFTD4 — no QS ENERGY_FORCE twin on that host.
