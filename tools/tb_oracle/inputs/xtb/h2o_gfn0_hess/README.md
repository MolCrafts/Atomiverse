# xtb GFN0 water Hessian twin (`gfn0-xtb-analytic-hessian`)

```text
oracle: xtb
oracle_version: 6.7.1
```

Re-run (pin host):

```bash
module load xtb/6.7.1-gcc-2025b-eb
export XTBPATH="$(dirname $(dirname $(which xtb)))/share/xtb"
export OMP_NUM_THREADS=1
xtb mol.xyz --gfn 0 --chrg 0 --uhf 0 --acc 1.0 --hess
```

Archived transcript: `xtb.out.txt` (verbatim). Cartesian Hessian:
`hessian.txt` (xtb writes `hessian`; renamed so `*.out` gitignore does not
apply). Projected frequencies: `vibspectrum.txt` (xtb writes `vibspectrum`).

Pinned TOTAL ENERGY: **−4.366816412151 Eh** (same geometry as
`h2o_gfn0_offeq_grad`).
Pinned projected ω: **1563.14, 3497.87, 3505.03** cm⁻¹.
Pinned \(H_{O_x,O_x}\): **0.4793058695** Eh/a0².
xtb numerical Hessian step: **0.005 Bohr**.
xtb printed wall: **0.007 s** (1 thread). `/usr/bin/time` process wall:
**0.02 s**.
