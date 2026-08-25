# xtb GFN1 water Hessian twin (`gfn-xtb-analytic-hessian`)

```text
oracle: xtb
oracle_version: 6.7.1
```

Re-run (pin host):

```bash
module load xtb/6.7.1-gcc-2025b-eb
export XTBPATH="$(dirname $(dirname $(which xtb)))/share/xtb"
export OMP_NUM_THREADS=1
xtb mol.xyz --gfn 1 --chrg 0 --uhf 0 --acc 0.01 --hess
```

Archived transcript: `xtb.out.txt` (verbatim). Cartesian Hessian:
`hessian.txt` (xtb writes `hessian`; renamed so `*.out` gitignore does not
apply). Projected frequencies: `vibspectrum.txt` (xtb writes `vibspectrum`).

Pinned TOTAL ENERGY: **−5.768641651774 Eh** (same geometry as
`h2o_gfn1_offeq_grad`).
Pinned projected ω: **1474.96, 3657.78, 3761.56** cm⁻¹.
Pinned \(H_{O_x,O_x}\): **0.5466593694** Eh/a0².
xtb Hessian path: **analytical** (log: “analytical hessian”).
xtb printed wall: **0.020 s** (1 thread). `/usr/bin/time` process wall:
**0.03 s**.
