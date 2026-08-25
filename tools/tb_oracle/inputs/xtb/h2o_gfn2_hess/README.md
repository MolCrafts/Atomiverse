# xtb GFN2 water Hessian twin (`gfn-xtb-analytic-hessian`)

```text
oracle: xtb
oracle_version: 6.7.1
```

Re-run (pin host):

```bash
module load xtb/6.7.1-gcc-2025b-eb
export XTBPATH="$(dirname $(dirname $(which xtb)))/share/xtb"
export OMP_NUM_THREADS=1
xtb mol.xyz --gfn 2 --chrg 0 --uhf 0 --acc 0.01 --hess
```

Geometry is the **0.96 Å** water of `h2o_gfn2/` / `h2o_gfn0_hess/` (not the
1.10 Å stretch of `h2o_gfn2_offeq_grad/`).

Archived transcript: `xtb.out.txt` (verbatim). Cartesian Hessian:
`hessian.txt` (xtb writes `hessian`; renamed so `*.out` gitignore does not
apply). Projected frequencies: `vibspectrum.txt` (xtb writes `vibspectrum`).

Pinned TOTAL ENERGY: **−5.070383095219 Eh** (same number as
`reg_gfn2-xtb-water` / `reg_gfn2-quadrupole-aes`).
Pinned projected ω: **1565.10, 3634.95, 3643.62** cm⁻¹.
Pinned \(H_{O_x,O_x}\): **0.5210291876** Eh/a0².
xtb Hessian path: **analytical** (log: “analytical hessian”).
xtb printed wall: **0.012 s** (1 thread). `/usr/bin/time` process wall:
**0.02 s**.

AES remains a post-SCC energy in ATV (`gfn2-aes-fock`); the ATV ω residual
vs this twin carries that Fock hole.
