# xtb 6.7.1 IPEA-xTB CH4 Hessian twin (`ipea-xtb-analytic-hessian`)

```text
oracle: xtb
oracle_version: 6.7.1 (compiled 2026-06-09)
```

Offline twin for `reg_ipea-xtb-analytic-hessian`.

Same tetrahedral CH4 as `ch4_ipea/` (\(r(\mathrm{C{-}H})=1.09\) Å). IPEA is
selected by pointing `XTBPATH` at a directory whose `param_gfn1-xtb.txt` is
a copy of xtb's `param_ipea-xtb.txt` (xtb prints `I P E A - x T B`). Polar
**water** is not this pin (0.5 eV kernel hole vs xtb).

Re-run (pin host, 2026-08-25):

```bash
module load xtb/6.7.1-gcc-2025b-eb
export XTBPATH=<dir with param_gfn1-xtb.txt copied from param_ipea-xtb.txt>
export OMP_NUM_THREADS=1
xtb mol.xyz --gfn 1 --chrg 0 --uhf 0 --acc 0.01 --hess
```

Archived transcript: `xtb.out.txt` (verbatim). Cartesian Hessian:
`hessian.txt` (xtb writes `hessian`; renamed so `*.out` gitignore does not
apply). Projected frequencies: `vibspectrum.txt` (xtb writes `vibspectrum`).

Pinned TOTAL ENERGY: **−4.670476737294 Eh** (same geometry as `ch4_ipea`).
Pinned projected ω (9 vib modes): **1541.68×3, 1705.89×2, 3548.78,
3576.13×3** cm⁻¹.
Pinned \(H_{C_x,C_x}\): **0.6813937129** Eh/a0².
xtb Hessian path: **analytical** (log: “analytical hessian”).
xtb printed wall: **0.170 s** (1 thread). `/usr/bin/time` process wall:
**0.43 s**.
