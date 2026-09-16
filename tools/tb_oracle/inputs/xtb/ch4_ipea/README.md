# xtb 6.7.1 IPEA-xTB methane pin twin

```text
oracle: xtb
oracle_version: 6.7.1 (compiled 2026-06-09)
```

Offline twin for `reg_ipea-xtb-small`.

- Input: `mol.xyz` (tetrahedral CH4, \(r(\mathrm{C{-}H})=1.09\) Å)
- IPEA selection: `XTBPATH` directory with `param_gfn1-xtb.txt` copied from
  xtb's `param_ipea-xtb.txt` (xtb prints `I P E A - x T B`)
- Command: `xtb mol.xyz --gfn 1 --chrg 0 --uhf 0 --sp` (`OMP_NUM_THREADS=1`)
- Archived stdout: `xtb.out.txt`
- Total energy: **−4.670476737294 Eh**
- Wall: 0.005 s (xtb) / 0.02 s (`/usr/bin/time`)

Never launched from ctest. Discriminates IPEA from GFN1-xTB on the same
geometry (GFN1 pin −4.274232224875 Eh).
