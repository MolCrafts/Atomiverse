# xtb 6.7.1 IPEA-xTB CH4 VIP/VEA twin (`ipea-xtb-vip`)

```text
oracle: xtb
oracle_version: 6.7.1 (compiled 2026-06-09)
```

Offline twin for `reg_ipea-xtb-vip`.

- Input: `mol.xyz` (same tetrahedral CH4 as `ch4_ipea/`, \(r(\mathrm{C{-}H})=1.09\) Å)
- IPEA selection: `XTBPATH` directory with `param_gfn1-xtb.txt` copied from
  xtb's `param_ipea-xtb.txt` (xtb prints `I P E A - x T B`)
- Command: `xtb mol.xyz --gfn 1 --chrg 0 --uhf 0 --vipea` (`OMP_NUM_THREADS=1`)
- Archived stdout: `xtb.out.txt`
- Neutral \(E\): **−4.670476737294 Eh** (same as `reg_ipea-xtb-small`)
- Empirical shift: **4.8455 eV** (`ipeashift`)
- delta SCC IP: **13.0088 eV**
- delta SCC EA: **−1.4559 eV**
- Wall: 0.005 s (xtb clock) / 0.02 s (`/usr/bin/time`)

Cation occupations (7 e): \(2.0000 + 1.6667\times 3\) (t2 hole, Fermi 300 K).
Anion occupations (9 e): \(2.0000\times 4 + 1.0000\).

Never launched from ctest. Polar water is not this pin.
