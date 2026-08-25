# xtb GFN1-xTB CH4 energy twin

```text
oracle: xtb
oracle_version: 6.7.1
```

## Command

```bash
module load xtb/6.7.1-gcc-2025b-eb
export OMP_NUM_THREADS=1
xtb mol.xyz --gfn 1 --chrg 0 --uhf 0 --acc 0.01
```

Geometry: tetrahedral CH4, \(r(\mathrm{C{-}H})=1.09\) Å (same as
`reg_dftb2-3ob-organic`). Isolated.

## Pin

`xtb.out.txt` is a verbatim transcript (`*.out` is gitignored). Line:

```
| TOTAL ENERGY               -4.274232224875 Eh   |
```

Public ATV pin: **−116.30778398 eV** (`Ha × 27.211386245988`).

1-thread wall: **0.02 s**.

Used by `regressions/semiempirical/gfn1-xtb-organic/`.
