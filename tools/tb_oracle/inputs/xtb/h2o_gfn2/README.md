# xtb 6.7.1 GFN2-xTB water pin twin

```text
oracle: xtb
oracle_version: 6.7.1 (edcfbbe)
```

Offline twin for `reg_gfn2-xtb-water` and `reg_gfn2-quadrupole-aes`.

- Input: `mol.xyz` (ATV off-eq water, Å)
- Command: `xtb mol.xyz --gfn 2 --chrg 0 --uhf 0`
- Archived stdout: `xtb.out` (gitignored `*.out` at capture; file is present
  on this clone)
- Total energy: **−5.070383095219 Eh**

Never launched from ctest.
