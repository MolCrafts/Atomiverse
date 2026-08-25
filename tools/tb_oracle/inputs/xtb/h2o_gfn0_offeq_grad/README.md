# xtb GFN0 off-eq water force twin (`gfn0-xtb-forces`)

```text
oracle: xtb
oracle_version: 6.7.1
```

Re-run (pin host):

```bash
module load xtb/6.7.1-gcc-2025b-eb
export XTBPATH="$(dirname $(dirname $(which xtb)))/share/xtb"
export OMP_NUM_THREADS=1
xtb mol.xyz --gfn 0 --chrg 0 --uhf 0 --acc 1.0 --grad
```

Archived transcript: `xtb.out.txt` (verbatim). Gradient file: `gradient.txt`
(xtb writes `gradient`; renamed so `*.out` gitignore does not apply).
Pinned TOTAL ENERGY: **−4.366816412151 Eh**.
Pinned \(\|\nabla E\|\): **0.007179285426 Eh/a0**.
Wall: **0.007 s** (1 thread).
