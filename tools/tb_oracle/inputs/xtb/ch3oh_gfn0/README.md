# xtb GFN0 methanol twin (`gfn0-beyond-toy`)

```text
oracle: xtb
oracle_version: 6.7.1
```

Re-run (pin host):

```bash
module load xtb/6.7.1-gcc-2025b-eb
export XTBPATH="$(dirname $(dirname $(which xtb)))/share/xtb"
export OMP_NUM_THREADS=1
xtb mol.xyz --gfn 0 --chrg 0 --uhf 0 --acc 1.0
```

Archived transcript: `xtb.out.txt` (verbatim). `*.out` is gitignored.
Pinned TOTAL ENERGY: **−7.597234498685 Eh**.
Optional `--grad` archive on this geometry: `gradient.txt` (wave-8
`gfn0-xtb-forces`; not the force-regression pin — that is off-eq water).
