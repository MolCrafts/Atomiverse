# xtb GFN2 off-eq water force twin (`gfn2-xtb-forces`, `gfn2-aes-fock`)

```text
oracle: xtb
oracle_version: 6.7.1
```

Re-run (pin host):

```bash
module load xtb/6.7.1-gcc-2025b-eb
export XTBPATH="$(dirname $(dirname $(which xtb)))/share/xtb"
export OMP_NUM_THREADS=1
xtb mol.xyz --gfn 2 --chrg 0 --uhf 0 --acc 1.0 --grad
```

Geometry: gas-phase water, O–H stretched 0.96 → **1.10** Å (same stretch
used by the on-disk 6.7.0 `--grad` transcript; energy identical, gradient
norm is **not** — pin 6.7.1).

Archived transcript: `xtb.out.txt` (verbatim). Gradient file: `gradient.txt`
(xtb writes `gradient`; renamed so `*.out` gitignore does not apply).
Pinned TOTAL ENERGY: **−5.058625058504 Eh**.
Pinned \(\|\nabla E\|\): **0.109121448851 Eh/a0**.
Wall: **0.018 s** (1 thread, this transcript).
