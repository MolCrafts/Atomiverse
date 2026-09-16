# xtb 6.7.1 IPEA-xTB off-eq CH4 force twin

```text
oracle: xtb
oracle_version: 6.7.1 (compiled 2026-06-09)
```

Offline twin for `reg_ipea-xtb-forces`.

Same tetrahedral CH4 as `ch4_ipea/` except one C–H stretched to 1.20 Å
so \(|F|\neq 0\). IPEA is selected by pointing `XTBPATH` at a directory
whose `param_gfn1-xtb.txt` is a copy of xtb's `param_ipea-xtb.txt`
(xtb prints `I P E A - x T B`).

Off-eq **water** is not this pin: IPEA polar Δq still carries a ~0.5 eV
kernel hole vs xtb (named in `reg_ipea-xtb-small` Gaps). Analytic \(F\)
on water is gated by ATV energy FD in `test_cpu_xtb_ipea`.

Re-run (pin host, 2026-08-25):

```bash
module load xtb/6.7.1-gcc-2025b-eb
export XTBPATH=<dir with param_gfn1-xtb.txt copied from param_ipea-xtb.txt>
export OMP_NUM_THREADS=1
xtb mol.xyz --gfn 1 --chrg 0 --uhf 0 --acc 0.01 --grad
```

Archived transcript: `xtb.out.txt` (verbatim). Gradient file: `gradient.txt`.
Pinned TOTAL ENERGY: **−4.662477307157 Eh**.
Pinned \(\|\nabla E\|\): **0.093201061537 Eh/a0**.
Wall: **0.004 s** (xtb 1-thread clock).
