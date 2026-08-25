---
title: GFN1-xTB organic energy — acceptance
status: in-progress
slug: gfn1-xtb-organic
---

# Acceptance

1. `cpu::semiempirical::GFN1` is a public Semiempirical Component. No `*Config`,
   no `make_*` / `build_*`. Frame in / Frame out.
2. Static in-tree H/C/N/O tables. Unsupported Z throws with a how-to.
3. `with_forces = true` throws (energy foothold). Default is false.
4. Regression `reg_gfn1-xtb-organic` matches offline xtb 6.7.1 `--gfn 1` total
   energy of tetrahedral CH4 (\(r=1.09\) Å) to the tolerance in `REFERENCE.md`.
5. `REFERENCE.md` states `oracle: xtb` and `oracle_version: 6.7.1`.
6. Twin under `tools/tb_oracle/inputs/xtb/ch4_gfn1/` (xyz + transcript).
7. ctest never launches xtb. Fair wall \(T_{\mathrm{atv}}/T_{\mathrm{xtb}} \le 10\)
   or `perf_exempt` with reason.
