---
title: IPEA-xTB small-molecule energy — acceptance
status: in-progress
slug: ipea-xtb-small
---

# Acceptance

1. `cpu::semiempirical::IPEA` is a public Semiempirical Component. No `*Config`,
   no `make_*` / `build_*`. Frame in / Frame out. Own type/tables (not a flag
   on GFN1/GFN2).
2. Static in-tree H/C/N/O tables. Unsupported Z throws with a how-to.
3. `with_forces = true` throws (energy foothold). Default is false.
4. Regression `reg_ipea-xtb-small` matches offline xtb 6.7.1 IPEA-xTB total
   energy of tetrahedral CH4 (\(r=1.09\) Å) to the tolerance in `REFERENCE.md`.
5. `REFERENCE.md` states `oracle: xtb` and `oracle_version: 6.7.1`.
6. Twin under `tools/tb_oracle/inputs/xtb/ch4_ipea/` (xyz + transcript).
7. ctest never launches xtb. Fair wall \(T_{\mathrm{atv}}/T_{\mathrm{xtb}} \le 10\)
   or `perf_exempt` with reason.
