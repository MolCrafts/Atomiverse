# Regression normalization — live

Contract: `regressions/README.md`; binding summary: CLAUDE.md iron law 1a.
Delete this file when every phase below is closed.

## Done

- 172 cases without an oracle or redundant with a kept case removed (586 → 414),
  with their CP2K oracle inputs; roadmap cells that depended only on them are ◐.
- tests/: end-to-end tests removed (9 files, 39 cases inside mixed files).
- Registration is data: every case has `case.cmake` (`atv_regression(...)`),
  `regressions/CMakeLists.txt` only discovers them. Per-case path macros
  (`ATV_PW_UPF_PATH`, `ATV_GAFF_ORACLE_PATH`) replaced by
  `ATV_REG_DATA_DIR` / `ATV_REG_CASE_DIR`.

## Open

1. **Census gates** — remove or shrink every case over 60 s, and every case red
   on `dev` with no owner. Source: serial census on Arrhenius (merged tree vs
   dev, `census_regs.sh`, 300 s cap).
2. **CPU/CUDA = one case** (34 `cuda-*` / `*-cuda-*` directories). Fold each
   into the CPU case of the same capability as `DEVICES both`, or rename it to
   the capability when no CPU case exists. CUDA-vs-CPU gates stay inside the
   case (`cpu-twin` oracle).
3. **Capability names** — rename history-named cases (`*-gate`, `*-product-*`,
   `*-foothold`, `arch-*`) to system + method + quantity.
4. **`REFERENCE.md` template** — 10 of 413 follow it; 53 never name the oracle.
   Rewrite each into the template; a case whose oracle cannot be named is
   removed and its roadmap cell becomes ◐.
5. **`internal/` includes** (27 GPW cases: dispersion and GPW forces/stress).
   Blocked on the public `forces`/`stress` shape for GPW + dispersion — closes
   with the ES public-shape unification, not by moving the cases.
6. **Roadmap links** — every ✓ cell names its case; every case names its row.
