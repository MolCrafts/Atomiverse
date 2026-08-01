# Acceptance — gpw-vdw-uks-kpoint-multigrid

## Correctness

- [x] G1: multi-k UKS + DRSLL single-grid ↔ REL_CUTOFF multigrid
      \(\lvert E_{\mathrm{tot}}^{\mathrm{mg}}-E_{\mathrm{tot}}^{\mathrm{sg}}\rvert \le 2\times10^{-5}\) Ha
      and same for \(E_{\mathrm{disp}}\) (multigrid is a cost path, not physics).
      Measured |ΔE_tot|=4.27e-6, |ΔE_disp|=1.55e-7 Ha.
- [x] G2: multi-k UKS multigrid \(E_{\mathrm{disp}}\) vs offline CP2K
      \(\lvert\Delta\rvert \le 1\times10^{-4}\) Ha. Measured 4.00e-5 Ha.
- [x] G2: multi-k UKS multigrid \(E_{\mathrm{tot}}\) vs offline CP2K
      \(\lvert\Delta\rvert \le 1\times10^{-2}\) Ha (Γ-dimer twin-family systematic
      ~7.9e-3 Ha, same envelope as `gpw-vdw-kpoints` / `gpw-vdw-uks-kpoints`).
      Measured 7.93e-3 Ha.
- [x] \(E_{\mathrm{disp}} \ne 0\) and finite on the multigrid multi-k UKS path
      (E_c^nl actually folded, not dropped).
- [x] SCF converged on both grid paths.

## Perf

- [x] Fair wall on the **same** multigrid twin SCF only (not multi-gate reg wall):
      \(T_{\mathrm{atv}}/T_{\mathrm{cp2k}} \le 10\). Measured **0.85** (11.6 / 13.703 s, OMP=8).
- [x] `within_10x = true`; **not** `perf_exempt` (CP2K has the algorithm).

## Deliverables

- [x] `tools/cp2k_oracle/inputs/gpw-vdw-uks-kpoint-multigrid/` twin `.inp`
- [x] Offline CP2K run recorded in `REFERENCE.md` (binary + version + input +
      output lines + wall)
- [x] `regressions/gpw/gpw-vdw-uks-kpoint-multigrid/` + CMake registration
- [x] Roadmap residual prose flips product (`docs/roadmap/dft.md` §8)
- [x] Commit on `wf/gpw-vdw-uks-kpoint-multigrid`
