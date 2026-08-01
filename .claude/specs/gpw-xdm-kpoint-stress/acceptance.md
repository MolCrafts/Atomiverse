# Acceptance — gpw-xdm-kpoint-stress

## Correctness

- [x] Gate B: `xdm_pbe` at 6 strained cells matches dual pure-Python pins
      (\(\lvert\Delta E\rvert \le 10^{-12}\) Ha; pin Å→Bohr). Measured 8.67e−19.
- [x] Gate A: multi-k product SCF `E_disp` matches unstrained pin
      (\(\lvert\Delta E\rvert \le 10^{-8}\) Ha; Frame CODATA residual). Measured 8.62e−12.
- [x] Gate C: analytic fixed-\(P_k\) \(\sigma_{\mathrm{diag}}\) vs multi-SCF
      strain-FD within \(2\times10^{-5}\) Ha/Bohr³; diagonal nonzero;
      off-diagonals exactly 0. Measured max|an−fd| 1.37e−09.
- [x] Gate D: \(\sigma(\mathrm{xdm})-\sigma(\mathrm{plain})\) equals dual-py
      strain-FD term per axis (\(\lvert\Delta\rvert \le 10^{-8}\) Ha/Bohr³). Measured 1.10e−15.
- [x] Gate E: `kStress` unit conversion; multi-k payload (`C_occ_k`) present.
- [x] Unit test: `GpwStress.XdmMultiKAnalyticMatchesFd` +
      `XdmChangesMultiKDiagonal` green.

## Perf

- [x] **perf_exempt** — CP2K has no stock XDM algorithm (no fair twin wall).
- [x] Product wall ≤ 60 s e2e budget (measured 17.8 s).

## Deliverables

- [x] `tools/cp2k_oracle/inputs/gpw-xdm-kpoint-stress/` pin script + note
- [x] `regressions/gpw/gpw-xdm-kpoint-stress/` + CMake registration
- [x] Unit test in `tests/cpu/gpw/test_cpu_gpw_stress.cpp`
- [x] Commit on `wf/gpw-xdm-kpoint-stress`
