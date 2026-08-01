# Acceptance — `gpw-hirshfeld-kpoint-forces`

## Gates

| # | Gate | Criterion |
| - | ---- | --------- |
| A | Isolated MBD@rsSCS + fixed representative \(v\) energy pin | \|E − pin\| ≤ 1×10⁻⁸ Ha |
| B | MBD fixed-\(v\) force dual-gate (product vs FD \(h=10^{-5}\) Bohr) | max\|ΔF\| ≤ 1×10⁻⁶ Ha/Bohr; \|ΣF\| ≤ 1×10⁻⁸ |
| C | TS fixed-\(v\) force dual-gate | max\|ΔF\| ≤ 1×10⁻⁶ Ha/Bohr; max\|F_h−F_free\| > 1×10⁻⁸ |
| D | Multi-k product TS+hirshfeld wiring (H₂, k-mesh 1×1×2) | max\|F^{k+ts+h}−F^{plain}−F_geom(v)\| ≤ 1×10⁻⁷ Ha/Bohr; residual vs free-atom nontrivial |
| E | Multi-k product MBD+hirshfeld wiring (same H₂ cell) | max\|F^{k+mbd+h}−F^{plain}−F_geom(v)\| ≤ 1×10⁻⁷ Ha/Bohr; volumes size==2; e_disp attractive |
| F | Multi-k payload present | SCF converged; `n_kpts ≥ 1` and `C_occ_k` non-empty (or equivalent k-path payload) |
| G | Wall | regression ≤ 60 s host budget |

## Oracle

- **oracle:** literature / libMBD (volume-scaled gradient dual)
- **perf_exempt:** true — CP2K 2026.1 has no stock TS/MBD pair potential
  (verified `cp2k.psmp` git:5e54ba2)

## Deliverables

- [x] `regressions/gpw/gpw-hirshfeld-kpoint-forces/` (reg + REFERENCE)
- [x] Unit test multi-k product fold in `test_cpu_gpw_vol_forces.cpp`
- [x] CMake `atv_add_regression(gpw gpw-hirshfeld-kpoint-forces …)`
- [x] Oracle note under `tools/cp2k_oracle/inputs/gpw-hirshfeld-kpoint-forces/`
- [x] Scoped build + `run_reg_gpw-hirshfeld-kpoint-forces` green

## Result (2026-08-01)

| Gate | Result |
| ---- | ------ |
| MBD fixed-\(v\) dual | max\|F−F_fd\| = 7.55e-11; \|ΣF\| = 2.43e-11; E pin match |
| TS fixed-\(v\) dual | max\|F−F_fd\| = 2.51e-14; max\|F_h−F_free\| = 1.79e-4 |
| multi-k TS+Hirsh H₂ | max\|wire_h\| = 5.41e-17; free residual 2.38e-7; wall 0.35 s total |
| multi-k MBD+Hirsh H₂ | max\|wire_h\| = 5.55e-17; free residual 3.86e-4 |
| unit `test_cpu_gpw_vol_forces` | Passed (0.88 s) |

**oracle:** literature / libMBD · **perf_exempt:** true (CP2K 2026.1 git:5e54ba2 no stock TS/MBD)
