# GPW analytic cell stress with periodic MBD@rsSCS (Γ)

*(Technical-report page; CP2K capability loop wave 14.)*

| Field | Value |
| ----- | ----- |
| Slug | `gpw-mbd-stress` |
| Regression | `regressions/gpw/gpw-mbd-stress/` |
| Public API | `cpu::gpw::GPWEvaluator{mbd = true, mbd_k_mesh = {2,2,2}, compute_stress = true, stress_use_fd = false}` |
| Roadmap / queue | `docs/roadmap/dft.md` §8 Dispersion; queue done 2026-07-26 |

## Summary

**MBD@rsSCS** many-body dispersion (free-atom polarizabilities, PBE
\(\beta = 0.83\)) folded into the **Γ analytic cell stress**. The stress is the
Hellmann–Feynman fixed-\(P\) path: one SCF, then a central FD of the
non-self-consistent Lagrangian under diagonal strain at fixed fractional
coordinates, with \(E_{\mathrm{disp}}^{\mathrm{MBD}}\) **rebuilt at every
strained cell**.

Rebuilding at each strain is the whole point. Holding \(E_{\mathrm{disp}}\) fixed
would differentiate a different functional than the SCF converged — the failure
mode that once shipped as a 0.53 Ha/Bohr ZnO force error.

## Oracles

| Gate | Oracle |
| ---- | ------ |
| \(E_{\mathrm{disp}}\) | offline **pymbd** `mbd_energy`, the same pin `gpw-mbd-periodic` uses |
| \(\sigma\) | multi-SCF strain-FD (`gpw_stress_fd_impl`), full re-SCF per strain |
| MBD actually folded | \(\sigma(\mathrm{mbd}) \neq \sigma(\mathrm{plain})\) |

CP2K ships no MBD@rsSCS stress class, so there is no external stress twin: the
correctness statement is "the analytic stress is the strain derivative of an
externally pinned functional" — the same structure as `gpw-d4-stress`.

## Measured (2026-07-26)

| Gate | Tol | Measured |
| ---- | --- | -------- |
| \(E_{\mathrm{disp}}\) vs pymbd | 1e−8 Ha | **1.85e−10** |
| \(\sigma_{\mathrm{diag}}\) analytic vs multi-SCF FD | 2e−5 Ha/Bohr³ | **7.39e−07** |
| MBD shift \(\lvert\sigma(\mathrm{mbd})-\sigma(\mathrm{plain})\rvert\) | > 1e−10 | **1.72e−07** |
| Off-diagonals (orthorhombic) | exactly 0 | **0** |
| Wall | ≤ 60 s | **23.3 s** |

Per axis: \(\sigma_{xx} = -2.185\times10^{-5}\), \(\sigma_{yy} = 2.007\times10^{-5}\),
\(\sigma_{zz} = 5.823\times10^{-5}\) Ha/Bohr³. Unit gates:
`GpwStress.AnalyticMatchesFdWithMbd` (6.1e−9 Ha/Bohr³ on cheap MT H₂),
`MbdChangesGammaDiagonal`, `MbdMultiKStressThrows`.

## The defect this gate exposed

The **multi-k** stress path never folded \(E_{\mathrm{disp}}^{\mathrm{MBD}}\)
into its k-summed Lagrangian, and `mbd` was missing from its mutual-exclusion
check — so a multi-k run with `mbd = true` silently returned a stress tensor with
the whole dispersion contribution missing. It now **throws**, naming the Γ path
and the strain-FD oracle as supported routes. Implementing the fold is the queued
capability `gpw-mbd-kpoint-stress`.

Performance: **perf_exempt** (no oracle binary to time against). For scale, the
fixed-\(P\) pass is 23.3 s versus 23.2 s for its own multi-SCF FD twin on this
small cell; the advantage grows with atom count.
