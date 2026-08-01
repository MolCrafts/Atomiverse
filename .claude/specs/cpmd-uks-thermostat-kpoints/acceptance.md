# Acceptance — `cpmd-uks-thermostat-kpoints`

## Correctness

| ID | Gate | Where |
| -- | ---- | ----- |
| A1 | Multi-k UKS + CPMD + electronic NH builds (no throw) — already unit-pinned as `BuildAcceptsMultiKUksCpmdThermostat` | unit (product) |
| A2 | E2E periodic H₂, mesh \(1\times1\times2\) unreduced, UKS closed-shell fillings \(n_\alpha=n_\beta=1\), electronic NH (\(K_0=10^{-5}\) Ha, \(\tau_e=50\) fs): cold seed converged; baseline + production `iters == 0`, `n_{\mathrm{kpts}} \ge 2`; \(E_{\mathrm{ext}}\) relative drift ≤ 0.10; max \(T_e\) ≤ 0.05 eV; \(E_{\mathrm{therm}}\) finite every production step; PE bound after warm-up; finite coords; \(0.4 < r_{\mathrm{HH}} < 1.5\) Å | `reg_cpmd-uks-thermostat-kpoints` |
| A3 | UKS-vs-RKS closed-shell single-point identity at `tol = 1e-7` ≤ \(3\times10^{-4}\) eV (same geometry / mesh / cutoff) | same regression |

## Perf

| Field | Value |
| ----- | ----- |
| `perf_exempt` | **true** — oracle family lock: CP2K 2026.1 has no fictitious-mass CP propagator; no fair wall twin of thermostatted multi-k UKS CP AIMD exists |
| Else | \(T_{\mathrm{atv}}/T_{\mathrm{oracle}} \le 10\) |

## Dual gate

Green only when A1–A3 hold and (`within_10x` or `perf_exempt`).
