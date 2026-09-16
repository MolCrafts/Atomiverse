# Reference — capacitance from electrode charge fluctuations (parallel plate)

## oracle

```text
oracle: literature
oracle_version: Limmer et al. J. Chem. Phys. 139, 184704 (2013) eq. (charge fluctuation C); Scalfi et al. Phys. Chem. Chem. Phys. 22, 10480 (2020); MetalWalls JOSS 5(53) 2373 (2020) charge-fluctuation analysis tools
```

No MetalWalls binary and no LAMMPS ELECTRODE / built-in charge-fluctuation \(C\)
compute on the agent host. The offline pin is the closed-form evaluation of the
literature fluctuation formula on a synthetic electrode-\(Q\) sample series
(proxy for a short const-pot MD charge time series).

## Physical statement

At fixed applied electrode potential, the differential capacitance of a
parallel-plate electrochemical cell follows from electrode charge fluctuations
(Limmer et al.; Scalfi / MetalWalls post-processing):

\[
C_{\mathrm{diff}}
  = \frac{\langle Q^2\rangle - \langle Q\rangle^2}{k_{\mathrm{B}}T}
  = \frac{\mathrm{Var}(Q)}{k_{\mathrm{B}}T}
  \qquad[\mathrm{e}/\mathrm{V}],
\]

with \(Q\) the total charge on one electrode [e], \(T\) [K], and
\(k_{\mathrm{B}} = 8.617333262\times 10^{-5}\,\mathrm{eV\,K^{-1}}\)
(CODATA 2018, `units::kB_ev_per_k`) so that \(k_{\mathrm{B}}T\) is in eV and
\(1\,e\cdot 1\,\mathrm{V} = 1\,\mathrm{eV}\).

When a fixed potential difference \(\Delta\Psi\) [V] is known, the integral
capacitance is

\[
C_{\mathrm{int}} = \frac{\langle Q\rangle}{\Delta\Psi}
  \qquad[\mathrm{e}/\mathrm{V}].
\]

## Offline pin procedure (synthetic \(Q\) series)

Geometry: parallel-plate toy — 4 left wall atoms (type 1), 4 right (type 2),
box \(L=20\) Å. Electrolyte atoms are absent (not needed for charge accounting).
Each sample assigns equal per-atom charges so that wall totals are \(Q_L\) and
\(Q_R=-Q_L\).

| Quantity | Value |
| --- | ---: |
| \(T\) | \(300\) K |
| \(\Delta\Psi\) | \(1.0\) V |
| \(Q_L\) samples [e] | \(0.4,\;0.6,\;0.4,\;0.6\) |
| \(N\) | \(4\) |

Closed form:

\[
\langle Q\rangle = 0.5\,\mathrm{e},\qquad
\mathrm{Var}(Q) = 0.01\,\mathrm{e}^2,
\]

\[
k_{\mathrm{B}}T = 8.617333262\times 10^{-5}\times 300 = 0.025851999786\,\mathrm{eV},
\]

\[
C_{\mathrm{diff}}^{\mathrm{pin}} = \frac{0.01}{0.025851999786}
  = 0.38681727072485284\,\mathrm{e/V},
\]

\[
C_{\mathrm{int}}^{\mathrm{pin}} = \frac{0.5}{1.0} = 0.5\,\mathrm{e/V}.
\]

Recorded offline (Python / CODATA \(k_B\) matching ATV):

```text
PIN_MEAN_Q      0.5
PIN_VAR_Q       0.01
PIN_C_DIFF      0.38681727072485284
PIN_C_INT       0.5
```

## Assertion

Atomiverse `cpu::CapacitanceChargeFluctuation({1, 2}, 300.0, 1.0)` driven with
the same four wall-charge assignments:

| Observable | Pin | Tol |
| --- | ---: | ---: |
| \(\langle Q\rangle\) [e] | \(0.5\) | \(10^{-12}\) |
| \(\mathrm{Var}(Q)\) [e²] | \(0.01\) | \(10^{-12}\) |
| \(C_{\mathrm{diff}}\) [e/V] | \(0.38681727072485284\) | \(10^{-12}\) |
| \(C_{\mathrm{int}}\) [e/V] | \(0.5\) | \(10^{-12}\) |

Tolerance is machine-eps scale: exact rational moments + the shared \(k_B\)
constant. Catches wrong electrode selection, double-counting, unit mistakes in
\(k_B T\), or inverted \(C=\mathrm{Var}/(kT)\) algebra.

## Why literature (not LAMMPS / live MetalWalls)

- LAMMPS core has no built-in electrode charge-fluctuation capacitance compute;
  the local `lmp` binary also lacks ELECTRODE.
- MetalWalls analysis tools implement this formula; no MW binary is installed
  on the agent host for a live twin.
- The formula identity is textbook for constant-potential electrodes (Limmer
  2013; Scalfi 2020; MetalWalls JOSS 2020).

## Perf

```text
perf_exempt: true
perf_exempt_reason: >
  Analysis Observer is O(N_electrode) charge sums + O(1) moment updates over
  four synthetic samples. No MetalWalls or LAMMPS twin binary for this named
  capacitance post-process on the agent host; fair wall vs an oracle binary is
  not applicable.
```

ratio_atv_over_oracle: n/a
within_10x: exempt
