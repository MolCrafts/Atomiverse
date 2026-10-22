# md/electrode-observables

One case; every section below is run by `atv_regressions --case md/electrode-observables` and keeps its own oracle record.


## Reference — capacitance from electrode charge fluctuations (parallel plate)

### oracle

```text
oracle: literature
oracle_version: Limmer et al. J. Chem. Phys. 139, 184704 (2013) eq. (charge fluctuation C); Scalfi et al. Phys. Chem. Chem. Phys. 22, 10480 (2020); MetalWalls JOSS 5(53) 2373 (2020) charge-fluctuation analysis tools
```

No MetalWalls binary and no LAMMPS ELECTRODE / built-in charge-fluctuation \(C\)
compute on the agent host. The offline pin is the closed-form evaluation of the
literature fluctuation formula on a synthetic electrode-\(Q\) sample series
(proxy for a short const-pot MD charge time series).

### Physical statement

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

### Offline pin procedure (synthetic \(Q\) series)

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

### Assertion

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

### Why literature (not LAMMPS / live MetalWalls)

- LAMMPS core has no built-in electrode charge-fluctuation capacitance compute;
  the local `lmp` binary also lacks ELECTRODE.
- MetalWalls analysis tools implement this formula; no MW binary is installed
  on the agent host for a live twin.
- The formula identity is textbook for constant-potential electrodes (Limmer
  2013; Scalfi 2020; MetalWalls JOSS 2020).

### Perf

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


## Reference — charging work \(-\sum_i \Psi_i Q_i\) (parallel plate)

oracle: literature
oracle_version: MetalWalls JOSS 5(53) 2373 (2020) eq.6; Scalfi et al. PCCP 22 10480 (2020) / arXiv:1911.09351 eq.12

### Physical statement

Under constant applied electrode potentials, the classical potential energy
includes the **charging work** exchanged with the external circuit. MetalWalls
(Marin-Laflèche et al., *JOSS* **5**, 2373 (2020), DOI 10.21105/joss.02373)
writes

\[
V(Q) = V_c(Q) - \sum_{i=1}^{N} \Psi_i Q_i
\tag{JOSS eq. 6}
\]

where \(V_c\) is the Coulomb energy of the charge distribution and the second
term is the work to charge the electrodes. Scalfi et al. (*PCCP* **22**,
10480 (2020)) identify \(q^{\mathrm{T}}\Psi = Q_L\Delta\psi\) as that work and
use \(-\,q^{\mathrm{T}}\Psi\) in the constant-potential Boltzmann weight.

This regression pins **only** the second term (the bookkeeping Component
`cpu::electrodes::ChargingWork`). Coulomb \(V_c\) and the BO charge solve are
other E1 specs.

### Geometry (offline pin)

Symmetric parallel-plate toy with **fixed** charges (proxy for post-const-pot
\(Q\)):

| Region | type | \(N\) | \(Q_i\) [e] | \(\Psi\) [V] |
| ------ | ---: | ----: | ----------: | -----------: |
| Left wall | 1 | 4 | \(+0.10\) | \(+0.50\) |
| Right wall | 2 | 4 | \(-0.10\) | \(-0.50\) |

Positions: left \(z = 0\), right \(z = 10\) Å; cubic box \(L = 20\) Å (unused
by the bookkeeping term).

### Closed-form pin

\[
E_{\mathrm{charge}}
  = -\bigl(4\cdot 0.50\cdot 0.10 + 4\cdot(-0.50)\cdot(-0.10)\bigr)
  = -0.4~\mathrm{eV}.
\]

Scalfi check: \(Q_L = 0.4\,\mathrm{e}\), \(\Delta\psi = 1.0\,\mathrm{V}\),
\(q^{\mathrm{T}}\Psi = Q_L\Delta\psi = 0.4\), \(E = -q^{\mathrm{T}}\Psi = -0.4\) eV.

**Units:** ATV metal — \(Q\) [e], \(\Psi\) [V], energy [eV]
(\(1\,e\cdot 1\,\mathrm{V} = 1\,\mathrm{eV}\)).

### Tolerance

Absolute \(1\times 10^{-12}\) eV (exact rational arithmetic).

### What is asserted

```text
| frame.meta.get(kEnergy) − (−0.4) | ≤ 1e-12 eV
| frame.meta.get(kEnergyChargingWork) − (−0.4) | ≤ 1e-12 eV
```

via public `cpu::Evaluate` + `cpu::electrodes::ChargingWork` only.

### Perf

perf_exempt: true

reason: No MetalWalls/LAMMPS twin binary for this named energy term in the
agent environment; the Component is pure \(O(N_{\mathrm{electrode}})\) arithmetic
with no force evaluation. Fair wall vs an oracle binary is not applicable
until a MetalWalls install is wired under `tools/metalwalls_oracle/`.

ratio_atv_over_oracle: n/a
within_10x: exempt


## thermo-integration-capacitance

oracle: literature
oracle_version: Reed/Scalfi Gaussian electrode linear-response capacitance;
Beutler et al. Chem. Phys. Lett. 222, 529 (1994) soft-core Kirkwood TI
oracle_kind: closed-form linear capacitor + soft-core TI identity

### Physics (map E10)

Soft-core / λ **thermodynamic integration workflow** for electrode capacitance:

1. **Differential C** of an empty const-pot parallel plate: \(C = Q/\Delta\Psi\)
   independent of \(\Psi\) for linear Gaussian BO.
2. **Kirkwood identity** on `PairCoulSoft`: \(\int_0^1 \partial U/\partial\lambda\,d\lambda = U(1)-U(0)\).
3. **Charging free energy** \(A = \tfrac12 C\Psi^2 = \tfrac12 Q\Psi\), consistent
   with `ChargingWork` \(W = -\sum_i \Psi_i Q_i\).

### Assertions

| gate | check |
|------|-------|
| A | C(0.5 V) ≈ C(1 V) within 1%; Q ∝ Ψ |
| B | soft-Coulomb TI identity ≤ 1e-5 eV |
| C | W = −QΨ; A = ½ QΨ |

### Perf

`perf_exempt` — algebraic + fixed-geometry TI, not a wall-clock twin.


## Reference — slab electrostatic potential profile \(\Psi(z)\) (E10)

### oracle

```text
oracle: literature
oracle_version: planar Poisson / parallel-plate capacitor (Jackson Classical Electrodynamics ch.1 continuum); MetalWalls JOSS 5(53) 2373 (2020) potential-profile analysis tools; Scalfi et al. Phys. Chem. Chem. Phys. 22, 10480 (2020) EDL profiles
```

No MetalWalls binary and no LAMMPS built-in planar Poisson \(\Psi(z)\) compute
on the agent host. The offline pin is the closed-form planar Poisson
integration (MD metal units) on a synthetic two-sheet parallel-plate geometry
(proxy for a const-pot electrode slab charge density profile).

### Physical statement

For planar symmetry the electrostatic potential satisfies

\[
\frac{\mathrm{d} E_z}{\mathrm{d} z}
  = 4\pi\,k_{\mathrm{e}}\,\rho(z)
  = \frac{\rho(z)}{\varepsilon_0},
\qquad
E_z = -\frac{\mathrm{d}\Psi}{\mathrm{d} z},
\]

with \(k_{\mathrm{e}} = 14.399645\,\mathrm{eV\cdot Å\cdot e^{-2}}\)
(`units::coulomb`, LAMMPS metal `qqr2e`), \(\rho\) [e/Å³], \(E_z\) [V/Å],
\(\Psi\) [V]. Gauge: \(\Psi(z_{\min})=0\), \(E_z(z_{\min})=0\).

Ideal infinite parallel-plate capacitor with surface charge \(\pm\sigma\):

\[
E_{\mathrm{gap}} = 4\pi k_{\mathrm{e}}\,\sigma
  \qquad\text{(between plates; 0 outside)}.
\]

### Offline pin geometry

| Quantity | Value |
| --- | ---: |
| \(L_x = L_y\) | \(10\) Å |
| \(L_z\) | \(20\) Å |
| Area \(A\) | \(100\) Å² |
| Bin window | \(z\in[0,20)\) Å |
| \(n_{\mathrm{bins}}\) | \(20\) |
| \(\Delta z\) | \(1\) Å |
| Left sheet | \(Q=+1\,e\) at \(z=5.5\) Å (bin 5) |
| Right sheet | \(Q=-1\,e\) at \(z=14.5\) Å (bin 14) |
| \(\sigma = Q/A\) | \(0.01\) e/Å² |

### Closed-form / discrete pin

\[
E_{\mathrm{gap}}^{\mathrm{pin}}
  = 4\pi\times 14.399645\times 0.01
  = 1.80951275785204\,\mathrm{V/Å}.
\]

Piecewise-constant \(\rho\) integration (implementation algorithm) with the
two sheets as uniform density in bins 5 and 14:

| Quantity | Pin value |
| --- | ---: |
| \(E_z\) mid-gap (bin 9, \(z=9.5\)) | \(1.80951275785204\) V/Å |
| \(E_z\) vacuum left (bin 0) | \(0\) |
| \(E_z\) vacuum right (bin 19) | \(0\) |
| \(\Psi\) bin 0 | \(0\) V |
| \(\Psi\) bin 19 (= potential drop) | \(-16.28561482066836\) V \(= -E_{\mathrm{gap}}\times 9\) |
| \(\rho\) bin 5 | \(0.01\) e/Å³ |
| \(\rho\) bin 14 | \(-0.01\) e/Å³ |

Recorded offline (Python, same \(k_{\mathrm{e}}\) / \(\pi\) as ATV):

```text
PIN_E_GAP           1.80951275785204
PIN_PSI_DROP        -16.28561482066836
PIN_RHO_LEFT        0.01
PIN_RHO_RIGHT       -0.01
```

Continuum cross-check: mid-gap \(E\) matches \(4\pi k_{\mathrm{e}}\sigma\)
exactly under this binning; potential drop equals \(E_{\mathrm{gap}}\) times
center-to-center plate separation (\(9\) Å).

### Tolerances

| Check | Tol | Rationale |
| --- | ---: | --- |
| \(E_{\mathrm{gap}}\), \(\Psi\) drop, \(\rho\) bins | \(1\times 10^{-12}\) absolute | Exact double arithmetic of discrete scheme |
| Continuum \(E\) identity | \(1\times 10^{-12}\) | Algebraic identity with `units::coulomb` |

### Perf

```text
perf_exempt: true
perf_exempt_reason: Analysis Observer is O(N_atoms) charge binning + O(n_bins) planar Poisson integration on a 2-atom synthetic capacitor. No MetalWalls or LAMMPS twin binary for this named potential-profile post-process on the agent host; fair wall vs an oracle binary is not applicable.
```
