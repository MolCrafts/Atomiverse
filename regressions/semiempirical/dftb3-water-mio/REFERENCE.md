# H2O SCC-DFTB3 (mio SK + 3ob \(U^{\mathrm d}\), mixed set)

```text
oracle: literature
oracle_version: Gaus, Cui, Elstner, J. Chem. Theory Comput. 7, 931 (2011),
  DOI 10.1021/ct100684s — DFTB3 E^(3)/Γ formulation
family_oracle: dftb+
```

**This pin is not `oracle: dftb+` and must not be labelled as such.** Two
named blockers:

1. **Mixed parameter set.** The run declares
   `d3.sk_set = cpu::semiempirical::DftbSkSet::MioSketch` and therefore pairs
   mio-1-1 SK/repulsive (`src/cpu/semiempirical/dftb3.cpp`) with 3ob-class
   Hubbard derivatives (`d3.hubbard_du_ha = basis_data::k3obHubbardDerivsHa`).
   The engine `ATV_WARN`s (`[dftb3] … 3ob supersedes the MIO parameter set`).
   This is a user-declared mixed-set pairing, **not** a DFTB+-reproducible
   Hamiltonian. 3ob-3-1 README: 3ob "supersedes the MIO parameter set".
2. **Missing \(\gamma^h\) X–H damping (\(\zeta=4.00\)).** DFTB3/3ob is
   undefined without DFTB+ `DampXHExponent`. Follow-on slug:
   `dftb-gamma-h-damping` (includes wiring `DftbSkSet::ThreeOb` into
   `dftb3.cpp`).

The absolute energy below is an **ATV twin snapshot** of that mixed-set
Hamiltonian, not an oracle pin. The real gates are (i) the homonuclear
reduction to the DFTB+ 24.1 H₂ pin and (ii) \(E^{(3)}\) moving the energy
by its own magnitude.

Family-locked numerical oracle package is **DFTB+** (never OpenMOPAC / xtb).
ctest never launches it.

## Third-order convention (what the code computes)

With \(\gamma_{AB}\) the DFTB+ Elstner charge kernel and
\(\Gamma_{AB}=\partial\gamma_{AB}/\partial q_A\) (on-site \(\gamma_{AA}=U_A\),
\(\Gamma_{AA}=U^{\mathrm{d}}_A\); \(\Gamma_{AB}\neq\Gamma_{BA}\)):

\[
E^{(2)}=\tfrac12\sum_{AB}\gamma_{AB}\,\Delta q_A\Delta q_B,\qquad
E^{(3)}=\tfrac13\sum_{AB}\Delta q_A^2\,\Delta q_B\,\Gamma_{AB}
\]

(Gaus/Cui/Elstner 2011). The SCC potential is its exact derivative, so
\(\sum_A\Delta q_A V_A=2E^{(2)}+3E^{(3)}\). Pinned by
`tests/cpu/semiempirical/test_dftb3_scc.cpp`.

Published 3ob \(U^{\mathrm d}\) [Ha]: H −0.1857, C −0.1492, N −0.1535,
O −0.1575 (3ob-3-1 README, CC-BY-SA-4.0, Elstner/KIT). Assigned explicitly;
never a silent default.

## Geometry

Water, slightly off-equilibrium (Å):

| Atom | x    | y    | z   |
|------|------|------|-----|
| O    | 0.00 | 0.00 | 0.0 |
| H    | 0.96 | 0.00 | 0.0 |
| H    |−0.24 | 0.93 | 0.0 |

Cubic box 20 Å (isolated). `scc_tol = 1e-10`, `max_scc = 50`, forces off.

## ATV twin snapshot (not an oracle pin)

| Quantity | Value | Unit | Tol |
| -------- | ----- | ---- | --- |
| \(E_{\mathrm{tot}}\) | **−111.195830730395** | eV | \(10^{-6}\) |

Measured 2026-09-04, Release, `OMP_NUM_THREADS=1`. Same geometry, DFTB2:
**−110.958967574695 eV**. \(E^{(3)}=-3.048297\times10^{-1}\) eV,
\(|E_{\mathrm{DFTB3}}-E_{\mathrm{DFTB2}}|/|E^{(3)}|=0.777\).

Supersedes sketch-\(U^{\mathrm d}\) snapshot −111.204208212956 eV.

## Reduction check (a real oracle, not a snapshot)

For a homonuclear system \(\Delta q\equiv 0\), so \(E^{(2)}=E^{(3)}=0\) and
DFTB3 must reduce **exactly** to DFTB2/DFTB0. Verified on H₂ at \(r=0.74\) Å:
DFTB3 and DFTB2 agree at **−18.366350522958 eV**, which *is* the DFTB+ 24.1
+ mio-1-1 pin **−18.36635052 eV**
(`regressions/semiempirical/dftb2-dftbplus-energy-pin/`, residual 3.0e-09 eV).
Gated by `tests/cpu/semiempirical/test_dftb3.cpp::ReducesToDftb2ForHomonuclearH2`.

## Gates

- Finite, non-zero total energy [eV] on `frame.meta[kEnergy]`
- SCC residual < `scc_tol` (converged, not budget-exhausted)
- Hard-coded ATV-twin energy vs table above (tol \(10^{-6}\) eV)
- \(|E_{\mathrm{DFTB3}} - E_{\mathrm{DFTB2}}| > 10^{-6}\) eV (third-order active)
- \(0.5\,|E^{(3)}| \le |E_{\mathrm{DFTB3}} - E_{\mathrm{DFTB2}}| \le
  2\,|E^{(3)}|\)

## Perf

`perf_exempt: true` — mixed-set DFTB3 has no matched DFTB+ Hamiltonian to
wall against. Revisit when `dftb-gamma-h-damping` lands a 3ob + \(\zeta=4.00\)
path.
