# semiempirical/dftb3

One case; every section below is run by `atv_regressions --case semiempirical/dftb3` and keeps its own oracle record.


## H2O SCC-DFTB3 (mio SK + 3ob \(U^{\mathrm d}\), mixed set)

```text
oracle: literature
oracle_version: Gaus, Cui, Elstner, J. Chem. Theory Comput. 7, 931 (2011),
  DOI 10.1021/ct100684s — DFTB3 E^(3)/Γ formulation
family_oracle: dftb+
```

**This pin is not `oracle: dftb+` and must not be labelled as such.** Two
named blockers:

1. **Mixed parameter set.** The run constructs
   `DFTB3{cpu::semiempirical::DftbSkSet::MioSketch}` and therefore pairs
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

### Third-order convention (what the code computes)

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

### Geometry

Water, slightly off-equilibrium (Å):

| Atom | x    | y    | z   |
|------|------|------|-----|
| O    | 0.00 | 0.00 | 0.0 |
| H    | 0.96 | 0.00 | 0.0 |
| H    |−0.24 | 0.93 | 0.0 |

Cubic box 20 Å (isolated). `scc_tol = 1e-10`, `max_scc = 50`, forces off.

### ATV twin snapshot (not an oracle pin)

| Quantity | Value | Unit | Tol |
| -------- | ----- | ---- | --- |
| \(E_{\mathrm{tot}}\) | **−111.195830730395** | eV | \(10^{-6}\) |

Measured 2026-09-04, Release, `OMP_NUM_THREADS=1`. Same geometry, DFTB2:
**−110.958967574695 eV**. \(E^{(3)}=-3.048297\times10^{-1}\) eV,
\(|E_{\mathrm{DFTB3}}-E_{\mathrm{DFTB2}}|/|E^{(3)}|=0.777\).

Supersedes sketch-\(U^{\mathrm d}\) snapshot −111.204208212956 eV.

### Reduction check (a real oracle, not a snapshot)

For a homonuclear system \(\Delta q\equiv 0\), so \(E^{(2)}=E^{(3)}=0\) and
DFTB3 must reduce **exactly** to DFTB2/DFTB0. Verified on H₂ at \(r=0.74\) Å:
DFTB3 and DFTB2 agree at **−18.366350522958 eV**, which *is* the DFTB+ 24.1
+ mio-1-1 pin **−18.36635052 eV**
(`regressions/semiempirical/dftb2/`, residual 3.0e-09 eV).
Gated by `tests/cpu/semiempirical/test_dftb3.cpp::ReducesToDftb2ForHomonuclearH2`.

### Gates

- Finite, non-zero total energy [eV] on `frame.meta[kEnergy]`
- SCC residual < `scc_tol` (converged, not budget-exhausted)
- Hard-coded ATV-twin energy vs table above (tol \(10^{-6}\) eV)
- \(|E_{\mathrm{DFTB3}} - E_{\mathrm{DFTB2}}| > 10^{-6}\) eV (third-order active)
- \(0.5\,|E^{(3)}| \le |E_{\mathrm{DFTB3}} - E_{\mathrm{DFTB2}}| \le
  2\,|E^{(3)}|\)

### Perf

`perf_exempt: true` — mixed-set DFTB3 has no matched DFTB+ Hamiltonian to
wall against. Revisit when `dftb-gamma-h-damping` lands a 3ob + \(\zeta=4.00\)
path.


## Reference — DFTB3 analytic nuclear forces (H2O)

```text
oracle: literature
oracle_version: Gaus, Cui, Elstner, J. Chem. Theory Comput. 7, 931 (2011),
  DOI 10.1021/ct100684s — DFTB3 nuclear gradient
family_oracle: dftb+
```

**This pin is not `oracle: dftb+` and must not be labelled as such.** Two
named blockers:

1. **Mixed parameter set.** The run constructs
   `DFTB3{cpu::semiempirical::DftbSkSet::MioSketch}` and therefore pairs
   mio-1-1 SK/repulsive (`src/cpu/semiempirical/dftb3.cpp`) with 3ob-class
   Hubbard derivatives (`hubbard_du_ha = basis_data::k3obHubbardDerivsHa`).
   The engine `ATV_WARN`s. This is not a DFTB+-reproducible Hamiltonian.
2. **Missing \(\gamma^h\) X–H damping (\(\zeta=4.00\)).** Follow-on slug:
   `dftb-gamma-h-damping`.

The absolute \(E\)/\(F\) numbers below are an **ATV twin snapshot** of that
mixed-set Hamiltonian, not an oracle pin. The real gate is analytic vs
re-SCC FD of the same ATV energy.

Force-module boundary (formula): Hourahine et al., *J. Chem. Phys.* **152**,
124101 (2020). Family-locked oracle package is **DFTB+** (never OpenMOPAC,
never xtb). ctest never launches it.

### Method

Third-order self-consistent-charge DFTB. Forces are the **closed-form
gradient at the converged SCC density** (Hellmann–Feynman + Pulay +
fixed-\(\Delta q\) \(\partial(E^{(2)}+E^{(3)})/\partial R\) + repulsive).
\(\Gamma_{AB}\neq\Gamma_{BA}\), so both third-order orderings carry a radial
derivative. On-site \(\gamma_{AA}=U_A\) and \(\Gamma_{AA}=U^{\mathrm d}_A\)
carry no geometry ⇒ zero gradient. A converged SCC-DFTB3 density is
stationary, so there is no coupled-perturbed term; the FD identity is the
experiment that proves it.

### Hamiltonian / tables

- ATV class: `cpu::semiempirical::DFTB3`
- SK / Hubbard \(U\) / repulsive: transcribed mio-1-1.
- \(U^{\mathrm d}\): published 3ob values, assigned explicitly.

### Geometry (Å)

| atom | Z | x | y | z |
| ---- | - | - | - | - |
| O | 8 | 0.00 | 0.00 | 0.00 |
| H | 1 | 1.05 | 0.00 | 0.00 |
| H | 1 | −0.26 | 1.00 | 0.00 |

Isolated molecule, cubic box 20 Å.
`scc_tol = 1e-10`, `max_scc = 50`, `mix_beta = 0.3`, `with_forces = true`.

### Pins (hard-coded gates)

#### 1. FD identity — the physics oracle (formula-exact, no binary needed)

Analytic \(F\) vs central **re-SCC** finite difference of the same total energy
(\(h = 10^{-4}\) Å), tol **3×10⁻⁵ eV/Å** (lattice-walk residual vs re-SCC FD):

| component | analytic [eV/Å] | re-SCC FD [eV/Å] | \|d\| [eV/Å] |
| --------- | --------------- | ---------------- | ------------ |
| \(F_{O,x}\) | 2.662706377721 | 2.662725579441 | 1.9e-05 |
| \(F_{H1,y}\) | −0.175866697553 | −0.175856553852 | 1.0e-05 |

#### 2. Translational invariance

\(|\sum_i F_i| \le 10^{-10}\) eV/Å per Cartesian.

#### 3. Third order must move the *force*, not just the energy

\(F_{O,x}(\mathrm{DFTB3})-F_{O,x}(\mathrm{DFTB2})=0.221\) eV/Å at this
geometry (gate: > 1e-6).

#### 4. ATV-twin determinism snapshot (not an oracle)

| Quantity | Value | Unit | Tol |
| -------- | ----- | ---- | --- |
| \(E_{\mathrm{tot}}\) | −110.961689188230 | eV | 1e-6 |
| \(F_{O,x}\) | 2.662706377721 | eV/Å | 1e-6 |
| \(F_{H1,y}\) | −0.175866697553 | eV/Å | 1e-6 |

Measured 2026-09-04, Release, `OMP_NUM_THREADS=1`. \(E^{(3)}=-0.2935994\) eV.
Supersedes sketch-\(U^{\mathrm d}\) snapshot −110.972004227931 eV.

### Gaps

- Mixed-set + missing \(\gamma^h\) (the two blockers above) — no absolute
  DFTB+ 24.1 force is claimed.
- **Molecular / Γ-lattice only.** Open shell, ThreeOb SK, D3, and cell
  stress on the DFTB3 path are owed slugs.
- Water pairs O(s,p) only with H(s); the pp-σ/π SK derivative is unit-gated
  on C–O (`test_dftb3.cpp`). An N-containing DFTB3 force case is still owed.

### Perf

`perf_exempt: true` — no matched DFTB+ Hamiltonian to wall against.
Algorithmically the analytic path replaces \(6N\) full re-SCC solves with one
post-SCC contraction.


## DFTB3 Γ-PBC analytic forces

```text
oracle: literature
oracle_version: Gaus, Cui, Elstner, J. Chem. Theory Comput. 7, 931 (2011),
  DOI 10.1021/ct100684s; lattice walk mirrors DFTB2
family_oracle: dftb+
```

**This pin is not `oracle: dftb+` and must not be labelled as such.** Two
named blockers:

1. **Mixed parameter set.** The run constructs
   `DFTB3{cpu::semiempirical::DftbSkSet::MioSketch}` and therefore pairs
   mio-1-1 SK/repulsive (`src/cpu/semiempirical/dftb3.cpp`) with 3ob-class
   Hubbard derivatives (`hubbard_du_ha = basis_data::k3obHubbardDerivsHa`).
   The engine `ATV_WARN`s. This is not a DFTB+-reproducible Hamiltonian.
2. **Missing \(\gamma^h\) X–H damping (\(\zeta=4.00\)).** Follow-on slug:
   `dftb-gamma-h-damping`.

The absolute energy below is an **ATV twin snapshot**, not an oracle pin.
The real gate is analytic nuclear forces vs re-SCC central FD of the
lattice-summed energy.

### Gate

Off-eq C-chain (r = 1.20 Å, \(L_x=2.8\) Å, \(L_y=L_z=12\) Å) so left/right
lattice shells differ. Analytic \(F\) vs re-SCC FD, tol **1×10⁻⁵ eV/Å**.
Net force ~0.

This geometry is **homonuclear**, so \(\Delta q\equiv 0\), \(E^{(3)}=0\), and
DFTB3 reduces to DFTB2 at Γ. It still
exercises the lattice-image walk on the DFTB3 path.

### ATV twin snapshot (not an oracle pin)

Measured 2026-09-04, Release, `OMP_NUM_THREADS=1`:

```text
E = -93.1115376115 eV
F[0] = (-14.8265853692, 0, 0) eV/Å
F[1] = ( 14.8265853692, 0, 0) eV/Å
F0x analytic=-14.8265853692  fd=-14.8265872815  |d|=1.912e-06 eV/Å
```

### Perf

`perf_exempt: true` — no matched DFTB+ DFTB3 Hamiltonian to wall against.
