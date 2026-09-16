# Reference — DFTB3 analytic nuclear forces (H2O)

```text
oracle: literature
oracle_version: Gaus, Cui, Elstner, J. Chem. Theory Comput. 7, 931 (2011),
  DOI 10.1021/ct100684s — DFTB3 nuclear gradient
family_oracle: dftb+
```

**This pin is not `oracle: dftb+` and must not be labelled as such.** Two
named blockers:

1. **Mixed parameter set.** The run declares
   `d3.sk_set = cpu::semiempirical::DftbSkSet::MioSketch` and therefore pairs
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

## Method

Third-order self-consistent-charge DFTB. Forces are the **closed-form
gradient at the converged SCC density** (Hellmann–Feynman + Pulay +
fixed-\(\Delta q\) \(\partial(E^{(2)}+E^{(3)})/\partial R\) + repulsive).
\(\Gamma_{AB}\neq\Gamma_{BA}\), so both third-order orderings carry a radial
derivative. On-site \(\gamma_{AA}=U_A\) and \(\Gamma_{AA}=U^{\mathrm d}_A\)
carry no geometry ⇒ zero gradient. A converged SCC-DFTB3 density is
stationary, so there is no coupled-perturbed term; the FD identity is the
experiment that proves it.

## Hamiltonian / tables

- ATV class: `cpu::semiempirical::DFTB3`
- SK / Hubbard \(U\) / repulsive: transcribed mio-1-1.
- \(U^{\mathrm d}\): published 3ob values, assigned explicitly.

## Geometry (Å)

| atom | Z | x | y | z |
| ---- | - | - | - | - |
| O | 8 | 0.00 | 0.00 | 0.00 |
| H | 1 | 1.05 | 0.00 | 0.00 |
| H | 1 | −0.26 | 1.00 | 0.00 |

Isolated molecule, cubic box 20 Å.
`scc_tol = 1e-10`, `max_scc = 50`, `mix_beta = 0.3`, `with_forces = true`.

## Pins (hard-coded gates)

### 1. FD identity — the physics oracle (formula-exact, no binary needed)

Analytic \(F\) vs central **re-SCC** finite difference of the same total energy
(\(h = 10^{-4}\) Å), tol **3×10⁻⁵ eV/Å** (lattice-walk residual vs re-SCC FD):

| component | analytic [eV/Å] | re-SCC FD [eV/Å] | \|d\| [eV/Å] |
| --------- | --------------- | ---------------- | ------------ |
| \(F_{O,x}\) | 2.662706377721 | 2.662725579441 | 1.9e-05 |
| \(F_{H1,y}\) | −0.175866697553 | −0.175856553852 | 1.0e-05 |

### 2. Translational invariance

\(|\sum_i F_i| \le 10^{-10}\) eV/Å per Cartesian.

### 3. Third order must move the *force*, not just the energy

\(F_{O,x}(\mathrm{DFTB3})-F_{O,x}(\mathrm{DFTB2})=0.221\) eV/Å at this
geometry (gate: > 1e-6).

### 4. ATV-twin determinism snapshot (not an oracle)

| Quantity | Value | Unit | Tol |
| -------- | ----- | ---- | --- |
| \(E_{\mathrm{tot}}\) | −110.961689188230 | eV | 1e-6 |
| \(F_{O,x}\) | 2.662706377721 | eV/Å | 1e-6 |
| \(F_{H1,y}\) | −0.175866697553 | eV/Å | 1e-6 |

Measured 2026-09-04, Release, `OMP_NUM_THREADS=1`. \(E^{(3)}=-0.2935994\) eV.
Supersedes sketch-\(U^{\mathrm d}\) snapshot −110.972004227931 eV.

## Gaps

- Mixed-set + missing \(\gamma^h\) (the two blockers above) — no absolute
  DFTB+ 24.1 force is claimed.
- **Molecular / Γ-lattice only.** Open shell, ThreeOb SK, D3, and cell
  stress on the DFTB3 path are owed slugs.
- Water pairs O(s,p) only with H(s); the pp-σ/π SK derivative is unit-gated
  on C–O (`test_dftb3.cpp`). An N-containing DFTB3 force case is still owed.

## Perf

`perf_exempt: true` — no matched DFTB+ Hamiltonian to wall against.
Algorithmically the analytic path replaces \(6N\) full re-SCC solves with one
post-SCC contraction.
