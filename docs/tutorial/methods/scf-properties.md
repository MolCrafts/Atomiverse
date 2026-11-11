# Molecular SCF properties

> The regression runner has been removed. Regression commands below are historical; case sources and oracle records are retained as archived evidence, not current executable checks.

Properties of a converged molecular mean-field determinant are computed by
companion types that bind a concrete method rather than by members of it:
`cpu::scf::WannierCenters::compute(rhf.result_handle(), frame)` localizes the
occupied orbitals (Foster–Boys), `cpu::scf::Polarizability<cpu::scf::RHF>{rhf}.run()`
solves the CPHF electric-field response, and `cpu::MultipoleObserver` under a
`cpu::Evaluate` reads dipole moments from the RHF it hosts.

All sections run on CPU. Besides the two documented below, the case carries
`h2o_rhf_dipole` (H₂O RHF/6-31G* dipole moment) and `h2o_charges_pyscf`
(H₂O RHF/6-31G* Löwdin population and MK/CHELPG ESP charges), whose oracle
records live in the case `REFERENCE.md`.

Case: `ctest -R '^scf/properties$'` — sources and oracle record in [`regressions/scf/properties/`](../../../regressions/scf/properties/). One section of the case per `##` heading below: `atv_regressions --case scf/properties --section <name>`.

## Common setup

- H₂O geometry (Å): O (0, 0, 0.1173); H (0, ±0.7572, −0.4692), the same in
  every section; RHF / STO-3G for the two sections below.
- Oracle protocol: `docs/tutorial/appendix-pyscf-oracle.md`; offline pins
  only, never PySCF at ctest time.

## wannier-centers — Foster–Boys Wannier centers {#wannier-centers}

### Summary

**Maximally localized Wannier functions (MLWF)** turn delocalized canonical
occupied MOs into compact orbitals whose centroids are the **Wannier
centers**. `WannierCenters` (`cpu/scf/wannier.h`, result `WannierResult`)
implements **Foster–Boys** localization (maximize
\(\sum_i|\langle\mathbf r\rangle_i|^2\)) for **closed-shell molecular**
RHF/RKS via Jacobi pair rotations on the occupied MO block. AO multipole
matrices reuse `scf::MultipoleIntegrals` about the Cartesian origin.
Roadmap: [dft.md](../../roadmap/dft.md) §11 MLWF. Speed is **`perf_exempt`**.

**Not covered:** periodic multi-k Marzari–Vanderbilt MLWF, GPW-grid or
plane-wave Wannier, Pipek–Mezey / ER criteria (roadmap ○ beyond the
molecular path).

### Method

Occupied spatial MOs \(\{\varphi_i\}\) from a restricted determinant. Define

\[
\mathbf R_{ij} = \langle\varphi_i|\mathbf r|\varphi_j\rangle,
\qquad
\Omega_i = \langle\varphi_i|r^2|\varphi_i\rangle - |\mathbf R_{ii}|^2.
\]

Boys maximizes \(\sum_i|\mathbf R_{ii}|^2\) (equivalently minimizes
\(\sum_i\Omega_i\)) under unitary mixing of occupied orbitals. Each pair
\((i,j)\) is rotated by

\[
\gamma = \tfrac14\operatorname{atan2}(B,A),
\]

with \(A^a=R_{ii}^a-R_{jj}^a\), \(B^a=2R_{ij}^a\),
\(A=\sum_a((A^a)^2-(B^a)^2)\), \(B=\sum_a 2A^a B^a\).

Outputs: centers \(\mathbf R_{ii}\) **[Å]**, spreads \(\Omega_i\) **[Å²]**.

### Usage

```cpp
cpu::scf::RHF rhf{cpu::scf::BasisSet::STO3G};
rhf.tol = 1e-10;
rhf.run(frame);   // closed-shell water, H2, …

cpu::scf::WannierCenters wb;   // members: max_sweeps, tol
auto r = wb.compute(rhf.result_handle(), frame);

// r.centers[i]  — Wannier center of occupied orbital i [Å]
// r.spreads[i]  — Boys spread Ω_i [Å²]
// r.total_spread, r.total_spread_initial, r.converged, r.sweeps
```

`compute` throws `std::invalid_argument` for open-shell / GHF / not-ready
results (no silently degraded path).

### Pin

- H₂ / STO-3G: single center at the bond midpoint (±1e-4 Å).
- H₂O / STO-3G: five centers; \(\Omega_\mathrm{tot}\) decreases; O-core +
  two OH-bond + two opposite-\(x\) lone pairs (rabbit ears); C₂ᵥ sum of
  \(x,y\) components ≈ 0.

These are geometric / symmetry pins (case `REFERENCE.md`); a CP2K `LOCALIZE`
numerical oracle is offline-only when a binary is available.

## static-polariz-h2o — RHF static polarizability, H₂O / STO-3G {#static-polariz-h2o}

Dipole polarizability \(\alpha_{ab} = \partial \mu_a / \partial F_b\) from
**CPHF** electric-field response (singlet A+B).

### Theory

Static polarizability is a second energy derivative w.r.t. a uniform field.
CPHF solves the first-order orbital response \(U^{F_b}\) and contracts with
dipole integrals. Equivalent checks: finite-field \(\mathrm{d}\mu/\mathrm{d}F\)
and \(-\partial^2 E/\partial F^2\).

### Practice

```cpp
cpu::scf::RHF rhf{cpu::scf::BasisSet::STO3G};
rhf.tol = 1e-12;
rhf.run(frame);
const auto alpha = cpu::scf::Polarizability<cpu::scf::RHF>{rhf}.run(); // 3×3 [a.u.]
```

### Results

#### Accuracy vs PySCF 2.14.0 (CPHF / FF cross-check)

The pin is analytic CPHF on PySCF MOs/ERIs, cross-checked by finite field
(\(h\sim 2\times 10^{-4}\) a.u.). Pins (case `REFERENCE.md`): α_xx =
0.04190863, α_yy = 5.20973255, α_zz = 2.12715820 a.u., gated at
\(10^{-5}\) a.u. per diagonal component; off-diagonals ≲ 1e-4 (C₂v). STO-3G
makes α_xx unphysically small — the comparison is engine-vs-engine at
identical theory, not vs experiment.

#### Speed

| Twin | \(T_{\mathrm{PySCF}}\) | \(T_{\mathrm{ATV}}\) | \(r\) | ≤10× |
| ---- | ---------------------: | -------------------: | ----: | ---- |
| vacuum `rhf` | 0.63 s | 0.13 s | **0.20** | ✓ |

ATV includes the CPHF polarizability work while the twin is SCF-only, so the
ratio is optimistic.
