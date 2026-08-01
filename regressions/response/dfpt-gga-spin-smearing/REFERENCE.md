# Reference — Γ DFPT phonons, joint collinear spin × pure GGA × metallic smear (diamond Si, PW)

## Method

Density-functional perturbation theory at **q = 0** for the concrete
plane-wave method `cpu::pw::PW` with the **three-way joint** product path:

- collinear spin (`spin_polarized = true`, `multiplicity = 1` ⇒ \(M=0\)),
- pure GGA kernel (`xc = XCType::PBE`),
- Fermi–Dirac metallic occupation smearing (`smearing = fermi_dirac`,
  `degauss = 0.02` Ry, `extra_bands = 0`):

\[
\Delta V_{\mathrm{Hxc},\sigma}[\Delta n_\uparrow,\Delta n_\downarrow]
  = \Delta V_{\mathrm{H}}[\Delta n]
  + \sum_{\sigma'} K_{\mathrm{xc}}^{\sigma\sigma'}[\Delta n_{\sigma'}],
\]

with spinor occupations \(f_{\sigma nk}\in[0,1]\) (shared \(\mu\)), Hartree
from total \(\Delta n\), and \(K_{\mathrm{xc}}^{\sigma\sigma'}\) from central
FD of the **spin-GGA** XC operator (`XCBuilder::step_spin`). Gapped fill at
these settings keeps near-integer FD occupations (\(f\in\{0,1\}\)), so free
energy \(F\approx E_{\mathrm{KS}}\) and results track the fixed-occ
`dfpt-gga-spin-si` parent.

Cite: Baroni, de Gironcoli, Dal Corso, Giannozzi, *Rev. Mod. Phys.* **73**,
515 (2001); de Gironcoli, *Phys. Rev. B* **51**, 6773 (1995); Perdew, Burke,
Ernzerhof, *Phys. Rev. Lett.* **77**, 3865 (1996).

Public path: `cpu::pw::DfptPhonon::compute(method, frame)` with
`method.spin_polarized = true`, `method.xc = XCType::PBE`, and
`method.smearing = Smearing::fermi_dirac`.

Siblings: `dfpt-gga-spin-si` (spin×GGA, fixed occ), `dfpt-spin-smearing`
(spin×smear, LSDA), `dfpt-gga-si` (restricted GGA).

## System (regression twin)

| Setting | Value |
| ------- | ----- |
| Structure | diamond Si, 2-atom fcc primitive cell, a = **5.43 Å** (experimental) |
| Pseudopotential | `tests/data/pw/Si_lda_nc.upf` (ONCVPSP NC, in-tree) |
| XC | `XCType::PBE` (true spin-GGA kernel) |
| Spin | collinear, multiplicity = 1 (\(N_\uparrow=N_\downarrow=4\) seed; free under smear) |
| Smearing | Fermi–Dirac, `degauss = 0.02` Ry, `extra_bands = 0` |
| Cutoff | **10 Ry** (minimal-twin budget) |
| k-mesh | **2×2×2** MP (unshifted) |
| SCF | tol 1e-8 (density L2), linear mix on spin channels |
| Response | `response_tol` 1e-5, `mix_beta` 0.7, CG tol 1e-9 |

## Offline pins (no live CP2K at regression time)

| Field | Value |
| ----- | ----- |
| Oracle 1 (physics anchor) | **CP2K 2026.1** (rev `git:5e54ba2`, source build `…/cp2k/install/bin/cp2k.psmp`, this host, 2026-08-01) — `VIBRATIONAL_ANALYSIS` (FD of analytic UKS k-point forces, `FULLY_PERIODIC T`, DX 0.01 bohr, `EPS_SCF 1e-8`) on the same cell: LSD + `MULTIPLICITY 1` + PBE + `&SMEAR FERMI_DIRAC` at 3157.75 K (≡ 0.02 Ry), ADDED_MOS 4, DZVP-MOLOPT-SR-GTH / GTH-PBE-q4, CUTOFF 300 Ry, 2×2×2 MP. Input: `tools/cp2k_oracle/inputs/dfpt-gga-spin-smearing/si2_uks_pbe_smear_vib.inp` |
| Oracle 1 output line | `VIB\|Frequency (cm^-1)   463.983524           534.803847           534.803847` → optical Γ modes **463.98, 534.80, 534.80 cm⁻¹** (mean 511.2; translations projected out by CP2K) |
| Pin 2 (sharp, same-Hamiltonian) | central **second difference of the collinear-spin PBE free energy** (Mermin \(F\), gapped fill ⇒ \(F=E\)): C[(0x),(0x)] = (F(+h) − 2F₀ + F(−h))/h², h = 0.03 Å, SCF tol 1e-9, identical ecut/k/XC/UPF/spin/FD |
| Pin 3 (cross-parent consistency) | \(M=0\) + gapped FD must reproduce the fixed-occ joint spin×GGA parent (`dfpt-gga-spin-si`) to spin-doubling / near-integer-fill roundoff |

The CP2K twin necessarily differs in PP/basis representation (GPW
GTH/MOLOPT FD-supercell vs NC-UPF plane-wave DFPT at a 10 Ry minimal
cutoff), so it anchors the **physics window**, not the digit. Window gate
`[450, 650]` cm⁻¹ covers both engines' spread; the free-energy FD pin is
the sharp correctness gate for the joint kernel.

### Measured pins (this host, 2026-08-01, Release, OMP=1)

| Quantity | Value |
| -------- | ----- |
| Free-energy FD C[(0x),(0x)] (h = 0.03 Å, spin-PBE + FD) | **17.3454 eV/Å²** (F₀ = −7.820593875040 Ha, F± = −7.820306828150 / −7.820307232893 Ha; \|TS\| ~ 1e-15 Ry ⇒ F=E) |
| DFPT C[(0x),(0x)] (regression settings) | **17.3711 eV/Å²** → gate `\|Δ\| ≤ 0.30` (~0.15 % agreement) |
| DFPT acoustic ω (no ASR) | 12.09, 12.09, 12.09 cm⁻¹ → gate `\|ω\| ≤ 30` |
| DFPT optical triplet | **579.82, 579.82, 579.96 cm⁻¹** → gate `[450, 650]`, spread ≤ 10 |
| Max row sum (translational invariance) | 0.0151 eV/Å² → gate ≤ 0.05 |
| Pre-symmetrization asymmetry | 9.6e-10 eV/Å² → gate ≤ 1e-5 |
| Regression wall | ≈ **48 s** (≤ 60 s budget, 1 thread) |

Cross-parent consistency (\(M=0\) gapped-FD limit): joint spin×GGA×smear
DFPT C00 17.3711 = fixed-occ spin×GGA parent 17.3711 (4-decimal identical);
optical triplet 579.82/579.96 = parent 579.82/579.96; free-energy FD Hessian
17.3454 = parent energy-FD 17.3454. The three-way product adds no spurious
response relative to the fixed-occ joint kernel while exercising the full
spin-resolved Sternheimer + `step_spin` FD + FD occupation weights path.

## Performance

| Item | Value |
| ---- | ----- |
| `perf_exempt` | **false** |
| ATV wall | **48.3 s** (2-atom Si, 6 responses × 2 spins, 8 k-points, GGA kernel FD + smear weights; 1 thread) |
| CP2K wall (same observable, comparable job: UKS k-point force evals + FD Hessian + SMEAR; committed input, serial 1 MPI × OMP=1) | **232.5 s** |
| `ratio_atv_over_cp2k` | **0.21** (48.3 / 232.5) |
| `within_10x` | **true** |
