# Reference — Γ-point collinear-spin DFPT phonons (diamond Si, PW LSDA)

## Method

Density-functional perturbation theory at **q = 0** for the concrete
plane-wave method `cpu::pw::PW` with **collinear spin**
(`spin_polarized = true`, `multiplicity = 1` ⇒ \(M=0\)):

\[
\Delta V_{\mathrm{Hxc},\sigma}[\Delta n_\uparrow,\Delta n_\downarrow]
  = \Delta V_{\mathrm{H}}[\Delta n]
  + \sum_{\sigma'} K_{\mathrm{xc}}^{\sigma\sigma'}[\Delta n_{\sigma'}],
\]

with \(f=1\) per spin-orbital, Hartree from total \(\Delta n\), and a local
spin \(f_{\mathrm{xc}}^{\sigma\sigma'}\) table for pure LSDA (GGA uses FD of
`XCBuilder::step_spin`).

Cite: Baroni, de Gironcoli, Dal Corso, Giannozzi, *Rev. Mod. Phys.* **73**,
515 (2001). Sibling closed-shell LSDA product: `regressions/response/dfpt-phonon-si`.

Public path: `cpu::pw::DfptPhonon::compute(method, frame)` with
`method.spin_polarized = true`.

## System (regression twin)

| Setting | Value |
| ------- | ----- |
| Structure | diamond Si, 2-atom fcc primitive cell, a = **5.43 Å** (experimental) |
| Pseudopotential | `tests/data/pw/Si_lda_nc.upf` (ONCVPSP NC, in-tree) |
| XC | `XCType::LSDA` (collinear LSDA kernel) |
| Spin | collinear, multiplicity = 1 (\(N_\uparrow=N_\downarrow=4\)) |
| Cutoff | **10 Ry** (minimal-twin budget) |
| k-mesh | **2×2×2** MP (unshifted) |
| SCF | tol 1e-8 (density L2), linear mix on spin channels |
| Response | `response_tol` 1e-5, `mix_beta` 0.7, CG tol 1e-9 |

## Offline pins (no live CP2K/QE at regression time)

| Field | Value |
| ----- | ----- |
| Preferred oracle | CP2K DFPT / `VIBRATIONAL_ANALYSIS` on the same cell (LSD / UKS) |
| CP2K binary this host | **not available** (`module avail cp2k` empty) |
| Pin 1 (sharp, same-Hamiltonian) | central **second difference of the collinear-spin SCF total energy**: C[(0x),(0x)] = (E(+h) − 2E₀ + E(−h))/h², h = 0.03 Å, SCF tol 1e-9, identical ecut/k/XC/UPF/spin |
| Pin 2 (physics) | optical Γ triplet in a literature-style LDA window; acoustic ≈ 0 without ASR |

### Measured pins (this host, 2026-07-26, Release, OMP=8)

| Quantity | Value |
| -------- | ----- |
| Energy-FD C[(0x),(0x)] (h = 0.03 Å, spin SCF) | **17.3390 eV/Å²** (E₀ = −7.805522314842 Ha, E± = −7.805235374758 / −7.805235779501 Ha) |
| DFPT C[(0x),(0x)] (regression settings) | **17.3415 eV/Å²** → gate `\|Δ\| ≤ 0.30` (~0.015 % agreement) |
| DFPT acoustic ω (no ASR) | 1.33, 1.34, 1.34 cm⁻¹ → gate `\|ω\| ≤ 30` |
| DFPT optical triplet | **579.45, 579.45, 579.59 cm⁻¹** → gate `[500, 650]`, spread ≤ 10 |
| Max row sum (translational invariance) | 0.0002 eV/Å² → gate ≤ 0.05 |
| Pre-symmetrization asymmetry | 7.3e-10 eV/Å² → gate ≤ 1e-5 |
| Regression wall | ≈ **34 s** (≤ 60 s budget) |

Closed-shell collinear (\(M=0\)) matches the restricted LSDA optical window
to within the minimal-settings bias; the energy-FD pin is the sharp
correctness gate for the spin kernel.

## CP2K twin (offline only)

See `tools/cp2k_oracle/inputs/dfpt-spin-si.note`. When a CP2K binary is
available, record ω(Γ) [cm⁻¹] + wall with version and optional fair ratio
\(r = T_{\mathrm{atv}}/T_{\mathrm{cp2k}}\) (require ≤ 10).

## Performance

| Item | Value |
| ---- | ----- |
| `perf_exempt` | **true** |
| Reason | No CP2K binary/module on the build host for a fair DFPT wall twin this shift |
| ATV wall (informative) | ≈ 34 s (2-atom Si, 6 responses × 2 spins, 8 k-points) |
| `ratio_atv_over_cp2k` | n/a (exempt) |
| `within_10x` | n/a (exempt; correctness gates only) |
