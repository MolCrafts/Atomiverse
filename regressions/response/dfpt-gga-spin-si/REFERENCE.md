# Reference — Γ DFPT phonons, joint collinear spin × pure GGA (diamond Si, PW)

## Method

Density-functional perturbation theory at **q = 0** for the concrete
plane-wave method `cpu::pw::PW` with the **joint** collinear-spin × pure-GGA
response kernel (`spin_polarized = true`, `multiplicity = 1` ⇒ \(M=0\),
`xc = XCType::PBE`):

\[
\Delta V_{\mathrm{Hxc},\sigma}[\Delta n_\uparrow,\Delta n_\downarrow]
  = \Delta V_{\mathrm{H}}[\Delta n]
  + \sum_{\sigma'} K_{\mathrm{xc}}^{\sigma\sigma'}[\Delta n_{\sigma'}],
\]

with \(f=1\) per spin-orbital, Hartree from total \(\Delta n\), and
\(K_{\mathrm{xc}}^{\sigma\sigma'}\) from central FD of the **spin-GGA** XC
operator (`XCBuilder::step_spin`, spectral \(\nabla\rho_\sigma\) + GGA flux)
— the joint product of the two shipped single-axis pins
`regressions/response/dfpt-gga-si` (restricted PBE kernel) and
`regressions/response/dfpt-spin-si` (collinear LSDA kernel).

Cite: Baroni, de Gironcoli, Dal Corso, Giannozzi, *Rev. Mod. Phys.* **73**,
515 (2001); Perdew, Burke, Ernzerhof, *Phys. Rev. Lett.* **77**, 3865 (1996).

Public path: `cpu::pw::DfptPhonon::compute(method, frame)` with
`method.spin_polarized = true` and `method.xc = XCType::PBE`.

## System (regression twin)

| Setting | Value |
| ------- | ----- |
| Structure | diamond Si, 2-atom fcc primitive cell, a = **5.43 Å** (experimental) |
| Pseudopotential | `tests/data/pw/Si_lda_nc.upf` (ONCVPSP NC, in-tree; same GGA-smoke PP as `pw-pbe-gga-si`) |
| XC | `XCType::PBE` (true spin-GGA kernel) |
| Spin | collinear, multiplicity = 1 (\(N_\uparrow=N_\downarrow=4\)) |
| Cutoff | **10 Ry** (minimal-twin budget) |
| k-mesh | **2×2×2** MP (unshifted) |
| SCF | tol 1e-8 (density L2), linear mix on spin channels |
| Response | `response_tol` 1e-5, `mix_beta` 0.7, CG tol 1e-9 |

## Offline pins (no live CP2K at regression time)

| Field | Value |
| ----- | ----- |
| Oracle 1 (physics anchor) | **CP2K 2026.1** (rev `85b8a9b49c`, `~/.local/bin/cp2k`, this host, 2026-07-31) — `VIBRATIONAL_ANALYSIS` (FD of analytic UKS k-point forces, `FULLY_PERIODIC T`, DX 0.01 bohr, `EPS_SCF 1e-8`) on the same cell: LSD + `MULTIPLICITY 1` + PBE, DZVP-MOLOPT-SR-GTH / GTH-PBE-q4, CUTOFF 300 Ry, 2×2×2 MP. Input: `tools/cp2k_oracle/inputs/dfpt-gga-spin-si/si2_uks_pbe_vib.inp` |
| Oracle 1 output line | `VIB\|Frequency (cm^-1)  4.6344295423E+02    5.5219216579E+02    5.5219217231E+02` → optical Γ modes **463.44, 552.19, 552.19 cm⁻¹** (mean 522.6; translations projected out by CP2K; the [111]-polarized member sits 89 cm⁻¹ below its FD-split partners — CP2K's own FD symmetry breaking, identical at `EPS_SCF` 1e-7 and 1e-8, so systematic to its displaced-supercell k-point FD, not SCF noise) |
| Pin 2 (sharp, same-Hamiltonian) | central **second difference of the collinear-spin PBE SCF total energy**: C[(0x),(0x)] = (E(+h) − 2E₀ + E(−h))/h², h = 0.03 Å, SCF tol 1e-9, identical ecut/k/XC/UPF/spin |
| Pin 3 (cross-parent consistency) | \(M=0\) joint kernel must reproduce the restricted-GGA parent (`dfpt-gga-si`) values to the spin-doubling roundoff |

The CP2K twin necessarily differs in PP/basis representation (GPW
GTH/MOLOPT FD-supercell vs NC-UPF plane-wave DFPT at a 10 Ry minimal
cutoff), so it anchors the **physics window**, not the digit: CP2K's
spin×PBE optical Γ modes span 463–552 cm⁻¹ (mean 522.6, near the ~517 cm⁻¹
experimental Raman line); the minimal-twin bias of the in-tree LDA-generated
NC UPF at 10 Ry shifts ATV to ~580 cm⁻¹, the same offset both parent pins
show. Window gate `[450, 650]` cm⁻¹ covers both engines' spread;
the energy-FD pin is the sharp correctness gate for the joint kernel.

### Measured pins (this host, 2026-07-31, Release, OMP=1)

| Quantity | Value |
| -------- | ----- |
| Energy-FD C[(0x),(0x)] (h = 0.03 Å, spin-PBE SCF) | **17.3454 eV/Å²** (E₀ = −7.820593875040 Ha, E± = −7.820306828150 / −7.820307232893 Ha) |
| DFPT C[(0x),(0x)] (regression settings) | **17.3711 eV/Å²** → gate `\|Δ\| ≤ 0.30` (~0.15 % agreement) |
| DFPT acoustic ω (no ASR) | 12.09, 12.09, 12.09 cm⁻¹ → gate `\|ω\| ≤ 30` |
| DFPT optical triplet | **579.82, 579.82, 579.96 cm⁻¹** → gate `[450, 650]`, spread ≤ 10 |
| Max row sum (translational invariance) | 0.0151 eV/Å² → gate ≤ 0.05 |
| Pre-symmetrization asymmetry | 9.6e-10 eV/Å² → gate ≤ 1e-5 |
| Regression wall | ≈ **45 s** (≤ 60 s budget, 1 thread) |

Cross-parent consistency (\(M=0\) limit): joint spin×GGA DFPT C00
17.3711 = restricted-GGA parent 17.3711 (4-decimal identical); optical
triplet 579.82/579.96 = parent 579.82/579.96; spin-FD energy Hessian
17.3454 vs restricted-FD 17.3456 (2e-4 apart). The joint kernel adds no
spurious spin response at \(M=0\) while exercising the full spin-resolved
Sternheimer + `step_spin` FD path.

## Performance

| Item | Value |
| ---- | ----- |
| `perf_exempt` | **false** |
| ATV wall | **44.7 s** (2-atom Si, 6 responses × 2 spins, 8 k-points, GGA kernel FD; 1 thread) |
| CP2K wall (same observable, comparable job: 13 UKS k-point force evals, FD Hessian; committed input, serial 1 thread) | **221.8 s** (`EPS_SCF 1e-8`; 143.7 s at 1e-7) |
| `ratio_atv_over_cp2k` | **0.20** (44.7 / 221.8; 0.31 vs the 1e-7 run) |
| `within_10x` | **true** |
