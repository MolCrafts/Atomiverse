# Reference — Γ-point collinear-spin DFPT + metallic smearing (diamond Si, PW LSDA)

## Method

Density-functional perturbation theory at **q = 0** for `cpu::pw::PW` with
**collinear spin** and **Fermi–Dirac metallic occupation smearing**:

\[
f_{\sigma nk}\in[0,1],\quad
\Delta n_\sigma=\sum_{nk} w_k\,f_{\sigma nk}\,2\,\mathrm{Re}\,\psi^*_{\sigma nk}\,\Delta\psi_{\sigma nk},
\]

shared chemical potential \(\mu\) over spin-orbitals, spin XC kernel as in
`dfpt-spin-si`, Sternheimer projector on bands with \(f\ge 10^{-8}\), and
optional \(\delta\mu\) charge neutrality (de Gironcoli) when
\(\partial N/\partial\mu\) is non-negligible.

Public path: `cpu::pw::DfptPhonon::compute(method, frame)` with
`method.spin_polarized = true` and `method.smearing = Smearing::fermi_dirac`.

Cite: Baroni et al., *Rev. Mod. Phys.* **73**, 515 (2001);
de Gironcoli, *Phys. Rev. B* **51**, 6773 (1995). Siblings:
`dfpt-spin-si`, `dfpt-metallic-si`.

## System (regression twin)

| Setting | Value |
| ------- | ----- |
| Structure | diamond Si, 2-atom fcc primitive cell, a = **5.43 Å** |
| Pseudopotential | `tests/data/pw/Si_lda_nc.upf` (ONCVPSP NC, in-tree) |
| XC | `XCType::LSDA` (collinear LSDA kernel) |
| Spin | collinear, multiplicity = 1 (\(M=0\)) |
| Smearing | Fermi–Dirac, `degauss = 0.02` Ry, `extra_bands = 0` |
| Cutoff | **10 Ry** (minimal-twin budget) |
| k-mesh | **2×2×2** MP (unshifted) |
| SCF | tol 1e-8 (density L2) |
| Response | `response_tol` 1e-5, `mix_beta` 0.7, CG tol 1e-9 |

Gapped Si at these settings keeps near-integer FD occupations
(\(f_{\sigma nk}\in\{0,1\}\)), so free energy \(F\approx E_{\mathrm{KS}}\) and
results track the collinear fixed-occ twin.

## Offline pins (no live CP2K at regression time)

| Field | Value |
| ----- | ----- |
| Preferred oracle | CP2K `VIBRATIONAL_ANALYSIS` + LSD + `&SMEAR` (GTH-GPW ≠ NC-UPF absolute E) |
| CP2K binary this host | **not available** (`module avail cp2k` empty) |
| Pin 1 (sharp, same-Hamiltonian) | central second difference of collinear-spin free energy: C[(0x),(0x)] with h = 0.03 Å (identical ecut/k/XC/UPF/spin/FD) |
| Pin 2 (physics) | optical Γ triplet in LDA window; acoustic ≈ 0 without ASR |

### Measured pins (this host, 2026-07-26, Release, OMP=8)

| Quantity | Value |
| -------- | ----- |
| Free-energy / energy FD C[(0x),(0x)] (h = 0.03 Å, spin+FD) | **17.3390 eV/Å²** (gapped fill ⇒ \(F=E\); same discrete Hessian as `dfpt-spin-si`) |
| DFPT C[(0x),(0x)] | **17.3415 eV/Å²** → gate `\|Δ\| ≤ 0.30` (~0.015 % agreement) |
| DFPT acoustic ω (no ASR) | 1.33, 1.34, 1.34 cm⁻¹ → gate `\|ω\| ≤ 30` |
| DFPT optical triplet | **579.45, 579.45, 579.59 cm⁻¹** → gate `[500, 650]`, spread ≤ 10 |
| Max row sum (translational invariance) | 0.0002 eV/Å² → gate ≤ 0.05 |
| Pre-symmetrization asymmetry | 7.3e-10 eV/Å² → gate ≤ 1e-5 |
| Regression wall | ≈ **35 s** (≤ 60 s budget; OMP=8 Release re-verify 2026-07-26) |

## CP2K twin (offline only)

See `tools/cp2k_oracle/inputs/dfpt-spin-smearing.note`. When a CP2K binary is
available, record ω(Γ) [cm⁻¹] + wall with version and optional fair ratio
\(r = T_{\mathrm{atv}}/T_{\mathrm{cp2k}}\) (require ≤ 10).

## Performance

| Item | Value |
| ---- | ----- |
| `perf_exempt` | **true** |
| Reason | No CP2K binary/module on the build host for a fair DFPT spin+smear wall twin this shift |
| ATV wall (informative) | ≈ 30 s (2-atom Si, 6 responses × 2 spins, 8 k-points, OMP=8) |
| `ratio_atv_over_cp2k` | n/a (exempt) |
| `within_10x` | n/a (exempt; correctness gates only) |
