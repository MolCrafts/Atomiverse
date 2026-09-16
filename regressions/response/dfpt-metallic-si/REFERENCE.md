# Reference — Γ-point DFPT phonons with metallic smearing (diamond Si, PW LSDA)

## Method

Density-functional perturbation theory at **q = 0** for `cpu::pw::PW` with
**Fermi–Dirac metallic occupation smearing**:

\[
f_{nk} = 2\,f_{\mathrm{FD}}\bigl((\varepsilon_{nk}-\mu)/\sigma\bigr),\quad
\Delta n = \sum_{nk} w_k\,f_{nk}\,2\,\mathrm{Re}\,\psi^*_{nk}\,\Delta\psi_{nk},
\]

ground-state SCF uses the Mermin free energy \(F = E_{\mathrm{KS}} - TS\).
Sternheimer projector removes bands with \(f \ge 10^{-8}\). At Γ a
Fermi-level shift \(\delta\mu\) restores \(\int\Delta n=0\) when
\(\partial N/\partial\mu\) is non-negligible (de Gironcoli).

Cite: Baroni et al., *Rev. Mod. Phys.* **73**, 515 (2001);
de Gironcoli, *Phys. Rev. B* **51**, 6773 (1995). Sibling fixed-occ:
`regressions/response/dfpt-phonon-si`. Sibling PW smear:
`regressions/pw/pw-smearing-al`.

Public path: `cpu::pw::DfptPhonon::compute(method, frame)` with
`method.smearing = Smearing::fermi_dirac`.

## System (regression twin)

| Setting | Value |
| ------- | ----- |
| Structure | diamond Si, 2-atom fcc primitive cell, a = **5.43 Å** |
| Pseudopotential | `tests/data/pw/Si_lda_nc.upf` (ONCVPSP NC, in-tree) |
| XC | `XCType::LSDA` |
| Smearing | Fermi–Dirac, `degauss = 0.02` Ry, `extra_bands = 0` |
| Cutoff | **10 Ry** (minimal-twin budget) |
| k-mesh | **2×2×2** MP (unshifted) |
| SCF | tol 1e-8 (density L2) |
| Response | `response_tol` 1e-5, `mix_beta` 0.7, CG tol 1e-9 |

Gapped Si at these settings keeps near-integer FD occupations, so free
energy \(F \approx E_{\mathrm{KS}}\) and results track the fixed-occ twin.
`extra_bands = 0` is intentional for this textbook twin (valence manifold
only); open metals use `extra_bands > 0` with denser k/ecut.

## Offline pins (no live CP2K at regression time)

| Field | Value |
| ----- | ----- |
| Preferred oracle | CP2K `VIBRATIONAL_ANALYSIS` + `&SMEAR` (GTH-GPW ≠ NC-UPF absolute E) |
| CP2K binary this host | **not available** (`module avail cp2k` empty) |
| Pin 1 (sharp, same-Hamiltonian) | central second difference of free energy F: C[(0x),(0x)] with h = 0.03 Å (identical ecut/k/XC/UPF/FD) |
| Pin 2 (physics) | optical Γ triplet in LDA window; acoustic ≈ 0 without ASR |

### Measured pins (this host, 2026-07-26, Release, OMP=8)

| Quantity | Value |
| -------- | ----- |
| Free-energy FD C[(0x),(0x)] (h = 0.03 Å) | **17.2393 eV/Å²** (same discrete Hessian as fixed-occ twin: integer FD fill ⇒ \(F=E\)) |
| DFPT C[(0x),(0x)] | **17.3416 eV/Å²** → gate `\|Δ\| ≤ 0.30` (~0.6 % agreement) |
| DFPT acoustic ω (no ASR) | −2.01, −2.00, −2.00 cm⁻¹ → gate `\|ω\| ≤ 30` |
| DFPT optical triplet | **579.45, 579.45, 579.60 cm⁻¹** → gate `[500, 650]`, spread ≤ 10 |
| Max row sum (translational invariance) | 0.0004 eV/Å² → gate ≤ 0.05 |
| Pre-symmetrization asymmetry | 3.4e-10 eV/Å² → gate ≤ 1e-5 |
| Regression wall | ≈ **16.5 s** (≤ 60 s budget) |

## CP2K twin (offline only)

See `tools/cp2k_oracle/inputs/dfpt-metallic-si.note`. When a CP2K binary is
available, record ω(Γ) [cm⁻¹] + wall with version and optional fair ratio
\(r = T_{\mathrm{atv}}/T_{\mathrm{cp2k}}\) (require ≤ 10).

## Performance

| Item | Value |
| ---- | ----- |
| `perf_exempt` | **true** |
| Reason | No CP2K binary/module on the build host for a fair DFPT+smear wall twin this shift |
| ATV wall (informative) | ≈ 16.5 s (2-atom Si, 6 responses, 8 k-points, OMP=8) |
| `ratio_atv_over_cp2k` | n/a (exempt) |
| `within_10x` | n/a (exempt; correctness gates only) |
