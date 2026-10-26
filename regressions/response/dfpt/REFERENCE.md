# response/dfpt

One case; every section below is run by `atv_regressions --case response/dfpt` and keeps its own oracle record.

CPU PW GGA uses private 3/2-padded XC quadrature and a weighted-adjoint
potential projection. Restricted and joint two-spin densities share one Pulay
mixing implementation; convergence uses the true unmixed spin residuals.
For the Si Γ PBE fixtures, the row-sum residual is 0.0201 eV/Å² against the
unchanged 0.05 bound, without acoustic sum-rule projection. The complete
commensurate output is the Fourier sum of the assembled supercell IFCs;
its upstream IFC symmetrization still reports preprojection asymmetry.

The seven Γ sections below retain historical Atomiverse energy/free-energy
finite-difference constants as **internal consistency supplements only**.
They were produced by native SCF, are not external oracle pins, and do not
independently qualify the Hessian. Their existing comparison bounds remain
unchanged. Oracle content is limited to each section's identified external
spectral reference and analytic translational/symmetry identities; the frozen
FD values are historical diagnostics rather than newly validated outputs.

## Reference — Γ-point DFPT phonons (diamond Si, PW LSDA, NC-UPF)

### Method

Density-functional perturbation theory at **q = 0** for the concrete
plane-wave method `cpu::pw::PW`: analytic Cartesian force constants

\[
C_{I\alpha,J\beta} = C^{\mathrm{Ewald}}
  + \sum_{k,i} w_k f \,\langle\psi_i|\partial^2 V_{\mathrm{ext}}|\psi_i\rangle\,\delta_{IJ}
  + \sum_{k,i} w_k f \cdot 2\,\mathrm{Re}
      \langle\Delta\psi_i^{(J\beta)}|\partial V_{\mathrm{ext}}^{(I\alpha)}|\psi_i\rangle ,
\]

with the self-consistent Sternheimer response solved by conduction-space
projected preconditioned CG, Hartree + LDA-kernel screening
(`ΔV_H = 4πΔρ/G²`, `f_xc = ∂V_xc/∂n` from the same `evaluate_xc(LSDA)` the
SCF uses), and analytic Ewald second derivatives. No acoustic-sum-rule
imposition anywhere. Public path: `cpu::pw::DfptPhonon::compute(method, frame)`.

Cite: Baroni, de Gironcoli, Dal Corso, Giannozzi, *Rev. Mod. Phys.* **73**,
515 (2001); Giannozzi, de Gironcoli, Pavone, Baroni, *Phys. Rev. B* **43**,
7231 (1991) — DFPT-LDA Si Γ optical **517 cm⁻¹** (exp. 517.3 cm⁻¹, Raman).

### System (regression twin)

| Setting | Value |
| ------- | ----- |
| Structure | diamond Si, 2-atom fcc primitive cell, a = **5.431 Å** (experimental) |
| Pseudopotential | `tests/data/pw/Si_lda_nc.upf` (ONCVPSP LDA NC, in-tree) |
| XC | `XCType::LSDA` (Slater + VWN5), the DFPT-supported kernel |
| Cutoff | **10 Ry** (minimal-twin budget; see below) |
| k-mesh | **2×2×2** MP (unshifted, 8 points) |
| SCF | tol 1e-8 (density L2), Pulay mixing |
| Response | `response_tol` 1e-5, `mix_beta` 0.7, CG tol 1e-9 |

### Offline pins (no live CP2K/QE at regression time)

| Field | Value |
| ----- | ----- |
| Preferred oracle | CP2K DFPT / `VIBRATIONAL_ANALYSIS` on the same cell |
| CP2K binary this host | **not available** (no CP2K module on the pin host this shift — same as every prior wave) |
| Supplement (native frozen FD, not an oracle) | central **second difference of the SCF total energy**: C[(0x),(0x)] = (E(+h) − 2E₀ + E(−h))/h², h = 0.03 Å, SCF tol 1e-9, identical ecut/k/XC |
| Pin 2 (physics) | literature DFPT-LDA / Raman Si TO(Γ) window |

#### Measured pins (this host, 2026-07-25, Release, OMP=8)

| Quantity | Value |
| -------- | ----- |
| Energy-FD C[(0x),(0x)] (h = 0.03 Å) | **17.2393 eV/Å²** (E₀ = −15.6073409534 Ry, E± = −15.6067583112 / −15.6067832378 Ry) |
| Energy-FD C[(0x),(0x)] (h = 0.02 Å, step-size check) | 17.2512 eV/Å² (Δ vs h = 0.03: 0.012) |
| Energy-FD C[(0x),(0x)] (ecut 15 Ry cross-check, h = 0.02 Å) | 17.2289 eV/Å² vs DFPT 17.2204 (0.05%) |
| DFPT C[(0x),(0x)] (regression settings) | **17.2277 eV/Å²** → gate `|Δ| ≤ 0.30` |
| DFPT acoustic ω (no ASR) | −4.40, −3.26, −2.16 cm⁻¹ → gate `|ω| ≤ 30` |
| DFPT optical triplet | **575.82, 578.49, 578.49 cm⁻¹** → gate `[500, 650]`, spread ≤ 10 |
| Max row sum (translational invariance) | 0.0022 eV/Å² → gate ≤ 0.02 |
| Pre-symmetrization asymmetry | 1.8e-9 eV/Å² → gate ≤ 1e-6 |
| Regression wall | ≈ **24 s** (≤ 60 s budget) |

The +12 % optical offset vs the 517 cm⁻¹ anchor is the expected
minimal-settings bias (10 Ry NC cutoff, 2×2×2 MP, experimental lattice
constant); the window [500, 650] rejects unit disasters and factor errors
The frozen native energy-FD comparison is an internal consistency supplement,
not an independent oracle.

#### Why the FD twin is energy-based, not force-based

`pw_forces_analytic` deviates from the exact gradient of the discretized
total energy by ≈ 2 % for this system (measured: F_x analytic −0.8442 vs
FD-of-energy −0.8601 eV/Å at a 0.05 Å displacement) — an egg-box-scale
inconsistency of the v1 analytic-force implementation (aliasing of the
density grid). FD of those forces therefore does **not** reproduce the true
discrete Hessian (row sums off by up to 0.6 eV/Å²), while DFPT matches the
energy second difference to 0.07 %. Central *energy* differences are the
honest oracle.

### CP2K twin (offline only)

When a CP2K binary is available: run Quickstep DFPT (or
`VIBRATIONAL_ANALYSIS`) for diamond Si at the same lattice constant and a
comparable NC/GTH setup, record ω(Γ) [cm⁻¹] + wall here with the CP2K
version, and optionally the fair wall ratio r = T_atv/T_cp2k (require ≤ 10).

### Performance

| Item | Value |
| ---- | ----- |
| `perf_exempt` | **true** |
| Reason | No CP2K binary/module on the build host for a fair DFPT wall twin this shift |
| ATV wall (informative) | ≈ 24 s (2-atom Si, 6 responses, 8 k-points) |
| `ratio_atv_over_cp2k` | n/a (exempt) |
| `within_10x` | n/a (exempt; correctness gates only) |


## Reference — Γ-point DFPT phonons (diamond Si, PW PBE GGA kernel)

### Method

Density-functional perturbation theory at **q = 0** for the concrete
plane-wave method `cpu::pw::PW` with a **true GGA XC kernel** (PBE):

\[
\Delta V_{\mathrm{Hxc}}[\Delta n]
  = \Delta V_{\mathrm{H}}[\Delta n]
  + K_{\mathrm{xc}}[\Delta n],
\]

where \(K_{\mathrm{xc}}\) is the second functional derivative of the same
`XCBuilder` operator used in the ground-state SCF (central finite difference
of \(V_{\mathrm{xc}}[n]\) with spectral \(\nabla\rho\) and GGA flux). Pure
LDA keeps the local \(f_{\mathrm{xc}}\) table. Hybrids remain unsupported.

Cite: Baroni, de Gironcoli, Dal Corso, Giannozzi, *Rev. Mod. Phys.* **73**,
515 (2001); Perdew, Burke, Ernzerhof, *Phys. Rev. Lett.* **77**, 3865 (1996).
Sibling LSDA product: `regressions/response/dfpt`.

Public path: `cpu::pw::DfptPhonon::compute(method, frame)` with
`method` constructed as `PW{pp, ecut_ry, k_mesh, XCType::PBE}`.

### System (regression twin)

| Setting | Value |
| ------- | ----- |
| Structure | diamond Si, 2-atom fcc primitive cell, a = **5.43 Å** (experimental) |
| Pseudopotential | `tests/data/pw/Si_lda_nc.upf` (ONCVPSP NC, in-tree; same GGA-smoke PP as `pw-pbe-gga-si`) |
| XC | `XCType::PBE` (true GGA kernel) |
| Cutoff | **10 Ry** (minimal-twin budget) |
| k-mesh | **2×2×2** MP (unshifted) |
| SCF | tol 1e-8 (density L2), Pulay mixing |
| Response | `response_tol` 1e-5, `mix_beta` 0.7, CG tol 1e-9 |

### Offline pins (no live CP2K/QE at regression time)

| Field | Value |
| ----- | ----- |
| Preferred oracle | CP2K DFPT / `VIBRATIONAL_ANALYSIS` on the same cell (PBE) |
| CP2K binary this host | **not available** (`module avail cp2k` empty) |
| Supplement (native frozen FD, not an oracle) | central **second difference of the SCF total energy**: C[(0x),(0x)] = (E(+h) − 2E₀ + E(−h))/h², h = 0.03 Å, SCF tol 1e-9, identical ecut/k/XC |
| Pin 2 (physics) | optical Γ triplet in a wide literature-style window; acoustic ≈ 0 without ASR |

#### Measured pins (this host, 2026-07-26, Release, OMP=8)

| Quantity | Value |
| -------- | ----- |
| Energy-FD C[(0x),(0x)] (h = 0.03 Å) | **17.3456 eV/Å²** (E₀ = −15.6411877416 Ry, E± = −15.6406136403 / −15.6406144498 Ry) |
| DFPT C[(0x),(0x)] (regression settings) | **17.3711 eV/Å²** → gate `\|Δ\| ≤ 0.30` (~0.15 % agreement) |
| DFPT acoustic ω (no ASR) | 12.08, 12.08, 12.09 cm⁻¹ → gate `\|ω\| ≤ 30` |
| DFPT optical triplet | **579.82, 579.82, 579.96 cm⁻¹** → gate `[450, 650]`, spread ≤ 10 |
| Max row sum (translational invariance) | 0.0151 eV/Å² → gate ≤ 0.05 |
| Pre-symmetrization asymmetry | 9.6e-10 eV/Å² → gate ≤ 1e-5 |
| Regression wall | ≈ **15 s** (≤ 60 s budget) |

The optical offset vs the experimental ~517 cm⁻¹ Raman line is the expected
minimal-settings bias (10 Ry NC cutoff, 2×2×2 MP, experimental lattice,
LDA-generated NC UPF used as fixed valence PP). The frozen native energy-FD comparison is an internal consistency
supplement, not an independent oracle.

### CP2K twin (offline only)

See `tools/cp2k_oracle/inputs/dfpt-gga-si.note`. When a CP2K binary is
available, record ω(Γ) [cm⁻¹] + wall with version and optional fair ratio
\(r = T_{\mathrm{atv}}/T_{\mathrm{cp2k}}\) (require ≤ 10).

### Performance

| Item | Value |
| ---- | ----- |
| `perf_exempt` | **true** |
| Reason | No CP2K binary/module on the build host for a fair DFPT wall twin this shift |
| ATV wall (informative) | ≈ 15 s (2-atom Si, 6 responses, 8 k-points, GGA kernel FD) |
| `ratio_atv_over_cp2k` | n/a (exempt) |
| `within_10x` | n/a (exempt; correctness gates only) |


## Reference — Γ-point collinear-spin DFPT phonons (diamond Si, PW LSDA)

### Method

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
515 (2001). Sibling closed-shell LSDA product: `regressions/response/dfpt`.

Public path: `cpu::pw::DfptPhonon::compute(method, frame)` with
`method.spin_polarized = true`.

### System (regression twin)

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

### Offline pins (no live CP2K/QE at regression time)

| Field | Value |
| ----- | ----- |
| Preferred oracle | CP2K DFPT / `VIBRATIONAL_ANALYSIS` on the same cell (LSD / UKS) |
| CP2K binary this host | **not available** (`module avail cp2k` empty) |
| Supplement (native frozen FD, not an oracle) | central **second difference of the collinear-spin SCF total energy**: C[(0x),(0x)] = (E(+h) − 2E₀ + E(−h))/h², h = 0.03 Å, SCF tol 1e-9, identical ecut/k/XC/UPF/spin |
| Pin 2 (physics) | optical Γ triplet in a literature-style LDA window; acoustic ≈ 0 without ASR |

#### Measured pins (this host, 2026-07-26, Release, OMP=8)

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
to within the minimal-settings bias; the frozen native energy-FD comparison is an internal consistency
supplement, not an independent oracle.

### CP2K twin (offline only)

See `tools/cp2k_oracle/inputs/dfpt-spin-si.note`. When a CP2K binary is
available, record ω(Γ) [cm⁻¹] + wall with version and optional fair ratio
\(r = T_{\mathrm{atv}}/T_{\mathrm{cp2k}}\) (require ≤ 10).

### Performance

| Item | Value |
| ---- | ----- |
| `perf_exempt` | **true** |
| Reason | No CP2K binary/module on the build host for a fair DFPT wall twin this shift |
| ATV wall (informative) | ≈ 34 s (2-atom Si, 6 responses × 2 spins, 8 k-points) |
| `ratio_atv_over_cp2k` | n/a (exempt) |
| `within_10x` | n/a (exempt; correctness gates only) |


## Reference — Γ DFPT phonons, joint collinear spin × pure GGA (diamond Si, PW)

### Method

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
`regressions/response/dfpt` (restricted PBE kernel) and
`regressions/response/dfpt` (collinear LSDA kernel).

Cite: Baroni, de Gironcoli, Dal Corso, Giannozzi, *Rev. Mod. Phys.* **73**,
515 (2001); Perdew, Burke, Ernzerhof, *Phys. Rev. Lett.* **77**, 3865 (1996).

Public path: `cpu::pw::DfptPhonon::compute(method, frame)` with
`method` constructed as `PW{pp, ecut_ry, k_mesh, XCType::PBE}` and
`method.spin_polarized = true`.

### System (regression twin)

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

### Offline pins (no live CP2K at regression time)

| Field | Value |
| ----- | ----- |
| Oracle 1 (physics anchor) | **CP2K 2026.1** (rev `85b8a9b49c`, `~/.local/bin/cp2k`, this host, 2026-07-31) — `VIBRATIONAL_ANALYSIS` (FD of analytic UKS k-point forces, `FULLY_PERIODIC T`, DX 0.01 bohr, `EPS_SCF 1e-8`) on the same cell: LSD + `MULTIPLICITY 1` + PBE, DZVP-MOLOPT-SR-GTH / GTH-PBE-q4, CUTOFF 300 Ry, 2×2×2 MP. Input: `tools/cp2k_oracle/inputs/dfpt-gga-spin-si/si2_uks_pbe_vib.inp` |
| Oracle 1 output line | `VIB\|Frequency (cm^-1)  4.6344295423E+02    5.5219216579E+02    5.5219217231E+02` → optical Γ modes **463.44, 552.19, 552.19 cm⁻¹** (mean 522.6; translations projected out by CP2K; the [111]-polarized member sits 89 cm⁻¹ below its FD-split partners — CP2K's own FD symmetry breaking, identical at `EPS_SCF` 1e-7 and 1e-8, so systematic to its displaced-supercell k-point FD, not SCF noise) |
| Supplement (native frozen FD, not an oracle) | central **second difference of the collinear-spin PBE SCF total energy**: C[(0x),(0x)] = (E(+h) − 2E₀ + E(−h))/h², h = 0.03 Å, SCF tol 1e-9, identical ecut/k/XC/UPF/spin |
| Pin 3 (cross-parent consistency) | \(M=0\) joint kernel must reproduce the restricted-GGA parent (`dfpt-gga-si`) values to the spin-doubling roundoff |

The CP2K twin necessarily differs in PP/basis representation (GPW
GTH/MOLOPT FD-supercell vs NC-UPF plane-wave DFPT at a 10 Ry minimal
cutoff), so it anchors the **physics window**, not the digit: CP2K's
spin×PBE optical Γ modes span 463–552 cm⁻¹ (mean 522.6, near the ~517 cm⁻¹
experimental Raman line); the minimal-twin bias of the in-tree LDA-generated
NC UPF at 10 Ry shifts ATV to ~580 cm⁻¹, the same offset both parent pins
show. Window gate `[450, 650]` cm⁻¹ covers both engines' spread;
the frozen native energy-FD comparison is an internal consistency
supplement, not an independent oracle.

#### Measured pins (this host, 2026-07-31, Release, OMP=1)

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

### Performance

| Item | Value |
| ---- | ----- |
| `perf_exempt` | **false** |
| ATV wall | **44.7 s** (2-atom Si, 6 responses × 2 spins, 8 k-points, GGA kernel FD; 1 thread) |
| CP2K wall (same observable, comparable job: 13 UKS k-point force evals, FD Hessian; committed input, serial 1 thread) | **221.8 s** (`EPS_SCF 1e-8`; 143.7 s at 1e-7) |
| `ratio_atv_over_cp2k` | **0.20** (44.7 / 221.8; 0.31 vs the 1e-7 run) |
| `within_10x` | **true** |


## Reference — Γ-point DFPT phonons with metallic smearing (diamond Si, PW LSDA)

### Method

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
`regressions/response/dfpt`. Sibling PW smear:
`regressions/pw/bands-dos`.

Public path: `cpu::pw::DfptPhonon::compute(method, frame)` with
`method.smearing = Smearing::fermi_dirac`.

### System (regression twin)

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

### Offline pins (no live CP2K at regression time)

| Field | Value |
| ----- | ----- |
| Preferred oracle | CP2K `VIBRATIONAL_ANALYSIS` + `&SMEAR` (GTH-GPW ≠ NC-UPF absolute E) |
| CP2K binary this host | **not available** (`module avail cp2k` empty) |
| Supplement (native frozen FD, not an oracle) | central second difference of free energy F: C[(0x),(0x)] with h = 0.03 Å (identical ecut/k/XC/UPF/FD) |
| Pin 2 (physics) | optical Γ triplet in LDA window; acoustic ≈ 0 without ASR |

#### Measured pins (this host, 2026-07-26, Release, OMP=8)

| Quantity | Value |
| -------- | ----- |
| Free-energy FD C[(0x),(0x)] (h = 0.03 Å) | **17.2393 eV/Å²** (same discrete Hessian as fixed-occ twin: integer FD fill ⇒ \(F=E\)) |
| DFPT C[(0x),(0x)] | **17.3416 eV/Å²** → gate `\|Δ\| ≤ 0.30` (~0.6 % agreement) |
| DFPT acoustic ω (no ASR) | −2.01, −2.00, −2.00 cm⁻¹ → gate `\|ω\| ≤ 30` |
| DFPT optical triplet | **579.45, 579.45, 579.60 cm⁻¹** → gate `[500, 650]`, spread ≤ 10 |
| Max row sum (translational invariance) | 0.0004 eV/Å² → gate ≤ 0.05 |
| Pre-symmetrization asymmetry | 3.4e-10 eV/Å² → gate ≤ 1e-5 |
| Regression wall | ≈ **16.5 s** (≤ 60 s budget) |

### CP2K twin (offline only)

See `tools/cp2k_oracle/inputs/dfpt-metallic-si.note`. When a CP2K binary is
available, record ω(Γ) [cm⁻¹] + wall with version and optional fair ratio
\(r = T_{\mathrm{atv}}/T_{\mathrm{cp2k}}\) (require ≤ 10).

### Performance

| Item | Value |
| ---- | ----- |
| `perf_exempt` | **true** |
| Reason | No CP2K binary/module on the build host for a fair DFPT+smear wall twin this shift |
| ATV wall (informative) | ≈ 16.5 s (2-atom Si, 6 responses, 8 k-points, OMP=8) |
| `ratio_atv_over_cp2k` | n/a (exempt) |
| `within_10x` | n/a (exempt; correctness gates only) |


## Reference — Γ-point collinear-spin DFPT + metallic smearing (diamond Si, PW LSDA)

### Method

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

### System (regression twin)

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

### Offline pins (no live CP2K at regression time)

| Field | Value |
| ----- | ----- |
| Preferred oracle | CP2K `VIBRATIONAL_ANALYSIS` + LSD + `&SMEAR` (GTH-GPW ≠ NC-UPF absolute E) |
| CP2K binary this host | **not available** (`module avail cp2k` empty) |
| Supplement (native frozen FD, not an oracle) | central second difference of collinear-spin free energy: C[(0x),(0x)] with h = 0.03 Å (identical ecut/k/XC/UPF/spin/FD) |
| Pin 2 (physics) | optical Γ triplet in LDA window; acoustic ≈ 0 without ASR |

#### Measured pins (this host, 2026-07-26, Release, OMP=8)

| Quantity | Value |
| -------- | ----- |
| Free-energy / energy FD C[(0x),(0x)] (h = 0.03 Å, spin+FD) | **17.3390 eV/Å²** (gapped fill ⇒ \(F=E\); same discrete Hessian as `dfpt-spin-si`) |
| DFPT C[(0x),(0x)] | **17.3415 eV/Å²** → gate `\|Δ\| ≤ 0.30` (~0.015 % agreement) |
| DFPT acoustic ω (no ASR) | 1.33, 1.34, 1.34 cm⁻¹ → gate `\|ω\| ≤ 30` |
| DFPT optical triplet | **579.45, 579.45, 579.59 cm⁻¹** → gate `[500, 650]`, spread ≤ 10 |
| Max row sum (translational invariance) | 0.0002 eV/Å² → gate ≤ 0.05 |
| Pre-symmetrization asymmetry | 7.3e-10 eV/Å² → gate ≤ 1e-5 |
| Regression wall | ≈ **35 s** (≤ 60 s budget; OMP=8 Release re-verify 2026-07-26) |

### CP2K twin (offline only)

See `tools/cp2k_oracle/inputs/dfpt-spin-smearing.note`. When a CP2K binary is
available, record ω(Γ) [cm⁻¹] + wall with version and optional fair ratio
\(r = T_{\mathrm{atv}}/T_{\mathrm{cp2k}}\) (require ≤ 10).

### Performance

| Item | Value |
| ---- | ----- |
| `perf_exempt` | **true** |
| Reason | No CP2K binary/module on the build host for a fair DFPT spin+smear wall twin this shift |
| ATV wall (informative) | ≈ 30 s (2-atom Si, 6 responses × 2 spins, 8 k-points, OMP=8) |
| `ratio_atv_over_cp2k` | n/a (exempt) |
| `within_10x` | n/a (exempt; correctness gates only) |


## Reference — Γ DFPT phonons, joint collinear spin × pure GGA × metallic smear (diamond Si, PW)

### Method

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
`method` constructed as `PW{pp, ecut_ry, k_mesh, XCType::PBE}`,
`method.spin_polarized = true`, and `method.smearing = Smearing::fermi_dirac`.

Siblings: `dfpt-gga-spin-si` (spin×GGA, fixed occ), `dfpt-spin-smearing`
(spin×smear, LSDA), `dfpt-gga-si` (restricted GGA).

### System (regression twin)

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

### Offline pins (no live CP2K at regression time)

| Field | Value |
| ----- | ----- |
| Oracle 1 (physics anchor) | **CP2K 2026.1** (rev `git:5e54ba2`, source build `…/cp2k/install/bin/cp2k.psmp`, this host, 2026-08-01) — `VIBRATIONAL_ANALYSIS` (FD of analytic UKS k-point forces, `FULLY_PERIODIC T`, DX 0.01 bohr, `EPS_SCF 1e-8`) on the same cell: LSD + `MULTIPLICITY 1` + PBE + `&SMEAR FERMI_DIRAC` at 3157.75 K (≡ 0.02 Ry), ADDED_MOS 4, DZVP-MOLOPT-SR-GTH / GTH-PBE-q4, CUTOFF 300 Ry, 2×2×2 MP. Input: `tools/cp2k_oracle/inputs/dfpt-gga-spin-smearing/si2_uks_pbe_smear_vib.inp` |
| Oracle 1 output line | `VIB\|Frequency (cm^-1)   463.983524           534.803847           534.803847` → optical Γ modes **463.98, 534.80, 534.80 cm⁻¹** (mean 511.2; translations projected out by CP2K) |
| Supplement (native frozen FD, not an oracle) | central **second difference of the collinear-spin PBE free energy** (Mermin \(F\), gapped fill ⇒ \(F=E\)): C[(0x),(0x)] = (F(+h) − 2F₀ + F(−h))/h², h = 0.03 Å, SCF tol 1e-9, identical ecut/k/XC/UPF/spin/FD |
| Pin 3 (cross-parent consistency) | \(M=0\) + gapped FD must reproduce the fixed-occ joint spin×GGA parent (`dfpt-gga-spin-si`) to spin-doubling / near-integer-fill roundoff |

The CP2K twin necessarily differs in PP/basis representation (GPW
GTH/MOLOPT FD-supercell vs NC-UPF plane-wave DFPT at a 10 Ry minimal
cutoff), so it anchors the **physics window**, not the digit. Window gate
`[450, 650]` cm⁻¹ covers both engines' spread; the frozen native free-energy FD comparison is an internal consistency
supplement, not an independent oracle.

#### Measured pins (this host, 2026-08-01, Release, OMP=1)

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

### Performance

| Item | Value |
| ---- | ----- |
| `perf_exempt` | **false** |
| ATV wall | **48.3 s** (2-atom Si, 6 responses × 2 spins, 8 k-points, GGA kernel FD + smear weights; 1 thread) |
| CP2K wall (same observable, comparable job: UKS k-point force evals + FD Hessian + SMEAR; committed input, serial 1 MPI × OMP=1) | **232.5 s** |
| `ratio_atv_over_cp2k` | **0.21** (48.3 / 232.5) |
| `within_10x` | **true** |


## Reference — incommensurate-q DFPT phonons (diamond Si, PW LSDA, NC-UPF)

### Method

Density-functional perturbation theory at a **non-rational** fractional phonon
wavevector \(\mathbf{q}\) for `cpu::pw::PW`. Public path selection in
`cpu::pw::DfptPhonon::compute`:

| `q_frac` | Path |
| -------- | ---- |
| \(\approx\mathbf{0}\) | Γ Sternheimer (`dfpt_gamma_force_constants`) |
| rational, \(n_a n_b n_c\le 8\), \(n_i\le 6\) | supercell Γ-DFPT + IFC Fourier |
| otherwise | **direct** `internal::dfpt_q_force_constants` + Hermitian mass-weighted diagonalization |

This regression pins the **direct Sternheimer** branch at
\(\mathbf{q}=(0.3,0,0)\) (not commensurate under the \(n\le 6\) gate).

Also fixed in this slug: **finite-q Ewald** dynamical matrix rewritten to
mirror the Γ pair accumulation (phases + ASR); finite-q response density
drops the extra TR factor of 2 (conjugate lives at \(-\mathbf{q}\)).

Cite: Baroni et al., *Rev. Mod. Phys.* **73**, 515 (2001); Giannozzi et al.,
*Phys. Rev. B* **43**, 7231 (1991).

### System

| Setting | Value |
| ------- | ----- |
| Structure | diamond Si primitive, a = **5.43 Å** |
| \(\mathbf{q}\) | **(0.3, 0, 0)** fractional (incommensurate under n≤6) |
| Pseudopotential | `tests/data/pw/Si_lda_nc.upf` |
| XC | LSDA |
| Cutoff | **10 Ry** |
| k-mesh | **1×1×1** |
| Response | tol 1e-4, mix 0.2, max_iter 120 |

### Offline pins (measured 2026-07-25, Release, OMP=8)

| Quantity | Value |
| -------- | ----- |
| SCF energy | **−14.544 Ry** |
| ATV ω(q) [cm⁻¹] | **692.9, 726.9, 726.9, 944.1, 1023.5, 1023.5** |
| \|ω\| sorted | 693, 727, 727, 944, 1023, 1023 |
| n_imag | 0 |
| max asymmetry | ≈ **29.6 eV/Å²** (gate ≤ 50) |
| Wall | ≈ **7 s** (≤ 60 s) |
| CP2K binary | not on pin host |

Gates: \|ω\| low-3 ∈ [400, 900], high-3 ∈ [700, 1300] cm⁻¹; response must
converge; n_imag = 0. Minimal twin (10 Ry, Γ-only k) biases the spectrum
high vs literature mid-ΓX — windows reject unit disasters, not claim
quantitative LDA dispersion.

### Performance

| Item | Value |
| ---- | ----- |
| `perf_exempt` | **true** |
| Reason | No CP2K DFPT wall twin on the build host this shift |
| ATV wall | ≈ 7 s |
| `ratio_atv_over_oracle` | n/a |
| `within_10x` | n/a (exempt) |

### CP2K twin (offline only)

When a CP2K binary is available: phonon / DFPT at fractional q ≈ (0.3,0,0)
on the same cell (or closest commensurate mesh), record ω [cm⁻¹] + wall in
`tools/cp2k_oracle/inputs/dfpt-incommensurate-si.note`.


## Reference — collinear-spin DFPT at incommensurate q (diamond Si, PW LSDA)

### Method

Density-functional perturbation theory at a **non-rational** fractional phonon
wavevector \(\mathbf{q}\) for `cpu::pw::PW` with **collinear spin**
(`spin_polarized = true`, `multiplicity = 1` ⇒ \(M=0\)):

- Direct Sternheimer on the primitive cell
  (`internal::dfpt_q_force_constants` → `dfpt_q_spin_impl`)
- Occupation \(f=1\) per spin-orbital
- Hartree from total \(\Delta n\); spin \(K_{\mathrm{xc}}\) (local LSDA table
  or FD of `XCBuilder::step_spin` for pure GGA)

Cite: Baroni et al., *Rev. Mod. Phys.* **73**, 515 (2001). Siblings:
`dfpt-spin-si` (Γ collinear), `dfpt-incommensurate-si` (restricted finite q).

### System

| Setting | Value |
| ------- | ----- |
| Structure | diamond Si primitive, a = **5.43 Å** |
| \(\mathbf{q}\) | **(0.3, 0, 0)** fractional (incommensurate under n≤6) |
| Pseudopotential | `tests/data/pw/Si_lda_nc.upf` |
| XC | LSDA |
| Spin | collinear, multiplicity = 1 (\(N_\uparrow=N_\downarrow=4\)) |
| Cutoff | **10 Ry** |
| k-mesh | **1×1×1** |
| SCF | tol 1e-6, mix 0.4 |
| Response | tol 1e-4, mix 0.2, max_iter 120 |

### Offline pins (measured 2026-07-26, Release, OMP=8)

| Quantity | Value |
| -------- | ----- |
| ATV ω(q) [cm⁻¹] | **692.90, 726.85, 726.85, 944.07, 1023.46, 1023.46** |
| \|ω\| sorted | 693, 727, 727, 944, 1023, 1023 |
| n_imag | 0 |
| max asymmetry | ≈ **29.6 eV/Å²** (gate ≤ 50) |
| Wall | ≈ **11 s** (≤ 60 s) |
| Restricted twin (no spin) | same spectrum within 0.1 cm⁻¹ (`dfpt-incommensurate-si`) |
| CP2K binary | not on pin host |

Gates: \|ω\| low-3 ∈ [400, 900], high-3 ∈ [700, 1300] cm⁻¹; response must
converge; n_imag = 0. Closed-shell collinear (\(M=0\)) must match the
restricted incommensurate twin under identical ecut/k/q/UPF. Minimal twin
(10 Ry, Γ-only k) biases the spectrum high vs literature mid-ΓX — windows
reject unit disasters, not claim quantitative LDA dispersion.

### Performance

| Item | Value |
| ---- | ----- |
| `perf_exempt` | **true** |
| Reason | No CP2K DFPT wall twin on the build host this shift |
| ATV wall | ≈ 11 s |
| `ratio_atv_over_oracle` | n/a |
| `within_10x` | n/a (exempt) |

### CP2K twin (offline only)

When a CP2K binary is available: phonon / DFPT at fractional q ≈ (0.3,0,0)
with LSD/UKS on the same cell; record ω [cm⁻¹] + wall in
`tools/cp2k_oracle/inputs/dfpt-spin-incommensurate.note`.


## Reference — finite-q DFPT phonons (diamond Si at X, PW LSDA, NC-UPF)

### Method

Density-functional perturbation theory at fractional phonon wavevector
\(\mathbf{q}\) for `cpu::pw::PW`. **Commensurate product path:** rational
\(\mathbf{q}=(h/n_a,k/n_b,l/n_c)\) is evaluated as Γ-point DFPT on the
\(n_a\times n_b\times n_c\) supercell (self-consistent Sternheimer, same
engine as `dfpt-phonon-si`) followed by IFC Fourier
(`PhononDispersion::frequencies_at_q`). This is analytically equivalent to
finite-\(q\) DFPT for Born–von Karman commensurate \(q\).

```cpp
cpu::pw::DfptPhonon dfpt;
dfpt.q_frac = {0.5, 0.0, 0.0}; // X → 2×1×1 supercell
auto r = dfpt.compute(method, frame);
```

Also fixed in this slug: **Ewald real-space lattice sum** used the transpose
of the cell matrix (harmless for the symmetric diamond primitive, fatal for
non-symmetric supercells). `EwaldSummation` and DFPT Ewald Hessians now build
\(L = h\mathbf{a}+k\mathbf{b}+l\mathbf{c}\) from **rows**.

Cite: Baroni et al., *Rev. Mod. Phys.* **73**, 515 (2001); Giannozzi et al.,
*Phys. Rev. B* **43**, 7231 (1991).

### System

| Setting | Value |
| ------- | ----- |
| Structure | diamond Si primitive, a = **5.43 Å** |
| \(\mathbf{q}\) | \(X=(1/2,0,0)\) → **2×1×1** supercell (4 atoms) |
| Pseudopotential | `tests/data/pw/Si_lda_nc.upf` |
| XC | LSDA |
| Cutoff | **10 Ry** |
| k-mesh | **1×1×1** on the supercell (Γ+X of primitive folded) |
| Response | tol 1e-4, mix 0.25, max_iter 120 |

### Offline pins (measured 2026-07-25, Release, OMP=8)

| Quantity | Value |
| -------- | ----- |
| SCF energy (4-atom SC) | **−30.164 Ry** (≈ 2 × primitive scale) |
| ATV ω(X) [cm⁻¹] | **−130.3, −130.3, 229.8, 448.2, 672.6, 672.6** |
| \|ω\| sorted | 130, 130, 230, 448, 673, 673 |
| Literature LDA X | TA≈146, LA≈397, LO≈413, TO≈455 |
| n_imag | 2 (soft TA pair under minimal ecut/k — accepted ≤ 2) |
| Wall | ≈ **43 s** (≤ 60 s) |
| CP2K binary | not on pin host |

Gates use \|ω\| windows [50,400] (low 3) and [300,850] (high 3), n_imag≤2,
asymmetry ≤ 1e-3 eV/Å².

### Performance

| Item | Value |
| ---- | ----- |
| `perf_exempt` | **true** |
| Reason | No CP2K DFPT wall twin on the build host this shift |
| ATV wall | ≈ 43 s |
| `ratio_atv_over_oracle` | n/a |
| `within_10x` | n/a (exempt) |


## Reference — finite-q DFPT phonons with pure GGA kernel (diamond Si at X, PW PBE)

### Method

Density-functional perturbation theory at finite phonon wavevector
\(\mathbf{q}=X=(1/2,0,0)\) for the concrete plane-wave method `cpu::pw::PW`
with the **pure GGA (PBE) exchange-correlation kernel**. Commensurate product
path: rational \(\mathbf{q}\) is evaluated as Γ-point DFPT on the 2×1×1
supercell (self-consistent Sternheimer, same engine as `dfpt-gga-si`)
followed by IFC Fourier (`PhononDispersion::frequencies_at_q`) —
analytically equivalent to finite-\(q\) DFPT for Born–von Karman
commensurate \(q\). \(K_{\mathrm{xc}}\) is the second functional derivative
of the same `XCBuilder` GGA operator the ground-state SCF converged (central
FD with spectral \(\nabla\rho\) and GGA flux) — the joint product of the two
shipped single-axis pins `regressions/response/dfpt`
(commensurate X path, LSDA kernel) and `regressions/response/dfpt`
(Γ, PBE kernel).

```cpp
cpu::pw::PW pw{cpu::pw::UpfPseudopotential{ATV_REG_DATA_DIR "/pw/Si_lda_nc.upf"},
               /*ecut_ry=*/10.0, /*k_mesh=*/1,
               cpu::scf::XCType::PBE}; // pure GGA kernel
pw.tol = 1e-8;
cpu::pw::DfptPhonon dfpt;
dfpt.q_frac = {0.5, 0.0, 0.0};   // X → 2×1×1 supercell Γ-DFPT
auto r = dfpt.compute(pw, frame);
```

Cite: Baroni, de Gironcoli, Dal Corso, Giannozzi, *Rev. Mod. Phys.* **73**,
515 (2001); Giannozzi et al., *Phys. Rev. B* **43**, 7231 (1991); Perdew,
Burke, Ernzerhof, *Phys. Rev. Lett.* **77**, 3865 (1996).

### System (regression twin)

| Setting | Value |
| ------- | ----- |
| Structure | diamond Si, 2-atom fcc primitive cell, a = **5.43 Å** (experimental) |
| \(\mathbf{q}\) | \(X=(1/2,0,0)\) → **2×1×1** supercell (4 atoms) |
| Pseudopotential | `tests/data/pw/Si_lda_nc.upf` (ONCVPSP NC, in-tree; same GGA-smoke PP as `pw-pbe-gga-si`) |
| XC | `XCType::PBE` (true GGA kernel) |
| Cutoff | **10 Ry** (minimal-twin budget) |
| k-mesh | **1×1×1** on the supercell (Γ+X of primitive folded) |
| SCF | tol 1e-8 (density L2) |
| Response | `response_tol` 1e-4, `mix_beta` 0.25, `cg_tol` 1e-8, max_iter 120 |

### Offline pins (no live CP2K at regression time)

| Field | Value |
| ----- | ----- |
| Oracle 1 (physics anchor) | **CP2K 2026.1** (rev `85b8a9b49c`, `~/.local/bin/cp2k`, this host, 2026-07-31) — `VIBRATIONAL_ANALYSIS` (FD of analytic RKS k-point forces, `FULLY_PERIODIC T`, DX 0.01 bohr, `EPS_SCF 1e-8`) on the **2×1×1 supercell** (4 atoms): PBE, DZVP-MOLOPT-SR-GTH / GTH-PBE-q4, CUTOFF 300 Ry, 1×2×2 MP (≡ 2×2×2 on the primitive). Input: `tools/cp2k_oracle/inputs/dfpt-gga-finite-q-si/si4_pbe_vib.inp` |
| Oracle 1 output lines | `VIB\|Frequency (cm^-1)  9.7979172103E+01 1.3126606201E+02 3.7058084483E+02` / `4.2823632486E+02 4.5748021881E+02 4.5868390952E+02` / `5.3831838074E+02 5.3831995025E+02 5.4364491240E+02` — the supercell Γ FD spectrum folds primitive Γ ∪ X; eigenvector phases identify modes 1–6 (cells out of phase = X branch) → **X: 97.98, 131.27 (TA, FD-split doublet), 370.58, 428.24 (LA/LO), 457.48, 458.68 (TO) cm⁻¹**; modes 7–9 = Γ optical 538.32, 538.32, 543.64 |
| Supplemental same-quantity check | Runtime three-energy curvature of an atom0x (+h,−h) pattern across the two X supercell images, h=0.03 Å. Divide by N_cells=2 to obtain Re C00(X); bound 0.30 eV/Å². This is not an external oracle. |
| Pin 3 (cross-parent consistency) | PBE spectrum must sit near the LSDA parent (`dfpt-finite-q-si`: \|ω\| 130, 130, 230, 448, 673, 673 cm⁻¹) — an XC-kernel change, not new lattice dynamics |

The CP2K twin necessarily differs in PP/basis representation (GPW GTH/MOLOPT
FD-supercell at 300 Ry vs the LDA-generated NC-UPF plane-wave twin at a
10 Ry minimal cutoff with a Γ-only supercell mesh), so it anchors the
**physics window**, not the digit. The same \|ω\| windows the LSDA parent
uses — low 3 ∈ [50, 400], high 3 ∈ [300, 850] cm⁻¹ — contain both the CP2K
X branch (98, 131, 371 | 428, 457, 459) and the biased ATV minimal twin
(108, 108, 239 | 451, 669, 669). The runtime q-pattern curvature is an internal consistency supplement,
while the external spectrum anchors the stated physics window.

### Force-constant observable and acceptance scope

The commensurate result now publishes raw complex C(q), obtained by the same
`PhononDispersion::force_constants_at_q` home-row IFC Fourier sum used by the
vibration spectrum path. It does not publish the top-left supercell block.
Frequency diagonalization retains the existing Hermitian LAPACK route;
raw matrix real/imaginary parts and preprojection asymmetry remain observable.

The original numerical bounds remain: external spectral windows low 3
[50,400] cm⁻¹/high 3 [300,850] cm⁻¹, at most 2 imaginary modes, asymmetry
≤1e-4 eV/Å², and runtime q-pattern curvature agreement within 0.30 eV/Å².
The entire existing section, including supplemental SCFs and parent
comparisons, enforces 60 s. No parameter section or performance exemption is
introduced. The previous frozen native supercell-diagonal constants were the
wrong quantity and have been removed rather than repinned from native output.

The q-pattern calculation uses the unchanged method controls and h=0.03 Å.
For displacement amplitudes (+h,−h),

\[
\operatorname{Re} C_{00}(X) \simeq
\frac{E(+h)-2E(0)+E(-h)}{2h^2}.
\]

The factor 2 is the number of displaced primitive images. Under smearing,
PW's returned energy includes its entropy term, so the same expression is a
free-energy curvature. This is an internal consistency supplement. The
external CP2K frequency data and analytic symmetry identities remain the
oracle content; there is no independent external digit pin for the complete
NC-UPF C(q) matrix at these minimal settings.

### Validation and historical timing

The new raw-matrix spring-chain units target the exact q-dependent matrix,
including complex off-diagonal phases and unprojected asymmetry, without SCF.
CPU validation on 2026-10-25 passed all 1785 units and all 15 DFPT
sections. This complete section took 12.50 s at the declared one thread;
Re C00(X) was 19.46231972 and the q-pattern curvature was 19.45278291 eV/Å².
These are observed native checks, not replacement oracle pins. Raw output and
source hashes: [XC/response validation](../../../tools/diagnostics/pw-xc-response/README.md).
The existing external CP2K reference process wall was 288.7 s.
It is retained as historical provenance, not a refreshed fair timing ratio.


## Reference — finite-q DFPT + metallic smearing (diamond Si at X, PW LSDA)

### Method

Density-functional perturbation theory at fractional phonon wavevector
\(\mathbf{q}\) for `cpu::pw::PW` with **Fermi–Dirac metallic occupation
smearing**. **Commensurate product path:** rational
\(\mathbf{q}=(h/n_a,k/n_b,l/n_c)\) is evaluated as Γ-point DFPT on the
supercell (smear-aware Sternheimer, same engine as `dfpt-metallic-si`)
followed by IFC Fourier (`PhononDispersion::frequencies_at_q`).

Direct finite-\(q\) Sternheimer (`dfpt_q_force_constants`) also accepts
restricted smear: fractional \(f_{nk}\) weights, projector on
\(f\ge 10^{-8}\) at \(\mathbf{k}+\mathbf{q}\); no \(\delta\mu\) at
\(q\neq 0\).

```cpp
cpu::pw::PW method{cpu::pw::UpfPseudopotential{ATV_REG_DATA_DIR "/pw/Si_lda_nc.upf"},
                   /*ecut_ry=*/10.0, /*k_mesh=*/1, cpu::scf::XCType::LSDA};
method.smearing = Smearing::fermi_dirac;
method.degauss = 0.02; // Ry
cpu::pw::DfptPhonon dfpt;
dfpt.q_frac = {0.5, 0.0, 0.0}; // X → 2×1×1 supercell
auto r = dfpt.compute(method, frame);
```

Cite: Baroni et al., *Rev. Mod. Phys.* **73**, 515 (2001);
de Gironcoli, *Phys. Rev. B* **51**, 6773 (1995). Sibling fixed-occ:
`regressions/response/dfpt`. Sibling Γ smear:
`regressions/response/dfpt`.

### System (regression twin)

| Setting | Value |
| ------- | ----- |
| Structure | diamond Si primitive, a = **5.43 Å** |
| \(\mathbf{q}\) | \(X=(1/2,0,0)\) → **2×1×1** supercell (4 atoms) |
| Pseudopotential | `tests/data/pw/Si_lda_nc.upf` |
| XC | LSDA |
| Smearing | Fermi–Dirac, `degauss = 0.02` Ry, `extra_bands = 0` |
| Cutoff | **10 Ry** |
| k-mesh | **1×1×1** on the supercell |
| Response | tol 1e-4, mix 0.25, max_iter 120 |

Gapped Si at these settings keeps near-integer FD occupations, so free
energy \(F \approx E_{\mathrm{KS}}\) and results track the fixed-occ twin.

### Offline pins

| Field | Value |
| ----- | ----- |
| Preferred oracle | CP2K `VIBRATIONAL_ANALYSIS` + `&SMEAR` (GTH-GPW ≠ NC-UPF absolute E) |
| CP2K binary this host | **not available** (`module avail cp2k` empty) |
| Physics pin | literature LDA Si X windows (Giannozzi et al.) via \(\lvert\omega\rvert\) gates |
| Same-Hamiltonian | spectra track `dfpt-finite-q-si` fixed-occ twin within gate windows |

#### Measured pins (this host, 2026-07-26, Release, OMP=8)

| Quantity | Value |
| -------- | ----- |
| SCF free energy F (4-atom SC) | **−30.164 Ry** (matches fixed-occ twin scale) |
| ATV ω(X) [cm⁻¹] | **−130.30, −130.30, 229.79, 448.20, 672.63, 672.63** |
| \(\lvert\omega\rvert\) sorted | 130, 130, 230, 448, 673, 673 |
| n_imag | 2 (soft TA under minimal ecut/k — accepted ≤ 2) |
| asymmetry | 8.4e-8 eV/Å² → gate ≤ 1e-3 |
| Wall | ≈ **40 s** (≤ 60 s budget) |

Gates: \(\lvert\omega\rvert\) low 3 ∈ [50, 400], high 3 ∈ [300, 850], n_imag≤2,
asymmetry ≤ 1e-3 eV/Å². Spectra track fixed-occ twin `dfpt-finite-q-si` to
printed precision (gapped FD fill).

### Performance

| Item | Value |
| ---- | ----- |
| `perf_exempt` | **true** |
| Reason | No CP2K DFPT+smear wall twin on the build host this shift |
| ATV wall | ≈ 40 s (2-atom primitive → 4-atom SC, 12 responses, OMP=8) |
| `ratio_atv_over_oracle` | n/a (exempt) |
| `within_10x` | n/a (exempt; correctness gates only) |


## Reference — finite-q DFPT collinear spin + metallic smearing (diamond Si, PW LSDA)

### Method

Density-functional perturbation theory at fractional phonon wavevector
\(\mathbf{q}\) for `cpu::pw::PW` with **collinear spin** and **Fermi–Dirac
metallic occupation smearing**.

| Path | Product |
|------|---------|
| Commensurate (e.g. \(X\)) | supercell Γ-DFPT (spin+smear Sternheimer) + IFC Fourier |
| **Direct** (this regression) | `dfpt_q_spin_impl`: shared \(\mu\), spinor \(f\in[0,1]\), projector \(f\ge 10^{-8}\) at \(\mathbf{k}+\mathbf{q}\); **no** \(\delta\mu\) at \(q\neq 0\) |

```cpp
cpu::pw::PW method{cpu::pw::UpfPseudopotential{ATV_REG_DATA_DIR "/pw/Si_lda_nc.upf"},
                   /*ecut_ry=*/10.0, /*k_mesh=*/1, cpu::scf::XCType::LSDA};
method.spin_polarized = true;
method.multiplicity = 1;
method.smearing = Smearing::fermi_dirac;
method.degauss = 0.02; // Ry
cpu::pw::DfptPhonon dfpt;
dfpt.q_frac = {0.3, 0.0, 0.0}; // incommensurate → direct Sternheimer
auto r = dfpt.compute(method, frame);
```

Cite: Baroni et al., *Rev. Mod. Phys.* **73**, 515 (2001);
de Gironcoli, *Phys. Rev. B* **51**, 6773 (1995). Siblings:
`dfpt-spin-incommensurate`, `dfpt-finite-q-smearing`, `dfpt-spin-smearing`.

### System (regression twin)

| Setting | Value |
| ------- | ----- |
| Structure | diamond Si primitive, a = **5.43 Å** |
| \(\mathbf{q}\) | **(0.3, 0, 0)** fractional (incommensurate under n≤6) |
| Pseudopotential | `tests/data/pw/Si_lda_nc.upf` |
| XC | LSDA |
| Spin | collinear, multiplicity = 1 (M = 0) |
| Smearing | Fermi–Dirac, `degauss = 0.02` Ry, `extra_bands = 0` |
| Cutoff | **10 Ry** |
| k-mesh | **1×1×1** |
| Response | tol 1e-4, mix 0.2, max_iter 120 |

Gapped Si at these settings keeps near-integer FD occupations, so free
energy \(F \approx E_{\mathrm{KS}}\) and results track the fixed-occ spin
incommensurate twin.

### Offline pins

| Field | Value |
| ----- | ----- |
| Preferred oracle | CP2K phonon / DFPT + LSD + `&SMEAR` (GTH-GPW ≠ NC-UPF absolute E) |
| CP2K binary this host | **not available** (`module avail cp2k` empty) |
| Physics pin | same \|ω\| windows as `dfpt-spin-incommensurate` / `dfpt-incommensurate-si` |
| Same-Hamiltonian | spectra match fixed-occ spin incommensurate twin to printed precision |

#### Measured pins (this host, 2026-07-26, Release, OMP=8)

| Quantity | Value |
| -------- | ----- |
| ATV ω(q) [cm⁻¹] | **692.90, 726.85, 726.85, 944.07, 1023.46, 1023.46** |
| \(\lvert\omega\rvert\) sorted | 693, 727, 727, 944, 1023, 1023 |
| n_imag | 0 |
| asymmetry | **29.6 eV/Å²** → gate ≤ 50 |
| Wall | ≈ **11 s** (≤ 60 s budget) |
| Twin match | identical to `dfpt-spin-incommensurate` fixed-occ print (gapped FD fill) |

Gates: \(\lvert\omega\rvert\) low 3 ∈ [400, 900], high 3 ∈ [700, 1300] cm⁻¹,
n_imag = 0, response converged, asymmetry ≤ 50 eV/Å².

### Performance

| Item | Value |
| ---- | ----- |
| `perf_exempt` | **true** |
| Reason | No CP2K DFPT spin+smear wall twin on the build host this shift |
| ATV wall | ≈ 11 s (2-atom primitive, 6 responses × 2 spin, OMP=8) |
| `ratio_atv_over_oracle` | n/a (exempt) |
| `within_10x` | n/a (exempt; correctness gates only) |


## Reference — finite-q DFPT phonons with joint collinear spin × pure GGA (diamond Si at X, PW)

### Method

Density-functional perturbation theory at finite phonon wavevector
\(\mathbf{q}=X=(1/2,0,0)\) for the concrete plane-wave method `cpu::pw::PW`
with the **joint** collinear-spin × pure-GGA response kernel
(`spin_polarized = true`, `multiplicity = 1` ⇒ \(M=0\),
`xc = XCType::PBE`). Commensurate product path: rational \(\mathbf{q}\) is
evaluated as Γ-point DFPT on the 2×1×1 supercell (self-consistent
spin-resolved Sternheimer, same engine as `dfpt-gga-spin-si`) followed by
IFC Fourier (`PhononDispersion::frequencies_at_q`).
\(K_{\mathrm{xc}}^{\sigma\sigma'}\) is the central FD of the spin-GGA
`XCBuilder::step_spin` operator the ground-state SCF converged — the triple
product of the two shipped joint parents
`regressions/response/dfpt` (Γ, spin × PBE) and
`regressions/response/dfpt` (X supercell, restricted PBE).

```cpp
cpu::pw::PW pw{cpu::pw::UpfPseudopotential{ATV_REG_DATA_DIR "/pw/Si_lda_nc.upf"},
               /*ecut_ry=*/10.0, /*k_mesh=*/1,
               cpu::scf::XCType::PBE}; // pure GGA kernel
pw.tol = 1e-8;
pw.spin_polarized = true;        // collinear, M = 0
pw.multiplicity = 1;
cpu::pw::DfptPhonon dfpt;
dfpt.q_frac = {0.5, 0.0, 0.0};   // X → 2×1×1 supercell Γ-DFPT
auto r = dfpt.compute(pw, frame);
```

Cite: Baroni, de Gironcoli, Dal Corso, Giannozzi, *Rev. Mod. Phys.* **73**,
515 (2001); Giannozzi et al., *Phys. Rev. B* **43**, 7231 (1991); Perdew,
Burke, Ernzerhof, *Phys. Rev. Lett.* **77**, 3865 (1996).

### System (regression twin)

| Setting | Value |
| ------- | ----- |
| Structure | diamond Si, 2-atom fcc primitive cell, a = **5.43 Å** (experimental) |
| \(\mathbf{q}\) | \(X=(1/2,0,0)\) → **2×1×1** supercell (4 atoms) |
| Pseudopotential | `tests/data/pw/Si_lda_nc.upf` (ONCVPSP NC, in-tree; same GGA-smoke PP as `pw-pbe-gga-si`) |
| XC | `XCType::PBE` (true spin-GGA kernel) |
| Spin | collinear, multiplicity = 1 (\(N_\uparrow=N_\downarrow=8\) on the supercell) |
| Cutoff | **10 Ry** (minimal-twin budget) |
| k-mesh | **1×1×1** on the supercell (Γ+X of primitive folded) |
| SCF | tol 1e-8 (density L2), linear mix on spin channels |
| Response | `response_tol` 1e-4, `mix_beta` 0.25, `cg_tol` 1e-8, max_iter 120 |

### Offline pins (no live CP2K at regression time)

| Field | Value |
| ----- | ----- |
| Oracle 1 (physics anchor) | **CP2K 2026.1** (rev `85b8a9b49c`, `~/.local/bin/cp2k`, this host, 2026-07-31) — `VIBRATIONAL_ANALYSIS` (FD of analytic UKS k-point forces, `FULLY_PERIODIC T`, DX 0.01 bohr, `EPS_SCF 1e-8`) on the **2×1×1 supercell** (4 atoms): LSD + `MULTIPLICITY 1` + PBE, DZVP-MOLOPT-SR-GTH / GTH-PBE-q4, CUTOFF 300 Ry, 1×2×2 MP (≡ 2×2×2 on the primitive). Input: `tools/cp2k_oracle/inputs/dfpt-gga-spin-finite-q-si/si4_uks_pbe_vib.inp` |
| Oracle 1 output lines | `VIB\|Frequency (cm^-1)  9.7979171441E+01 1.3126605303E+02 3.7058085178E+02` / `4.2823632737E+02 4.5748022016E+02 4.5868391248E+02` / `5.3831837242E+02 5.3831998162E+02 5.4364489119E+02` — supercell Γ FD folds primitive Γ ∪ X; eigenvector phases identify modes 1–6 (cells out of phase = X branch) → **X: 97.98, 131.27 (TA, FD-split doublet), 370.58, 428.24 (LA/LO), 457.48, 458.68 (TO) cm⁻¹**; modes 7–9 = Γ optical 538.32, 538.32, 543.64. Digit-identical to the restricted supercell twin at \(M=0\). |
| Supplemental same-quantity check | Runtime three-energy curvature of an atom0x (+h,−h) pattern across the two X supercell images, h=0.03 Å. Divide by N_cells=2 to obtain Re C00(X); bound 0.30 eV/Å². This is not an external oracle. |
| Analytic M=0 identity | Runtime restricted DFPT at the same q with all other controls, including smearing, unchanged: difference in Re C00(X) ≤ 0.05 eV/Å². No frozen native parent value. |

The CP2K twin necessarily differs in PP/basis representation (GPW GTH/MOLOPT
FD-supercell at 300 Ry vs the LDA-generated NC-UPF plane-wave twin at a
10 Ry minimal cutoff with a Γ-only supercell mesh), so it anchors the
**physics window**, not the digit. The same \|ω\| windows the finite-q
parents use — low 3 ∈ [50, 400], high 3 ∈ [300, 850] cm⁻¹ — contain both
the CP2K X branch (98, 131, 371 \| 428, 457, 459) and the biased ATV minimal
twin (109, 109, 239 \| 451, 669, 669). The runtime q-pattern curvature supplements the external spectral
window; it is not an independent numerical oracle.

### Force-constant observable and acceptance scope

The commensurate result now publishes raw complex C(q), obtained by the same
`PhononDispersion::force_constants_at_q` home-row IFC Fourier sum used by the
vibration spectrum path. It does not publish the top-left supercell block.
Frequency diagonalization retains the existing Hermitian LAPACK route;
raw matrix real/imaginary parts and preprojection asymmetry remain observable.

The original numerical bounds remain: external spectral windows low 3
[50,400] cm⁻¹/high 3 [300,850] cm⁻¹, at most 2 imaginary modes, asymmetry
≤1e-4 eV/Å², and runtime q-pattern curvature agreement within 0.30 eV/Å².
The runtime same-q restricted-parent M=0 comparison retains 0.05 eV/Å².
The entire existing section, including supplemental SCFs and parent
comparisons, enforces 60 s. No parameter section or performance exemption is
introduced. The previous frozen native supercell-diagonal constants were the
wrong quantity and have been removed rather than repinned from native output.

The q-pattern calculation uses the unchanged method controls and h=0.03 Å.
For displacement amplitudes (+h,−h),

\[
\operatorname{Re} C_{00}(X) \simeq
\frac{E(+h)-2E(0)+E(-h)}{2h^2}.
\]

The factor 2 is the number of displaced primitive images. Under smearing,
PW's returned energy includes its entropy term, so the same expression is a
free-energy curvature. This is an internal consistency supplement. The
external CP2K frequency data and analytic symmetry identities remain the
oracle content; there is no independent external digit pin for the complete
NC-UPF C(q) matrix at these minimal settings.

### Validation and historical timing

The new raw-matrix spring-chain units target the exact q-dependent matrix,
including complex off-diagonal phases and unprojected asymmetry, without SCF.
CPU validation on 2026-10-25 passed all 1785 units and all 15 DFPT
sections. This complete section took 33.31 s at the declared one thread;
Re C00(X) was 19.45605115 and the q-pattern curvature was 19.45278288 eV/Å².
These are observed native checks, not replacement oracle pins. Raw output and
source hashes: [XC/response validation](../../../tools/diagnostics/pw-xc-response/README.md).
The existing external CP2K reference process wall was 577.8 s.
It is retained as historical provenance, not a refreshed fair timing ratio.


## Reference — finite-q DFPT phonons, joint collinear spin × pure GGA × metallic smear (diamond Si at X, PW)

### Method

Density-functional perturbation theory at finite phonon wavevector
\(\mathbf{q}=X=(1/2,0,0)\) for the concrete plane-wave method `cpu::pw::PW`
with the **four-way joint** product path:

- collinear spin (`spin_polarized = true`, `multiplicity = 1` ⇒ \(M=0\)),
- pure GGA kernel (`xc = XCType::PBE`),
- Fermi–Dirac metallic occupation smearing (`smearing = fermi_dirac`,
  `degauss = 0.02` Ry, `extra_bands = 0`),
- commensurate finite \(\mathbf{q}\) via 2×1×1 supercell Γ-DFPT + IFC Fourier:

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
`dfpt-gga-spin-finite-q-si` parent.

```cpp
cpu::pw::PW pw{cpu::pw::UpfPseudopotential{ATV_REG_DATA_DIR "/pw/Si_lda_nc.upf"},
               /*ecut_ry=*/10.0, /*k_mesh=*/1, cpu::scf::XCType::PBE};
pw.spin_polarized = true;
pw.multiplicity = 1;
pw.smearing = Smearing::fermi_dirac;
pw.degauss = 0.02;
pw.extra_bands = 0;
cpu::pw::DfptPhonon dfpt;
dfpt.q_frac = {0.5, 0.0, 0.0};   // X → 2×1×1 supercell Γ-DFPT
auto r = dfpt.compute(pw, frame);
```

Cite: Baroni, de Gironcoli, Dal Corso, Giannozzi, *Rev. Mod. Phys.* **73**,
515 (2001); de Gironcoli, *Phys. Rev. B* **51**, 6773 (1995); Giannozzi et
al., *Phys. Rev. B* **43**, 7231 (1991); Perdew, Burke, Ernzerhof, *Phys.
Rev. Lett.* **77**, 3865 (1996).

Siblings: `dfpt-gga-spin-smearing` (Γ, spin×GGA×smear),
`dfpt-gga-spin-finite-q-si` (X, spin×GGA, fixed occ),
`dfpt-finite-q-spin-smearing` (direct q, spin×LSDA×smear).

### System (regression twin)

| Setting | Value |
| ------- | ----- |
| Structure | diamond Si, 2-atom fcc primitive cell, a = **5.43 Å** (experimental) |
| \(\mathbf{q}\) | \(X=(1/2,0,0)\) → **2×1×1** supercell (4 atoms) |
| Pseudopotential | `tests/data/pw/Si_lda_nc.upf` (ONCVPSP NC, in-tree) |
| XC | `XCType::PBE` (true spin-GGA kernel) |
| Spin | collinear, multiplicity = 1 (\(N_\uparrow=N_\downarrow=8\) on the supercell) |
| Smearing | Fermi–Dirac, `degauss = 0.02` Ry, `extra_bands = 0` |
| Cutoff | **10 Ry** (minimal-twin budget) |
| k-mesh | **1×1×1** on the supercell (Γ+X of primitive folded) |
| SCF | tol 1e-8 (density L2), linear mix on spin channels |
| Response | `response_tol` 1e-4, `mix_beta` 0.25, `cg_tol` 1e-8, max_iter 120 |

### Offline pins (no live CP2K at regression time)

| Field | Value |
| ----- | ----- |
| Oracle 1 (physics anchor) | **CP2K 2026.1** (rev `git:5e54ba2`, source build `…/cp2k/install/bin/cp2k.psmp`, env `…/install/env.sh`, this host, 2026-08-01) — `VIBRATIONAL_ANALYSIS` (FD of analytic UKS k-point forces, `FULLY_PERIODIC T`, DX 0.01 bohr, `EPS_SCF 1e-8`) on the **2×1×1 supercell** (4 atoms): LSD + `MULTIPLICITY 1` + PBE + `&SMEAR FERMI_DIRAC` at 3157.75 K (≡ 0.02 Ry), ADDED_MOS 4, DZVP-MOLOPT-SR-GTH / GTH-PBE-q4, CUTOFF 300 Ry, 1×2×2 MP (≡ 2×2×2 on the primitive). Input: `tools/cp2k_oracle/inputs/dfpt-gga-spin-finite-q-smearing/si4_uks_pbe_smear_vib.inp` |
| Oracle 1 output lines | `VIB\|Frequency (cm^-1)    98.223723           130.533918           364.045396` / `427.180368           452.075468           458.190888` / `536.999556           537.000981           541.694572` — supercell Γ FD folds primitive Γ ∪ X; modes 1–6 (cells out of phase) = **X branch: 98.22, 130.53 (TA, FD-split doublet), 364.05, 427.18 (LA/LO), 452.08, 458.19 (TO) cm⁻¹**; modes 7–9 = Γ optical 537.00, 537.00, 541.69. Matches the fixed-occ spin×GGA supercell twin at \(M=0\) to the expected gapped-smear digit. |
| Supplemental same-quantity check | Runtime three-energy curvature of an atom0x (+h,−h) pattern across the two X supercell images, h=0.03 Å. Divide by N_cells=2 to obtain Re C00(X); bound 0.30 eV/Å². This is not an external oracle. |
| Analytic M=0 identity | Runtime restricted DFPT at the same q with all other controls, including smearing, unchanged: difference in Re C00(X) ≤ 0.05 eV/Å². No frozen native parent value. |

The CP2K twin necessarily differs in PP/basis representation (GPW
GTH/MOLOPT FD-supercell at 300 Ry vs the LDA-generated NC-UPF plane-wave twin
at a 10 Ry minimal cutoff with a Γ-only supercell mesh), so it anchors the
**physics window**, not the digit. The same \|ω\| windows the finite-q
parents use — low 3 ∈ [50, 400], high 3 ∈ [300, 850] cm⁻¹ — contain both
the CP2K X branch (98, 131, 364 \| 427, 452, 458) and the biased ATV minimal
twin (109, 109, 239 \| 451, 669, 669). The runtime free-energy q-pattern curvature supplements the external
spectral window and does not supply an independent numerical oracle.

### Force-constant observable and acceptance scope

The commensurate result now publishes raw complex C(q), obtained by the same
`PhononDispersion::force_constants_at_q` home-row IFC Fourier sum used by the
vibration spectrum path. It does not publish the top-left supercell block.
Frequency diagonalization retains the existing Hermitian LAPACK route;
raw matrix real/imaginary parts and preprojection asymmetry remain observable.

The original numerical bounds remain: external spectral windows low 3
[50,400] cm⁻¹/high 3 [300,850] cm⁻¹, at most 2 imaginary modes, asymmetry
≤1e-4 eV/Å², and runtime q-pattern curvature agreement within 0.30 eV/Å².
The runtime same-q restricted-parent M=0 comparison retains 0.05 eV/Å².
The entire existing section, including supplemental SCFs and parent
comparisons, enforces 60 s. No parameter section or performance exemption is
introduced. The previous frozen native supercell-diagonal constants were the
wrong quantity and have been removed rather than repinned from native output.

The q-pattern calculation uses the unchanged method controls and h=0.03 Å.
For displacement amplitudes (+h,−h),

\[
\operatorname{Re} C_{00}(X) \simeq
\frac{E(+h)-2E(0)+E(-h)}{2h^2}.
\]

The factor 2 is the number of displaced primitive images. Under smearing,
PW's returned energy includes its entropy term, so the same expression is a
free-energy curvature. This is an internal consistency supplement. The
external CP2K frequency data and analytic symmetry identities remain the
oracle content; there is no independent external digit pin for the complete
NC-UPF C(q) matrix at these minimal settings.

### Validation and historical timing

The new raw-matrix spring-chain units target the exact q-dependent matrix,
including complex off-diagonal phases and unprojected asymmetry, without SCF.
CPU validation on 2026-10-25 passed all 1785 units and all 15 DFPT
sections. This complete section took 32.58 s at the declared one thread;
Re C00(X) was 19.45605115 and the q-pattern curvature was 19.45278288 eV/Å².
These are observed native checks, not replacement oracle pins. Raw output and
source hashes: [XC/response validation](../../../tools/diagnostics/pw-xc-response/README.md).
The existing external CP2K reference process wall was 585.9 s.
It is retained as historical provenance, not a refreshed fair timing ratio.
