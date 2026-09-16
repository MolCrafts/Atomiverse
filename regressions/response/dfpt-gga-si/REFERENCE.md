# Reference — Γ-point DFPT phonons (diamond Si, PW PBE GGA kernel)

## Method

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
Sibling LSDA product: `regressions/response/dfpt-phonon-si`.

Public path: `cpu::pw::DfptPhonon::compute(method, frame)` with
`method.xc = XCType::PBE`.

## System (regression twin)

| Setting | Value |
| ------- | ----- |
| Structure | diamond Si, 2-atom fcc primitive cell, a = **5.43 Å** (experimental) |
| Pseudopotential | `tests/data/pw/Si_lda_nc.upf` (ONCVPSP NC, in-tree; same GGA-smoke PP as `pw-pbe-gga-si`) |
| XC | `XCType::PBE` (true GGA kernel) |
| Cutoff | **10 Ry** (minimal-twin budget) |
| k-mesh | **2×2×2** MP (unshifted) |
| SCF | tol 1e-8 (density L2), Pulay mixing |
| Response | `response_tol` 1e-5, `mix_beta` 0.7, CG tol 1e-9 |

## Offline pins (no live CP2K/QE at regression time)

| Field | Value |
| ----- | ----- |
| Preferred oracle | CP2K DFPT / `VIBRATIONAL_ANALYSIS` on the same cell (PBE) |
| CP2K binary this host | **not available** (`module avail cp2k` empty) |
| Pin 1 (sharp, same-Hamiltonian) | central **second difference of the SCF total energy**: C[(0x),(0x)] = (E(+h) − 2E₀ + E(−h))/h², h = 0.03 Å, SCF tol 1e-9, identical ecut/k/XC |
| Pin 2 (physics) | optical Γ triplet in a wide literature-style window; acoustic ≈ 0 without ASR |

### Measured pins (this host, 2026-07-26, Release, OMP=8)

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
LDA-generated NC UPF used as fixed valence PP). The energy-FD pin is the
sharp correctness gate for the GGA kernel.

## CP2K twin (offline only)

See `tools/cp2k_oracle/inputs/dfpt-gga-si.note`. When a CP2K binary is
available, record ω(Γ) [cm⁻¹] + wall with version and optional fair ratio
\(r = T_{\mathrm{atv}}/T_{\mathrm{cp2k}}\) (require ≤ 10).

## Performance

| Item | Value |
| ---- | ----- |
| `perf_exempt` | **true** |
| Reason | No CP2K binary/module on the build host for a fair DFPT wall twin this shift |
| ATV wall (informative) | ≈ 15 s (2-atom Si, 6 responses, 8 k-points, GGA kernel FD) |
| `ratio_atv_over_cp2k` | n/a (exempt) |
| `within_10x` | n/a (exempt; correctness gates only) |
