# Reference — Γ-point DFPT phonons (diamond Si, PW LSDA, NC-UPF)

## Method

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

## System (regression twin)

| Setting | Value |
| ------- | ----- |
| Structure | diamond Si, 2-atom fcc primitive cell, a = **5.431 Å** (experimental) |
| Pseudopotential | `tests/data/pw/Si_lda_nc.upf` (ONCVPSP LDA NC, in-tree) |
| XC | `XCType::LSDA` (Slater + VWN5), the DFPT-supported kernel |
| Cutoff | **10 Ry** (minimal-twin budget; see below) |
| k-mesh | **2×2×2** MP (unshifted, 8 points) |
| SCF | tol 1e-8 (density L2), Pulay mixing |
| Response | `response_tol` 1e-5, `mix_beta` 0.7, CG tol 1e-9 |

## Offline pins (no live CP2K/QE at regression time)

| Field | Value |
| ----- | ----- |
| Preferred oracle | CP2K DFPT / `VIBRATIONAL_ANALYSIS` on the same cell |
| CP2K binary this host | **not available** (no CP2K module on the pin host this shift — same as every prior wave) |
| Pin 1 (sharp, same-Hamiltonian) | central **second difference of the SCF total energy**: C[(0x),(0x)] = (E(+h) − 2E₀ + E(−h))/h², h = 0.03 Å, SCF tol 1e-9, identical ecut/k/XC |
| Pin 2 (physics) | literature DFPT-LDA / Raman Si TO(Γ) window |

### Measured pins (this host, 2026-07-25, Release, OMP=8)

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
while the energy-FD pin (0.07 % agreement) is the sharp correctness gate.

### Why the FD twin is energy-based, not force-based

`pw_forces_analytic` deviates from the exact gradient of the discretized
total energy by ≈ 2 % for this system (measured: F_x analytic −0.8442 vs
FD-of-energy −0.8601 eV/Å at a 0.05 Å displacement) — an egg-box-scale
inconsistency of the v1 analytic-force implementation (aliasing of the
density grid). FD of those forces therefore does **not** reproduce the true
discrete Hessian (row sums off by up to 0.6 eV/Å²), while DFPT matches the
energy second difference to 0.07 %. Central *energy* differences are the
honest oracle.

## CP2K twin (offline only)

When a CP2K binary is available: run Quickstep DFPT (or
`VIBRATIONAL_ANALYSIS`) for diamond Si at the same lattice constant and a
comparable NC/GTH setup, record ω(Γ) [cm⁻¹] + wall here with the CP2K
version, and optionally the fair wall ratio r = T_atv/T_cp2k (require ≤ 10).

## Performance

| Item | Value |
| ---- | ----- |
| `perf_exempt` | **true** |
| Reason | No CP2K binary/module on the build host for a fair DFPT wall twin this shift |
| ATV wall (informative) | ≈ 24 s (2-atom Si, 6 responses, 8 k-points) |
| `ratio_atv_over_cp2k` | n/a (exempt) |
| `within_10x` | n/a (exempt; correctness gates only) |
