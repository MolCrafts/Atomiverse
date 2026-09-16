# Reference — collinear-spin DFPT at incommensurate q (diamond Si, PW LSDA)

## Method

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

## System

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

## Offline pins (measured 2026-07-26, Release, OMP=8)

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

## Performance

| Item | Value |
| ---- | ----- |
| `perf_exempt` | **true** |
| Reason | No CP2K DFPT wall twin on the build host this shift |
| ATV wall | ≈ 11 s |
| `ratio_atv_over_oracle` | n/a |
| `within_10x` | n/a (exempt) |

## CP2K twin (offline only)

When a CP2K binary is available: phonon / DFPT at fractional q ≈ (0.3,0,0)
with LSD/UKS on the same cell; record ω [cm⁻¹] + wall in
`tools/cp2k_oracle/inputs/dfpt-spin-incommensurate.note`.
