# Reference — incommensurate-q DFPT phonons (diamond Si, PW LSDA, NC-UPF)

## Method

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

## System

| Setting | Value |
| ------- | ----- |
| Structure | diamond Si primitive, a = **5.43 Å** |
| \(\mathbf{q}\) | **(0.3, 0, 0)** fractional (incommensurate under n≤6) |
| Pseudopotential | `tests/data/pw/Si_lda_nc.upf` |
| XC | LSDA |
| Cutoff | **10 Ry** |
| k-mesh | **1×1×1** |
| Response | tol 1e-4, mix 0.2, max_iter 120 |

## Offline pins (measured 2026-07-25, Release, OMP=8)

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

## Performance

| Item | Value |
| ---- | ----- |
| `perf_exempt` | **true** |
| Reason | No CP2K DFPT wall twin on the build host this shift |
| ATV wall | ≈ 7 s |
| `ratio_atv_over_oracle` | n/a |
| `within_10x` | n/a (exempt) |

## CP2K twin (offline only)

When a CP2K binary is available: phonon / DFPT at fractional q ≈ (0.3,0,0)
on the same cell (or closest commensurate mesh), record ω [cm⁻¹] + wall in
`tools/cp2k_oracle/inputs/dfpt-incommensurate-si.note`.
