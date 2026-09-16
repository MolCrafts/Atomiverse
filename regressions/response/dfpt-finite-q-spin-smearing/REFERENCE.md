# Reference — finite-q DFPT collinear spin + metallic smearing (diamond Si, PW LSDA)

## Method

Density-functional perturbation theory at fractional phonon wavevector
\(\mathbf{q}\) for `cpu::pw::PW` with **collinear spin** and **Fermi–Dirac
metallic occupation smearing**.

| Path | Product |
|------|---------|
| Commensurate (e.g. \(X\)) | supercell Γ-DFPT (spin+smear Sternheimer) + IFC Fourier |
| **Direct** (this regression) | `dfpt_q_spin_impl`: shared \(\mu\), spinor \(f\in[0,1]\), projector \(f\ge 10^{-8}\) at \(\mathbf{k}+\mathbf{q}\); **no** \(\delta\mu\) at \(q\neq 0\) |

```cpp
cpu::pw::PW method;
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

## System (regression twin)

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

## Offline pins

| Field | Value |
| ----- | ----- |
| Preferred oracle | CP2K phonon / DFPT + LSD + `&SMEAR` (GTH-GPW ≠ NC-UPF absolute E) |
| CP2K binary this host | **not available** (`module avail cp2k` empty) |
| Physics pin | same \|ω\| windows as `dfpt-spin-incommensurate` / `dfpt-incommensurate-si` |
| Same-Hamiltonian | spectra match fixed-occ spin incommensurate twin to printed precision |

### Measured pins (this host, 2026-07-26, Release, OMP=8)

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

## Performance

| Item | Value |
| ---- | ----- |
| `perf_exempt` | **true** |
| Reason | No CP2K DFPT spin+smear wall twin on the build host this shift |
| ATV wall | ≈ 11 s (2-atom primitive, 6 responses × 2 spin, OMP=8) |
| `ratio_atv_over_oracle` | n/a (exempt) |
| `within_10x` | n/a (exempt; correctness gates only) |
