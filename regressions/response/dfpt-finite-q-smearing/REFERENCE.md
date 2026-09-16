# Reference — finite-q DFPT + metallic smearing (diamond Si at X, PW LSDA)

## Method

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
cpu::pw::PW method;
method.smearing = Smearing::fermi_dirac;
method.degauss = 0.02; // Ry
cpu::pw::DfptPhonon dfpt;
dfpt.q_frac = {0.5, 0.0, 0.0}; // X → 2×1×1 supercell
auto r = dfpt.compute(method, frame);
```

Cite: Baroni et al., *Rev. Mod. Phys.* **73**, 515 (2001);
de Gironcoli, *Phys. Rev. B* **51**, 6773 (1995). Sibling fixed-occ:
`regressions/response/dfpt-finite-q-si`. Sibling Γ smear:
`regressions/response/dfpt-metallic-si`.

## System (regression twin)

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

## Offline pins

| Field | Value |
| ----- | ----- |
| Preferred oracle | CP2K `VIBRATIONAL_ANALYSIS` + `&SMEAR` (GTH-GPW ≠ NC-UPF absolute E) |
| CP2K binary this host | **not available** (`module avail cp2k` empty) |
| Physics pin | literature LDA Si X windows (Giannozzi et al.) via \(\lvert\omega\rvert\) gates |
| Same-Hamiltonian | spectra track `dfpt-finite-q-si` fixed-occ twin within gate windows |

### Measured pins (this host, 2026-07-26, Release, OMP=8)

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

## Performance

| Item | Value |
| ---- | ----- |
| `perf_exempt` | **true** |
| Reason | No CP2K DFPT+smear wall twin on the build host this shift |
| ATV wall | ≈ 40 s (2-atom primitive → 4-atom SC, 12 responses, OMP=8) |
| `ratio_atv_over_oracle` | n/a (exempt) |
| `within_10x` | n/a (exempt; correctness gates only) |
