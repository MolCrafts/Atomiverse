# Reference — phonon dispersion q-path (diamond Si, SW IFC Fourier)

## Method

Interatomic force constants Φ of a **2×2×2 primitive supercell** of diamond
Si are assembled by finite-displacement central differences
(`cpu::vib::Phonon` + Stillinger–Weber). The mass-weighted dynamical matrix
of the **2-atom primitive cell** at fractional wavevector **q** is the
Fourier sum

\[
D(\mathbf{q})
  = \frac{1}{\sqrt{m_i m_j}}
    \sum_{\mathbf{R}}
    \Phi(0i,\,\mathbf{R}j)\,e^{i\mathbf{q}\cdot\mathbf{R}},
\]

with Φ converted eV/Å² → Hartree/Bohr² using the same NIST factors as
`MassWeighting` / `PhononDiagonalizer`. Eigenvalues → ω [cm⁻¹]. Public path:
`cpu::vib::PhononDispersion::compute(hessian, unit, na, nb, nc)`.

This is the phonopy-style **supercell IFC Fourier** product surface for
full dispersions (q ≠ 0). Finite-q DFPT Sternheimer remains a follow-on.

Cite: Stillinger & Weber, *Phys. Rev. B* **31**, 5262 (1985) (potential);
Baroni et al., *Rev. Mod. Phys.* **73**, 515 (2001) (DFPT context);
phonopy / Togo–Tanaka for the Fourier IFC construction.

## System (regression twin)

| Setting | Value |
| ------- | ----- |
| Structure | diamond Si, **conventional cubic 8-atom** cell, a = **5.431 Å** |
| Force engine | `StillingerWeber` / `SWPotential(Si, "stillinger-weber")` |
| Supercell | **2×2×2** conventional (64 atoms) for IFCs — same SW twin as `fd-phonon-supercell` |
| FD step | \(h = 10^{-3}\) Å (`kHessianStepAngstrom`) |
| q-path | conventional-cubic **Γ → X → L** (fractional): (0,0,0), (½,0,0), (½,½,½) + midpoints |
| Masses | IUPAC-2013 via `cpu::vib::mass_amu(Z)` |
| Branches | \(3N = 24\) (conventional cell) |

**Why conventional, not 2-atom primitive:** the classical neighbor cell list
takes `Box::length(d) = cell(d,d)`, which is **0** for the diamond primitive
triclinic cell and can infinite-loop in orthorhombic min-image helpers. The
Fourier API itself is cell-agnostic; cubic conventional is the proven SW
geometry from `fd-phonon-supercell`. (Triclinic neighbor fix is out of scope
for this slug.)

## Offline pins (no live CP2K at regression time)

| Field | Value |
| ----- | ----- |
| Preferred oracle | CP2K `VIBRATIONAL_ANALYSIS` + phonon band structure on the same SW (or GTH) cell |
| CP2K binary this host | **not available** (no CP2K module on the pin host this shift) |
| Pin 1 (Γ acoustic) | \(\max|\omega_\mathrm{ac}(\Gamma)| < 30\) cm⁻¹ (no ASR projection) — same bound as `fd-phonon-supercell` / `dfpt-phonon-si` |
| Pin 2 (Γ optical) | \(\omega_\mathrm{max}(\Gamma) \in [500, 900]\) cm⁻¹ — SW Si optical band (same window as `reg_fd-phonon-supercell`) |
| Pin 3 (X, L) | \(\omega_\mathrm{max}(X),\omega_\mathrm{max}(L) \in [300, 900]\) cm⁻¹ — finite optical/acoustic zone-boundary band for SW Si |

### Measured pins (this host, 2026-07-25, Release, OMP=4)

| Quantity | Value |
| -------- | ----- |
| Γ \|acoustic\|_max | **0.00** cm⁻¹ |
| Γ ω_max (optical) | **594.82** cm⁻¹ |
| X ω_max | **562.56** cm⁻¹ |
| L ω_max | **559.27** cm⁻¹ |
| Branches / path length | 24 / 5 |
| Regression wall | **0.07 s** (≤ 60 s budget) |

### Physics notes

- SW is short-ranged (cutoff \(a\sigma \approx 3.77\) Å). The 2×2×2
  conventional supercell (L = 10.862 Å > 2×cutoff) is the same geometry as
  `reg_fd-phonon-supercell`; IFC support is complete for this twin.
- True experimental Si LO(Γ) ≈ 520 cm⁻¹ is **not** the pin — SW is classical
  and systematically stiffer; the window is the same SW band used by the FD
  Γ foothold.
- Γ acoustic near-zero is a hard correctness gate (broken ASR → broken Φ).
- At Γ, the Fourier spectrum of the conventional cell must recover the same
  acoustic/optical physics as the supercell-Γ FD spectrum of the 64-atom cell
  (zone-folded branches sit among the 24 modes).

## Performance

| Item | Value |
| ---- | ----- |
| `perf_exempt` | **true** |
| Reason | No CP2K binary/module on the build host for a fair VIBRATIONAL_ANALYSIS wall twin this shift |
| ATV wall (informative) | same FD cost class as `fd-phonon-supercell` (64-atom SW, 6N displacements) + cheap Fourier path; unit test baseline ~30 ms for FD alone; expect ≪ 60 s |
| `ratio_atv_over_oracle` | n/a (exempt) |
| `within_10x` | n/a (exempt; correctness gates only) |

## CP2K twin (offline only)

When a CP2K binary is available: run Quickstep / classical SW vibrational
analysis for diamond Si at a = 5.431 Å, record ω(Γ/X/L) [cm⁻¹] + wall with
the CP2K version, and optionally the fair wall ratio
\(r = T_\mathrm{ATV}/T_\mathrm{CP2K}\) (require ≤ 10).
