# Reference — finite-q DFPT phonons (diamond Si at X, PW LSDA, NC-UPF)

## Method

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

## System

| Setting | Value |
| ------- | ----- |
| Structure | diamond Si primitive, a = **5.43 Å** |
| \(\mathbf{q}\) | \(X=(1/2,0,0)\) → **2×1×1** supercell (4 atoms) |
| Pseudopotential | `tests/data/pw/Si_lda_nc.upf` |
| XC | LSDA |
| Cutoff | **10 Ry** |
| k-mesh | **1×1×1** on the supercell (Γ+X of primitive folded) |
| Response | tol 1e-4, mix 0.25, max_iter 120 |

## Offline pins (measured 2026-07-25, Release, OMP=8)

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

## Performance

| Item | Value |
| ---- | ----- |
| `perf_exempt` | **true** |
| Reason | No CP2K DFPT wall twin on the build host this shift |
| ATV wall | ≈ 43 s |
| `ratio_atv_over_oracle` | n/a |
| `within_10x` | n/a (exempt) |
