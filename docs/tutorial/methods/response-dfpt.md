# Plane-wave DFPT phonons

> The regression runner has been removed. Regression commands below are historical; case sources and oracle records are retained as archived evidence, not current executable checks.

Density-functional perturbation theory (DFPT) computes the phonon dynamical
matrix analytically from self-consistent first-order density response
(Sternheimer equations) instead of \(6N\) finite-displacement SCF solves. It
runs on top of the plane-wave method `cpu::pw::PW`
(`PW{std::make_shared<UpfPseudopotential>(path), ecut_Ry, mesh, cpu::scf::XCType}` plus public
members `tol`, `max_iter`, `mix_beta`, `spin_polarized`, `multiplicity`,
`smearing`, `degauss` [Ry], `extra_bands`). The calculator is
`cpu::pw::DfptPhonon` — public members `response_tol`, `response_max_iter`,
`cg_tol`, `cg_max_iter`, `mix_beta`, `q_frac` (default Γ) — and
`dfpt.compute(method, frame)` returns a `cpu::pw::DfptPhononResult` with
wavenumbers [cm⁻¹] and force constants [eV/Å²]. No `*Config` bag, no
factories; the input Frame is not mutated.

The sections pin diamond Si on the CPU backend across every axis the
calculator supports: wavevector (Γ, commensurate X via a supercell,
incommensurate \(\mathbf{q}=(0.3,0,0)\) via direct Sternheimer), XC kernel
(LSDA, pure-GGA PBE), spin (restricted, collinear \(M=0\)) and occupations
(fixed, Fermi–Dirac smearing). Finite-displacement crystal phonons and IFC
dispersions are on [response-phonons.md](response-phonons.md#fd-phonon-supercell).

Case: `ctest -R '^response/dfpt$'` — sources and oracle record in [`regressions/response/dfpt/`](../../../regressions/response/dfpt/). One section of the case per `##` heading below: `atv_regressions --case response/dfpt --section <name>`.
Every section requires the PW module (`ATV_BUILD_PW=ON`, `ATV_BUILD_REGRESSIONS=ON`;
standard CPU configure in module environment `buildenv-gcc/2026.03-mpich`).

## Common setup

### Theory

For each Cartesian ionic displacement direction at wavevector \(\mathbf{q}\),
solve the Sternheimer equation in the conduction space

\[
P_c\,(H - \varepsilon)\,P_c\,|\Delta\psi\rangle
=
-P_c\,\Delta V_{\mathrm{ext+Hxc}}\,|\psi\rangle,
\]

with conduction-space projected CG per \((k,\text{band})\), and screen the
induced density self-consistently with Hartree + XC kernel

\[
\Delta V_{\mathrm{Hxc}}[\Delta n]
=
\Delta V_{\mathrm{H}}[\Delta n]
+
K_{\mathrm{xc}}[\Delta n].
\]

The Cartesian force constants \(C_{I\alpha,J\beta}\) combine the electronic
response with analytic Ewald second derivatives and KB nonlocal first/second
derivative terms; they are Hermitian-symmetrized, mass-weighted (IUPAC masses
by \(Z\), same convention as `cpu::vib`) and diagonalized to \(\omega\) [cm⁻¹]
via `nist::au2wavenumber()`, ascending, imaginary eigenvalues reported as
**negative** \(\omega\). Internal assembly uses Ha/Bohr²; public force
constants are eV/Å². The acoustic sum rule is **never imposed** — residual
acoustic \(|\omega|\) and force-constant row sums are validation observables.
Reference: Baroni, de Gironcoli, Dal Corso, Giannozzi, *Rev. Mod. Phys.*
**73**, 515 (2001).

**Wavevector path** (selected inside `DfptPhonon::compute` from `q_frac`):

| `q_frac` | Engine path |
| -------- | ----------- |
| \(\approx\mathbf{0}\) | Γ Sternheimer (`internal::dfpt_gamma_force_constants`) |
| rational, \(n_a n_b n_c\le 8\), \(n_i\le 6\) | supercell Γ-DFPT + IFC Fourier |
| otherwise | **direct** `internal::dfpt_q_force_constants` on the primitive cell |

A phonon at rational
\(\mathbf{q} = \tfrac{h}{n_a}\mathbf{b}_1 + \tfrac{k}{n_b}\mathbf{b}_2 + \tfrac{l}{n_c}\mathbf{b}_3\)
is exactly a Γ phonon of the \(n_a\times n_b\times n_c\) supercell folded back
by \(e^{i\mathbf{q}\cdot\mathbf{R}}\), so the commensurate path is
analytically equivalent to direct finite-\(q\) Sternheimer for Born–von Karman
commensurate \(\mathbf{q}\):

1. Resolve the smallest positive \(n_a,n_b,n_c\le 6\) with product \(\le 8\)
   such that \(n_a q_a\in\mathbb{Z}\) (etc.).
2. Expand the primitive `Frame` with `vib::expand_supercell`.
3. Run the Γ assembly on the supercell (same response tolerances).
4. Fourier-transform the supercell IFC with
   `PhononDispersion::frequencies_at_q` at `q_frac` of the **primitive** cell.
5. Mass-weight and report the primitive \(3N\) branches.

The direct path solves finite-\(q\) Sternheimer on the primitive cell with
Bloch phases in the Ewald second derivatives (+ ASR) and returns a complex
Hermitian \(C(\mathbf{q})\); the response density uses the single-\(q\)
convention (the conjugate sector lives at \(-\mathbf{q}\); no extra TR factor
of 2). With `spin_polarized` it runs `dfpt_q_spin_impl`.

**XC kernel.** Pure LDA keeps a local \(f_{\mathrm{xc}}\) table. Pure GGA
(`XCType::PBE` and siblings with `hybrid_exchange_fraction == 0`) applies
\(K_{\mathrm{xc}}\) by central finite difference of the ground-state
\(V_{\mathrm{xc}}[n]\) from the same spectral-gradient `XCBuilder` path used in
SCF (spectral \(\nabla\rho\) + GGA flux; true GGA, not \(\sigma=0\)) along the
induced density direction (`add_xc_kernel_gamma`,
`src/cpu/planewave/internal/dfpt.cpp`). Hybrids are rejected (no
exact-exchange response).

**Collinear spin.** With `spin_polarized = true` the ground state is dual-spin
KS (\(n_{\mathrm{spin}}=2\)), each Sternheimer response is spin-resolved
(\(f=1\) per spin-orbital at fixed occupation), and the induced spin densities
are screened jointly,

\[
\Delta V_{\mathrm{Hxc},\sigma}[\Delta n_\uparrow,\Delta n_\downarrow]
  = \Delta V_{\mathrm{H}}[\Delta n]
  + \sum_{\sigma'} K_{\mathrm{xc}}^{\sigma\sigma'}[\Delta n_{\sigma'}],
\qquad \Delta n = \Delta n_\uparrow + \Delta n_\downarrow ,
\]

with the local collinear \(f_{\mathrm{xc}}^{\sigma\sigma'}\) table for LSDA or
central FD of `XCBuilder::step_spin` for pure GGA (`apply_xc_kernel_spin_fd`,
`dfpt_gamma_spin_impl` on Γ/supercell, `dfpt_q_spin_impl` on the direct
path). All spin sections use `multiplicity = 1`, so
\(N_\uparrow = N_\downarrow = 4\) for diamond Si (8 valence electrons): the
collinear framework (two spin channels, spin XC kernel, spin density mixing)
is fully exercised on a non-magnetic insulator, and at \(M=0\) the spin result
must reproduce its restricted twin.

**Metallic occupations.** With `smearing` set (Fermi–Dirac / Gaussian /
Methfessel–Paxton, width `degauss` [Ry]) the ground state minimizes the
Mermin free energy \(F = E_{\mathrm{KS}} - TS\) (same machinery as
[pw-smearing-al](pw-smearing.md#pw-smearing-al)). With reduced energy
\(x = (\varepsilon-\mu)/\sigma\), restricted band occupations
\(f_b\in[0,2]\) double the spinless fill; spin-orbital occupations are
\(f_{\sigma nk}\in[0,1]\) with a shared \(\mu\). The response density is
occupation-weighted,

\[
\Delta n_{(\sigma)}
  = \sum_{nk} w_k\, f_{(\sigma) nk}\,
    2\,\mathrm{Re}\bigl(\psi_{(\sigma) nk}^*\,\Delta\psi_{(\sigma) nk}\bigr),
\]

and the Sternheimer projector (at \(\mathbf{k}+\mathbf{q}\)) removes bands with
\(f \ge 10^{-8}\). At Γ, when \(\partial N/\partial\mu\) is non-negligible, a
Fermi-level shift \(\delta\mu\) restores \(\int\Delta n = 0\) (de Gironcoli,
*Phys. Rev. B* **51**, 6773 (1995)); there is **no** \(\delta\mu\) at
\(q\neq 0\). `extra_bands` pads empty states above the valence manifold (open
metals); every smearing section uses `extra_bands = 0` because gapped Si keeps
near-integer occupations at these settings, so \(F \approx E_{\mathrm{KS}}\)
and each smeared section tracks its fixed-occupation twin. Cold /
Marzari–Vanderbilt smearing is not gated by any DFPT section (Fermi–Dirac is
the e2e path).

**Guards.** `compute` throws `std::invalid_argument` unless `method.xc` is pure
LSDA or pure GGA (`hybrid_exchange_fraction == 0`) and `method.upf_path` is set
(the empty-`upf_path` toy \(V_\mathrm{loc}\) is rejected). Restricted
smearing is allowed at Γ, commensurate \(\mathbf{q}\) and direct finite-\(q\);
collinear spin, with fixed occupations or smearing, is allowed at Γ,
commensurate and incommensurate \(\mathbf{q}\).

### Observables

All on `DfptPhononResult`, never on Frame (Frame purity; a Frame energy from a
separate `PW` run stays eV on `kEnergy`).

| Symbol | Member | Unit |
| ------ | ------ | ---- |
| \(\omega_k\) | `frequencies` (length \(3N\), ascending) | cm⁻¹ |
| \(C\), \(\mathrm{Re}\,C(q)\) | `force_constants`, shape \([3N,3N]\), index `3·I + α` | eV/Å² |
| \(\mathrm{Im}\,C(q)\) | `force_constants_imag` (≈0 at Γ) | eV/Å² |
| \(\max\lvert C-C^\dagger\rvert/2\) | `max_asymmetry` (before Hermitian projection) | eV/Å² |
| — | `scf_converged`, `response_converged` (GS + all \(3N\) responses) | bool |

### Common system and code

| Setting | Value |
| ------- | ----- |
| Structure | diamond Si, 2-atom fcc primitive, \(a = 5.43\) Å (experimental) |
| Basis positions [Å] | \((0,0,0)\), \((a/4,a/4,a/4)\); mass 28.0855 amu |
| Pseudopotential | `tests/data/pw/Si_lda_nc.upf` (ONCVPSP NC, in-tree; LDA-generated, used as fixed valence PP for PBE too) |
| Cutoff | **10 Ry** (minimal-twin budget) |
| k-mesh | Monkhorst–Pack, unshifted: **2×2×2** at Γ; **1×1×1** on the supercell (X) and on the primitive cell (incommensurate) |
| SCF | density L2 tol \(10^{-8}\), `max_iter = 200` (incommensurate spin sections: tol \(10^{-6}\), mix 0.4) |
| Response (Γ) | `response_tol = 10^{-5}`, `mix_beta = 0.7`, CG tol \(10^{-9}\) |
| Response (X) | `response_tol = 10^{-4}`, `mix_beta = 0.25`, CG tol \(10^{-8}\), max_iter 120 |
| Response (\(q=(0.3,0,0)\)) | `response_tol = 10^{-4}`, `mix_beta = 0.2`, CG tol \(10^{-8}\), max_iter 120 |
| Smearing (where used) | Fermi–Dirac, `degauss = 0.02` Ry, `extra_bands = 0` |

Shared frame and calculator (each section lists only what it changes):

```cpp
using namespace ATV;
using namespace ATV::cpu::pw;

constexpr double kA = 5.43;  // Å
cpu::Frame frame(2);
frame.span(kAtom_atomic_number)[0] = frame.span(kAtom_atomic_number)[1] = 14;
frame.span(kAtom_mass)[0] = frame.span(kAtom_mass)[1] = 28.0855;
frame.span(kAtom_x)[1] = frame.span(kAtom_y)[1] = frame.span(kAtom_z)[1] = kA / 4.0;
const double h = kA / 2.0;
Mat3 cell;  // fcc primitive: (0,h,h), (h,0,h), (h,h,0)
cell(0, 0) = 0; cell(0, 1) = h; cell(0, 2) = h;
cell(1, 0) = h; cell(1, 1) = 0; cell(1, 2) = h;
cell(2, 0) = h; cell(2, 1) = h; cell(2, 2) = 0;
frame.box.set_cell(cell);

PW method{std::make_shared<UpfPseudopotential>("tests/data/pw/Si_lda_nc.upf"),
          10.0,                      // [Ry]
          2,                         // 2×2×2 MP (1 for X / incommensurate)
          cpu::scf::XCType::LSDA};   // or XCType::PBE
method.tol = 1e-8;
method.max_iter = 200;

DfptPhonon dfpt;                     // q_frac default {0,0,0} = Γ
dfpt.response_tol = 1e-5;
dfpt.cg_tol = 1e-9;
dfpt.mix_beta = 0.7;
const DfptPhononResult r = dfpt.compute(method, frame);
// r.frequencies [cm⁻¹], length 3N; r.force_constants / _imag [eV/Å²];
// r.scf_converged, r.response_converged, r.max_asymmetry
```

Headers: public `dfpt_phonon.h`, `pw_solver.h` (smear/spin members); private
assembly `cpu/planewave/internal/dfpt.h`, `src/cpu/planewave/internal/dfpt.cpp`
(Γ Sternheimer assembly + Ewald), `src/cpu/planewave/dfpt_phonon.cpp`
(Γ / commensurate / incommensurate dispatch), `smearing.*`;
`phonon_dispersion.h` (IFC Fourier). The Ewald real-space lattice sum in
`EwaldSummation` and the DFPT Ewald Hessians builds
\(L = h\mathbf{a}+k\mathbf{b}+l\mathbf{c}\) from the lattice vectors, not the
transposed cell matrix (harmless for the symmetric diamond primitive, fatal
for non-symmetric supercells).

Recorded properties, with no unit test in the tree gating them: analytic Ewald
dynamical matrix vs FD of Ewald forces (≤ 5e-6 Ha/Bohr²), Sternheimer CG exact
on the free-electron Hamiltonian, pure-GGA acceptance / hybrid reject, spin +
smear / finite-\(q\) acceptance, collinear + incommensurate \(q\) reaching UPF
I/O (an empty UPF throws).

### Oracle

- **Sharp same-Hamiltonian pin (Γ and X sections).** The diagonal force
  constant \(C[(0x),(0x)]\) is compared with the offline central second
  difference of the SCF total (free) energy at identical ecut/k/XC/UPF/spin/FD
  settings,
  \[
  C[(0x),(0x)] = \frac{E(+h)-2E_0+E(-h)}{h^2},\qquad h = 0.03\,\text{Å},
  \]
  hard-coded in the regression with tolerance **0.30** eV/Å². This is the true
  discrete Hessian; the analytic `PW::forces` gradient deviates from it by
  ≈ 2 % (egg-box/aliasing), so FD-of-forces is *not* the oracle.
- **Literature windows.** DFPT-LDA / Raman Si optical Γ ≈ 517 cm⁻¹ and LDA
  X-point TA≈146, LA≈397, LO≈413, TO≈455 cm⁻¹ (Giannozzi, de Gironcoli,
  Pavone, Baroni, *Phys. Rev. B* **43**, 7231 (1991)). The minimal twin
  (10 Ry, small k-mesh, experimental \(a\)) biases optical \(\omega(\Gamma)\)
  ~+12 % high; the windows reject unit disasters and wrong path selection,
  they do not claim production-converged frequencies.
- **CP2K.** The preferred external oracle is offline CP2K Quickstep
  `VIBRATIONAL_ANALYSIS` / DFPT on the same cell (never live CP2K in ctest).
  The PBE spin sections carry a real CP2K 2026.1 FD-vib anchor and a fair wall
  ratio \(r = T_{\mathrm{ATV}}/T_{\mathrm{CP2K}} \le 10\). The other sections
  are `perf_exempt` (no CP2K binary/module on their pin host,
  `module avail cp2k` empty): when a binary is available, record \(\omega\)
  [cm⁻¹], version and wall in `REFERENCE.md` and require \(r\le 10\) on the
  same host/core count. The CP2K smearing sketches are **not**
  absolute-\(\omega\)/\(E\) twins: GTH-GPW and NC-UPF PW are different
  Hamiltonians; they document that CP2K can combine `VIBRATIONAL_ANALYSIS`
  with LSD and `&SMEAR METHOD FERMI_DIRAC` on a Si-like solid.

### Section matrix

| Section | \(\mathbf{q}\) | XC | Spin | Smear | ATV wall | CP2K \(r\) |
| ------- | -------------- | -- | ---- | ----- | -------: | ---------- |
| [dfpt-phonon-si](#dfpt-phonon-si) | Γ | LSDA | — | — | ≈ 25 s | exempt |
| [dfpt-gga-si](#dfpt-gga-si) | Γ | PBE | — | — | ≈ 15 s | exempt |
| [dfpt-spin-si](#dfpt-spin-si) | Γ | LSDA | ✓ | — | ≈ 34 s | exempt |
| [dfpt-gga-spin-si](#dfpt-gga-spin-si) | Γ | PBE | ✓ | — | 44.7 s | 0.20 |
| [dfpt-metallic-si](#dfpt-metallic-si) | Γ | LSDA | — | FD | ≈ 16.5 s | exempt |
| [dfpt-spin-smearing](#dfpt-spin-smearing) | Γ | LSDA | ✓ | FD | ≈ 35 s | exempt |
| [dfpt-gga-spin-smearing](#dfpt-gga-spin-smearing) | Γ | PBE | ✓ | FD | 48.3 s | 0.21 |
| [dfpt-incommensurate-si](#dfpt-incommensurate-si) | (0.3,0,0) | LSDA | — | — | ≈ 7 s | exempt |
| [dfpt-spin-incommensurate](#dfpt-spin-incommensurate) | (0.3,0,0) | LSDA | ✓ | — | ≈ 11 s | exempt |
| [dfpt-finite-q-si](#dfpt-finite-q-si) | X | LSDA | — | — | ≈ 43 s | exempt |
| [dfpt-gga-finite-q-si](#dfpt-gga-finite-q-si) | X | PBE | — | — | 49.8 s | 0.17 |
| [dfpt-finite-q-smearing](#dfpt-finite-q-smearing) | X | LSDA | — | FD | ≈ 40 s | exempt |
| [dfpt-finite-q-spin-smearing](#dfpt-finite-q-spin-smearing) | (0.3,0,0) | LSDA | ✓ | FD | ≈ 11 s | exempt |
| [dfpt-gga-spin-finite-q-si](#dfpt-gga-spin-finite-q-si) | X | PBE | ✓ | — | 107.9 s | 0.19 |
| [dfpt-gga-spin-finite-q-smearing](#dfpt-gga-spin-finite-q-smearing) | X | PBE | ✓ | FD | 148.1 s | 0.25 |

### Common limitations

- **Minimal twin.** 10 Ry / small k-mesh / experimental lattice / LDA-generated
  NC UPF: optical \(\omega\) sits ~12 % above the experimental Raman TO(Γ) and
  the X / incommensurate spectra are biased; that offset is settings bias, not
  the correctness gate. Production dispersions need denser settings outside the
  ≤ 60 s e2e budget.
- **Gapped Si, not an open metal.** Smearing sections exercise the
  occupation-weight machinery near the integer-fill limit; true multi-band
  metals (Fermi-surface terms on Al/Fe) need denser k/ecut, `extra_bands > 0`
  and a separate pin (ground-state smearing:
  [pw-smearing-al](pw-smearing.md#pw-smearing-al),
  [pw-uks-fe-smearing](pw-smearing.md#pw-uks-fe-smearing)).
- **Closed-shell collinear only** (\(M=0\)); open-shell magnetic phonons are
  not pinned.
- No LO–TO splitting / Born effective charges (nonpolar Si does not need them).
- No hybrid / meta-GGA / GAPW kernels, no noncollinear / SOC response, no GPW
  DFPT.
- CPU only; no CUDA DFPT.

### References

- S. Baroni, S. de Gironcoli, A. Dal Corso, P. Giannozzi, *Rev. Mod. Phys.*
  **73**, 515 (2001).
- P. Giannozzi, S. de Gironcoli, P. Pavone, S. Baroni, *Phys. Rev. B* **43**,
  7231 (1991) (DFPT-LDA Si optical Γ ~517 cm⁻¹; X-point and mid-ΓX bands).
- J. P. Perdew, K. Burke, M. Ernzerhof, *Phys. Rev. Lett.* **77**, 3865 (1996)
  (PBE sections).
- S. de Gironcoli, *Phys. Rev. B* **51**, 6773 (1995) (metallic DFPT,
  \(\delta\mu\)).
- N. D. Mermin, *Phys. Rev.* **137**, A1441 (1965) (finite-temperature free
  energy).
- Roadmap: [dft.md](../../roadmap/dft.md) §11 DFPT.
- Background: [ch08-plane-wave](../ch08-plane-wave.md); ground-state PBE twin
  [pw-pbe-gga-si](pw-rks.md#pw-pbe-gga-si); FD crystal phonons
  [fd-phonon-supercell](response-phonons.md#fd-phonon-supercell) and IFC
  dispersion [phonon-dispersion-si](response-phonons.md#phonon-dispersion-si).

## dfpt-phonon-si — Γ phonons, restricted LSDA {#dfpt-phonon-si}

Γ-point DFPT on the restricted LSDA ground state with fixed occupations: the
reference point every other section varies. Settings are exactly the shared
code (LSDA, 2×2×2 MP, Γ response tolerances). The primitive cell uses the
experimental \(a = 5.43\) Å (the SW phonon twins use 5.431 Å).

### Accuracy

| Quantity | Atomiverse / pin | Gate | Notes |
| -------- | ---------------: | ---- | ----- |
| Energy-FD \(C[(0x),(0x)]\) [eV/Å²] | **17.2393** (hard-coded) | oracle | \(h = 0.03\) Å, SCF tol 1e-9 |
| DFPT \(C[(0x),(0x)]\) [eV/Å²] | **17.2277** | \(\lvert\Delta\rvert \le 0.30\) | **0.07 %** |
| Acoustic \(\lvert\omega\rvert\) [cm⁻¹] | ≤ 4.4 | ≤ 30 | ASR emerges, never imposed |
| Max force-constant row sum [eV/Å²] | ≤ 0.0022 | ≤ 0.02 | translational invariance |
| Optical triplet [cm⁻¹] | **575.8–578.5** | each in [500, 650] | +12 % over 517 = minimal-settings bias |
| Optical degeneracy spread [cm⁻¹] | 2.7 | ≤ 10 | |
| Pre-symmetrization asymmetry [eV/Å²] | — | ≤ 1e-6 | Hermitian diagnostic |
| Wall | ≈ 25 s | ≤ 60 s | |

## dfpt-gga-si — Γ phonons, pure GGA (PBE) kernel {#dfpt-gga-si}

Extends the Γ path beyond the LSDA kernel to a **true pure-GGA XC kernel**
(PBE): \(K_{\mathrm{xc}}\) is the second functional derivative of the same
`XCBuilder` operator used in the ground-state SCF. Restricted, fixed
occupations, closed-shell insulator — the same Hamiltonian family as the
ground-state twin [pw-pbe-gga-si](pw-rks.md#pw-pbe-gga-si) (true PBE on the
FFT grid).

```cpp
PW method{std::make_shared<UpfPseudopotential>("tests/data/pw/Si_lda_nc.upf"),
          10.0, 2, cpu::scf::XCType::PBE};   // pure GGA kernel
```

### Accuracy

Measured pins (Release, OMP=8):

| Quantity | Atomiverse / pin | Gate | Notes |
| -------- | ---------------: | ---- | ----- |
| Energy-FD \(C[(0x),(0x)]\) [eV/Å²] | **17.3456** (hard-coded) | oracle for DFPT \(C_{00}\) | \(h=0.03\) Å; \(E_0=-15.6411877416\) Ry; \(E_\pm=-15.6406136403/-15.6406144498\) Ry |
| DFPT \(C[(0x),(0x)]\) [eV/Å²] | **17.3711** | \(\|C_{00}-17.3456\| \le 0.30\) (~0.15 %) | true discrete Hessian twin |
| Acoustic \(\omega\) [cm⁻¹] | 12.08, 12.08, 12.09 | \(\|\omega\| \le 30\) | no ASR imposition |
| Optical triplet [cm⁻¹] | **579.82, 579.82, 579.96** | each \(\in [450,650]\); spread \(\le 10\) | lit. Raman ~517 cm⁻¹; minimal-settings bias |
| Max force-constant row sum [eV/Å²] | 0.0151 | \(\le 0.05\) | translational invariance |
| Pre-symmetrization \(\max\|C-C^T\|/2\) [eV/Å²] | \(9.6\times10^{-10}\) | \(\le 10^{-5}\) | Hermitian diagnostic |
| Wall | ≈ 15 s | ≤ 60 s e2e budget | 2 atoms, 6 responses, 8 k |

### Performance

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | ------------------------: | ----------------------: | ----: | ---- | ------ |
| Γ DFPT Si PBE (2-atom, 10 Ry, 2×2×2) | n/a | ≈ 15 | n/a | n/a | **`perf_exempt`** — no CP2K binary on pin host |

## dfpt-spin-si — Γ phonons, collinear spin, LSDA {#dfpt-spin-si}

Γ-point DFPT with `method.spin_polarized = true` and the collinear LSDA
\(f_{\mathrm{xc}}^{\sigma\sigma'}\) kernel, on the same NC-UPF / 10 Ry /
2×2×2 twin as the restricted gate, so the spin kernel is checked against a
same-Hamiltonian energy second difference without changing the physical
system. Optical frequencies land in the same LDA window as restricted LSDA.

```cpp
method.spin_polarized = true;   // collinear DFPT path
method.multiplicity = 1;        // M = 0
```

### Accuracy

Measured twin (Release, `OMP_NUM_THREADS=8`):

| Quantity | Atomiverse / pin | External / rule | Tol / band | Notes |
| -------- | ---------------: | --------------- | ---------- | ----- |
| Energy-FD \(C[(0x),(0x)]\) | **17.3390 eV/Å²** | same-Ham. \((E(+h)-2E_0+E(-h))/h^2\), \(h=0.03\) Å, spin SCF | pin definition | \(E_0=-7.805522314842\) Ha |
| DFPT \(C[(0x),(0x)]\) | **17.3415 eV/Å²** | energy-FD pin | \(\lvert\Delta\rvert\le\mathbf{0.30}\) (~0.015 %) | sharp correctness gate |
| Acoustic \(\omega\) (no ASR) | 1.33, 1.34, 1.34 cm⁻¹ | Γ acoustic sum rule | \(\lvert\omega\rvert\le\mathbf{30}\) cm⁻¹ | ASR never imposed |
| Optical triplet | **579.45, 579.45, 579.59 cm⁻¹** | LDA / Raman window (~517 lit.) | **[500, 650]** cm⁻¹ | +~12 % minimal-settings bias |
| Optical spread | 0.14 cm⁻¹ | degeneracy | \(\le\mathbf{10}\) cm⁻¹ | |
| Max row sum (ASR) | 0.0002 eV/Å² | translational invariance | \(\le\mathbf{0.05}\) | |
| Pre-sym \(\lvert C-C^\top\rvert/2\) | \(7.3\times10^{-10}\) eV/Å² | Hermitian \(C\) | \(\le\mathbf{10^{-5}}\) | |

The restricted LSDA section [dfpt-phonon-si](#dfpt-phonon-si) sits in the
same class (energy-FD pin 17.2393 eV/Å²; optical ~576–578 cm⁻¹); closed-shell
collinear \(M=0\) matches that optical window within the minimal-settings
bias.

### Performance

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | ------------------------: | ----------------------: | ----: | ---- | ------ |
| Si Γ collinear DFPT (10 Ry, 2×2×2) | n/a | ≈ **34** (informative) | n/a | n/a (**exempt**) | No CP2K DFPT binary on pin host |

Informative wall: 2-atom Si, 6 responses × 2 spins, 8 k-points; e2e ≤ 60 s
single-rank CPU (PASS).

## dfpt-gga-spin-si — Γ phonons, collinear spin × pure GGA {#dfpt-gga-spin-si}

The joint spin × GGA kernel at Γ: spin-resolved Sternheimer response where
\(K_{\mathrm{xc}}^{\sigma\sigma'}\) is the second functional derivative of the
**spin-GGA** operator, obtained by central FD of `XCBuilder::step_spin`
around the frozen spin ground state (spinor occupations \(f\in[0,1]\), linear
mix on spin channels). At \(M=0\) the joint kernel must reproduce the
restricted-GGA [dfpt-gga-si](#dfpt-gga-si), a consistency limit the
regression checks against both parents' pins
([dfpt-spin-si](#dfpt-spin-si) is the LSDA spin parent).

```cpp
PW method{std::make_shared<UpfPseudopotential>("tests/data/pw/Si_lda_nc.upf"),
          10.0, 2, cpu::scf::XCType::PBE};   // pure GGA kernel
method.spin_polarized = true;              // joint spin × GGA response
method.multiplicity   = 1;
```

### Accuracy

Measured (Release, OMP=1):

| Quantity | Atomiverse / pin | Gate | Notes |
| -------- | ---------------: | ---- | ----- |
| Energy-FD \(C[(0x),(0x)]\) [eV/Å²] (spin-PBE SCF) | **17.3454** (hard-coded) | oracle for DFPT \(C_{00}\) | \(h=0.03\) Å; \(E_0=-7.820593875040\) Ha |
| DFPT \(C[(0x),(0x)]\) [eV/Å²] | **17.3711** | \(\|C_{00}-17.3454\|\le0.30\) (~0.15 %) | = restricted parent 17.3711 |
| Acoustic \(\omega\) [cm⁻¹] | 12.09 ×3 | \(\|\omega\|\le30\) | no ASR imposition |
| Optical triplet [cm⁻¹] | **579.82, 579.82, 579.96** | each \(\in[450,650]\); spread \(\le10\) | = restricted parent |
| CP2K 2026.1 anchor [cm⁻¹] | 463.44, 552.19, 552.19 | inside the same window | UKS-PBE k-point FD vib, GTH/MOLOPT twin |
| Max row sum [eV/Å²] | 0.0151 | \(\le0.05\) | translational invariance |
| Pre-symmetrization asymmetry [eV/Å²] | \(9.6\times10^{-10}\) | \(\le10^{-5}\) | Hermitian diagnostic |
| Wall | **44.7 s** | ≤ 60 s | 6 responses × 2 spins, 8 k |

Cross-parent consistency at \(M=0\): joint DFPT C00 and optical triplet are
4-decimal identical to `dfpt-gga-si`; the spin-FD energy pin 17.3454 sits
2e-4 from the restricted-FD 17.3456 — the joint kernel adds no spurious spin
response while exercising the full spin-resolved `step_spin` FD path.

### Performance

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× |
| ---- | ------------------------: | ----------------------: | ----: | ---- |
| Γ spin×PBE phonons Si (CP2K FD vib, `EPS_SCF 1e-8`, serial) | 221.8 | 44.7 | **0.20** | yes |

CP2K binary `~/.local/bin/cp2k`, CP2K 2026.1 rev `85b8a9b49c`.

## dfpt-metallic-si — Γ phonons, restricted metallic smearing {#dfpt-metallic-si}

Γ-point DFPT over a restricted PW KS ground state with fractional
(Fermi–Dirac) occupations: Mermin free energy, \(f\)-weighted response
density, Sternheimer CG in the subspace orthogonal to occupied or fractionally
occupied bands, and the de Gironcoli \(\delta\mu\) at Γ. LSDA kernel.
Sharp acceptance is the same-Hamiltonian **free-energy** central second
difference for \(C[(0x),(0x)]\), not a GTH-GPW absolute-\(E\) match to CP2K.
Gapped Si keeps near-integer FD occupations, so results track the
fixed-occupation [dfpt-phonon-si](#dfpt-phonon-si).

```cpp
method.smearing = Smearing::fermi_dirac;
method.degauss = 0.02;   // Ry
method.extra_bands = 0;  // gapped twin; open metals use > 0
// r.frequencies [cm⁻¹] length 6; r.force_constants [eV/Å²] 6×6
```

### Accuracy

Pinned on Arrhenius x86_64, Release, OMP=8:

| Quantity | Atomiverse / pin | Oracle / twin | Tol | Notes |
| -------- | ---------------: | ------------: | --- | ----- |
| Free-energy FD \(C[(0x),(0x)]\) | **17.2393 eV/Å²** | same-Hamiltonian central FD of \(F\), \(h=0.03\) Å | pin definition | identical ecut/k/XC/UPF/FD |
| DFPT \(C[(0x),(0x)]\) | **17.3416 eV/Å²** | FD pin 17.2393 | \(\lvert\Delta\rvert\le 0.30\) | ~0.6 % agreement |
| Acoustic \(\omega\) (no ASR) | −2.01, −2.00, −2.00 cm⁻¹ | 0 (translational) | \(\lvert\omega\rvert\le 30\) | emergent ASR |
| Optical triplet | **579.45, 579.45, 579.60** cm⁻¹ | LDA / Raman window ~517 cm⁻¹ | \([500,650]\), spread ≤ 10 | +12 % = minimal-settings bias (10 Ry, \(2^3\) k) |
| Max row sum (ASR) | 0.0004 eV/Å² | 0 | ≤ 0.05 | translational invariance |
| Pre-symmetrization asymmetry | \(3.4\times10^{-10}\) eV/Å² | 0 | ≤ \(10^{-5}\) | \(C\) Hermitian to roundoff |

**CP2K sketch** (not an absolute twin): `VIBRATIONAL_ANALYSIS` + `&SMEAR
METHOD FERMI_DIRAC`, diamond Si primitive \(a \approx 5.43\) Å, GPW PADE +
GTH-PADE + DZVP-MOLOPT-SR-GTH (Si q4), \(T_e \approx 3158\) K for
`degauss ≈ 0.02` Ry, `ADDED_MOS` ≥ 4, MP \(2\times2\times2\). No CP2K module
on the Arrhenius login node used for this pin.

### Performance

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | ------------------------: | ----------------------: | ----: | ---- | ------ |
| DFPT+`&SMEAR` vs `dfpt-metallic-si` | n/a | **≈ 16.5** | n/a | n/a | **`perf_exempt`** — no CP2K binary on pin host |

Informative wall: 2-atom Si, 6 responses, 8 k-points, OMP=8.

## dfpt-spin-smearing — Γ phonons, collinear spin + metallic smearing {#dfpt-spin-smearing}

Collinear spin and Fermi–Dirac smearing in one ground-state + Sternheimer
path at Γ: spin-resolved occupations \(f_{\sigma nk}\in[0,1]\),

\[
n_{\mathrm{elec}}
  = \sum_{\sigma nk} w_k\, f_{\sigma nk}
  = N_\uparrow + N_\downarrow ,
\]

a shared \(\mu\), the collinear LSDA kernel, and the optional \(\delta\mu\).
Gapped Si keeps \(f_{\sigma nk}\in\{0,1\}\), so \(F\) collapses to
\(E_{\mathrm{KS}}\) and the force constants track the collinear
fixed-occupation [dfpt-spin-si](#dfpt-spin-si); the free-energy FD Hessian
matches its energy-FD pin (**17.3390 eV/Å²**).

```cpp
method.spin_polarized = true;
method.multiplicity = 1;
method.smearing = Smearing::fermi_dirac;
method.degauss = 0.02;   // Ry
method.extra_bands = 0;  // valence Sternheimer manifold (gapped twin)
```

### Accuracy

Release, OMP=8. Sharp pin \(C[(0x),(0x)] = (F(+h)-2F_0+F(-h))/h^2\),
\(h=0.03\) Å, at identical ecut/k/XC/UPF/spin/FD.

| Quantity | Atomiverse / pin | Tol | Notes |
| -------- | ---------------: | --- | ----- |
| Free-energy FD \(C[(0x),(0x)]\) | **17.3390 eV/Å²** | — | offline central FD (sharp oracle) |
| DFPT \(C[(0x),(0x)]\) | **17.3415 eV/Å²** | \(\lvert\Delta\rvert\le 0.30\) | ~0.015 % vs FD pin |
| Acoustic \(\omega\) (no ASR) | 1.33, 1.34, 1.34 cm⁻¹ | \(\lvert\omega\rvert\le 30\) | emerges, not imposed |
| Optical triplet | **579.45, 579.45, 579.59** cm⁻¹ | \([500,650]\), spread \(\le 10\) | LDA minimal-settings window (lit. ~517 cm⁻¹ Giannozzi *et al.*) |
| Max row sum (ASR) | 0.0002 eV/Å² | \(\le 0.05\) | translational invariance |
| Pre-symmetrization \(\lvert C-C^\top\rvert/2\) | \(7.3\times10^{-10}\) eV/Å² | \(\le 10^{-5}\) | force-constant symmetry |
| Regression wall | ≈ **35 s** | \(\le 60\) s | OMP=8 Release |

**CP2K sketch** (not an absolute twin): `VIBRATIONAL_ANALYSIS` + LSD /
unrestricted (mult. 1) + `&SMEAR` FERMI_DIRAC \(\sim 0.02\) Ry \(k_B T\)
(~3158 K), `ADDED_MOS` ≥ 4; GPW, PADE/LDA, GTH-PADE + DZVP-MOLOPT-SR-GTH
(Si q4); diamond Si primitive \(a=5.43\) Å, 2 atoms, MP \(2\times2\times2\).

### Performance

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | ------------------------: | ----------------------: | ----: | ---- | ------ |
| CP2K vib+LSD+SMEAR vs `dfpt-spin-smearing` | n/a | ≈ **30–35** | n/a | **perf_exempt** | no CP2K module on pin host; correctness gates only |

Informative wall ~30 s (2-atom Si, 6 responses × 2 spins, 8 k-points, OMP=8).

## dfpt-gga-spin-smearing — Γ phonons, collinear spin × pure GGA × smearing {#dfpt-gga-spin-smearing}

The three-way Γ path: spin-resolved Sternheimer with a pure-GGA
\(K_{\mathrm{xc}}^{\sigma\sigma'}\) (FD of `XCBuilder::step_spin`) **and**
Fermi–Dirac occupation weights (shared \(\mu\)). Parents:
[dfpt-gga-spin-si](#dfpt-gga-spin-si) (spin × GGA, fixed occupations),
[dfpt-spin-smearing](#dfpt-spin-smearing) (spin × smear, LSDA),
[dfpt-gga-si](#dfpt-gga-si). Gapped fill keeps \(f\in\{0,1\}\), so numbers
match the fixed-occupation spin × GGA parent to four decimals.

```cpp
PW method{std::make_shared<UpfPseudopotential>("tests/data/pw/Si_lda_nc.upf"),
          10.0, 2, cpu::scf::XCType::PBE};
method.spin_polarized = true;
method.multiplicity = 1;
method.smearing = Smearing::fermi_dirac;
method.degauss = 0.02;
method.extra_bands = 0;
```

### Accuracy

Measured (Release, OMP=1):

| Gate | Value |
| ---- | ----- |
| Free-energy FD C00 | 17.3454 eV/Å² |
| DFPT C00 | 17.3711 eV/Å² (\|Δ\| ≤ 0.30) |
| Optical Γ | 579.82 / 579.82 / 579.96 cm⁻¹ ∈ [450, 650] |
| Acoustic | 12.09 cm⁻¹ (≤ 30) |
| Row sum / asymmetry | ≤ 0.05 eV/Å² / ≤ 1e-5 eV/Å² |
| CP2K optical anchor (UKS-PBE+SMEAR `VIBRATIONAL_ANALYSIS`) | 463.98 / 534.80 / 534.80 cm⁻¹ |

### Performance

| Item | Value |
| ---- | ----- |
| ATV | 48.3 s (serial) |
| CP2K twin | 232.5 s (1 MPI × OMP=1) |
| ratio | **0.21** (`within_10x`, not `perf_exempt`) |

### Limitations

- The finite-\(q\) three-way combination is
  [dfpt-gga-spin-finite-q-smearing](#dfpt-gga-spin-finite-q-smearing).

## dfpt-incommensurate-si — Incommensurate q = (0.3, 0, 0), restricted LSDA {#dfpt-incommensurate-si}

Direct finite-\(q\) Sternheimer on the **primitive** cell without a supercell.
For \(\mathbf{q}=(0.3,0,0)\) no integer \(n\le 6\) satisfies
\(n\cdot 0.3\in\mathbb{Z}\), so the commensurate gate fails and
`DfptPhonon::compute` runs `internal::dfpt_q_force_constants`: one
ground-state SCF, \(3N\) complex responses, Hermitian \(C(\mathbf{q})\),
Hartree + LDA \(f_{xc}\) screening, Bloch-phased Ewald + ASR, KB nonlocal
terms. Restricted LSDA, fixed occupations, 1×1×1 k-mesh.

```cpp
PW method{std::make_shared<UpfPseudopotential>("tests/data/pw/Si_lda_nc.upf"),
          10.0, 1, cpu::scf::XCType::LSDA};
method.tol = 1e-8;
method.max_iter = 200;

DfptPhonon dfpt;
dfpt.q_frac = {0.3, 0.0, 0.0};   // incommensurate along Γ–X
dfpt.response_tol = 1e-4;
dfpt.response_max_iter = 120;
dfpt.cg_tol = 1e-8;
dfpt.mix_beta = 0.2;
```

### Accuracy

No CP2K \(\omega(q)\) digits exist on the pin host, so the gate uses measured
ATV spectrum windows under the minimal twin (Release, OMP=8):

| Quantity | Atomiverse | Oracle / pin | Tol / gate | Notes |
| -------- | ---------: | -----------: | ---------- | ----- |
| SCF energy (GS) | **−14.544 Ry** | ATV measured | reported | Minimal twin |
| \(\omega(q)\) [cm⁻¹] | **692.9, 726.9, 726.9, 944.1, 1023.5, 1023.5** | ATV measured | — | Degenerate pairs (2–3), (5–6) |
| low-3 \(\lvert\omega\rvert\) | 693, 727, 727 | window | **[400, 900]** cm⁻¹ | Unit / path integrity |
| high-3 \(\lvert\omega\rvert\) | 944, 1023, 1023 | window | **[700, 1300]** cm⁻¹ | Same |
| \(n_{\mathrm{imag}}\) | **0** | — | \(\le 0\) | Hard |
| max asymmetry | ≈ **29.6** eV/Å² | — | \(\le 50\) eV/Å² | Hermitian residual |
| SCF / response | both converged | — | `true` | Hard |
| Literature mid-ΓX (LDA) | — | ~100–500 cm⁻¹ acoustic/optic band (converged) | **not** a tight pin | Twin biases **high** at 10 Ry / Γ-only \(k\) |

When CP2K DFPT / phonon at fractional \(q\approx(0.3,0,0)\) is available
offline, record \(\omega\) + wall into the oracle note and tighten the
windows.

### Performance

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | -----------------------: | ----------------------: | ----: | ---- | ------ |
| Si DFPT \(q=(0.3,0,0)\), 10 Ry, \(k=1\) | — | ≈ 7 | **exempt** | n/a | No CP2K binary at pin; remeasure when available |

### Limitations

- One fractional \(q\) on a **minimal** twin (10 Ry, \(1\times1\times1\)) —
  spectrum sits high vs Giannozzi mid-ΓX; not a production phonon band.
- `max_asymmetry` ~ 30 eV/Å² is accepted (≤ 50); tighter hermiticity may need
  denser response settings.

## dfpt-spin-incommensurate — Incommensurate q, collinear spin, LSDA {#dfpt-spin-incommensurate}

Collinear-spin response on the direct finite-\(q\) path
(`dfpt_q_spin_impl`) at \(\mathbf{q}=(0.3,0,0)\), closed-shell \(M=0\). The
collinear twin of [dfpt-incommensurate-si](#dfpt-incommensurate-si); the
acceptance is identity with that restricted spectrum plus the same windows.

```cpp
PW method{std::make_shared<UpfPseudopotential>("tests/data/pw/Si_lda_nc.upf"),
          10.0, 1, cpu::scf::XCType::LSDA};
method.spin_polarized = true;
method.multiplicity = 1;   // M = 0 closed-shell collinear
method.tol = 1e-6;
method.mix_beta = 0.4;
method.max_iter = 200;

DfptPhonon dfpt;
dfpt.q_frac = {0.3, 0.0, 0.0};   // incommensurate → direct Sternheimer
dfpt.response_tol = 1e-4;
dfpt.response_max_iter = 120;
dfpt.cg_tol = 1e-8;
dfpt.mix_beta = 0.2;
```

### Accuracy

Measured (Release, OMP=8):

| Quantity | Atomiverse / pin | Oracle / twin | Tol / gate | Notes |
| -------- | ---------------: | ------------: | ---------- | ----- |
| \(\omega(q)\) [cm⁻¹] | **692.90, 726.85, 726.85, 944.07, 1023.46, 1023.46** | restricted twin same within **0.1** cm⁻¹ (`dfpt-incommensurate-si`) | identity \(M=0\) | 6 branches ascending |
| \(\lvert\omega\rvert\) sorted | 693, 727, 727, 944, 1023, 1023 | — | low-3 ∈ [400, 900]; high-3 ∈ [700, 1300] | unit-disaster windows |
| \(n_{\mathrm{imag}}\) | **0** | 0 | \(\le 0\) | no soft instability at pin |
| max asymmetry \(A\) | ≈ **29.6** eV/Å² | — | \(\le 50\) | before Hermitian projection |
| SCF / response | both converged | required | flags true | ground-state + \(3N\) responses |
| Literature mid-ΓX | biased **high** (10 Ry, Γ-\(k\)) | Giannozzi *et al.* PRB **43**, 7231 (1991) | not a hard pin | minimal twin, not quantitative LDA |

### Performance

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | ------------------------: | ----------------------: | ----: | ---- | ------ |
| diamond Si collinear incommensurate DFPT | n/a (no binary) | ≈ **11** | n/a | **perf_exempt** | no fair CP2K DFPT wall |

## dfpt-finite-q-si — Commensurate q = X, restricted LSDA {#dfpt-finite-q-si}

The X point \(X=(1/2,0,0)\) ⇒ \(n_a=2\), \(n_b=n_c=1\): Γ-DFPT on the
2×1×1 supercell (4 Si atoms) folded to six primitive branches — TA (doublet),
LA, LO, TO (doublet) in the literature LDA ordering. The minimal settings
(10 Ry, Γ on the supercell) admit a soft/imaginary TA pair, so the gates use
\(|\omega|\) windows and \(n_\mathrm{imag}\le 2\).

```cpp
PW method{std::make_shared<UpfPseudopotential>("tests/data/pw/Si_lda_nc.upf"),
          10.0, 1, cpu::scf::XCType::LSDA};  // Γ on 2×1×1 SC ≡ fold of primitive X

DfptPhonon dfpt;
dfpt.q_frac = {0.5, 0.0, 0.0};       // X of the primitive cell
dfpt.response_tol = 1e-4;
dfpt.response_max_iter = 120;
dfpt.cg_tol = 1e-8;
dfpt.mix_beta = 0.25;
```

### Accuracy

Release, OMP=8; k-mesh 1×1×1 on the 2×1×1 supercell:

| Quantity | Atomiverse / pin | Reference | Tol / gate | Notes |
| -------- | ---------------: | --------: | ---------- | ----- |
| SCF energy (4-atom SC) | −30.164 Ry | ~2 × primitive scale | finite | internal Ry OK |
| \(\omega(X)\) [cm⁻¹] | −130.3, −130.3, 229.8, 448.2, 672.6, 672.6 | — | hard pin | measured ATV |
| \(\|\omega\|\) sorted | 130, 130, 230, 448, 673, 673 | — | windows below | regression gates |
| Low 3 \(\|\omega\|\) | in [50, 400] | lit. TA/LA biased | hard | soft TA allowed |
| High 3 \(\|\omega\|\) | in [300, 850] | lit. LO/TO biased | hard | incomplete basis |
| Literature LDA X | — | TA≈146, LA≈397, LO≈413, TO≈455 | qualitative | Giannozzi PRB 1991 |
| \(n_\mathrm{imag}\) | 2 | ≤ 2 accepted | hard | minimal ecut/k |
| max_asymmetry | ≤ 1e-3 eV/Å² | Hermitian residual | hard | before projection |
| Wall | ≈ 43 s | ≤ 60 s e2e | hard | OMP=8 |

Gates use magnitude windows rather than signed literature digits: the
10 Ry / Γ-SC twin is tutorial-class and can report a soft TA doublet. When a
CP2K Quickstep DFPT / `VIBRATIONAL_ANALYSIS` twin is available, record
\(\omega(X)\) for all 6 branches + wall under matched OMP.

### Performance

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | ------------------------: | -----------------------: | ----: | ---- | ------ |
| Si X finite-q DFPT | n/a (no binary) | ≈ 43 | n/a | n/a | **`perf_exempt`** |

## dfpt-gga-finite-q-si — Commensurate q = X, pure GGA (PBE) kernel {#dfpt-gga-finite-q-si}

Finite \(\mathbf{q}=X\) where every supercell Sternheimer response (12 on the
2×1×1 supercell) is screened with the PBE \(K_{\mathrm{xc}}\) from central FD
of the converged ground-state GGA operator — the Γ machinery of
[dfpt-gga-si](#dfpt-gga-si) exercised on the supercell whose Hessian carries
the X modes; LSDA parent [dfpt-finite-q-si](#dfpt-finite-q-si). Dual gate: the
offline same-Hamiltonian **supercell** PBE energy second difference (sharp)
plus a CP2K 2026.1 supercell `VIBRATIONAL_ANALYSIS` physics anchor whose
eigenvector phases identify the folded X branch.

```cpp
PW method{std::make_shared<UpfPseudopotential>("tests/data/pw/Si_lda_nc.upf"),
          10.0, 1, cpu::scf::XCType::PBE};  // pure GGA kernel
// dfpt: X response settings (q_frac = {0.5, 0, 0}, 1e-4 / 0.25 / 1e-8 / 120)
```

### Accuracy

Measured (Release; identical numbers at OMP=1 and OMP=8):

| Quantity | Atomiverse / pin | Gate | Notes |
| -------- | ---------------: | ---- | ----- |
| Energy-FD \(C[(0x),(0x)]\) [eV/Å²] (supercell PBE SCF) | **21.2006** (hard-coded) | oracle for DFPT \(C_{00}\) | \(h=0.03\) Å; \(E_0=-15.121055415608\) Ha |
| DFPT \(C[(0x),(0x)]\) [eV/Å²] | **21.1974** | \(\|C_{00}-21.2006\|\le0.30\) (~0.015 %) | supercell diagonal |
| \(\omega(X)\) [cm⁻¹] | −108.49, −107.98, 239.26, 451.36, 668.88, 668.94 | \(\|\omega\|\) low 3 ∈ [50, 400]; high 3 ∈ [300, 850] | soft TA pair ≤ 2 imaginary (minimal twin, same as LSDA parent) |
| CP2K 2026.1 X anchor [cm⁻¹] | 97.98, 131.27, 370.58, 428.24, 457.48, 458.68 | inside the same windows | supercell FD vib; X branch by eigenvector phase |
| LSDA parent \(\lvert\omega\rvert\) [cm⁻¹] | 130, 130, 230, 448, 673, 673 | cross-parent sanity | XC-kernel change only |
| Pre-symmetrization asymmetry [eV/Å²] | \(7.3\times10^{-6}\) | \(\le10^{-4}\) | supercell Hermitian diagnostic |
| Wall | **49.8 s** (serial) | ≤ 60 s | 12 supercell responses, Γ-only mesh |

### Performance

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× |
| ---- | ------------------------: | ----------------------: | ----: | ---- |
| X phonons Si 2×1×1 supercell (CP2K FD vib, `EPS_SCF 1e-8`, serial) | 288.7 | 49.8 | **0.17** | yes |

CP2K binary `~/.local/bin/cp2k`, CP2K 2026.1 rev `85b8a9b49c`.

### Limitations

- No dedicated incommensurate-\(q\) GGA pin (the direct path shares the
  kernel code).

## dfpt-finite-q-smearing — Commensurate q = X, restricted metallic smearing {#dfpt-finite-q-smearing}

Fermi–Dirac smearing on the commensurate X path: smear-aware Γ-DFPT on the
2×1×1 supercell (the engine of [dfpt-metallic-si](#dfpt-metallic-si)) plus
IFC Fourier. Gapped fill keeps near-integer occupations, so the spectrum
tracks the fixed-occupation [dfpt-finite-q-si](#dfpt-finite-q-si) inside the
same \(\lvert\omega\rvert\) windows. LSDA kernel (pure GGA is accepted by the
API but not gated here).

```cpp
PW method{std::make_shared<UpfPseudopotential>("tests/data/pw/Si_lda_nc.upf"),
          10.0, 1, cpu::scf::XCType::LSDA};  // on the 2×1×1 supercell path
method.smearing = Smearing::fermi_dirac;
method.degauss = 0.02;             // Ry
method.extra_bands = 0;            // gapped twin; open metals use > 0
// dfpt: X response settings (q_frac = {0.5, 0, 0}, 1e-4 / 0.25 / 1e-8 / 120)
```

### Accuracy

Pinned on Arrhenius x86_64, Release, OMP=8:

| Quantity | Pin / gate | Tol / window | Notes |
| -------- | ---------: | ------------ | ----- |
| SCF free energy \(F\) (4-atom SC) | **−30.164 Ry** | scale check | matches fixed-occ twin scale |
| ATV \(\omega(X)\) [cm⁻¹] | **−130.30, −130.30, 229.79, 448.20, 672.63, 672.63** | hard-coded print | signed spectrum |
| \(\lvert\omega\rvert\) sorted | 130, 130, 230, 448, 673, 673 | — | gate inputs |
| Low 3 \(\lvert\omega\rvert\) | ∈ **[50, 400]** cm⁻¹ | window | TA-like / soft under minimal ecut |
| High 3 \(\lvert\omega\rvert\) | ∈ **[300, 850]** cm⁻¹ | window | LO/TO-like band |
| \(n_{\mathrm{imag}}\) | **2** | ≤ 2 | soft TA pair accepted at 10 Ry / Γ-only SC k |
| Max \(\lvert C-C^\dagger\rvert/2\) | **8.4×10⁻⁸** eV/Å² | ≤ **10⁻³** | Hermitian residual |
| Same-Hamiltonian twin | tracks `dfpt-finite-q-si` | print precision | gapped FD fill |

**CP2K sketch** (not an absolute twin; absolute \(E\)/\(\omega\) not
compared): CP2K + `&SMEAR` (FD \(T_e\) matched to `degauss ≈ 0.02` Ry
≈ 3158 K), diamond Si primitive, PADE/LDA, GTH-PADE + DZVP-MOLOPT-SR-GTH
(Si q4), supercell \(2\times1\times1\) or a finite-\(q\) path sampling **X**.

### Performance

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | ------------------------: | ----------------------: | ----: | ---- | ------ |
| CP2K DFPT+smear wall (when available) | n/a | **≈ 40** | n/a | n/a | **`perf_exempt`** — no CP2K binary on pin host |

ATV wall: 2-atom primitive → 4-atom SC, 12 responses, OMP=8; ≤ 60 s budget.

### Limitations

- No \(\delta\mu\) at \(q\neq 0\): direct finite-\(q\) metals with partial
  occupations do not restore \(\int\Delta n=0\) via a Fermi shift.

## dfpt-finite-q-spin-smearing — Incommensurate q, collinear spin + metallic smearing {#dfpt-finite-q-spin-smearing}

Collinear spin with Fermi–Dirac smearing on the **direct** finite-\(q\) path
(`internal::dfpt_q_force_constants` → `dfpt_q_spin_impl`) at
\(\mathbf{q}=(0.3,0,0)\): spin-resolved Sternheimer with projector on bands
with \(f\ge 10^{-8}\) at \(\mathbf{k}+\mathbf{q}\), occupation weights from the
ground-state \(f_{\sigma nk}\) at the shared \(\mu\), no \(\delta\mu\). Gates
freeze ATV spectra on the same Hamiltonian as the fixed-occupation
[dfpt-spin-incommensurate](#dfpt-spin-incommensurate). The commensurate
spin + smear path is covered by
[dfpt-finite-q-smearing](#dfpt-finite-q-smearing) at X and
[dfpt-spin-smearing](#dfpt-spin-smearing) at Γ.

```cpp
PW method{std::make_shared<UpfPseudopotential>("tests/data/pw/Si_lda_nc.upf"),
          10.0, 1, cpu::scf::XCType::LSDA};
method.spin_polarized = true;
method.multiplicity = 1;
method.smearing = Smearing::fermi_dirac;
method.degauss = 0.02; // Ry
method.extra_bands = 0;
method.tol = 1e-6;
method.mix_beta = 0.4;
// dfpt: incommensurate response settings (q_frac = {0.3, 0, 0}, 1e-4 / 0.2 / 1e-8 / 120)
```

### Accuracy

Release, OMP=8:

| Quantity | Pin | Tol / gate | Notes |
| -------- | --: | ---------- | ----- |
| \(\omega(\mathbf{q})\) [cm⁻¹] | **692.90, 726.85, 726.85, 944.07, 1023.46, 1023.46** | — | measured ATV print |
| \(\lvert\omega\rvert\) low 3 | 693, 727, 727 | ∈ **[400, 900]** | unit-disaster window |
| \(\lvert\omega\rvert\) high 3 | 944, 1023, 1023 | ∈ **[700, 1300]** | same |
| \(n_{\mathrm{imag}}\) | **0** | \(\le 0\) | no soft modes on this twin |
| Asymmetry \(\max\lvert C-C^\dagger\rvert/2\) | **29.6 eV/Å²** | \(\le 50\) | pre-Hermitian projection |
| Twin match | identical print to `dfpt-spin-incommensurate` | printed precision | gapped FD fill |

**CP2K sketch** (not an absolute twin): Quickstep GPW, PADE/LDA, GTH-PADE +
DZVP-MOLOPT-SR-GTH (Si q4), LSD / unrestricted multiplicity 1, SMEAR
FERMI_DIRAC \(T_e \approx 3158\) K (\(\approx 0.02\) Ry \(k_B T\)), fractional
\(\mathbf{q}\approx(0.3, 0, 0)\) or the closest commensurate mesh.

### Performance

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | ------------------------: | ----------------------: | ----: | ---- | ------ |
| CP2K DFPT spin+smear @ q≈(0.3,0,0) vs `dfpt-finite-q-spin-smearing` | n/a | **≈ 11** | **perf_exempt** | n/a | no CP2K wall twin on pin host |

ATV wall: 2-atom primitive, 6 responses × 2 spin, OMP=8; ≤ 60 s budget.

## dfpt-gga-spin-finite-q-si — Commensurate q = X, collinear spin × pure GGA {#dfpt-gga-spin-finite-q-si}

The triple path: X → 2×1×1 supercell Γ-DFPT + IFC Fourier where every
supercell Sternheimer response (12 × 2 spins) is spin-resolved and screened
with the spin-GGA \(K_{\mathrm{xc}}^{\sigma\sigma'}\) (FD of `step_spin`;
the DFPT-internal spin SCF caps mixing at ≤ 0.4). Parents:
[dfpt-gga-spin-si](#dfpt-gga-spin-si) (Γ, spin × PBE) and
[dfpt-gga-finite-q-si](#dfpt-gga-finite-q-si) (X, restricted PBE). At \(M=0\)
the sharp pin is the restricted supercell energy-FD pin 21.2006 eV/Å² and the
joint kernel must reproduce the restricted finite-q GGA parent.

```cpp
PW method{std::make_shared<UpfPseudopotential>("tests/data/pw/Si_lda_nc.upf"),
          10.0, 1, cpu::scf::XCType::PBE};  // pure GGA kernel
method.spin_polarized = true;              // joint spin × GGA response
method.multiplicity   = 1;
// dfpt: X response settings (q_frac = {0.5, 0, 0}, 1e-4 / 0.25 / 1e-8 / 120)
```

### Accuracy

Measured (Release; OMP=1):

| Quantity | Atomiverse / pin | Gate | Notes |
| -------- | ---------------: | ---- | ----- |
| Energy-FD \(C[(0x),(0x)]\) [eV/Å²] (supercell, M=0) | **21.2006** (hard-coded) | oracle for DFPT \(C_{00}\) | same-Ham. pin; \(M=0\) ≡ restricted supercell parent |
| DFPT \(C[(0x),(0x)]\) [eV/Å²] | **21.1911** | \(\|C_{00}-21.2006\|\le0.30\) (~0.04 %); vs restricted parent 21.1974 \(\le0.05\) | supercell diagonal |
| \(\omega(X)\) [cm⁻¹] | −108.83, −108.32, 239.12, 451.33, 668.84, 668.90 | \(\|\omega\|\) low 3 ∈ [50, 400]; high 3 ∈ [300, 850] | soft TA pair ≤ 2 imaginary (minimal twin) |
| CP2K 2026.1 X anchor [cm⁻¹] | 97.98, 131.27, 370.58, 428.24, 457.48, 458.68 | inside the same windows | supercell UKS-PBE FD vib; X by eigenvector phase; digit-identical to restricted twin at \(M=0\) |
| Pre-symmetrization asymmetry [eV/Å²] | \(9.5\times10^{-6}\) | \(\le10^{-4}\) | supercell Hermitian diagnostic |
| Wall | **107.9 s** (serial) | fair ratio ≤ 10 | 12 supercell responses × 2 spins; exceeds the 60 s family target by the spin doubling |

### Performance

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× |
| ---- | ------------------------: | ----------------------: | ----: | ---- |
| X phonons Si 2×1×1 supercell UKS-PBE (CP2K FD vib, `EPS_SCF 1e-8`, serial) | 577.8 | 107.9 | **0.19** | yes |

CP2K binary `~/.local/bin/cp2k`, CP2K 2026.1 rev `85b8a9b49c`.

### Limitations

- No incommensurate-\(q\) joint spin-GGA pin (shared code with
  `dfpt_q_spin_impl`).

## dfpt-gga-spin-finite-q-smearing — Commensurate q = X, collinear spin × pure GGA × smearing {#dfpt-gga-spin-finite-q-smearing}

The four-way path: X → 2×1×1 supercell Γ-DFPT + IFC Fourier where every
supercell response is spin-resolved, screened with the spin-GGA
\(K_{\mathrm{xc}}^{\sigma\sigma'}\), **and** weighted by Fermi–Dirac
occupations (shared \(\mu\), no \(\delta\mu\) on the supercell path). Parents:
[dfpt-gga-spin-smearing](#dfpt-gga-spin-smearing) (Γ),
[dfpt-gga-spin-finite-q-si](#dfpt-gga-spin-finite-q-si) (X, fixed
occupations), [dfpt-finite-q-spin-smearing](#dfpt-finite-q-spin-smearing)
(direct \(q\), LSDA). Gapped fill gives \(F\approx E\); at \(M=0\) the joint
kernel must reproduce the fixed-occupation finite-q spin × GGA parent.

```cpp
PW method{std::make_shared<UpfPseudopotential>("tests/data/pw/Si_lda_nc.upf"),
          10.0, 1, cpu::scf::XCType::PBE};
method.spin_polarized = true;
method.multiplicity   = 1;
method.smearing = Smearing::fermi_dirac;
method.degauss  = 0.02;
method.extra_bands = 0;
// dfpt: X response settings (q_frac = {0.5, 0, 0}, 1e-4 / 0.25 / 1e-8 / 120)
```

### Accuracy

Measured (Release; OMP=1):

| Quantity | Atomiverse / pin | Gate | Notes |
| -------- | ---------------: | ---- | ----- |
| Free-energy FD \(C[(0x),(0x)]\) [eV/Å²] (supercell, M=0 + FD) | **21.2006** (hard-coded) | oracle for DFPT \(C_{00}\) | same-Ham. pin; gapped fill ⇒ F=E |
| DFPT \(C[(0x),(0x)]\) [eV/Å²] | **21.1911** | \(\|C_{00}-21.2006\|\le0.30\) (~0.04 %); vs fixed-occ parent 21.1911 \(\le0.05\) | supercell diagonal |
| \(\omega(X)\) [cm⁻¹] | −108.83, −108.32, 239.12, 451.33, 668.84, 668.90 | \(\|\omega\|\) low 3 ∈ [50, 400]; high 3 ∈ [300, 850] | soft TA pair ≤ 2 imaginary (minimal twin) |
| CP2K 2026.1 X anchor [cm⁻¹] | 98.22, 130.53, 364.05, 427.18, 452.08, 458.19 | inside the same windows | supercell UKS-PBE+SMEAR FD vib; X by eigenvector phase |
| Pre-symmetrization asymmetry [eV/Å²] | \(9.5\times10^{-6}\) | \(\le10^{-4}\) | supercell Hermitian diagnostic |
| Wall | **148.1 s** (serial) | fair ratio ≤ 10 | 12 supercell responses × 2 spins + smear weights |

### Performance

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× |
| ---- | ------------------------: | ----------------------: | ----: | ---- |
| X phonons Si 2×1×1 supercell UKS-PBE+SMEAR (CP2K FD vib, `EPS_SCF 1e-8`, serial) | 585.9 | 148.1 | **0.25** | yes |

CP2K binary `…/cp2k/install/bin/cp2k.psmp`, CP2K 2026.1 rev `git:5e54ba2`.

### Limitations

- No incommensurate-\(q\) joint spin-GGA × smear pin (shared code).
