# Collinear-spin DFPT at incommensurate \(\mathbf{q}\) — diamond Si (PW LSDA)

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `dfpt-spin-incommensurate` |
| Regression | `regressions/response/dfpt-spin-incommensurate/` |
| Public API | `cpu::pw::DfptPhonon` (`q_frac`, response members) + `cpu::pw::PW` (`spin_polarized`, `multiplicity`) + `DfptPhononResult` |
| Roadmap | [dft.md](../../roadmap/dft.md) §11 DFPT — collinear spin at incommensurate \(q\) ✓ |
| Dep | `dfpt-spin-si` ✓ (Γ collinear); `dfpt-incommensurate-si` ✓ (restricted finite \(q\)) |

## Summary

**Collinear-spin density-functional perturbation theory at a non-rational
phonon wavevector** combines two previously separate product surfaces:
spin-resolved Sternheimer response (`spin_polarized = true`, fixed
occupations) and **direct** finite-\(q\) assembly on the primitive cell
(incommensurate \(\mathbf{q}\), no supercell fold). Product path is
`cpu::pw::DfptPhonon::compute` over a `cpu::pw::PW` ground state with
NC-UPF and pure LSDA (or pure GGA kernel). This page pins diamond Si at
\(\mathbf{q}=(0.3,0,0)\) with closed-shell collinear \(M=0\)
(`multiplicity = 1`). Offline pins only — oracle family **CP2K** when a
binary is present (`tools/cp2k_oracle/inputs/dfpt-spin-incommensurate.note`);
never live CP2K in ctest. Performance: **`perf_exempt`** (no fair CP2K
DFPT wall on the pin host). Regression: `reg_dfpt-spin-incommensurate`.

Siblings: Γ collinear `dfpt-spin-si`; restricted incommensurate
`dfpt-incommensurate-si`; metallic smear DFPT `dfpt-metallic-si` (spin+smear
still out of scope).

## Method

**Level.** Analytic dynamical matrix \(C(\mathbf{q})\) from self-consistent
first-order density response instead of \(6N\) finite-displacement SCF
solves. Reference: Baroni, de Gironcoli, Dal Corso, Giannozzi,
*Rev. Mod. Phys.* **73**, 515 (2001).

**Path selection** inside `DfptPhonon::compute` (same table as restricted
finite-\(q\)):

| `q_frac` | Path |
| -------- | ---- |
| \(\approx\mathbf{0}\) | Γ Sternheimer (`dfpt_gamma_force_constants`) |
| rational, \(n_a n_b n_c\le 8\), \(n_i\le 6\) | supercell Γ-DFPT + IFC Fourier |
| otherwise | **direct** `internal::dfpt_q_force_constants` (this slug) |

For \(\mathbf{q}=(0.3,0,0)\) no \(n\le 6\) supercell is commensurate, so the
code takes the **direct Sternheimer** branch on the primitive cell
(`dfpt_q_spin_impl` when `method.spin_polarized`).

**Collinear spin response.** With `spin_polarized = true` and fixed
occupations:

- Ground-state SCF is dual-spin KS (\(n_{\mathrm{spin}}=2\)).
- Response densities \(\Delta n_\uparrow\), \(\Delta n_\downarrow\) are
  solved self-consistently per ionic perturbation direction.
- Occupation \(f=1\) per spin-orbital (closed-shell \(M=0\) for
  `multiplicity = 1` on Si: \(N_\uparrow=N_\downarrow=4\) valence).
- Hartree from total \(\Delta n=\Delta n_\uparrow+\Delta n_\downarrow\).
- Spin XC kernel \(K_{\mathrm{xc}}^{\sigma\sigma'}\): local LSDA
  \(f_{\mathrm{xc}}\) table, or central FD of `XCBuilder::step_spin` for
  pure GGA.

Mass-weighted Hermitian diagonalization of \(C(\mathbf{q})\) yields
\(3N\) wavenumbers [cm⁻¹] (imaginary eigenvalues reported negative).

**System (product pin).**

| Setting | Value |
| ------- | ----- |
| Structure | diamond Si primitive, \(a = 5.43\) Å (FCC vectors) |
| \(\mathbf{q}\) | \((0.3, 0, 0)\) fractional |
| Pseudopotential | `tests/data/pw/Si_lda_nc.upf` |
| XC | LSDA (`cpu::scf::XCType::LSDA`) |
| Spin | collinear, `multiplicity = 1` (\(M=0\)) |
| Cutoff | 10 Ry |
| k-mesh | \(1\times1\times1\) |
| SCF | tol \(10^{-6}\), mix 0.4, max_iter 200 |
| Response | tol \(10^{-4}\), mix 0.2, max_iter 120, cg_tol \(10^{-8}\) |

**Observables** (on `DfptPhononResult`, never on Frame):

| Symbol | Definition | Unit |
| ------ | ---------- | ---- |
| \(\omega_k\) | `frequencies` (length \(3N=6\), ascending) | cm⁻¹ |
| \(\mathrm{Re}\,C(q)\) | `force_constants` shape \([6,6]\) | eV/Å² |
| \(\mathrm{Im}\,C(q)\) | `force_constants_imag` | eV/Å² |
| \(A\) | `max_asymmetry` = max \(\lvert C-C^\dagger\rvert/2\) before Hermitian projection | eV/Å² |
| — | `scf_converged`, `response_converged` | flags |

**Key exclusions (this slug).**

- Spin **+** metallic smearing (throws); finite-\(q\) **+** smear (throws).
- Hybrid / meta-GGA XC kernels; noncollinear / SOC response.
- LO–TO / Born charges (nonpolar Si does not need them for this pin).
- Live CP2K `VIBRATIONAL_ANALYSIS` / DFPT at ctest time.
- CUDA DFPT; public `*Config` bags; `make_*` / `build_*` factories.
- Quantitative LDA dispersion claim under 10 Ry / Γ-only \(k\) — gates are
  unit-disaster windows, not literature mid-ΓX targets.

## Implementation

Public surface only: construct `cpu::pw::PW` and `cpu::pw::DfptPhonon`, set
**members** (no `DfptConfig`, no factories), call `compute(method, frame)`.
Headers: `include/atomiverse/cpu/planewave/dfpt_phonon.h`,
`pw_solver.h`. Frame in / result out; Frame not mutated.

```cpp
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/planewave/dfpt_phonon.h"
#include "atomiverse/cpu/planewave/pw_solver.h"

using namespace ATV;
using namespace ATV::cpu::pw;

// Diamond Si primitive, a = 5.43 Å
cpu::Frame frame(2);
constexpr double a = 5.43;
frame.span(kAtom_type)[0] = 14;
frame.span(kAtom_type)[1] = 14;
frame.span(kAtom_mass)[0] = 28.0855;
frame.span(kAtom_mass)[1] = 28.0855;
frame.span(kAtom_x)[1] = a / 4.0;
frame.span(kAtom_y)[1] = a / 4.0;
frame.span(kAtom_z)[1] = a / 4.0;
const double h = a / 2.0;
Mat3 cell;
cell(0, 0) = 0; cell(0, 1) = h; cell(0, 2) = h;
cell(1, 0) = h; cell(1, 1) = 0; cell(1, 2) = h;
cell(2, 0) = h; cell(2, 1) = h; cell(2, 2) = 0;
frame.box.set_cell(cell);

PW method;
method.upf_path = "tests/data/pw/Si_lda_nc.upf";
method.ecut_ry = 10.0;
method.k_mesh = 1;
method.xc = cpu::scf::XCType::LSDA;
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

const DfptPhononResult r = dfpt.compute(method, frame);
// r.frequencies [cm⁻¹]; r.force_constants [eV/Å²]
```

```bash
# After configure with -DATV_BUILD_PW=ON -DATV_BUILD_REGRESSIONS=ON
cmake --build build -j$(nproc) --target atomiverse reg_dfpt-spin-incommensurate
./build/regressions/reg_dfpt-spin-incommensurate
# or: cmake --build build --target run_reg_dfpt-spin-incommensurate
```

Unit guard (no full SCF): `tests/cpu/planewave/test_dfpt.cpp` —
`SpinIncommensurateQReachesUpfIo` (collinear + incommensurate \(q\) reaches
UPF I/O; empty UPF still throws).

## Validation (vs offline CP2K / twin pins)

Source: `regressions/response/dfpt-spin-incommensurate/REFERENCE.md`
(measured 2026-07-26, Release, OMP=8). CP2K binary **not** on pin host —
spectrum windows + closed-shell identity to the restricted incommensurate
twin. Oracle note: `tools/cp2k_oracle/inputs/dfpt-spin-incommensurate.note`.

| Quantity | Atomiverse / pin | Oracle / twin | Tol / gate | Notes |
| -------- | ---------------: | ------------: | ---------- | ----- |
| \(\omega(q)\) [cm⁻¹] | **692.90, 726.85, 726.85, 944.07, 1023.46, 1023.46** | restricted twin same within **0.1** cm⁻¹ (`dfpt-incommensurate-si`) | identity \(M=0\) | 6 branches ascending |
| \(\lvert\omega\rvert\) sorted | 693, 727, 727, 944, 1023, 1023 | — | low-3 ∈ [400, 900]; high-3 ∈ [700, 1300] | unit-disaster windows |
| \(n_{\mathrm{imag}}\) | **0** | 0 | \(\le 0\) | no soft instability at pin |
| max asymmetry \(A\) | ≈ **29.6** eV/Å² | — | \(\le 50\) | before Hermitian projection |
| SCF / response | both converged | required | flags true | ground-state + \(3N\) responses |
| Literature mid-ΓX | biased **high** (10 Ry, Γ-\(k\)) | Giannozzi *et al.* PRB **43**, 7231 (1991) | not a hard pin | minimal twin, not quantitative LDA |

Geometry / PP / ecut / \(k\) / \(q\) / spin bookkeeping as in **Method**
table. Public energies remain eV on Frame when SCF writes them; DFPT
spectra live only on `DfptPhononResult`.

## Performance

Source: `REFERENCE.md` +
the case `REFERENCE.md` (wave 8 staff table).

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | ------------------------: | ----------------------: | ----: | ---- | ------ |
| diamond Si collinear incommensurate DFPT | n/a (no binary) | ≈ **11** | n/a | **perf_exempt** | no fair CP2K DFPT wall this shift |

| Item | Value |
| ---- | ----- |
| `perf_exempt` | **true** |
| Reason | No CP2K DFPT wall twin on the build host |
| Wall budget (regression) | ≤ 60 s |
| `ratio_atv_over_oracle` | n/a |
| `within_10x` | n/a (exempt) |

Gate policy: \(r = T_{\mathrm{ATV}}/T_{\mathrm{CP2K}} \le 10\) when a fair
offline twin exists; this slug is exempt until a CP2K wall is recorded in
the oracle note.

## Limitations

- Minimal settings (10 Ry, \(1\times1\times1\) \(k\)) **do not** claim
  converged phonon dispersion along Γ–X; windows catch unit/sign bugs.
- Collinear **open-shell** (\(M\neq 0\)) DFPT at incommensurate \(q\) is not
  the product pin (Si \(M=0\) identity is the acceptance twin).
- Spin+smear and finite-\(q\)+smear remain **hard throws** — follow-ons,
  not silent approximations.
- No LO–TO splitting / \(\mathbf{Z}^*\) path on this page.
- CPU only; CUDA ports are out of the CP2K capability loop scope.
- When a CP2K binary becomes available, re-record \(\omega(q)\) + wall in
  `tools/cp2k_oracle/inputs/dfpt-spin-incommensurate.note` and lift
  `perf_exempt` if a fair ratio is measurable.

## References

- Offline pin: `regressions/response/dfpt-spin-incommensurate/REFERENCE.md`
- Oracle note: `tools/cp2k_oracle/inputs/dfpt-spin-incommensurate.note`
- Oracle rules: `tools/cp2k_oracle/README.md` (never live in ctest)
- API: `include/atomiverse/cpu/planewave/dfpt_phonon.h`
- Siblings: `docs/tutorial/methods/dfpt-phonon-si.md` (Γ restricted);
  regressions `dfpt-spin-si`, `dfpt-incommensurate-si`, `dfpt-metallic-si`
- Baroni et al., *Rev. Mod. Phys.* **73**, 515 (2001)
- Giannozzi et al., *Phys. Rev. B* **43**, 7231 (1991)
