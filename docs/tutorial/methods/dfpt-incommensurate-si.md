# Incommensurate-q DFPT phonons — diamond Si (PW LSDA, NC-UPF)

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `dfpt-incommensurate-si` |
| Regression | `regressions/response/dfpt-incommensurate-si/` |
| Public API | `cpu::pw::DfptPhonon` (members, incl. `q_frac`) + `cpu::pw::DfptPhononResult` |
| Roadmap | [dft.md](../../roadmap/dft.md) §11 DFPT — incommensurate-q ✓ |
| Dep | `dfpt-phonon-si` ✓ (Γ), `dfpt-finite-q-si` ✓ (commensurate X); closes direct Sternheimer branch |

## Summary

**Density-functional perturbation theory at an incommensurate phonon wavevector**
computes the complex dynamical matrix \(C(\mathbf{q})\) on the **primitive
cell** by self-consistent Sternheimer response, without building a supercell.
For `cpu::pw::DfptPhonon`, fractional \(\mathbf{q}\) that is **not** rational
with supercell product \(\le 8\) (or would need \(n_i>6\)) selects the direct
path `internal::dfpt_q_force_constants`. This product gate pins
\(\mathbf{q}=(0.3,0,0)\) on diamond Si under restricted PW LSDA + NC-UPF:
`regressions/response/dfpt-incommensurate-si` (`reg_dfpt-incommensurate-si`).
Oracle = **offline CP2K** when available (`tools/cp2k_oracle/`); pin host had
no CP2K binary — gates use measured ATV windows + literature bias notes.
Performance: **`perf_exempt`**.

## Method

**Level.** Analytic phonons at finite \(\mathbf{q}\) from DFPT (Baroni et al.,
*Rev. Mod. Phys.* **73**, 515 (2001)). One ground-state SCF, then \(3N\)
self-consistent first-order density responses for Cartesian ionic
perturbations at wavevector \(\mathbf{q}\); assemble Hermitian
\(C(\mathbf{q})\); mass-weight and diagonalize.

**Public path selection** (`DfptPhonon::compute`):

| `q_frac` | Engine path |
| -------- | ----------- |
| \(\approx\mathbf{0}\) | Γ Sternheimer (`dfpt_gamma_force_constants`) — `dfpt-phonon-si` |
| rational, \(n_a n_b n_c\le 8\), \(n_i\le 6\) | supercell Γ-DFPT + IFC Fourier — `dfpt-finite-q-si` (e.g. \(X=\tfrac12\hat{\mathbf{x}}\)) |
| **otherwise** | **direct** `dfpt_q_force_constants` on the primitive cell — **this slug** |

**Incommensurate branch.** For \(\mathbf{q}=(0.3,0,0)\), no integer \(n\le 6\)
satisfies \(n\cdot 0.3\in\mathbb{Z}\), so the commensurate gate fails and the
engine runs finite-\(q\) Sternheimer: conduction-space projected CG per
\((k,\text{band})\), Hartree + LDA \(f_{xc}\) screening of the first-order
density, analytic Ewald second derivatives with Bloch phases + ASR, KB
nonlocal first/second derivative terms. Response density uses the single-\(q\)
convention (conjugate sector lives at \(-\mathbf{q}\); no extra TR factor of
2). Force constants are complex; frequencies come from Hermitian
diagonalization of the mass-weighted matrix (IUPAC masses by \(Z\), same as
`cpu::vib`). Imaginary eigenvalues are reported as **negative** \(\omega\).

**XC / spin / PP.** Restricted LSDA kernel only (`XCType::LSDA`); fixed
occupations (no metallic smearing); UPF norm-conserving PP required (empty
`upf_path` / toy potential throws). Collinear spin unsupported.

**Observables** (on `DfptPhononResult`, never on Frame):

| Symbol | Definition | Unit |
| ------ | ---------- | ---- |
| \(\omega_i(q)\) | phonon wavenumbers, length \(3N\), ascending | cm⁻¹ |
| \(\mathrm{Re}\,C(q)\), \(\mathrm{Im}\,C(q)\) | Cartesian force constants | eV/Å² |
| `max_asymmetry` | \(\max|C-C^\dagger|/2\) before Hermitian projection | eV/Å² |
| `scf_converged` / `response_converged` | GS + all \(3N\) response SCFs | bool |

Internal assembly uses Ha/Bohr²; public FC are eV/Å². Frame energy (if read
from a separate `PW` run) remains eV on `kEnergy`.

**Key exclusions (this slug).**

- Commensurate rational \(q\) (supercell path) → `dfpt-finite-q-si`.
- \(\mathbf{q}=\mathbf{0}\) only → `dfpt-phonon-si`.
- Full band structure ω(q) from IFC Fourier → `phonon-dispersion-si` (FD IFC).
- GGA \(f_{xc}\), metals (Fermi-surface terms), LO–TO / Born charges, UKS.
- Quantitative match to literature mid-ΓX at **converged** ecut/k (this twin
  is 10 Ry, \(1\times1\times1\) — spectrum biased high; windows reject unit
  disasters only).
- Live CP2K at ctest time (offline pins only).
- CUDA DFPT (CPU only for this loop).

## Implementation

Public surface only (`cpu::Frame` + concrete method members; no `*Config`, no
`make_*` / `build_*`). Headers:
`include/atomiverse/cpu/planewave/dfpt_phonon.h`,
`pw_solver.h`; private assembly
`include/atomiverse/cpu/planewave/internal/dfpt.h`.

Minimal sequence (matches `reg_dfpt-incommensurate-si.cpp`):

```cpp
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/planewave/dfpt_phonon.h"
#include "atomiverse/cpu/planewave/pw_solver.h"

using namespace ATV;
using namespace ATV::cpu::pw;

constexpr double kA = 5.43;   // Å
constexpr double kEcutRy = 10.0;
constexpr int kMesh = 1;

cpu::Frame frame(2);
frame.span(kAtom_type)[0] = 14;
frame.span(kAtom_type)[1] = 14;
frame.span(kAtom_mass)[0] = 28.0855;
frame.span(kAtom_mass)[1] = 28.0855;
frame.span(kAtom_x)[1] = kA / 4.0;
frame.span(kAtom_y)[1] = kA / 4.0;
frame.span(kAtom_z)[1] = kA / 4.0;
const double h = kA / 2.0;
Mat3 cell;
// fcc primitive: (0,h,h), (h,0,h), (h,h,0)
cell(0, 0) = 0; cell(0, 1) = h; cell(0, 2) = h;
cell(1, 0) = h; cell(1, 1) = 0; cell(1, 2) = h;
cell(2, 0) = h; cell(2, 1) = h; cell(2, 2) = 0;
frame.box.set_cell(cell);

PW method;
method.upf_path = "tests/data/pw/Si_lda_nc.upf";
method.ecut_ry = kEcutRy;
method.k_mesh = kMesh;
method.xc = cpu::scf::XCType::LSDA;
method.tol = 1e-8;
method.max_iter = 200;

DfptPhonon dfpt;
dfpt.q_frac = {0.3, 0.0, 0.0};   // incommensurate along Γ–X
dfpt.response_tol = 1e-4;
dfpt.response_max_iter = 120;
dfpt.cg_tol = 1e-8;
dfpt.mix_beta = 0.2;
const DfptPhononResult r = dfpt.compute(method, frame);
// r.frequencies [cm⁻¹]; r.force_constants / _imag [eV/Å²]
```

```bash
# scoped build (never bare cmake --build build)
cmake --build build -j$(nproc) --target atomiverse reg_dfpt-incommensurate-si
cmake --build build --target run_reg_dfpt-incommensurate-si
# or:
./build/regressions/reg_dfpt-incommensurate-si
```

`compute` throws `std::invalid_argument` for non-LSDA XC, empty `upf_path`,
metallic smearing, or collinear spin — those regimes are follow-ons, not
silent approximations.

## Validation (vs CP2K / offline pins)

Source:
[`regressions/response/dfpt-incommensurate-si/REFERENCE.md`](../../../regressions/response/dfpt-incommensurate-si/REFERENCE.md).
Oracle note: `tools/cp2k_oracle/inputs/dfpt-incommensurate-si.note`.
Oracle status at pin time (2026-07-25): **no CP2K binary** on the worker host.
Absolute CP2K \(\omega(q)\) digits are therefore **not** invented; the product
gate uses measured ATV spectrum windows under the documented minimal twin.
Never live CP2K in ctest.

**Setup:**

| Item | Value |
| ---- | ----- |
| Structure | diamond Si primitive, 2 atoms, \(a = 5.43\) Å |
| \(\mathbf{q}\) | **(0.3, 0, 0)** fractional (incommensurate under \(n\le 6\)) |
| Pseudopotential | `tests/data/pw/Si_lda_nc.upf` |
| XC | LSDA (`XCType::LSDA`) |
| Cutoff | **10 Ry** |
| k-mesh | **1×1×1** |
| Response | tol \(10^{-4}\), mix 0.2, max_iter 120 |
| Masses | 28.0855 amu (Si) |
| Provenance | Baroni RMP 2001; Giannozzi PRB **43**, 7231 (1991) mid-ΓX context |

**Accuracy gate (product):**

| Quantity | Atomiverse (pin 2026-07-25) | Oracle / pin | Tol / gate | Notes |
| -------- | --------------------------: | -----------: | ---------- | ----- |
| SCF energy (GS) | **−14.544 Ry** | ATV measured | reported | Minimal twin |
| \(\omega(q)\) [cm⁻¹] | **692.9, 726.9, 726.9, 944.1, 1023.5, 1023.5** | ATV measured | — | Degenerate pairs (2–3), (5–6) |
| low-3 \(\lvert\omega\rvert\) | 693, 727, 727 | window | **[400, 900]** cm⁻¹ | Unit / path integrity |
| high-3 \(\lvert\omega\rvert\) | 944, 1023, 1023 | window | **[700, 1300]** cm⁻¹ | Same |
| \(n_{\mathrm{imag}}\) | **0** | — | \(\le 0\) | Hard |
| max asymmetry | ≈ **29.6** eV/Å² | — | \(\le 50\) eV/Å² | Hermitian residual |
| SCF / response | both converged | — | `true` | Hard |
| Literature mid-ΓX (LDA) | — | ~100–500 cm⁻¹ acoustic/optic band (converged) | **not** a tight pin | Twin biases **high** at 10 Ry / Γ-only \(k\) |

Windows reject unit disasters and wrong path selection; they **do not** claim
quantitative LDA dispersion at production cutoffs. When CP2K DFPT / phonon at
fractional \(q\approx(0.3,0,0)\) is available offline, record \(\omega\) + wall
into the oracle note and tighten REFERENCE.

Offline re-pin (human only; never ctest):

```bash
# when a CP2K binary is available — see tools/cp2k_oracle/inputs/dfpt-incommensurate-si.note
# run VIBRATIONAL_ANALYSIS / DFPT at fractional q ≈ (0.3,0,0) on the same cell
# record ω [cm⁻¹] + wall into REFERENCE.md; optionally tighten windows
```

## Performance

Source: [`REFERENCE.md`](../../../regressions/response/dfpt-incommensurate-si/REFERENCE.md)
(2026-07-25); queue **perf_exempt**.

| Item | Value |
| ---- | ----- |
| `perf_exempt` | **true** |
| Reason | No CP2K DFPT wall twin on the pin host this shift |
| \(T_{\mathrm{ATV}}\) (Release, OMP=8) | ≈ **7 s** (≤ 60 s e2e budget) |
| \(r = T_{\mathrm{ATV}}/T_{\mathrm{CP2K}}\) | **n/a (exempt)** |
| `within_10x` | n/a (exempt) |

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | -----------------------: | ----------------------: | ----: | ---- | ------ |
| Si DFPT \(q=(0.3,0,0)\), 10 Ry, \(k=1\) | — | ≈ 7 | **exempt** | n/a | No CP2K binary at pin; remeasure when available |

Campaign queue: ratio **perf_exempt**
. Gate when a twin
exists: \(r\le 10\) on the same host/core count (single-rank).

## Limitations

- **Direct Sternheimer at one fractional \(q\)** on a **minimal** twin (10 Ry,
  \(1\times1\times1\)) — spectrum sits high vs Giannozzi mid-ΓX; not a
  production phonon band.
- **No offline CP2K \(\omega\) digits** on the pin host — absolute oracle
  match deferred; windows + convergence + zero imaginary modes are the product
  gate.
- **LSDA / insulator / restricted / NC-UPF only** — GGA kernel, metals,
  collinear spin, LO–TO splitting are out of scope.
- **`max_asymmetry` ~ 30 eV/Å²** is accepted (≤ 50); tighter hermiticity may
  need denser response settings.
- **`perf_exempt`**: do not invent a speed ratio until CP2K is timed offline
  on the same host.
- **CPU only** — CUDA DFPT remains out of the CP2K loop.
- Commensurate and Γ product gates remain separate regressions
  (`dfpt-finite-q-si`, `dfpt-phonon-si`).

## References

- Offline pin: [`regressions/response/dfpt-incommensurate-si/REFERENCE.md`](../../../regressions/response/dfpt-incommensurate-si/REFERENCE.md)
- Oracle note: `tools/cp2k_oracle/inputs/dfpt-incommensurate-si.note`
- Oracle rules: `tools/cp2k_oracle/README.md` (offline only; never in ctest)
- Roadmap: [dft.md](../../roadmap/dft.md) §11 DFPT
- Related tutorials: [dfpt-phonon-si](dfpt-phonon-si.md) (Γ),
  [fd-phonon-supercell](fd-phonon-supercell.md) (FD Γ)
- Baroni, de Gironcoli, Dal Corso, Giannozzi, *Rev. Mod. Phys.* **73**, 515 (2001)
- Giannozzi et al., *Phys. Rev. B* **43**, 7231 (1991)
