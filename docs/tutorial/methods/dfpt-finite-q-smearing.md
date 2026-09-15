# Finite-q DFPT phonons with metallic smearing — diamond Si at X (PW LSDA)

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `dfpt-finite-q-smearing` |
| Regression | `regressions/response/dfpt-finite-q-smearing/` |
| Public API | `cpu::pw::DfptPhonon` (members: `q_frac`, response/CG tolerances) + `cpu::pw::DfptPhononResult`; ground state via `cpu::pw::PW` members `smearing`, `degauss` [Ry], `extra_bands` |
| Roadmap | [dft.md](../../roadmap/dft.md) §11 DFPT — finite-q + metallic smear ✓ |
| Dep | `dfpt-finite-q-si` ✓ (fixed-occ X twin); `dfpt-metallic-si` ✓ (Γ FD); `pw-smearing-al` ✓ (occupations) |
| Oracle | CP2K (product intent: `VIBRATIONAL_ANALYSIS` + `&SMEAR`); offline note only |

## Summary

**Density-functional perturbation theory** at fractional phonon wavevector
\(\mathbf{q}\) with **Fermi–Dirac metallic occupation smearing** on the
plane-wave method `cpu::pw::PW`. This slug productizes the combination that
earlier code rejected at finite \(\mathbf{q}\): restricted smear is allowed
on the **commensurate** path (supercell Γ-DFPT + IFC Fourier) and on the
**direct** finite-\(q\) Sternheimer assembly. Textbook twin: **diamond Si**
primitive cell at the zone-boundary **X** point
(\(\mathbf{q}=(1/2,0,0)\)), NC-UPF LSDA, `degauss = 0.02` Ry. Gapped fill
keeps near-integer occupations so spectra track the fixed-occupation sibling
`dfpt-finite-q-si` inside hard-coded \(\lvert\omega\rvert\) windows.
Regression: `reg_dfpt-finite-q-smearing`. Public energy on Frame remains eV
when Frame meta is written; DFPT frequencies are **cm⁻¹** on
`DfptPhononResult` (not parked on Frame).

## Method

**Level.** Analytic dynamical matrix \(C(\mathbf{q})\) from self-consistent
first-order density response (Sternheimer / DFPT) instead of \(6N\)
finite-displacement SCF solves. Pseudopotential path is **norm-conserving
UPF** (same as other PW DFPT slugs). XC on this pin is **LSDA**; pure GGA
kernels are supported by the facade but not gated here.

**Occupations.** Closed-shell convention (spinless \(f\in[0,1]\) doubled to
band occupation \(f_b\in[0,2]\)):

\[
f_{nk}
  = 2\,f_{\mathrm{FD}}\bigl((\varepsilon_{nk}-\mu)/\sigma\bigr),\qquad
\sigma = \texttt{degauss}\;[\mathrm{Ry}].
\]

Ground-state SCF uses the **Mermin free energy** \(F = E_{\mathrm{KS}}-TS\)
when `method.smearing != Smearing::none` (same machinery as
[pw-smearing-al](pw-smearing-al.md)). Density and force-constant response
terms weight orbitals by fractional \(f_{nk}\). The Sternheimer projector at
\(\mathbf{k}+\mathbf{q}\) removes bands with \(f \ge 10^{-8}\).

**Commensurate product path (this regression).** Rational
\(\mathbf{q}=(h/n_a,k/n_b,l/n_c)\) with small supercell product is evaluated
as:

1. Expand the primitive cell to an \(n_a\times n_b\times n_c\) supercell.
2. Run **smear-aware Γ-point DFPT** on that supercell
   (`internal::dfpt_gamma_force_constants` — same engine as Γ metallic
   sibling `regressions/response/dfpt-metallic-si`).
3. Fourier the supercell IFCs to the requested \(\mathbf{q}\) of the
   primitive cell (`PhononDispersion::frequencies_at_q`).

For \(X=(1/2,0,0)\) the supercell is **\(2\times1\times1\)** (4 Si atoms).
This is analytically equivalent to finite-\(q\) DFPT for Born–von Karman
commensurate \(\mathbf{q}\).

**Direct finite-\(q\) path (also smear-capable).** Incommensurate or
oversize-\(\mathbf{q}\) uses `internal::dfpt_q_force_constants` on the
primitive cell: complex Hermitian \(C(q)\), fractional \(f_{nk}\) weights,
projector at \(\mathbf{k}+\mathbf{q}\). **No de Gironcoli \(\delta\mu\)** at
\(q\neq 0\) (Γ-only charge-neutrality shift for metals).

**Mass weighting and spectrum.** Cartesian \(C\) [Ha/Bohr²] → public
`force_constants` [eV/Å²] (real part; imag part on
`force_constants_imag`); mass-weighted Hermitian diagonalization; eigenvalues
→ \(\omega\) [cm⁻¹] via `nist::au2wavenumber()` (same CODATA path as
`cpu::vib`). Imaginary eigenvalues report as **negative** \(\omega\).

**Excluded on this pin.** Collinear **spin + smear** (throws; residual
`dfpt-spin-smearing`); hybrids; GPW DFPT; CUDA DFPT; LO–TO / Born charges
(nonpolar Si); cold / Marzari–Vanderbilt smear enums as DFPT gates (FD is
the e2e path).

References: Baroni, de Gironcoli, Dal Corso, Giannozzi, *Rev. Mod. Phys.*
**73**, 515 (2001); de Gironcoli, *Phys. Rev. B* **51**, 6773 (1995);
Giannozzi et al., *Phys. Rev. B* **43**, 7231 (1991) (LDA Si phonon
windows at X).

## Implementation

Public surface only: construct `cpu::pw::PW`, set smear members, construct
`cpu::pw::DfptPhonon` with `q_frac`, call `compute(method, frame)`. **No**
public `*Config` bag and **no** `make_*` / `build_*` factory. Frame is not
mutated; spectra live on the owned result.

Headers:
`include/atomiverse/cpu/planewave/dfpt_phonon.h` (facade),
`include/atomiverse/cpu/planewave/pw_solver.h` (smear members);
internals under `planewave/internal/dfpt.*` (not a user type).

```cpp
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/planewave/dfpt_phonon.h"
#include "atomiverse/cpu/planewave/pw_solver.h"

using namespace ATV;
using namespace ATV::cpu::pw;

// Diamond Si primitive, a = 5.43 Å (REFERENCE.md)
cpu::Frame frame(2);
frame.span(kAtom_type)[0] = 14;
frame.span(kAtom_type)[1] = 14;
frame.span(kAtom_mass)[0] = 28.0855;
frame.span(kAtom_mass)[1] = 28.0855;
frame.span(kAtom_x)[1] = 5.43 / 4.0;
frame.span(kAtom_y)[1] = 5.43 / 4.0;
frame.span(kAtom_z)[1] = 5.43 / 4.0;
const double h = 5.43 / 2.0;
Mat3 cell;
cell(0, 0) = 0; cell(0, 1) = h; cell(0, 2) = h;
cell(1, 0) = h; cell(1, 1) = 0; cell(1, 2) = h;
cell(2, 0) = h; cell(2, 1) = h; cell(2, 2) = 0;
frame.box.set_cell(cell);

PW method;
method.upf_path = "tests/data/pw/Si_lda_nc.upf";
method.ecut_ry = 10.0;
method.k_mesh = 1;                 // on the 2×1×1 supercell path
method.xc = cpu::scf::XCType::LSDA;
method.smearing = Smearing::fermi_dirac;
method.degauss = 0.02;             // Ry
method.extra_bands = 0;            // gapped twin; open metals use > 0
method.tol = 1e-8;
method.max_iter = 200;

DfptPhonon dfpt;
dfpt.q_frac = {0.5, 0.0, 0.0};     // X → commensurate 2×1×1
dfpt.response_tol = 1e-4;
dfpt.response_max_iter = 120;
dfpt.cg_tol = 1e-8;
dfpt.mix_beta = 0.25;

const DfptPhononResult r = dfpt.compute(method, frame);
// r.frequencies [cm⁻¹] length 6; r.force_constants [eV/Å²];
// r.scf_converged, r.response_converged, r.max_asymmetry
```

`compute` throws `std::invalid_argument` for hybrid XC, empty `upf_path`,
or **spin + smear**. Restricted smear alone is accepted at Γ, commensurate
finite \(\mathbf{q}\), and direct finite-\(q\) Sternheimer.

```bash
# configure (login shell + Lmod — see AGENTS.md)
bash -lc '
  module load buildenv-gcc/2026.03-mpich
  export OPENBLAS_ROOT=$OPENBLASDIR
  export RUSTUP_TOOLCHAIN=stable-x86_64-unknown-linux-gnu
  export LD_LIBRARY_PATH=$LIBRARY_PATH:$LD_LIBRARY_PATH
  cmake -B build -DCMAKE_BUILD_TYPE=Release \
        -DATV_BUILD_TESTS=ON -DATV_BUILD_PW=ON \
        -DATV_BUILD_REGRESSIONS=ON -DATV_BUILD_CUDA=OFF \
        -DBLA_VENDOR=OpenBLAS
  cmake --build build -j$(nproc) --target atomiverse reg_dfpt-finite-q-smearing
  ./build/regressions/reg_dfpt-finite-q-smearing
  # or: cmake --build build --target run_reg_dfpt-finite-q-smearing
'
```

Source consumer:
`regressions/response/dfpt-finite-q-smearing/reg_dfpt-finite-q-smearing.cpp`.

## Validation (CP2K product intent; offline pins)

Source:
[`regressions/response/dfpt-finite-q-smearing/REFERENCE.md`](../../../regressions/response/dfpt-finite-q-smearing/REFERENCE.md)
(hard-coded gates only; **never** live CP2K in ctest).

### ATV system (frequency gates)

| Setting | Value |
| ------- | ----- |
| Geometry | diamond Si primitive, \(a=5.43\) Å, 2 atoms |
| \(\mathbf{q}\) | \(X=(1/2,0,0)\) → **2×1×1** supercell (4 atoms) |
| PP | `tests/data/pw/Si_lda_nc.upf` (NC, LDA) |
| XC | LSDA |
| Smearing | Fermi–Dirac, `degauss = 0.02` Ry, `extra_bands = 0` |
| \(E_{\mathrm{cut}}\) | 10 Ry |
| k-mesh | \(1\times1\times1\) on the supercell |
| Response | tol \(10^{-4}\), mix 0.25, max_iter 120 |
| Pin date | 2026-07-26 (Arrhenius x86_64, Release, OMP=8) |

Gapped Si at these settings keeps near-integer FD occupations, so
\(F \approx E_{\mathrm{KS}}\) and results track the fixed-occ twin.

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

Literature LDA Si at X (Giannozzi et al. 1991; not absolute twin at 10 Ry):
TA≈146, LA≈397, LO≈413, TO≈455 cm⁻¹ — windows intentionally wide for the
minimal-settings bias (same as fixed-occ finite-q slug).

### Offline CP2K twin (algorithm / product documentation)

**Not an absolute-\(\omega\) twin** (GTH-GPW ≠ NC-UPF PW). Documents that
CP2K `VIBRATIONAL_ANALYSIS` with `&SMEAR METHOD FERMI_DIRAC` is the product
counterpart of ATV finite-\(q\) DFPT under metallic occupations.

| Item | Value |
| ---- | ----- |
| Preferred oracle | CP2K + `&SMEAR` (FD \(T_e\) matched to `degauss ≈ 0.02` Ry ≈ 3158 K) |
| Note | `tools/cp2k_oracle/inputs/dfpt-finite-q-smearing.note` |
| Suggested structure | diamond Si primitive, PADE/LDA, GTH-PADE + DZVP-MOLOPT-SR-GTH (Si q4) |
| Phonon | supercell \(2\times1\times1\) or finite-\(q\) path sampling **X** |
| CP2K binary this pin host | **not available** (`module avail cp2k` empty) |
| Absolute \(E\) / \(\omega\) vs ATV | **not compared** (different Hamiltonian) |

Oracle contract: `tools/cp2k_oracle/README.md` (offline only).

## Performance

Source: same REFERENCE. Gate:
\(r = T_{\mathrm{ATV}}/T_{\mathrm{CP2K}} \le 10\) when a fair twin exists;
this slug is **`perf_exempt`**.

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | ------------------------: | ----------------------: | ----: | ---- | ------ |
| CP2K DFPT+smear wall (when available) | n/a | **≈ 40** | n/a | n/a | **`perf_exempt`** — no CP2K binary on pin host |

| Item | Value |
| ---- | ----- |
| `perf_exempt` | **true** |
| Reason | No CP2K DFPT+smear wall twin on the build host this shift |
| ATV wall | ≈ **40 s** (2-atom primitive → 4-atom SC, 12 responses, OMP=8; ≤ 60 s budget) |
| `ratio_atv_over_oracle` | n/a (exempt) |
| `within_10x` | n/a (exempt; correctness gates only) |

Handoff: the case `REFERENCE.md` records
`dfpt-finite-q-smearing` as **`perf_exempt`**.

## Limitations

- **Minimal ecut / k-mesh twin.** Soft TA pair (\(n_{\mathrm{imag}}=2\)) is
  accepted by design; not a production phonon band structure.
- **Absolute frequencies are ATV/literature-window pins, not CP2K-matched.**
  GTH-GPW and NC-UPF PW are different Hamiltonians; CP2K documents product
  SMEAR + phonon settings only until a binary records ω walls offline.
- **Gapped Si, not an open metal.** `extra_bands = 0` and near-integer FD
  fill; metallic Al/Fe DFPT remains a deeper product story (ground-state
  smear: `pw-smearing-al`, `pw-uks-fe-smearing`).
- **Restricted spin only on this path.** Collinear spin + smear is
  **rejected** and remains the open residual (`dfpt-spin-smearing`).
- **No \(\delta\mu\) at \(q\neq 0\).** Direct finite-\(q\) metals with
  partial occupations do not restore \(\int\Delta n=0\) via a Fermi shift
  (Γ de Gironcoli only).
- **LSDA pin.** Pure GGA DFPT is productized elsewhere (`dfpt-gga-si`); not
  re-gated under smear+X.
- **CPU only.** CUDA DFPT out of scope for this loop.
- **No public Config / factories.** Parameters are members on `PW` and
  `DfptPhonon` only.
- **Speed.** `perf_exempt`; informative ATV wall ≈ 40 s, no fair \(r\).

## References

- Offline pin: `regressions/response/dfpt-finite-q-smearing/REFERENCE.md`
- Regression: `regressions/response/dfpt-finite-q-smearing/reg_dfpt-finite-q-smearing.cpp`
- Fixed-occ sibling: `regressions/response/dfpt-finite-q-si/REFERENCE.md`
- Γ metallic sibling: `regressions/response/dfpt-metallic-si/REFERENCE.md`
- CP2K product note: `tools/cp2k_oracle/inputs/dfpt-finite-q-smearing.note`
- Oracle contract: `tools/cp2k_oracle/README.md`
- Handoff (`perf_exempt`): the case `REFERENCE.md`
- Roadmap: `docs/roadmap/dft.md` §11 DFPT (finite-q + metallic smear ✓)
- Public header: `include/atomiverse/cpu/planewave/dfpt_phonon.h`
- PW smear members: `include/atomiverse/cpu/planewave/pw_solver.h`
- Internals: `include|src/.../planewave/internal/dfpt.*`, `smearing.*`
- Related tutorials: [dfpt-phonon-si](dfpt-phonon-si.md) (Γ fixed-occ),
  [pw-smearing-al](pw-smearing-al.md) (metallic occupations),
  [fd-phonon-supercell](fd-phonon-supercell.md) (FD crystal path)
- Literature: Baroni et al., *Rev. Mod. Phys.* **73**, 515 (2001);
  de Gironcoli, *Phys. Rev. B* **51**, 6773 (1995);
  Giannozzi et al., *Phys. Rev. B* **43**, 7231 (1991)
