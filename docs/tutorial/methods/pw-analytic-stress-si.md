# Plane-wave Nielsen–Martin analytic stress — diamond Si (NC-UPF LDA)

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `pw-analytic-stress-si` |
| Regression | `regressions/pw/pw-analytic-stress-si/` |
| Public API | `cpu::pw::pw_stress` (analytic Voigt [eV/Å³]); oracle `cpu::pw::pw_stress_fd`; `PWEvaluator::with_stress(true)` → `frame.meta.get(kStress)` |
| Roadmap | [dft.md](../../roadmap/dft.md) § Periodic response — analytic stress|

## Summary

**Nielsen–Martin cell stress** for spin-unpolarized plane-wave LDA on bulk
silicon (diamond, 2-atom fcc primitive), same incomplete-basis twin as
`pw-lda-si` (ONCVPSP NC-UPF, \(E_\mathrm{cut}=15\) Ry, \(2\times2\times2\) MP).
The product path is **`cpu::pw::pw_stress`**: one SCF class of cost, then
Hellmann–Feynman cell derivatives under symmetric strain at fixed fractional
coordinates, returning Voigt \(\sigma\) in **eV/Å³**. Multi-SCF strain
finite-difference remains **`pw_stress_fd`** (dual-gate oracle only, not the
hot path). Product gate: `reg_pw-analytic-stress-si` vs an offline multi-SCF
FD pin (not a freeze of analytic output). Performance: **`perf_exempt`** (no
fair CP2K Quickstep NC-UPF PW twin on the pin host).

## Method

**Level.** Periodic plane-wave Kohn–Sham DFT (same engine as `pw-lda-si`). After
one converged SCF on the unstrained cell, the **Nielsen–Martin** stress is
formed as the cell derivative of the total energy at fixed fractional
coordinates (Nielsen & Martin, *Phys. Rev. B* **32**, 3780 (1985)). Engine
decomposition (public header / internal impl notes): kinetic (bands), Hartree,
local PP form-factor \(G\)-derivative, Ewald ion–ion, LDA XC volume term, and
Kleinman–Bylander nonlocal (UPF projectors). Sign convention:

\[
\sigma_{\alpha\beta}
  = \frac{1}{\Omega}\,
    \frac{\partial E}{\partial\varepsilon_{\alpha\beta}},
\qquad
\sigma = (\sigma_{xx},\sigma_{yy},\sigma_{zz},\sigma_{xy},\sigma_{xz},\sigma_{yz})
\quad [\mathrm{eV}/\mathrm{\AA}^3].
\]

**Oracle (not product).** `pw_stress_fd` rebuilds a full SCF for each of 12
central-difference strained cells (\(h = 2\times 10^{-3}\) default). Used only
for dual-gate unit tests and the offline Si pin.

**XC / spin / occupations.** `XCType::LSDA` (Slater + PW92 on the FFT grid);
restricted closed shell; no metallic smearing on this twin.

**BZ sampling.** Uniform Monkhorst–Pack `k_mesh`³ (regression: \(2\times 2\times 2\)).

**Observables.**

| Symbol | Definition | Unit |
| ------ | ---------- | ---- |
| \(\sigma_v\) | Analytic Voigt stress from `pw_stress` | eV/Å³ |
| \(\sigma_v^\mathrm{FD}\) | Multi-SCF strain-FD from `pw_stress_fd` / offline pin | eV/Å³ |
| `kStress` | Frame meta write when `PWEvaluator::with_stress(true)` | eV/Å³ |

**Key exclusions (this slug / closed SPEC).**

- GPW strain-FD stress → `gpw-stress-water`.
- Production GGA stress refinements beyond the LDA twin (PBE stress may follow).
- MPI multi-rank stress scaling.
- Variable-cell product drivers beyond existing isotropic cell-relax smoke.
- Absolute `STRESS_TENSOR` parity with **CP2K Quickstep GTH-PADE + DZVP-GTH**
  (different PP and discretization).
- Live CP2K / QE / ABINIT at ctest time (offline pins only).
- Production-converged ONCV cutoffs (~60 Ry) and dense k-meshes when they
  exceed the ≤ 60 s e2e budget.

## Implementation

Public surface only (`cpu::Frame` + concrete `cpu::pw::PW` members; no `*Config`,
no `make_*` / `build_*`). Headers:
`include/atomiverse/cpu/planewave/pw_solver.h` (`pw_stress`, `pw_stress_fd`,
`PW`), `pw_evaluator.h` (`with_stress`), Frame schema `kStress`.

Minimal sequence (matches `reg_pw-analytic-stress-si.cpp`):

```cpp
#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/planewave/pw_solver.h"
#include "atomiverse/cpu/scf/xc_functional.h"

using namespace ATV;
using namespace ATV::cpu::pw;

constexpr double kA = 5.43;       // Å
constexpr double kEcutRy = 15.0;  // tutorial incomplete basis
constexpr int kMesh = 2;          // 2×2×2 MP

cpu::Frame frame(2);
frame.span(kAtom_type)[0] = 14;
frame.span(kAtom_type)[1] = 14;
frame.span(kAtom_mass)[0] = 28.0855;
frame.span(kAtom_mass)[1] = 28.0855;
frame.span(kAtom_x)[0] = 0.0;
frame.span(kAtom_y)[0] = 0.0;
frame.span(kAtom_z)[0] = 0.0;
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
method.upf_path = "…/tests/data/pw/Si_lda_nc.upf";
method.ecut_ry = kEcutRy;
method.k_mesh = kMesh;
method.tol = 1e-6;
method.max_iter = 50;
method.xc = cpu::scf::XCType::LSDA;

// Analytic Nielsen–Martin stress [eV/Å³], Voigt xx,yy,zz,xy,xz,yz
const std::array<double, 6> san = pw_stress(frame, method);

// Dual-gate oracle only (12 SCFs) — not the product hot path:
// const auto sfd = pw_stress_fd(frame, method, /*h=*/2e-3);

// Driver binding (optional): write σ to frame.meta.get(kStress)
// PWEvaluator pwe{std::move(method)};
// pwe.with_stress(true);
// cpu::single_point(frame, pwe);
```

Build and run the regression (`ATV_BUILD_PW=ON`, `ATV_BUILD_REGRESSIONS=ON`):

```bash
bash -lc '
  module load buildenv-gcc/2026.03-mpich
  export OPENBLAS_ROOT="$OPENBLASDIR"
  export RUSTUP_TOOLCHAIN=stable-x86_64-unknown-linux-gnu
  export LD_LIBRARY_PATH="$LIBRARY_PATH:$LD_LIBRARY_PATH"
  cmake -B build -DCMAKE_BUILD_TYPE=Release \
        -DATV_BUILD_TESTS=ON -DATV_BUILD_HF=ON -DATV_BUILD_PW=ON \
        -DATV_BUILD_REGRESSIONS=ON -DATV_BUILD_CUDA=OFF \
        -DBLA_VENDOR=OpenBLAS
  cmake --build build --target reg_pw-analytic-stress-si -j$(nproc)
  ./build/regressions/reg_pw-analytic-stress-si
'
```

UPF path is injected at compile time as `ATV_PW_UPF_PATH` →
`${CMAKE_SOURCE_DIR}/tests/data/pw/Si_lda_nc.upf`.

| Path | Role |
| ---- | ---- |
| `include/atomiverse/cpu/planewave/pw_solver.h` | Public `pw_stress` / `pw_stress_fd` / `PW` |
| `include/atomiverse/cpu/planewave/pw_evaluator.h` | `with_stress` → `kStress` |
| `include/atomiverse/cpu/planewave/internal/pw_stress.h` | Private analytic + FD impl |
| `regressions/pw/pw-analytic-stress-si/reg_pw-analytic-stress-si.cpp` | Product gate |
| `regressions/pw/pw-analytic-stress-si/REFERENCE.md` | Offline FD pin + perf note |
| `tests/cpu/planewave/test_pw_stress.cpp` | Unit dual-gate (H toy, symmetry) |
| `tools/cp2k_oracle/inputs/si_lda_pw_note.inp` | Optional CP2K wall skeleton (not absolute stress twin) |

## Validation (CP2K)

Source:
[`regressions/pw/pw-analytic-stress-si/REFERENCE.md`](../../../regressions/pw/pw-analytic-stress-si/REFERENCE.md).
Primary science gate is **offline multi-SCF strain-FD** on the same NC-UPF twin
(`pw_stress_fd`, \(h=2\times 10^{-3}\)), measured **2026-07-24** Arrhenius epyc
(Release, `OMP_NUM_THREADS=16`). Hard-coded pins in the regression binary —
**never** live third-party oracles from ctest. Closed SPEC acceptance A5–A8.

### Pseudopotential

| Item | Value |
| ---- | ----- |
| File | `tests/data/pw/Si_lda_nc.upf` (in-tree) |
| Family | PseudoDojo ONCVPSP-LDA-SR-noCC v0.4 |
| Generator | ONCVPSP 3.3.0 (D. R. Hamann), \(Z_\mathrm{val}=4\), no NLCC |
| Cite | Hamann, *Phys. Rev. B* **88**, 085117 (2013); van Setten et al., *Comput. Phys. Commun.* **226**, 39 (2018) |

### Twin geometry and settings (regression)

| Setting | Value |
| ------- | ----- |
| Structure | diamond Si, 2-atom fcc primitive |
| \(a\) | 5.43 Å |
| Basis atoms | \((0,0,0)\), \((a/4,a/4,a/4)\) |
| Cell | \((a/2)(0,1,1)\), \((a/2)(1,0,1)\), \((a/2)(1,1,0)\) |
| \(E_\mathrm{cut}\) | 15 Ry (tutorial; ONCV production ~60 Ry) |
| k-mesh | \(2\times 2\times 2\) (`k_mesh = 2`) |
| SCF tol | \(10^{-6}\) |
| `max_iter` | 50 |
| XC | LSDA |

### Accuracy table

| Quantity | Atomiverse / pin | External / oracle | Tol | Notes |
| -------- | ---------------: | ----------------: | --- | ----- |
| \(\sigma_{xx},\sigma_{yy},\sigma_{zz}\) pin | dual-gate | **−0.05755** eV/Å³ (mean FD diagonals) | 0.05 eV/Å³ | Offline multi-SCF FD; not analytic self-golden |
| \(\sigma_{xy},\sigma_{xz},\sigma_{yz}\) pin | dual-gate | **0** | 0.05 eV/Å³ | Crystal symmetry zero |
| Live dual-gate \(\max_v\lvert\sigma^\mathrm{an}-\sigma^\mathrm{FD}\rvert\) | **\(7.3\times 10^{-5}\)** eV/Å³ | same twin, offline | ≪ 0.05 | REFERENCE live an↔FD (not in ctest) |
| Soft symmetry \(\lvert\sigma_{xy,xz,yz}\rvert\) | regression | 0 | **0.025** eV/Å³ | Incomplete ecut/k residual allowed |
| Soft symmetry \(\lvert\sigma_{xx}-\sigma_{yy}\rvert,\ldots\) | regression | 0 | **0.025** eV/Å³ | Cubic isotropy of diagonal |
| Sign of hydrostatic \(\sigma_{xx}\) | compressive **&lt; 0** | EOS context | qualitative | Twin \(a=5.43\) Å compressed vs PseudoDojo \(a_0\approx 5.468\) Å |

**CP2K note.** Quickstep bulk-Si decks use **GTH-PADE + DZVP-GTH**, not ONCV
plane-wave. Absolute `STRESS_TENSOR` values are **not** comparable to this
NC-UPF twin. No CP2K binary was available on the pin host for a fair PW wall
twin this shift. Optional skeleton:
`tools/cp2k_oracle/inputs/si_lda_pw_note.inp` (same geometry / k-mesh; not an
absolute-stress oracle).

**Literature / PseudoDojo context (EOS, not the hard stress pin).**

| Quantity | Value | Source |
| -------- | ----- | ------ |
| \(a_0\) (EOS) | ≈ 5.468 Å | PseudoDojo `Si.djrepo` deltafactor \(v_0\) |
| \(B_0\) | 89.96 GPa | same |
| Twin \(a\) | 5.43 Å (compressed vs \(a_0\)) | regression |

## Performance

Source: `regressions/pw/pw-analytic-stress-si/REFERENCE.md`. Queue / handoff:
**`perf_exempt`** (no fair CP2K / QE / ABINIT NC-UPF PW stress wall twin on pin
host).

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | ------------------------: | ----------------------: | ----: | ---- | ------ |
| PW analytic stress Si ecut=15 / k=2 | n/a | ≈ **30** (informative) | n/a | n/a (**exempt**) | 1 SCF + stress pass; no fair NC-UPF PW oracle binary |

| Item | Value |
| ---- | ----- |
| `perf_exempt` | **true** |
| Reason | No CP2K / QE / ABINIT binary on the pin host for a fair NC-UPF PW stress wall twin; absolute stress not a CP2K GTH twin |
| ATV analytic wall (informative) | ≈ 30 s (Release, 16 threads) |
| Live dual-gate wall (analytic + full FD) | ≈ 116 s offline only — **not** in ctest |
| `ratio_atv_over_cp2k` | n/a (exempt) |
| `within_10x` | n/a (exempt; dual-gate correctness only) |
| Wall budget | e2e ≤ 60 s single-rank CPU (PASS for analytic path) |

When a plane-wave oracle binary is available offline: time the same geometry,
append \(r = T_\mathrm{ATV}/T_\mathrm{oracle}\) to REFERENCE.md, require
\(r\le 10\).

## Limitations

- Tutorial ecut/k only — hydrostatic stress magnitude is incomplete-basis; not
  a production EOS / equilibrium-pressure productization.
- Dual-gate product pin uses offline multi-SCF FD (tol 0.05 eV/Å³); live
  an↔FD residual \(\sim 10^{-4}\) was measured offline only.
- Soft symmetry tols (0.025 eV/Å³) absorb residual shear at incomplete ecut/k.
- Absolute CP2K Quickstep stress is a different PP/basis world — fair science
  comparison is EOS / structural response, not \(\sigma\) digit-for-digit.
- GGA (PBE) stress refinements, MPI scaling, and full variable-cell drivers are
  out of this slug’s closed acceptance.
- CUDA PW surface is not a CPU compatibility API for this pin.

## References

- Offline pin: [`regressions/pw/pw-analytic-stress-si/REFERENCE.md`](../../../regressions/pw/pw-analytic-stress-si/REFERENCE.md)
- Regression: [`regressions/pw/pw-analytic-stress-si/reg_pw-analytic-stress-si.cpp`](../../../regressions/pw/pw-analytic-stress-si/reg_pw-analytic-stress-si.cpp)
- Closed SPEC (git history): `.claude/specs/pw-analytic-stress-si/SPEC.md` (deleted at harness done)
- Energy foothold twin: [pw-lda-si](pw-lda-si.md)
- Optional CP2K wall skeleton: `tools/cp2k_oracle/inputs/si_lda_pw_note.inp`
- Narrative style ref: [Chapter 8 — Plane-Wave DFT](../ch08-plane-wave.md)
- O. H. Nielsen and R. M. Martin, *Phys. Rev. B* **32**, 3780 (1985)
- D. R. Hamann, *Phys. Rev. B* **88**, 085117 (2013)
- M. J. van Setten et al., *Comput. Phys. Commun.* **226**, 39 (2018)
- PseudoDojo ONCVPSP-LDA-SR-noCCv0.4 `Si.djrepo` (EOS context)
