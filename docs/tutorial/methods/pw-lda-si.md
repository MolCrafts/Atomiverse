# Plane-wave LDA — bulk silicon (k-mesh energy)

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `pw-lda-si` |
| Regression | `regressions/pw/pw-lda-si/` |
| Public API | `cpu::pw::PW` (members) + `cpu::pw::PWEvaluator` + `cpu::single_point` |
| Roadmap | [dft.md](../../roadmap/dft.md) § Periodic DFT / plane-wave|

## Summary

Spin-unpolarized **plane-wave Kohn–Sham LDA** total energy of bulk silicon
(diamond, 2-atom fcc primitive cell) with a norm-conserving ONCVPSP UPF and a
uniform Monkhorst–Pack k-mesh. This is the CP2K-loop foothold for product PW
energy under k-point sampling: Frame in, concrete `cpu::pw::PW` members, energy
on `PWResult` [Ry/Ha] and `frame.meta.get(kEnergy)` [eV]. Absolute energy is
gated against a **PseudoDojo / ABINIT-class** same-pseudo \(E_\mathrm{min}\)
band (not a freeze of Atomiverse-only digits; not CP2K Quickstep absolute
energy). Product gate: `regressions/pw/pw-lda-si` (`reg_pw-lda-si`).
Performance: **`perf_exempt`** (no fair CP2K PW twin on the pin host).

## Method

**Level.** Periodic plane-wave KS-DFT. Wavefunctions

\[
\psi_{n\mathbf{k}}(\mathbf{r})
  = \sum_{\mathbf{G}} c_{n\mathbf{k}}(\mathbf{G})\,
    e^{i(\mathbf{k}+\mathbf{G})\cdot\mathbf{r}},
\qquad
\tfrac12|\mathbf{k}+\mathbf{G}|^2 \le E_\mathrm{cut},
\]

with Hartree solved in reciprocal space
(\(V_\mathrm{H}(\mathbf{G}) = 4\pi\rho(\mathbf{G})/G^2\), \(G\neq 0\)),
LDA XC on the FFT real-space grid, and a Davidson diagonalization of the
effective Hamiltonian. Core electrons are replaced by a **norm-conserving**
pseudopotential (UPF v1).

**XC.** Slater exchange + PW92 correlation (engine pure LDA /
`cpu::scf::XCType::LSDA`). The in-tree Si UPF was generated with ONCVPSP LDA
(iexc=3, PZ); the electronic XC on the grid is the engine LDA, not hybrid /
meta-GGA.

**Spin / occupations.** Restricted (spin-unpolarized); fixed closed-shell
occupations (`Smearing::none`). No metallic smearing on this twin.

**BZ sampling.** Uniform Monkhorst–Pack mesh of size
`PW::k_mesh`³ (regression: \(2\times 2\times 2\)).

**Observables.**

| Symbol | Definition | Unit |
| ------ | ---------- | ---- |
| \(E_\mathrm{tot}\) | PW total energy (`PWResult::energy`) | Ry |
| \(E_\mathrm{Ha}\) | same energy in Hartree (`PWResult`) | Ha |
| \(E\) | Frame meta energy (`kEnergy`) | eV |

Conversion used by the engine gate:
\(E[\mathrm{eV}] = E[\mathrm{Ry}]\times 13.605693\)
(`cpu/planewave/qe_units.h`).

**Key exclusions (this slug).**

- Analytic / FD forces and Nielsen–Martin stress → `pw-analytic-stress-si` and
  related force regressions, not this energy pin.
- True GGA (PBE) → `pw-pbe-gga-si`.
- Metallic smearing → `pw-smearing-al`.
- Collinear spin → `pw-spin-polarized-o2`.
- PW BOMD → `pw-bomd-h`.
- Production-converged ONCV cutoffs (~60 Ry) and dense k-meshes when they
  exceed the ≤ 60 s e2e budget — twin stays tutorial-class with an explicit
  incomplete-basis window.
- Absolute energy parity with **CP2K Quickstep GTH-PADE + DZVP-GTH** (different
  PP and discretization). Fair science comparison to CP2K is EOS
  (\(a_0\), \(B_0\)), not \(E_\mathrm{tot}\).
- Live CP2K / ABINIT / QE at ctest time (offline pins only).

## Implementation

Public surface only (`cpu::Frame` + concrete method members; no `*Config`, no
`make_*` / `build_*` factories). Headers:
`include/atomiverse/cpu/planewave/pw_solver.h`,
`pw_evaluator.h`, `qe_units.h`; evaluate entry
`include/atomiverse/cpu/evaluate/single_point.h`.

Minimal sequence (matches `reg_pw-lda-si.cpp`):

```cpp
#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/evaluate/single_point.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/planewave/pw_evaluator.h"
#include "atomiverse/cpu/planewave/pw_solver.h"

using namespace ATV;
using namespace ATV::cpu::pw;

constexpr double kA = 5.43;       // Å
constexpr double kEcutRy = 15.0;  // tutorial incomplete basis
constexpr int kMesh = 2;          // 2×2×2 MP

cpu::Frame frame(2);
frame.span(kAtom_type)[0] = 14;
frame.span(kAtom_type)[1] = 14;
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
// Pure LDA on the PW grid (engine default XC may be PBE; set explicitly):
method.xc = cpu::scf::XCType::LSDA;

PWEvaluator pwe{std::move(method)};
cpu::single_point(frame, pwe);

const auto& r = pwe.last_result(); // r.energy [Ry], r.converged, r.iterations
const double e_ev = frame.meta.get(kEnergy); // [eV]
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
  cmake --build build --target reg_pw-lda-si -j$(nproc)
  ./build/regressions/reg_pw-lda-si
'
```

UPF path is injected at compile time as `ATV_PW_UPF_PATH` →
`${CMAKE_SOURCE_DIR}/tests/data/pw/Si_lda_nc.upf`.

## Validation (vs ABINIT-class / CP2K note)

Source: `regressions/pw/pw-lda-si/REFERENCE.md` (offline pin; never live third
party in ctest). Closed acceptance (git `pw-lda-si` spec): incomplete-basis
band above PseudoDojo ABINIT-class \(E_\mathrm{min}\).

### Pseudopotential

| Item | Value |
| ---- | ----- |
| File | `tests/data/pw/Si_lda_nc.upf` (in-tree) |
| Family | PseudoDojo ONCVPSP-LDA-SR-noCC v0.4 |
| Generator | ONCVPSP 3.3.0 (D. R. Hamann), \(Z_\mathrm{val}=4\), no NLCC |
| Cite | Hamann, *Phys. Rev. B* **88**, 085117 (2013); van Setten et al., *Comput. Phys. Commun.* **226**, 39 (2018) |
| Upstream | <https://github.com/PseudoDojo/ONCVPSP-LDA-SR-noCCv0.4> (`Si/Si.upf`) |

### ABINIT-class same-pseudo pin (converged limit)

From PseudoDojo `Si/Si.djrepo` deltafactor key `"8.0"`, `num_sites = 2`
(retrieved 2026-07-23):

| Quantity | Value |
| -------- | ----: |
| \(E_\mathrm{min}\) | **−230.158202 eV/cell** = **−16.916316 Ry/cell** |
| \(v_0\) | 20.43445 Å³/atom → \(a_0 \approx 5.468\) Å (conventional cubic) |
| \(B_0\) | 89.96 GPa |

### Twin geometry and settings (regression)

| Setting | Value |
| ------- | ----- |
| Structure | diamond Si, 2-atom fcc primitive |
| \(a\) | 5.43 Å |
| Basis atoms | \((0,0,0)\), \((a/4,a/4,a/4)\) |
| Cell | \((a/2)(0,1,1)\), \((a/2)(1,0,1)\), \((a/2)(1,1,0)\) |
| \(E_\mathrm{cut}\) | 15 Ry (tutorial; ONCV production ~60 Ry) |
| k-mesh | \(2\times 2\times 2\) (`k_mesh = 2`) |
| SCF tol | \(10^{-6}\) (density change) |
| `max_iter` | 50 |

### Accuracy table

| Quantity | Atomiverse / pin | External | Tol / band | Notes |
| -------- | ---------------: | -------: | ---------- | ----- |
| \(E_\mathrm{min}\) (converged PP) | — | **−16.916316 Ry** | literature pin | PseudoDojo / ABINIT-class |
| \(E_\mathrm{tot}\) (twin) | band gate | same \(E_\mathrm{min}\) | \([-16.916316,\,-15.416316]\) Ry | \(E_\mathrm{min} \le E_\mathrm{ATV} \le E_\mathrm{min}+1.50\) Ry |
| Measured twin (informative) | **≈ −15.670898 Ry** (5 iters) | — | inside band (~1.25 Ry above \(E_\mathrm{min}\)) | REFERENCE host note 2026-07-23, Release, 16 OMP |
| Frame \(E\) [eV] | \(E_\mathrm{Ry}\times 13.605693\) | engine conversion | \(10^{-4}\) eV | `qe_units.h` |
| SCF | `converged == true` | — | — | `max_iter = 50` |

**Incomplete-basis rationale.** The twin is deliberately under-converged in
\(E_\mathrm{cut}\) and BZ sampling, so \(E_\mathrm{ATV}\) is **higher** (less
bound) than the PseudoDojo ABINIT \(E_\mathrm{min}\). The 1.50 Ry headroom is
not a golden of Atomiverse output: the lower edge is the external pin; the
window rejects unit disasters while allowing known incomplete-basis elevation
at \(a=5.43\) Å.

**CP2K.** Quickstep bulk-Si decks use **GTH-PADE + DZVP-GTH**. Absolute total
energies are **not** comparable to ONCV plane-wave. Optional offline wall-only
input skeleton: `tools/cp2k_oracle/inputs/si_lda_pw_note.inp` (same geometry /
\(2\times 2\times 2\) MP; not an absolute-energy oracle).

## Performance

Source: `regressions/pw/pw-lda-si/REFERENCE.md`. Queue row:
**`perf_exempt`** (ABINIT-class energy band; no fair CP2K PW twin on pin host).

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | ------------------------: | ----------------------: | ----: | ---- | ------ |
| PW LDA Si ecut=15 / k=2 | n/a | ≈ 7–8 (informative) | n/a | n/a (**exempt**) | No CP2K/ABINIT binary on pin host; absolute energy not a CP2K twin |

| Item | Value |
| ---- | ----- |
| `perf_exempt` | **true** |
| Reason | No CP2K (or ABINIT) module on the build host; fair wall twin not run offline this shift |
| `ratio_atv_over_cp2k` | n/a (exempt) |
| `within_10x` | n/a (exempt; correctness gate only) |
| Wall budget | e2e ≤ 60 s single-rank CPU (PASS) |

When a CP2K binary is available offline: time the same geometry with
`tools/cp2k_oracle/inputs/si_lda_pw_note.inp`, append
\(r = T_\mathrm{ATV}/T_\mathrm{CP2K}\) to REFERENCE.md, require \(r\le 10\).
Wall comparison is same-geometry only, not absolute-energy parity.

## Limitations

- Tutorial ecut/k only — not production-converged absolute energies or EOS
  productization (EOS is the fair multi-code science metric).
- Energy-only foothold; forces/stress/GGA/smearing/spin/BOMD are other slugs.
- No MPI multi-rank scaling gate here (`mpi-pw-scaling-4rank` later).
- `PW::xc` default in headers may be GGA-PBE for other product paths; LDA
  callers must set `XCType::LSDA` for pure local XC (see Method).
- Empty `upf_path` selects a toy \(V_\mathrm{loc}\) (unphysical); this
  regression always uses the in-tree ONCV Si UPF.
- CUDA PW surface is not a CPU compatibility API for this pin.

## References

- Offline pin: [`regressions/pw/pw-lda-si/REFERENCE.md`](../../../regressions/pw/pw-lda-si/REFERENCE.md)
- Regression: [`regressions/pw/pw-lda-si/reg_pw-lda-si.cpp`](../../../regressions/pw/pw-lda-si/reg_pw-lda-si.cpp)
- Optional CP2K wall skeleton: `tools/cp2k_oracle/inputs/si_lda_pw_note.inp`
- Narrative style ref: [Chapter 8 — Plane-Wave DFT](../ch08-plane-wave.md)
- D. R. Hamann, *Phys. Rev. B* **88**, 085117 (2013)
- M. J. van Setten et al., *Comput. Phys. Commun.* **226**, 39 (2018)
- PseudoDojo ONCVPSP-LDA-SR-noCCv0.4 `Si.djrepo` deltafactor \(E_\mathrm{min}\)
