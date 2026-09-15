# Plane-wave isotropic variable-cell relax — diamond Si (NC-UPF LDA)

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `pw-variable-cell-si` |
| Regression | `regressions/pw/pw-variable-cell-si/` |
| Roadmap | [optimization.md](../../roadmap/optimization.md) § cell relax; [dft.md](../../roadmap/dft.md) § PW CELL_OPT|

## Summary

Productizes **Nielsen–Martin analytic stress** (`pw-analytic-stress-si`) into a
**plane-wave isotropic CELL_OPT** path: `Minimize` drives
`IsotropicCellRelax` on intensive Voigt `kStress` [eV/Å³] written by
`PWEvaluator::with_stress(true)`, and **`StressConverged`** terminates on residual
hydrostatic stress (mechanical / CP2K convention). Twin is diamond Si
(2-atom fcc primitive), ONCVPSP-LDA NC-UPF, incomplete tutorial basis
(\(E_\mathrm{cut}=15\) Ry, \(2\times2\times2\) MP). Expanded start
\(a_0=5.58\) Å shrinks toward the ATV zero-stress basin
\(a^\*=5.510\) Å. Absolute \(a^\*\) / energy are **not** a CP2K Quickstep
GTH twin — methodology input only under `tools/cp2k_oracle/inputs/pw-variable-cell-si/`.
Product gate: `reg_pw-variable-cell-si`. Performance: **`perf_exempt`**.

## Method

**Level.** Same periodic plane-wave KS-DFT engine as `pw-lda-si` /
`pw-analytic-stress-si`. After each force-phase SCF, the evaluator writes
intensive cell stress from the Nielsen–Martin decomposition (Nielsen & Martin,
*Phys. Rev. B* **32**, 3780 (1985)). The optimizer does **not** recompute a
second SCF for stress: one production SCF supplies energy and stress.

**Stress convention (productization).** PW / GPW intensive stress is
**mechanical / CP2K**:

\[
\sigma_{\alpha\beta}
  = -\frac{1}{\Omega}\,
    \frac{\partial E}{\partial\varepsilon_{\alpha\beta}},
\qquad
\sigma = (\sigma_{xx},\sigma_{yy},\sigma_{zz},\sigma_{xy},\sigma_{xz},\sigma_{yz})
\quad [\mathrm{eV}/\mathrm{\AA}^3].
\]

Positive diagonal stress under compression drives `IsotropicCellRelax` to
**expand** the box (and vice versa). Pre-product energy-derivative pin signs in
`pw-analytic-stress-si` are flipped to this convention on the product path.
Hydrostatic pressure for the isotropic step:

\[
P = \frac{\sigma_{xx}+\sigma_{yy}+\sigma_{zz}}{3}
\quad [\mathrm{eV}/\mathrm{\AA}^3]
\quad\text{(intensive; } \texttt{stress\_is\_virial = false}\text{)}.
\]

**Cell motion.** Fixed fractional coordinates; isotropic **full-matrix** scale
via `Box::scale_isotropic` (required for the fcc primitive whose diagonal is
zero). Conventional cubic lattice constant is recovered as
\(a = 2\,\mathrm{cell}_{01}\) (off-diagonal entries \(= a/2\)).

**Optimizer / criterion.**

| Piece | Role |
| ----- | ---- |
| `IsotropicCellRelax(P_target, alpha)` | steepest-descent scale toward \(P_\mathrm{target}\) |
| `bulk_mod`, `max_scale` | soft \(B\) and per-step \(\lvert s-1\rvert\) cap (damp flat incomplete-basis EOS) |
| `StressConverged{0, stol}` | stop when \(\max_d\lvert\sigma_{dd}-\sigma^\*\rvert \le \mathrm{stol}\) |

**XC / spin / occupations.** `XCType::LSDA`; restricted closed shell; no metallic
smearing.

**BZ sampling.** Uniform Monkhorst–Pack `k_mesh`³ (regression: \(2\times2\times2\)).

**Observables.**

| Symbol | Definition | Unit |
| ------ | ---------- | ---- |
| \(a\) | Conventional cubic lattice constant | Å |
| \(a^\*\) | Zero-stress basin pin for this twin | Å |
| \(\sigma_v\) | Intensive Voigt stress on `kStress` | eV/Å³ |
| \(P\) | Hydrostatic \((\sigma_{xx}+\sigma_{yy}+\sigma_{zz})/3\) | eV/Å³ |
| \(E\) | Frame meta energy (`kEnergy`) | eV |

**Key exclusions (this slug / closed acceptance).**

- Anisotropic / constrained CELL_OPT → `AnisotropicCellRelax` / GPW
  `gpw-variable-cell-ice` (axes mask).
- Joint ion+cell relax, internal coordinates, PR NPT dynamics.
- Absolute lattice / energy parity with **CP2K Quickstep GTH-PADE + DZVP-GTH**
  (different PP and discretization).
- Production ONCV cutoffs (~60 Ry) and dense k-meshes when they exceed the
  ≤ 60 s e2e budget.
- Live CP2K / QE / ABINIT at ctest time (offline pins only).
- CUDA PW surface is not a CPU compatibility API for this pin.

## Implementation

Public surface only (`cpu::Frame` + concrete method members; no `*Config`, no
`make_*` / `build_*`). Headers:
`include/atomiverse/cpu/minimize/minimize.h`, `cell_relax.h`,
`include/atomiverse/cpu/driver/criteria.h` (`StressConverged`),
`include/atomiverse/cpu/planewave/pw_evaluator.h`, `pw_solver.h`.

Minimal sequence (matches `reg_pw-variable-cell-si.cpp`):

```cpp
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/mat3.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/driver/criteria.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/minimize/cell_relax.h"
#include "atomiverse/cpu/minimize/minimize.h"
#include "atomiverse/cpu/planewave/pw_evaluator.h"
#include "atomiverse/cpu/planewave/pw_solver.h"
#include "atomiverse/cpu/scf/xc_functional.h"

using namespace ATV;
using namespace ATV::cpu::pw;

constexpr double kA0 = 5.58;      // start expanded [Å]
constexpr double kEcutRy = 15.0;  // tutorial incomplete basis
constexpr int kMesh = 2;          // 2×2×2 MP
constexpr double kStol = 1.5e-2;  // residual stress gate [eV/Å³]

cpu::Frame frame(2);
// diamond Si fcc primitive at conventional a0 (see reg_ for full geometry)
frame.span(kAtom_type)[0] = 14;
frame.span(kAtom_type)[1] = 14;
frame.span(kAtom_mass)[0] = 28.0855;
frame.span(kAtom_mass)[1] = 28.0855;
frame.span(kAtom_x)[0] = 0.0;
frame.span(kAtom_y)[0] = 0.0;
frame.span(kAtom_z)[0] = 0.0;
frame.span(kAtom_x)[1] = kA0 / 4.0;
frame.span(kAtom_y)[1] = kA0 / 4.0;
frame.span(kAtom_z)[1] = kA0 / 4.0;
const double h = kA0 / 2.0;
Mat3 cell;
cell(0, 0) = 0; cell(0, 1) = h; cell(0, 2) = h;
cell(1, 0) = h; cell(1, 1) = 0; cell(1, 2) = h;
cell(2, 0) = h; cell(2, 1) = h; cell(2, 2) = 0;
frame.box.set_cell(cell);

PW method;
method.upf_path = "…/tests/data/pw/Si_lda_nc.upf";
method.ecut_ry = kEcutRy;
method.k_mesh = kMesh;
method.tol = 1e-6;
method.max_iter = 40;
method.xc = cpu::scf::XCType::LSDA;

cpu::minimizer::Minimize opt;
cpu::minimizer::IsotropicCellRelax cell(/*P_target*/ 0.0, /*alpha*/ 1.0);
cell.bulk_mod = 3.5;           // eV/Å³ — soft B damps incomplete-basis EOS
cell.max_scale = 0.012;        // |s−1| cap per step
opt.set_optimizer_ref(cell);
opt.add_criterion(cpu::StressConverged{0.0, kStol});
opt.on(minimizer::Phase::force)
    .add(PWEvaluator{std::move(method)}.with_stress(true));
opt.build();
opt.run(frame, 3);  // seed force + up to 2 scale steps

const double a1 = 2.0 * static_cast<double>(frame.box.cell()(0, 1));
const auto& st = frame.meta.get(kStress);  // Voigt [eV/Å³]
const double e_ev = frame.meta.get(kEnergy);  // eV
// opt.converged(), cell.pressure_residual()
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
  cmake --build build --target reg_pw-variable-cell-si -j$(nproc)
  ./build/regressions/reg_pw-variable-cell-si
'
```

UPF path is injected at compile time as `ATV_PW_UPF_PATH` →
`${CMAKE_SOURCE_DIR}/tests/data/pw/Si_lda_nc.upf`.

| Path | Role |
| ---- | ---- |
| `include/atomiverse/cpu/minimize/cell_relax.h` | `IsotropicCellRelax` |
| `include/atomiverse/cpu/driver/criteria.h` | `StressConverged` |
| `include/atomiverse/cpu/planewave/pw_evaluator.h` | `with_stress` → `kStress` |
| `include/atomiverse/cpu/planewave/pw_solver.h` | `PW`, analytic stress helpers |
| `regressions/pw/pw-variable-cell-si/reg_pw-variable-cell-si.cpp` | Product gate |
| `regressions/pw/pw-variable-cell-si/REFERENCE.md` | Offline \(a^\*\) pin + perf |
| `tools/cp2k_oracle/inputs/pw-variable-cell-si/si_cellopt.inp` | Optional CP2K CELL_OPT methodology twin (not absolute \(a^\*\)) |

## Validation (CP2K methodology; ATV zero-stress pin)

Source:
[`regressions/pw/pw-variable-cell-si/REFERENCE.md`](../../../regressions/pw/pw-variable-cell-si/REFERENCE.md).
Primary science gate is the **ATV residual-stress basin** on the same NC-UPF
twin (incomplete ecut/k), hard-coded in the regression binary — **never** a live
third-party oracle from ctest. CP2K Quickstep supplies an offline **CELL_OPT
methodology** twin only (GTH ≠ ONCV).

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
| Start \(a_0\) | **5.58 Å** (expanded) |
| Basis atoms | \((0,0,0)\), \((a/4,a/4,a/4)\) |
| Cell | \((a/2)(0,1,1)\), \((a/2)(1,0,1)\), \((a/2)(1,1,0)\) |
| \(E_\mathrm{cut}\) | 15 Ry (tutorial; ONCV production ~60 Ry) |
| k-mesh | \(2\times 2\times 2\) (`k_mesh = 2`) |
| SCF tol | \(10^{-6}\) |
| `max_iter` | 40 |
| XC | LSDA |
| Relax | α=1, `bulk_mod`=3.5 eV/Å³, `max_scale`=0.012, budget 3 |
| Criterion | `StressConverged{0, 1.5×10^{-2} eV/Å³}` |

### Accuracy table

| Quantity | Atomiverse / pin | External / oracle | Tol | Notes |
| -------- | ---------------: | ----------------: | --- | ----- |
| Relaxed conventional \(a^\*\) | **5.510 Å** | ATV residual-stress basin (this ecut/k) | ≤ 0.05 Å | Hard pin in `reg_pw-variable-cell-si` |
| Measured relaxed \(a\) (2026-07-25) | **5.5097 Å** | pin 5.510 | \(\lvert a-a^\*\rvert=3\times10^{-4}\) Å | Release, OMP=8 |
| Shrink from expanded start | \(a_1 < a_0\) | qualitative | \(a_1 < 5.58-10^{-4}\) | Productization direction gate |
| Residual \(\max_d\lvert\sigma_{dd}\rvert\) | **~8.0×10⁻³** eV/Å³ | 0 (target) | ≤ 1.2×stol = 1.8×10⁻² | Mechanical intensive stress |
| Frame energy | **≈ −213.27 eV** | finite | finite | Public eV on Frame |
| Mechanical stress context at \(a=5.43\) | σ ≈ **+0.0576** eV/Å³ | `pw-analytic-stress-si` twin | — | Compressed vs twin \(a^\*\); drives expansion if started there |

**CP2K note.** Quickstep bulk-Si CELL_OPT uses **GTH-PADE + DZVP-GTH**, not
ONCV plane-wave. Absolute \(a^\*\) / energy are **not** comparable to this
NC-UPF twin. Optional methodology input (KEEP_ANGLES / KEEP_SYMMETRY /
FIXED_ATOMS, \(2\times2\times2\), start \(a_0=5.55\) Å):
`tools/cp2k_oracle/inputs/pw-variable-cell-si/si_cellopt.inp` — offline wall /
workflow only.

**Literature / PseudoDojo context (EOS, not the hard \(a^\*\) pin).**

| Quantity | Value | Source |
| -------- | ----- | ------ |
| EOS \(a_0\) | ≈ 5.468 Å | PseudoDojo `Si.djrepo` |
| Twin \(a^\*\) (this ecut/k) | 5.510 Å | ATV residual-stress pin |
| Incomplete-basis shift | \(a^\* > a_0^\mathrm{EOS}\) | Expected at 15 Ry / \(2^3\) k |

### Hard gates (product)

| Gate | Tolerance | Measured (2026-07-25, Release, OMP=8) |
| ---- | --------- | ------------------------------------- |
| shrink from expanded start | \(a_1 < a_0\) | **5.5097 < 5.58** ✓ |
| \(\lvert a - a^\*\rvert\) | ≤ 0.05 Å | **3×10⁻⁴ Å** |
| residual \(\max\lvert\sigma_{dd}\rvert\) | ≤ 1.2×stol = 1.8×10⁻² | **8.0×10⁻³** |
| SCF / energy finite | — | E ≈ −213.27 eV ✓ |
| wall | ≤ 60 s | **50.2 s** |

## Performance

Source: `regressions/pw/pw-variable-cell-si/REFERENCE.md`. Queue / handoff:
**`perf_exempt`** (no fair CP2K NC-UPF PW CELL_OPT twin; CP2K GTH ≠ ONCV —
same class as `pw-analytic-stress-si` / `pw-lda-si`).

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | ------------------------: | ----------------------: | ----: | ---- | ------ |
| PW isotropic CELL_OPT Si ecut=15 / k=2 | n/a | **50.2** (informative) | n/a | n/a (**exempt**) | ≤3 force launches; no fair NC-UPF PW oracle twin |

| Item | Value |
| ---- | ----- |
| `perf_exempt` | **true** |
| Reason | No fair NC-UPF PW CELL_OPT twin (CP2K Quickstep GTH-PADE ≠ ONCV PW); absolute \(a^\*\) not a CP2K digit twin |
| ATV wall (informative) | **50.2 s** (Release, OMP=8) |
| \(T_{\mathrm{cp2k}}\) | n/a (no fair twin) |
| `ratio_atv_over_cp2k` | n/a (exempt) |
| `within_10x` | n/a (exempt) |
| Wall budget | e2e ≤ 60 s single-rank CPU (PASS) |

Offline CP2K methodology wall (not a ratio gate):

```bash
module load buildenv-gcc/2026.03-mpich
export CP2KROOT=…/CP2K/2026.1-…   # site install
export CP2K_DATA_DIR=$CP2KROOT/share/cp2k/data
export PATH=$CP2KROOT/bin:$PATH
export OMP_NUM_THREADS=4
cd tools/cp2k_oracle/inputs/pw-variable-cell-si
mpirun -np 1 cp2k.psmp -i si_cellopt.inp -o cellopt.out
# wall only — absolute a* not comparable to NC-UPF
```

## Limitations

- Tutorial ecut/k only — equilibrium \(a^\*\) is incomplete-basis relative to
  PseudoDojo EOS \(a_0\approx 5.468\) Å; not a production lattice-constant
  productization.
- Soft `bulk_mod` / small `max_scale` and a 3-iteration budget are tuned for
  the flat incomplete-basis EOS and the ≤ 60 s e2e gate — not a general
  production CELL_OPT schedule.
- Absolute CP2K Quickstep CELL_OPT lattice is a different PP/basis world —
  fair science comparison is structural response methodology, not digit-for-digit
  \(a^\*\).
- Isotropic full-matrix scale only; no anisotropic axes mask, no joint ionic
  minimization, no NPT dynamics on this slug.
- GGA (PBE) cell relax, MPI multi-rank CELL_OPT, and denser k / higher ecut
  remain out of closed acceptance.
- CUDA PW surface is not a CPU compatibility API for this pin.

## References

- Offline pin: [`regressions/pw/pw-variable-cell-si/REFERENCE.md`](../../../regressions/pw/pw-variable-cell-si/REFERENCE.md)
- Regression: [`regressions/pw/pw-variable-cell-si/reg_pw-variable-cell-si.cpp`](../../../regressions/pw/pw-variable-cell-si/reg_pw-variable-cell-si.cpp)
- Analytic stress foothold: [pw-analytic-stress-si](pw-analytic-stress-si.md)
- Energy foothold twin: [pw-lda-si](pw-lda-si.md)
- Optional CP2K CELL_OPT methodology input:
  `tools/cp2k_oracle/inputs/pw-variable-cell-si/si_cellopt.inp`
- Oracle rules: `tools/cp2k_oracle/README.md` (offline pins only)
- Narrative style ref: [Chapter 8 — Plane-Wave DFT](../ch08-plane-wave.md),
  [Chapter 12 — Geometry optimization](../ch12-geometry-optimization.md)
- O. H. Nielsen and R. M. Martin, *Phys. Rev. B* **32**, 3780 (1985)
- D. R. Hamann, *Phys. Rev. B* **88**, 085117 (2013)
- M. J. van Setten et al., *Comput. Phys. Commun.* **226**, 39 (2018)
- PseudoDojo ONCVPSP-LDA-SR-noCCv0.4 `Si.djrepo` (EOS context)
