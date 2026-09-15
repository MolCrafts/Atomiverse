# Orbital Transformation (OT) SCF — isolated H₂O / GPW PBE

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `ot-minimizer-water` |
| Regression | `regressions/gpw/ot-minimizer-water/` |
| Public API | `cpu::gpw::GPWEvaluator` + member `use_ot` (ctor arg or assign) |
| Roadmap | `docs/roadmap/dft.md` §10 GPW — OT minimizer ✓ (`reg_ot-minimizer-water`) |

## Summary

**Orbital Transformation (OT)** is a diagonalization-free SCF path for closed-shell
Γ-point **GPW** DFT: occupied MO coefficients are optimized on the Stiefel-type
manifold \(C^\top S C = I\) instead of a dense generalized eigensolve + Pulay
DIIS each outer step. Product surface is the public member
`cpu::gpw::GPWEvaluator::use_ot` (default `false`; no `*Config` bag, no env
required). This page pins **OT total energy** for the CP2K getting-started
isolated water (PBE / DZVP-MOLOPT-GTH / GTH-PBE / Martyna–Tuckerman) against an
offline **CP2K 2026.1** OT twin, and dual-checks OT vs in-tree DIIS on the same
grids. Product gate: `regressions/gpw/ot-minimizer-water`
(`reg_ot-minimizer-water`). Fair OT-only wall ratio
\(r = T_{\mathrm{ATV}}/T_{\mathrm{CP2K}} \approx 9.0\) (OMP=4, ≤10×).

## Method

**Setting.** Closed-shell Γ-point Gaussian-and-plane-waves (GPW) Kohn–Sham DFT
with PBE, contracted MOLOPT Gaussians, GTH pseudopotentials, and isolated
electrostatics (Martyna–Tuckerman Poisson, `nonperiodic = true`). Wavefunction
in the Gaussian AO basis; density/Hartree/XC on a uniform real-space grid with
plane-wave cutoff \(E_{\mathrm{cut}}\) [Ry]. Public energy on Frame is eV
(`kEnergy`); method channel is Hartree (`last_energy_hartree()` /
`GpwResult::e_total`).

**OT parameterization** (Kühne et al., *J. Chem. Phys.* **152**, 194103 (2020)
§VII C; VandeVondele & Hutter, *J. Chem. Phys.* **118**, 4365 (2003)). For
reference occupied coefficients \(C_0\) with \(C_0^\top S C_0 = I\) and an
\(S\)-tangent step \(x\) (\(C_0^\top S x = 0\)):

\[
C(x) = C_0\,\cos U + x\,U^{-1}\sin U,\qquad
U = (x^\top S x)^{1/2},
\]

which preserves \(C(x)^\top S C(x) = I\) for any such \(x\). Closed-shell
band-energy gradient consistent with \(P = 2\,C C^\top\):

\[
g = (I - S C_0 C_0^\top)\,(4 H C_0).
\]

**Minimization.** Preconditioned conjugate gradient on the OT manifold. In-tree
preconditioners (internal, not public API): FULL_KINETIC \(M = T + S\), and
FULL_SINGLE_INVERSE-style \(M = H + a S\) (CP2K default class for gapped
systems). Outer SCF rebuilds \(H\) from the collocated density until residual /
energy criteria meet `tol`.

**DIIS twin path.** `use_ot = false` (default): generalized diagonalization of
\((H,S)\) plus density mixing (`mix`). Same self-consistent fixed point for a
gapped closed-shell ground state when both converge on identical grids.

**Key exclusions (binding from closed spec).**

- Sparse / linear-scaling OT preconditioner ○ (dense \(O(n_b^3)\) inverse cost).
- Open-shell OT / UKS OT ○ (UKS energy path is separate: `gpw-uks-o2-gamma`).
- k-point OT ○.
- Driver-phase rewrite of the GPW SCF loop ○ (OT remains inside the GPW inner
  solver / evaluator).
- MPI multi-rank OT scaling ○.

## Implementation

Public surface only (`cpu::Frame` + `GPWEvaluator` members / ctor args; no
`*Config`, no factories). Match the regression (`250` Ry pin budget; tutorial
class in `examples/09_gpw.cpp` is `400` Ry):

```cpp
#include "atomiverse/cpu/evaluate/single_point.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_evaluator.h"
#include "atomiverse/cpu/gpw/gth_potential.h"

using namespace ATV;

// CP2K getting-started h2o [Å], 10 Å cubic cell (MT Poisson)
cpu::Frame frame(3);
const int z[3] = {8, 1, 1};
const double x[3]  = {5.0000, 5.7586, 4.2414};
const double y[3]  = {5.0000, 5.0000, 5.0000};
const double zc[3] = {5.0000, 5.5043, 5.5043};
for (int i = 0; i < 3; ++i) {
    frame.span(kAtom_type)[i] = z[i];
    frame.span(kAtom_x)[i] = x[i];
    frame.span(kAtom_y)[i] = y[i];
    frame.span(kAtom_z)[i] = zc[i];
}
frame.box = cpu::Box::cubic(10.0);

const cpu::gpw::Kind O{
    molrs::Element::O,
    cpu::gpw::GaussianBasis(molrs::Element::O, "DZVP-MOLOPT-GTH"),
    cpu::gpw::GthPotential(molrs::Element::O, "GTH-PBE-q6")};
const cpu::gpw::Kind H{
    molrs::Element::H,
    cpu::gpw::GaussianBasis(molrs::Element::H, "DZVP-MOLOPT-GTH"),
    cpu::gpw::GthPotential(molrs::Element::H, "GTH-PBE-q1")};

cpu::gpw::GPWEvaluator eval(
    /*kinds*/ {O, H}, /*cutoff_ry*/ 250.0, /*max_iter*/ 200, /*tol*/ 1e-7,
    /*mix*/ 0.3, cpu::gpw::XcKind::PBE, /*nonperiodic*/ true,
    /*compute_forces*/ false, /*fd_step*/ 1e-3, /*d3*/ false,
    /*use_ot*/ true);
// equivalent after construction: eval.use_ot = true;

cpu::single_point(frame, eval);
const double e_ha = eval.last_energy_hartree();   // [Hartree]
const bool ok = eval.last_result().converged;
// frame.meta.get(kEnergy) is eV (public Frame unit)
```

Diagnostic env `ATV_GPW_OT` may still force OT on (OR with the member); product
code and the regression **must not** depend on it.

Build and run the regression:

```bash
bash -lc '
  module load buildenv-gcc/2026.03-mpich
  export OPENBLAS_ROOT=$OPENBLASDIR
  export RUSTUP_TOOLCHAIN=stable-x86_64-unknown-linux-gnu
  export LD_LIBRARY_PATH=$LIBRARY_PATH:$LD_LIBRARY_PATH
  cmake -B build -DCMAKE_BUILD_TYPE=Release \
        -DATV_BUILD_TESTS=ON -DATV_BUILD_HF=ON \
        -DATV_BUILD_PW=ON -DATV_BUILD_GPW=ON \
        -DATV_BUILD_REGRESSIONS=ON -DATV_BUILD_CUDA=OFF \
        -DBLA_VENDOR=OpenBLAS
  cmake --build build --target reg_ot-minimizer-water -j$(nproc)
  ./build/regressions/reg_ot-minimizer-water
'
```

| Path | Role |
| ---- | ---- |
| `include/atomiverse/cpu/gpw/gpw_evaluator.h` | Public `use_ot` member + ctor |
| `include/atomiverse/cpu/gpw/gpw_solver.h` | `GpwDft::use_ot` (inner solver) |
| `include/atomiverse/cpu/gpw/internal/ot.h` | OT parameterization / preconds (internal) |
| `src/cpu/gpw/gpw_evaluator.cpp` | Wire member into `GpwDft` at `build()` |
| `tests/cpu/gpw/test_cpu_gpw_ot.cpp` | Unit: fixed-H OT, mini-H₂ OT≈DIIS, member toggle |
| `regressions/gpw/ot-minimizer-water/` | E2E product gate + `REFERENCE.md` |
| `tools/cp2k_oracle/inputs/ot-minimizer-water-ot.inp` | Offline CP2K OT twin |
| `tools/cp2k_oracle/inputs/ot-minimizer-water-diag.inp` | Offline CP2K diag twin |

## Validation (vs CP2K)

Source: [`regressions/gpw/ot-minimizer-water/REFERENCE.md`](../../../regressions/gpw/ot-minimizer-water/REFERENCE.md).
Oracle: offline **CP2K 2026.1** (`cp2k.psmp`, 1 MPI rank); **never** live in ctest.

**Setup:**

| Item | Value |
| ---- | ----- |
| System | Isolated H₂O, charge 0, closed shell |
| Geometry [Å] | O (5, 5, 5); H (5.7586, 5, 5.5043); H (4.2414, 5, 5.5043) |
| Cell | 10 Å cubic, `PERIODIC NONE`, Martyna–Tuckerman |
| Basis / PP | DZVP-MOLOPT-GTH; GTH-PBE-q6 (O), GTH-PBE-q1 (H) |
| XC | PBE |
| Cutoff (pin) | **250 Ry** (regression budget; 400 Ry tutorial class cross-check only) |
| SCF | `max_iter=200`, `tol=1e-7`; OT vs DIIS dual path |
| CP2K OT settings | `&OT` `MINIMIZER DIIS`, `PRECONDITIONER FULL_SINGLE_INVERSE`, outer SCF |

**Pinned energies [Hartree] (CUTOFF 250, 2026-07-23):**

| Quantity | Atomiverse / pin | CP2K 2026.1 | Tol | Notes |
| -------- | ---------------: | ----------: | --- | ----- |
| \(E_{\mathrm{tot}}\) OT | −17.2118583480 | **−17.212848291913666** | **1.5×10⁻³ Ha** | primary offline pin |
| \(E_{\mathrm{tot}}\) DIIS (ATV) | −17.2118583479 | — | — | same grids as OT |
| \(\lvert E_{\mathrm{OT}}-E_{\mathrm{CP2K}}\rvert\) | \(9.90\times10^{-4}\) | — | ≤ 1.5×10⁻³ | measured |
| \(\lvert E_{\mathrm{OT}}-E_{\mathrm{DIIS}}\rvert\) (ATV) | \(\sim 4\times10^{-11}\) | — | ≤ 1×10⁻⁴ | fixed-point parity |

Milli-Ha pin tol at 250 Ry absorbs residual grid/multigrid defaults (NGRIDS /
REL_CUTOFF / collocation) that still differ from CP2K; OT vs DIIS in-tree is
near machine parity on identical grids. At 400 Ry (not the regression pin)
CP2K OT was −17.212337205347978 (DIAG −17.212337010579070) — energy-class
cross-check only.

Offline oracle sketch (host with CP2K module; not ctest):

```bash
module load ... CP2K/2026.1
export CP2K_DATA_DIR=.../share/cp2k/data
cp2k.psmp -i tools/cp2k_oracle/inputs/ot-minimizer-water-ot.inp -o ot.out
# ENERGY| Total FORCE_EVAL ( QS ) energy [hartree]  -17.212848291913666
```

## Performance

Source: fair wall table in
`regressions/gpw/ot-minimizer-water/REFERENCE.md` (2026-07-23). Queue record:
ratio **~9.0** (fair OT-only, OMP=4). Gate \(r\le 10\) unless `perf_exempt`.

Comparable settings: cutoff 250 Ry, PBE, MT, single rank, `OMP_NUM_THREADS=4`.

| Twin | Wall [s] | \(r = T_{\mathrm{ATV}}/T_{\mathrm{CP2K}}\) | ≤10× | Caveat |
| ---- | -------: | ----------------------------------------: | ---- | ------ |
| CP2K OT (`cp2k.psmp`) | **4.31** | — | — | primary twin |
| ATV OT-only (est. from reg timestamps) | **~39** | **~9.0** | ✓ | fair OT vs OT |
| ATV OT+DIIS (full regression binary) | 41.73 | — | — | **not** a fair twin vs CP2K OT alone |
| ATV OT+DIIS OMP=1 | 54.01 | — | — | e2e ≤60 s budget |

**Caveats.** (1) Report the **OT-only** ratio; the regression binary runs both
OT and DIIS and must not be ratioed against CP2K OT alone. (2) Dense OT
preconditioner cost dominates small-basis water; ratio does not forecast
sparse/linear-scaling OT. (3) `within_10x=true` at handbook write time; no
optimize pass required for this foothold.

## Limitations

- Closed-shell Γ-point GPW only; no UKS OT, no k-point OT, no MPI industrial
  scaling gate on this slug.
- Dense \(O(n_b^3)\) preconditioner inverse — sparse OT is out of scope
  (closed-spec `out_of_scope`).
- Pin is energy-class at 250 Ry (milli-Ha vs CP2K); not a sub-µHa multigrid
  parity claim. Tighter agreement needs matched NGRIDS/REL_CUTOFF and higher
  cutoff (tutorial 400 Ry class).
- Forces / stress / BOMD with OT are separate queue rows (`gpw-forces-water` ✓
  energy+forces DIIS path; `gpw-stress-water`, `gpw-bomd-water` incomplete).
- PW backend has no OT product switch (`docs/roadmap/dft.md` OT column: GPW ✓,
  PW ○, CUDA ○).
- Speed: ~9× CP2K on this microcase under OMP=4 fair OT twin — acceptable for
  the wave-1 gate, not a claim of production Quickstep parity.

## References

- Offline pin + fair wall: `regressions/gpw/ot-minimizer-water/REFERENCE.md`
- Regression binary: `regressions/gpw/ot-minimizer-water/reg_ot-minimizer-water.cpp`
- Closed spec (git history): `.claude/specs/ot-minimizer-water/` (deleted at
  done; see commit `feat(gpw): productize OT SCF + CP2K water pin`)
- CP2K oracle inputs: `tools/cp2k_oracle/inputs/ot-minimizer-water-*.inp`
- Narrative GPW chapter: `docs/tutorial/ch09-gpw.md`
- Roadmap: `docs/roadmap/dft.md` §10 (OT ✓ on GPW)
- VandeVondele, J.; Hutter, J. *J. Chem. Phys.* **118**, 4365 (2003),
  doi:10.1063/1.1543154 (original OT)
- Kühne, T. D. et al. *J. Chem. Phys.* **152**, 194103 (2020) §VII C,
  doi:10.1063/5.0007045 (CP2K / formulation followed in-tree)
