# GPW analytic forces — H₂O / PBE / DZVP-MOLOPT-GTH (MT)

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `gpw-forces-water` |
| Regression | `regressions/gpw/gpw-forces-water/` |
| Public API | `cpu::gpw::GPWEvaluator` (`compute_forces = true`); `cpu::Evaluate` |
| Related narrative | [Ch. 9 GPW](../ch09-gpw.md) |
| Queue | CP2K capability loop — wave 1 done |

## Summary

Closed-shell **Γ-point Gaussian-and-plane-waves (GPW)** analytic nuclear
forces on isolated water, product path matching CP2K Quickstep
`ENERGY_FORCE`. The wavefunction lives in contracted Gaussians
(DZVP-MOLOPT-GTH); Hartree / XC / local short-range GTH PP are evaluated on a
uniform real-space grid (cutoff 400 Ry); electrostatics use
Martyna–Tuckerman isolated Poisson (`nonperiodic = true`). Forces are
Hellmann–Feynman + Pulay + grid/core terms written to
`frame.span(kAtom_f*)` in eV/Å after one SCF + gradient. Public surface:
`cpu::gpw::GPWEvaluator` constructor args / members (no `*Config` bag) bound
into `cpu::Evaluate`. Product gate:
`regressions/gpw/gpw-forces-water` (`reg_gpw-forces-water`), offline CP2K
**2026.1** pin. Fair wall ratio
\(r = T_{\mathrm{ATV}}/T_{\mathrm{CP2K}} = 3.56\) (≤10×).

## Method

**GPW Kohn–Sham (Quickstep)** (VandeVondele et al., *Comput. Phys. Commun.*
**167**, 103 (2005)): density matrix in the Gaussian AO basis; collocate
\(\rho\) onto a multigrid; FFT Poisson + grid XC; integrate potentials back
to the AO KS matrix; diagonalize / mix until residual \(\le\) `tol`.

**Ionic forces** (analytic product path, `gpw_forces_analytic`):

\[
\mathbf{F}_A = -\frac{\partial E}{\partial \mathbf{R}_A}
=
\mathbf{F}_A^{\mathrm{HF}}
+ \mathbf{F}_A^{\mathrm{Pulay}}
+ \mathbf{F}_A^{\mathrm{grid/core}},
\]

where Hellmann–Feynman collects explicit \(\mathbf{R}_A\)-dependence of the
Hamiltonian (local PP, nonlocal projectors, nuclear–electron terms), Pulay
is the energy-weighted density matrix contracted with \(\partial S/\partial R_A\),
and grid/core terms cover collocated short-range local PP and multigrid
discretization consistent with the SCF functional.

**Twin settings (binding):**

| Knob | Value |
| ---- | ----- |
| System | Isolated H₂O, CP2K getting-started geometry |
| Cell | Cubic 10 Å; MT / `PERIODIC NONE` |
| XC | PBE |
| Basis | DZVP-MOLOPT-GTH |
| PP | GTH-PBE-q6 (O), GTH-PBE-q1 (H) |
| Grid | CUTOFF 400 Ry (CP2K REL_CUTOFF 60 Ry, NGRIDS 4) |
| SCF | EPS_SCF / `tol` \(1\times 10^{-7}\) |
| `local_pp_on_grid` | **true** (product default; energy and gradient both read it) |
| Observable | Cart forces [Ha/Bohr]; \(E_{\mathrm{tot}}\) [Ha] soft check |

**Geometry [Å]:**

| Atom | \(x\) | \(y\) | \(z\) |
| ---- | ----: | ----: | ----: |
| O | 5.0000 | 5.0000 | 5.0000 |
| H1 | 5.7586 | 5.0000 | 5.5043 |
| H2 | 4.2414 | 5.0000 | 5.5043 |

**Key exclusions (this page / acceptance):**

- Cell stress (`gpw-stress-water`) — not gated here.
- BOMD trajectory pin (`gpw-bomd-water`) — forces only.
- Open-shell UKS GPW, k-point GPW, hybrid/GAPW — other queue rows.
- Live CP2K from ctest / regression binary — offline pin only.
- Finite-difference forces as the product gate (unit test only:
  analytic vs FD on a tiny system).

**Historical consistency note.** `local_pp_on_grid` must be the same for
energy and analytic gradient. Energy-only grid PP with a force path that
omits collocated local SR PP produced a ZnO analytic-vs-FD residual of
order 0.53 Ha/Bohr; the product path keeps both sides on grid PP
(`cpu/gpw/internal/local_pp.h`).

## Implementation

Public surface only (`cpu::Frame`, `cpu::gpw::Kind` / lookup constructors,
`GPWEvaluator` members; no public `*Config`, no `make_*` factories):

```cpp
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_evaluator.h"
#include "atomiverse/cpu/gpw/gth_potential.h"

using namespace ATV;

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
    frame.span(kAtom_fx)[i] = frame.span(kAtom_fy)[i] =
        frame.span(kAtom_fz)[i] = 0;
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
    /*kinds*/ {O, H}, /*cutoff_ry*/ 400.0, /*max_iter*/ 200, /*tol*/ 1e-7,
    /*mix*/ 0.3, /*xc*/ cpu::gpw::XcKind::PBE, /*nonperiodic*/ true,
    /*compute_forces*/ true);

cpu::Evaluate driver;
driver.on(evaluate::Phase::evaluate).add_ref(eval);
driver.build();
driver.run(frame);

// Energy: frame.meta.get(kEnergy) [eV]; also eval.last_energy_hartree() [Ha]
// Forces: kAtom_fx/y/z [eV/Å] → Ha/Bohr via units::kEvPerAngstromInHartreePerBohr
```

Build and run the regression (requires GPW + HF + PW + regressions):

```bash
bash -lc '
  module load buildenv-gcc/2026.03-mpich
  export OPENBLAS_ROOT=$OPENBLASDIR
  export RUSTUP_TOOLCHAIN=stable-x86_64-unknown-linux-gnu
  export LD_LIBRARY_PATH=$LIBRARY_PATH:$LD_LIBRARY_PATH
  cmake -B build -DCMAKE_BUILD_TYPE=Release \
        -DATV_BUILD_TESTS=ON -DATV_BUILD_HF=ON -DATV_BUILD_PW=ON \
        -DATV_BUILD_GPW=ON -DATV_BUILD_REGRESSIONS=ON \
        -DATV_BUILD_CUDA=OFF -DBLA_VENDOR=OpenBLAS
  cmake --build build --target reg_gpw-forces-water -j$(nproc)
  ./build/regressions/reg_gpw-forces-water
'
```

| Path | Role |
| ---- | ---- |
| `include/atomiverse/cpu/gpw/gpw_evaluator.h` | Public `GPWEvaluator` / `Kind` |
| `include/atomiverse/cpu/gpw/gaussian_basis.h` | MOLOPT lookup constructor |
| `include/atomiverse/cpu/gpw/gth_potential.h` | GTH-PBE lookup constructor |
| `src/cpu/gpw/internal/` | Private SCF + analytic force kernels |
| `regressions/gpw/gpw-forces-water/reg_gpw-forces-water.cpp` | Product gate |
| `regressions/gpw/gpw-forces-water/REFERENCE.md` | Offline CP2K pins + fair wall |
| `tools/cp2k_oracle/inputs/h2o_forces_mt.inp` | CP2K twin input |
| `tests/cpu/gpw/test_cpu_gpw_forces.cpp` | Unit analytic-vs-FD (not CP2K e2e) |

## Validation (CP2K)

Source:
[`regressions/gpw/gpw-forces-water/REFERENCE.md`](../../../regressions/gpw/gpw-forces-water/REFERENCE.md).
Oracle: offline **CP2K 2026.1** (`cp2k.psmp`, EasyBuild
`2026.1-gmpflf-2026.03`), Arrhenius epyc, `RUN_TYPE ENERGY_FORCE`. Hard-coded
pins in the regression binary — **never** live CP2K from ctest.

**Setup:**

| Item | Value |
| ---- | ----- |
| Engine | CP2K 2026.1 Quickstep GPW |
| Host | Arrhenius epyc (offline oracle) |
| Twin input | `tools/cp2k_oracle/inputs/h2o_forces_mt.inp` |
| Threads (pin wall) | `mpirun -n 1`, `OMP_NUM_THREADS=4` |
| Atomiverse path | `GPWEvaluator` + `Evaluate`, `compute_forces=true`, product MT + PBE |

**Offline CP2K FORCE_EVAL pins:**

| Quantity | CP2K pin | Tol (ATV gate) | Notes |
| -------- | -------: | -------------- | ----- |
| \(E_{\mathrm{tot}}\) [Ha] | **−17.21233715058673** | \(1\times 10^{-3}\) Ha | Soft check; multigrid residual ~\(4\times 10^{-4}\) Ha |
| \(F_{\mathrm{O}}\) [Ha/Bohr] | \((-6.62274644\times 10^{-8},\,-1.03414433\times 10^{-6},\,-8.96660013\times 10^{-2})\) | max component \(1\times 10^{-3}\) | Spec A6 |
| \(F_{\mathrm{H1}}\) [Ha/Bohr] | \((+5.16349669\times 10^{-2},\,-2.88080683\times 10^{-9},\,+4.49073908\times 10^{-2})\) | same | |
| \(F_{\mathrm{H2}}\) [Ha/Bohr] | \((-5.16349704\times 10^{-2},\,-2.33864544\times 10^{-10},\,+4.49073720\times 10^{-2})\) | same | |
| \(\|\sum_A\mathbf{F}_A\|\) [Ha/Bohr] | CP2K print residual \(\sim 1.5\times 10^{-4}\) | \(\le 5\times 10^{-4}\) | Spec A7; numerical residual, not free-molecule machine zero |

**Measured ATV vs pin (REFERENCE):**

| Metric | Value |
| ------ | ----: |
| max\|F_atv − F_cp2k\| | **3.42×10⁻⁴** Ha/Bohr |
| \|∑F\| | **2.1×10⁻⁴** Ha/Bohr |
| \|ΔE\| | **4.2×10⁻⁴** Ha |
| SCF iters | **9** |

**Tolerance rationale.** Same-method twin (Quickstep GPW / product MT + PBE +
MOLOPT/GTH). Component force tol \(1\times 10^{-3}\) Ha/Bohr is dominated by
grid / multigrid / mixing residuals, not chemical accuracy; physical
\(|F|\) on O\(_z\) is \(\sim 0.09\) Ha/Bohr. Energy soft check at
\(1\times 10^{-3}\) Ha covers product multigrid residual \(\sim 4\times 10^{-4}\) Ha
on this twin. Translational residual gate matches CP2K’s own printed residual
scale.

## Performance

Source: `REFERENCE.md` fair wall on Arrhenius epyc login (same host class as
pin), 4 OMP threads, `OMP_PROC_BIND=close`, single rank.

- **ATV:** `./build/regressions/reg_gpw-forces-water` (one Evaluate pass with
  analytic forces).
- **CP2K:** `mpirun -n 1 cp2k.psmp -i h2o_forces_mt.inp` (`ENERGY_FORCE`).

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | ------------------------: | -----------------------: | ----: | ---- | ------ |
| H₂O GPW ENERGY_FORCE | **11.2** | **39.9** | **3.56** | ✓ | Single-rank, 4 OMP; not multi-rank scaling |

Gate: \(r = T_{\mathrm{ATV}}/T_{\mathrm{CP2K}} \le 10\). `perf_exempt = false`.
Handoff / queue record the same ratio **3.56** for wave-1 merge.

## Limitations

- **Isolated MT water only** — periodic stress, supercell phonons, and BOMD
  trajectory pins are separate queue rows (`gpw-stress-water`,
  `fd-phonon-supercell`, `gpw-bomd-water`).
- **Closed-shell Γ RKS** — UKS / k-points / hybrids not covered
  (`gpw-uks-o2-gamma`, `gpw-kpoints-si`, blocked hybrid/GAPW).
- **Grid residual dominates** force/energy agreement at \(\sim 10^{-4}\) Ha
  / Ha/Bohr; not a sub-µHa absolute energy target (contrast molecular
  Gaussian SCF pins).
- **Wall ~3.6× CP2K** on this twin is within gate; further speed work is
  not required for acceptance A9.
- **Unit analytic-vs-FD** lives under `tests/cpu/gpw/` only; e2e CP2K
  parity is exclusively the regression binary.
- Frame forces are eV/Å; literature / pin comparison must convert to
  Ha/Bohr (`units::kEvPerAngstromInHartreePerBohr`).

## References

- Offline pin: `regressions/gpw/gpw-forces-water/REFERENCE.md`
- Twin input: `tools/cp2k_oracle/inputs/h2o_forces_mt.inp`
- Oracle notes: `tools/cp2k_oracle/README.md`
- Narrative GPW: `docs/tutorial/ch09-gpw.md`
- Spec history (closed, deleted at done): git `wf/gpw-forces-water` /
  commit `94e64ea9` (`.claude/specs/gpw-forces-water/spec.md`)
- J. VandeVondele, M. Krack, F. Mohamed, M. Parrinello, T. Chassaing,
  J. Hutter, *Comput. Phys. Commun.* **167**, 103 (2005)
