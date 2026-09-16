# GPW cell stress (strain-FD) — H₂O / PBE / DZVP-MOLOPT-GTH (MT)

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `gpw-stress-water` |
| Regression | `regressions/gpw/gpw-stress-water/` |
| Public API | `cpu::gpw::GPWEvaluator` (`compute_stress`, `stress_use_fd`, `stress_strain_h`); product analytic default — see [gpw-analytic-stress](gpw-analytic-stress.md); free `gpw_stress` / `gpw_stress_fd` |
| Related narrative | [Ch. 9 GPW](../ch09-gpw.md) |
| Queue | CP2K capability loop — wave 2 done (2026-07-24) |

## Summary

Closed-shell **Γ-point Gaussian-and-plane-waves (GPW)** **cell stress** on
isolated water, product path matching CP2K Quickstep
`STRESS_TENSOR NUMERICAL`. v1 is **central finite difference of the SCF
energy under orthorhombic strain** at fixed fractional coordinates (same
strategy as plane-wave `pw_stress`), not analytic Nielsen–Martin / Pulay
stress. Wavefunction in contracted Gaussians (DZVP-MOLOPT-GTH); Hartree /
XC / local short-range GTH PP on a uniform grid (cutoff 400 Ry);
electrostatics Martyna–Tuckerman isolated Poisson (`nonperiodic = true`).
Public surface: `cpu::gpw::GPWEvaluator` members `compute_stress` and
`stress_strain_h` (no `*Config` bag) bound into `cpu::Evaluate`; stress on
Frame as `kStress` [eV/Å³] Voigt, pin channel
`last_stress_ha_bohr3()` [Ha/Bohr³]. Product gate:
`regressions/gpw/gpw-stress-water` (`reg_gpw-stress-water`), offline CP2K
**2026.1** NUMERICAL pin. Fair wall ratio
\(r = T_{\mathrm{ATV}}/T_{\mathrm{CP2K}} = 4.75\) vs ENERGY_FORCE baseline
(≤10×).

## Method

**GPW Kohn–Sham (Quickstep)** (VandeVondele et al., *Comput. Phys. Commun.*
**167**, 103 (2005)): density matrix in the Gaussian AO basis; collocate
\(\rho\) onto a multigrid; FFT Poisson + grid XC; integrate potentials back
to the AO KS matrix; diagonalize / mix until residual \(\le\) `tol`.

**Cell stress v1 (strain-FD).** For orthorhombic cell edge lengths \(L_a\)
and volume \(\Omega\), apply a dimensionless diagonal strain step \(h\) at
**fixed fractional coordinates** (scale \(L_a\) and Cartesian \(r_a\) by
\(1\pm h\)), recompute the SCF energy, and form a central difference.
Sign convention matches CP2K `STRESS_TENSOR`:

\[
\sigma_{aa}
=
-\Omega^{-1}
\frac{\partial E}{\partial\varepsilon_{aa}}
\approx
-\Omega^{-1}
\frac{E(+h_a)-E(-h_a)}{2h},
\qquad a\in\{x,y,z\},
\]

with \(E\) in Hartree and \(\Omega\) in Bohr³. Positive \(\sigma_{xx}\) when
expanding the box lowers the energy. Voigt order:
\((\sigma_{xx},\sigma_{yy},\sigma_{zz},\sigma_{xy},\sigma_{xz},\sigma_{yz})\).
Default \(h = 2\times 10^{-3}\) (`stress_strain_h`). Product cost is
**6 strained SCF** (2 per diagonal) **+ 1** unperturbed energy SCF.

**Why NUMERICAL, not ANALYTICAL (binding pin choice).** On this nonperiodic
MT water twin, CP2K `STRESS_TENSOR ANALYTICAL` reports diagonals of order
\(-3\times 10^{5}\) bar while `NUMERICAL` reports order \(+10^{3}\) bar.
Atomiverse strain-FD matches **NUMERICAL** in magnitude and sign after the
CP2K convention above. Analytic GPW stress is a follow-on; the product pin
is the same-method FD twin.

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
| Stress | CP2K `STRESS_TENSOR NUMERICAL`; ATV strain-FD \(h=2\times 10^{-3}\) |
| `local_pp_on_grid` | **true** (product default) |
| Observable | Voigt \(\sigma\) [Ha/Bohr³]; \(E_{\mathrm{tot}}\) [Ha] soft check |

**Geometry [Å]:**

| Atom | \(x\) | \(y\) | \(z\) |
| ---- | ----: | ----: | ----: |
| O | 5.0000 | 5.0000 | 5.0000 |
| H1 | 5.7586 | 5.0000 | 5.5043 |
| H2 | 4.2414 | 5.0000 | 5.5043 |

**Key exclusions (closed-spec `out_of_scope`):**

- Analytic Nielsen–Martin / Pulay stress decomposition — follow-on.
- Triclinic / off-diagonal shear strain — GPW cell orthorhombic only; v1
  off-diagonal Voigt components are **exactly 0**.
- BOMD trajectory pin (`gpw-bomd-water`).
- UKS stress, k-point GPW, D3/D3BJ/D4 stress contributions.
- Public `*Config` bags / `make_*` factories; live CP2K from ctest.
- CUDA GPW.

## Implementation

Public surface only (`cpu::Frame`, `cpu::gpw::Kind` / lookup constructors,
`GPWEvaluator` members; no public `*Config`, no `make_*` factories). Match
the regression:

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
    /*compute_forces*/ false);
eval.compute_stress = true;
eval.stress_strain_h = 2e-3;

cpu::Evaluate driver;
driver.on(evaluate::Phase::evaluate).add_ref(eval);
driver.build();
driver.run(frame);

// Energy: frame.meta.get(kEnergy) [eV]; eval.last_energy_hartree() [Ha]
// Stress: frame.meta.get(kStress) [eV/Å³] Voigt;
//         eval.last_stress_ha_bohr3() [Ha/Bohr³] for pin compare
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
  cmake --build build --target reg_gpw-stress-water -j$(nproc)
  ./build/regressions/reg_gpw-stress-water
'
```

| Path | Role |
| ---- | ---- |
| `include/atomiverse/cpu/gpw/gpw_evaluator.h` | Public `compute_stress`, `stress_strain_h`, `last_stress_ha_bohr3()` |
| `include/atomiverse/cpu/gpw/gpw_solver.h` | Public free `gpw_stress` [Ha/Bohr³] |
| `include/atomiverse/cpu/gpw/internal/gpw_stress.h` | Private strain-FD implementation |
| `src/cpu/gpw/internal/gpw_stress.cpp` | Orthorhombic central FD of SCF energy |
| `src/cpu/gpw/gpw_evaluator.cpp` | Wire stress into Evaluate launch + `kStress` |
| `tests/cpu/gpw/test_cpu_gpw_stress.cpp` | Unit: finite diagonals, off-diag ≈ 0 (not CP2K e2e) |
| `regressions/gpw/gpw-stress-water/reg_gpw-stress-water.cpp` | Product gate |
| `regressions/gpw/gpw-stress-water/REFERENCE.md` | Offline CP2K pins + fair wall |
| `tools/cp2k_oracle/inputs/h2o_stress_mt.inp` | CP2K twin (`STRESS_TENSOR NUMERICAL`) |

## Validation (vs CP2K)

Source:
[`regressions/gpw/gpw-stress-water/REFERENCE.md`](../../../regressions/gpw/gpw-stress-water/REFERENCE.md).
Oracle: offline **CP2K 2026.1** (`cp2k.psmp`, EasyBuild
`2026.1-gmpflf-2026.03`), Arrhenius epyc, `RUN_TYPE ENERGY_FORCE` +
`STRESS_TENSOR NUMERICAL`. Hard-coded pins in the regression binary —
**never** live CP2K from ctest.

**Setup:**

| Item | Value |
| ---- | ----- |
| Engine | CP2K 2026.1 Quickstep GPW |
| Host | Arrhenius epyc (offline oracle) |
| Twin input | `tools/cp2k_oracle/inputs/h2o_stress_mt.inp` |
| Threads (pin wall) | single rank, `OMP_NUM_THREADS=4`, `OMP_PROC_BIND=close` |
| Atomiverse path | `GPWEvaluator` + `Evaluate`, `compute_stress=true`, \(h=2\times 10^{-3}\) |

**Offline CP2K pins (unperturbed energy + NUMERICAL stress):**

CP2K prints stress in **bar**; conversion factor from CP2K’s internal table:
\(1\,\mathrm{Ha/Bohr}^3 = 2.94210107994716\times 10^{8}\,\mathrm{bar}\).

| Quantity | CP2K pin | Tol (ATV gate) | Notes |
| -------- | -------: | -------------- | ----- |
| \(E_{\mathrm{tot}}\) [Ha] | **−17.212337150586624** | \(1\times 10^{-3}\) Ha | Soft check (same class as forces twin) |
| \(\sigma_{xx}\) [Ha/Bohr³] | **2.195224914616×10⁻⁵** | max diag \(5\times 10^{-6}\) | From \(+6.45857359202\times 10^{3}\) bar |
| \(\sigma_{yy}\) [Ha/Bohr³] | **1.321554147453×10⁻⁸** | same | From \(+3.88814588443\) bar |
| \(\sigma_{zz}\) [Ha/Bohr³] | **1.270036962196×10⁻⁵** | same | From \(+3.73657711805\times 10^{3}\) bar |
| \(\sigma_{xy},\sigma_{xz},\sigma_{yz}\) | \(\sim 10^{-14}\)–\(10^{-15}\) | ATV **exactly 0**; soft \(\lvert\sigma_{\mathrm{cp2k}}\rvert\le 10^{-8}\) | Orthorhombic v1 |

**Measured ATV vs pin (REFERENCE):**

| Metric | Value |
| ------ | ----: |
| max\|\(\sigma_{\mathrm{diag,atv}}-\sigma_{\mathrm{cp2k}}\)\| | **9.1×10⁻⁹** Ha/Bohr³ |
| \|ΔE\| | **4.2×10⁻⁴** Ha |
| SCF iters (unperturbed) | **9** |

**Tolerance rationale.** Same-method twin (ATV strain-FD vs CP2K
`NUMERICAL`). Diagonal gate \(5\times 10^{-6}\) Ha/Bohr³ is generous vs
measured residual \(\sim 10^{-8}\) (sub-percent of \(\lvert\sigma_{xx}\rvert\sim 2\times 10^{-5}\));
it absorbs strain step / SCF / multigrid differences while requiring correct
order and sign. Energy soft check \(1\times 10^{-3}\) Ha matches
`gpw-forces-water`. Do **not** pin against CP2K `ANALYTICAL` on this MT
cluster twin (gross bar-scale mismatch; documented in REFERENCE).

Offline oracle sketch (host with CP2K module; not ctest):

```bash
module load ... CP2K/2026.1
export CP2K_DATA_DIR=.../share/cp2k/data
cp2k.psmp -i tools/cp2k_oracle/inputs/h2o_stress_mt.inp -o stress.out
# ENERGY| Total FORCE_EVAL ( QS ) energy [hartree]  -17.212337150586624
# STRESS| Numerical stress tensor [bar]  (xx ~ +6.46e3, zz ~ +3.74e3, yy ~ +3.89)
```

## Performance

Source: `REFERENCE.md` fair wall on Arrhenius epyc login (same host class as
pin), 4 OMP threads, `OMP_PROC_BIND=close`, single rank. Handoff / queue
record ratio **4.75**.

- **ATV:** `./build/regressions/reg_gpw-stress-water` (strain-FD: 6+1 SCF).
- **CP2K fair denominator:** ENERGY_FORCE single-SCF baseline (**10.15 s**),
  not the full NUMERICAL multi-SCF wall — same convention as product gate
  (ATV stress pass is compared to one energy+force-class twin wall).
- **CP2K NUMERICAL full stress** (~**87 s**) is recorded for context; ATV is
  faster than that full FD twin (\(r\approx 0.55\)) but that is **not** the
  acceptance ratio.

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | ------------------------: | -----------------------: | ----: | ---- | ------ |
| H₂O GPW ENERGY_FORCE baseline | **10.15** | **48.2** | **4.75** | ✓ | Primary fair gate |
| H₂O GPW full NUMERICAL stress | **~87** | **48.2** | **~0.55** | ✓ | FD vs FD context only |

Gate: \(r = T_{\mathrm{ATV}}/T_{\mathrm{CP2K}} \le 10\). `perf_exempt = false`.
`within_10x = true`. Strain-FD multi-SCF cost is expected; analytic stress
would change both accuracy path and wall.

## Limitations

- **Orthorhombic strain-FD only** — no triclinic shear; off-diagonal Voigt
  components are hard zeros in v1. Analytic Nielsen–Martin / Pulay GPW
  stress is a separate follow-on (and a different CP2K twin:
  `STRESS_TENSOR ANALYTICAL` is **not** validated on this MT water pin).
- **Isolated MT water** — periodic solid stress, variable-cell ice, and
  phonons are other queue rows (`gpw-variable-cell-ice` dep satisfied by
  this foothold; `fd-phonon-supercell`, `pw-analytic-stress-si`).
- **Closed-shell Γ RKS** — UKS / k-points / hybrids / dispersion stress not
  covered.
- **Cost** — 6+1 SCF per stress evaluation; not a single-pass analytic
  gradient. Fair ratio uses ENERGY_FORCE baseline, not CP2K NUMERICAL wall.
- **Grid residual** still dominates absolute energy agreement
  (\(\sim 4\times 10^{-4}\) Ha); stress parity on this twin is tight because
  both sides are numerical FD of the same functional class.
- Frame stress is eV/Å³; literature / pin comparison must convert to
  Ha/Bohr³ (`kHartreeInEv`, `kBohrInAngstrom`).
- Unit tests under `tests/cpu/gpw/` only; e2e CP2K parity is exclusively
  the regression binary.

## References

- Offline pin + fair wall: `regressions/gpw/gpw-stress-water/REFERENCE.md`
- Twin input: `tools/cp2k_oracle/inputs/h2o_stress_mt.inp`
- Oracle notes: `tools/cp2k_oracle/README.md`
- Related force pin: `docs/tutorial/methods/gpw-forces-water.md`,
  `regressions/gpw/gpw-forces-water/`
- Narrative GPW: `docs/tutorial/ch09-gpw.md`
- Roadmap: `docs/roadmap/dft.md` (strain-FD cell stress ✓;
  `GPWEvaluator::compute_stress`)
- Spec history (closed, deleted at done): git `wf/gpw-stress-water` /
  commit `79de745b` (`.claude/specs/gpw-stress-water/{spec,acceptance}.md`);
  pin/docs update `d5e67a44`; feature `c20e5145` / `2a52d0e4`
- J. VandeVondele, M. Krack, F. Mohamed, M. Parrinello, T. Chassaing,
  J. Hutter, *Comput. Phys. Commun.* **167**, 103 (2005)
- CP2K Quickstep stress (`STRESS_TENSOR NUMERICAL` / `ANALYTICAL`):
  CP2K manual, FORCE_EVAL / QS stress tensor section
