# GPW PBE-D4 analytic ionic forces — periodic water dimer

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `gpw-d4-forces` |
| Regression | `regressions/gpw/gpw-d4-forces/` |
| Public API | `cpu::gpw::GPWEvaluator` (`d4 = true`, `compute_forces = true`); `cpu::scf::dftd4` / `D4Result`; `GpwResult::e_disp` [Ha] |
| Roadmap | `docs/roadmap/dft.md` §8 Dispersion |

## Summary

Product **Γ-point GPW** (PBE) **analytic ionic forces** with additive
**DFT-D4** (Becke–Johnson two-body, PBE parameters; ATM off) on the same
**periodic water-dimer** cell as `gpw-d4-water` / `gpw-d4-stress`. Electronic
Hellmann–Feynman + Pulay forces from `gpw_forces_*` are **augmented** by the
geometry-only dispersion gradient

\[
\mathbf{F}_A^{\mathrm{D4}} = -\frac{\partial E_{\mathrm{disp}}}{\partial\mathbf{R}_A}
\quad[\mathrm{Ha/Bohr}],
\]

from `cpu::scf::dftd4` — the exact gradient of the same two-body energy pinned
by `gpw-d4-water` against offline **dftd4 3.7.0**.

**Why dftd4 rather than live CP2K QS ENERGY_FORCE?** CP2K 2026.1 on Arrhenius
was built **without** the DFTD4 library (`TYPE DFTD4` aborts). Hard pins are
**libdftd4 / dftd4 3.7.0** energy + printed gradient (`-g`). Twin POSCAR /
notes live under `tools/cp2k_oracle/inputs/gpw-d4-forces/` (and geometry under
`gpw-d4-water/`) for hosts that link libdftd4.

Public surface: bool members `d4` and `compute_forces` on
`cpu::gpw::GPWEvaluator` / `GpwDft` (no `*Config` bag). Product gate:
`regressions/gpw/gpw-d4-forces` (`reg_gpw-d4-forces`). Wall ratio is
**perf_exempt** (no fair CP2K QS ENERGY_FORCE twin on the pin host — same policy
as `gpw-d4-water` / `gpw-d4-stress`).

## Method

**Product ionic force** on atom \(A\):

\[
\mathbf{F}_A =
\mathbf{F}_A^{\mathrm{elec}}
+ \mathbf{F}_A^{\mathrm{D4}},
\]

where \(\mathbf{F}_A^{\mathrm{elec}}\) is the GPW analytic electronic force
(Hellmann–Feynman + Pulay + grid/core terms consistent with the SCF functional
— see [gpw-forces-water](gpw-forces-water.md)), and \(\mathbf{F}_A^{\mathrm{D4}}\)
is geometry-only.

**Two-body DFT-D4 with BJ damping** (Caldeweyher et al., *J. Chem. Phys.*
**150**, 154122 (2019)), periodic lattice sums over image vectors
\(\mathbf{T}\):

\[
E_{\mathrm{disp}}^{(2)} = -\sum_{A\le B}\sum_{\mathbf{T}}{}^{\prime}
\left[
  \frac{s_6\,C_6^{AB}}{R_{AB,T}^6 + f^6}
  + \frac{s_8\,C_8^{AB}}{R_{AB,T}^8 + f^8}
\right],
\quad
f = a_1\sqrt{3\,r_4^{A} r_4^{B}} + a_2,
\]

with \(C_8^{AB} = 3\,C_6^{AB}\,r_4^{A} r_4^{B}\). The prime excludes the
singular \(R=0\) self term. **ATM / MBD scale off** (\(s_9 = 0\); pin flag
`--mbdscale 0.0`).

**Charge / CN model (periodic).** Partial charges from **EEQ2019**;
EN-weighted erf coordination numbers; \(C_6\) from reference polarizability
interpolation (in-tree tables). Periodic EEQ uses an Ewald-split A-matrix.
Default cutoffs match dftd4: CN **30 Bohr**, two-body disp **60 Bohr**.

**PBE-D4 parameters** (SAW190103 BJ set; dftd4 `pbe-D4`):

| Parameter | Value | Unit |
| --------- | ----: | ---- |
| \(s_6\) | 1.0 | dimensionless |
| \(s_8\) | 0.95948085 | dimensionless |
| \(a_1\) | 0.38574991 | dimensionless |
| \(a_2\) | 4.80688534 | Bohr |

**Force implementation note.** `cpu::scf::dftd4` returns the **analytic**
\(F = -\nabla E_{\mathrm{disp}}\) of the two-body functional (BJ pair +
CN\(_\mathrm{d4}\) weight chain + EEQ charge-response adjoint). Offline
dftd4 prints **gradient** \(g = +\partial E/\partial R\); the product
comparison is \(F_{\mathrm{ATV}} \stackrel{?}{=} -g_{\mathrm{dftd4}}\).
Because D4 is geometry-only, it does **not** change the SCF density —
product wiring is therefore

\[
\mathbf{F}^{\mathrm{d4\,on}} - \mathbf{F}^{\mathrm{plain}}
\;\approx\;
\mathbf{F}^{\mathrm{D4\,standalone}}
\]

within SCF residual (regression tol \(1\times 10^{-7}\) Ha/Bohr).

**Relation to sibling slugs.**

| Slug | Deliverable |
| ---- | ----------- |
| `gpw-d4-water` | \(E_{\mathrm{disp}}\) pin only |
| `gpw-d4-stress` | Analytic fixed-P cell stress + D4 |
| **`gpw-d4-forces`** | Analytic **ionic** forces + D4 (this page) |
| `dft-d4-h2o-pbe` | Molecular (non-periodic) D4 energy vs PySCF |

**Key exclusions.**

- Three-body ATM / MBD scale (\(s_9 \neq 0\))
- Full ice Ih supercell / phonons
- Absolute QS \(E_{\mathrm{tot}}\) parity vs CP2K (DFTD4 N/A on pin host)
- k-point GPW forces with D4 as a separate multi-k product twin (Γ product path)
- Multi-SCF finite-difference dual-gate of full product forces on this 6-atom
  cell (budget; unit dual-gate on cheap MT H₂ in `test_cpu_gpw_forces`)
- CUDA GPW
- Live dftd4 / CP2K from ctest

## Implementation

Public surface only (`Frame`, `Kind` lookup constructors, `GPWEvaluator`
members; no `*Config`, no `make_*` / `build_*` factories):

```cpp
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_evaluator.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/cpu/scf/dftd4.h"
#include "atomiverse/molrs.hpp"

using namespace ATV;

// Water dimer, atom order O H H O H H (REFERENCE.md); cell 12 Å cubic.
cpu::Frame frame(6);
// … fill type / x / y / z [Å]; zero fx/fy/fz …
frame.box = cpu::Box::cubic(12.0);

const cpu::gpw::Kind O{
    molrs::Element::O,
    cpu::gpw::GaussianBasis(molrs::Element::O, "DZVP-MOLOPT-SR-GTH"),
    cpu::gpw::GthPotential(molrs::Element::O, "GTH-PBE-q6")};
const cpu::gpw::Kind H{
    molrs::Element::H,
    cpu::gpw::GaussianBasis(molrs::Element::H, "DZVP-MOLOPT-SR-GTH"),
    cpu::gpw::GthPotential(molrs::Element::H, "GTH-PBE-q1")};

cpu::gpw::GPWEvaluator eval({O, H});
eval.cutoff_ry = 200.0;
eval.max_iter = 120;
eval.tol = 1e-6;
eval.mix = 0.4;
eval.xc = cpu::gpw::XcKind::PBE;
eval.nonperiodic = false;
eval.d4 = true;              // exclusive vs d3 / d3bj / ts / mbd / xdm
eval.compute_forces = true;
eval.compute_stress = false;

cpu::Evaluate driver;
driver.on(evaluate::Phase::evaluate).add_ref(eval);
driver.build();
driver.run(frame);

// eval.last_result().e_disp  [Hartree]  — pin vs dftd4
// frame.span(kAtom_f*)       [eV/Å]    — product forces (elec + D4)
// Standalone D4 forces [Ha/Bohr]:
//   auto r = cpu::scf::dftd4(Z, pos_bohr, cell_bohr, /*periodic=*/true);
//   r.forces[A] = −∂E_disp/∂R_A
```

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
  cmake --build build -j$(nproc) --target atomiverse reg_gpw-d4-forces
  ./build/regressions/reg_gpw-d4-forces
'
```

| Path | Role |
| ---- | ---- |
| `include/atomiverse/cpu/gpw/gpw_evaluator.h` | `GPWEvaluator::d4`, `compute_forces` |
| `include/atomiverse/cpu/gpw/gpw_solver.h` | `GpwDft::d4`, `GpwResult::e_disp` |
| `include/atomiverse/cpu/scf/dftd4.h` | `dftd4` / `dftd4_energy` / `D4Result` |
| `src/cpu/gpw/internal/gpw_forces.cpp` | Adds `d4.forces` into product forces |
| `tests/cpu/gpw/test_cpu_gpw_dftd4.cpp` | Unit E_disp + force pin |
| `tests/cpu/gpw/test_cpu_gpw_forces.cpp` | H₂ analytic-vs-FD dual-gate with `d4` |
| `regressions/gpw/gpw-d4-forces/reg_gpw-d4-forces.cpp` | Product gate |
| `regressions/gpw/gpw-d4-forces/REFERENCE.md` | Offline pins + perf_exempt |
| `tools/cp2k_oracle/inputs/gpw-d4-forces/` | Pin notes + dftd4 JSON dumps |
| `tools/cp2k_oracle/inputs/gpw-d4-water/` | POSCAR + CP2K twin (DFTD4 when linked) |

## Validation (CP2K / dftd4)

Source:
[`regressions/gpw/gpw-d4-forces/REFERENCE.md`](../../../regressions/gpw/gpw-d4-forces/REFERENCE.md).
Oracle: offline **dftd4 3.7.0** (Arrhenius EasyBuild
`dftd4/3.7.0-gfbf-2025b`), CLI gradient on
`tools/cp2k_oracle/inputs/gpw-d4-water/POSCAR`. External engine pin — not a
golden of Atomiverse’s own output; **never** launch live dftd4 or CP2K from
ctest.

**Setup:**

| Item | Value |
| ---- | ----- |
| System | Water dimer, **6 atoms**, order **O H H O H H** |
| Geometry [Å] | O (0,0,0); H (0.758602, 0, ±0.504284); O (3,0,0); H (3.758602, 0, ±0.504284) |
| Cell | ABC **12.0 12.0 12.0 Å**, PERIODIC XYZ |
| Basis / PP | `DZVP-MOLOPT-SR-GTH` + `GTH-PBE-q6` (O) / `GTH-PBE-q1` (H) |
| Occupations | RKS, 16 valence e⁻ → `n_occ = 8` |
| XC + disp | PBE host + DFT-D4 two-body (`s9=0`) |
| Grid | CUTOFF **200** Ry (E_disp / F_D4 are basis-independent; SCF only needs to run) |
| Pin command | `dftd4 -f pbe --mbdscale 0.0 --grad pin_grad.txt --json pin.json POSCAR` |

**Note on atom order.** POSCAR / dftd4 JSON is **O O H H H H**; ATV
regressions and force tables use **O H H O H H**. Gradients below are already
remapped.

### Hard pins (dftd4 3.7.0)

| Quantity | Offline pin | Tol (ATV gate) | Notes |
| -------- | ----------: | -------------- | ----- |
| \(E_{\mathrm{disp}}\) [Ha] | **−0.00099458842745846774** | \(1\times 10^{-8}\) Ha | Same as `gpw-d4-water` |
| max\|F_d4 + g_dftd4\| [Ha/Bohr] | (identity) | \(1\times 10^{-7}\) | Force = −gradient |
| max\|(F_d4_on − F_plain) − F_standalone\| [Ha/Bohr] | (wiring) | \(1\times 10^{-7}\) | SCF residual |
| Unit dual-gate max\|F_an − F_fd\| [Ha/Bohr] | H₂ MT | \(5\times 10^{-5}\) | Unit suite only |

**Remapped offline gradient** \(g = \partial E/\partial R\) [Ha/Bohr]
(force pin \(F = -g\)):

| Atom | \(g_x\) | \(g_y\) | \(g_z\) |
| ---- | ------- | ------- | ------- |
| O0 | −5.3738961271913915e−05 | ~0 | ~0 |
| H0 | −5.9550148958160894e−05 | ~0 | −1.0810592129949165e−05 |
| H1 | −5.9550148958160921e−05 | ~0 | +1.0810592129949182e−05 |
| O1 | +1.3609823922940861e−04 | ~0 | ~0 |
| H2 | +1.8370509979413634e−05 | ~0 | −8.5497124425277334e−06 |
| H3 | +1.8370509979413539e−05 | ~0 | +8.5497124425278096e−06 |

Gradient norm ≈ **1.719×10⁻⁴** Ha/Bohr.

### Measured ATV (first green, 2026-07-26)

| Quantity | Value |
| -------- | ----: |
| \(E\) (Γ PBE-D4 200 Ry) | **−34.3313259463** Ha (SCF 11 iters) |
| \(E_{\mathrm{disp}}\) | **−9.9458849003×10⁻⁴** Ha (\|Δ pin\| = 6.3×10⁻¹¹) |
| max\|F_d4 + g_dftd4\| | **3.0×10⁻¹¹** Ha/Bohr |
| max\|(F_d4 − F_plain) − F_standalone\| | **2.9×10⁻⁹** Ha/Bohr |
| max\|F_product\| | **0.136** Ha/Bohr |
| \(T_{\mathrm{product}}\) (OMP=8, Release) | **17.6** s |
| \(T_{\mathrm{plain}}\) twin (OMP=8) | **17.1** s |

**Tolerance rationale.** Matching geometry, PBE-D4 parameters, EEQ2019, and
periodic lattice sums track dftd4 to nano-Hartree / sub-nHa/Bohr on the
standalone D4 path. Wiring tol \(10^{-7}\) Ha/Bohr absorbs SCF residual when
subtracting two full product force evaluations. Product electronic force
magnitude (~0.1 Ha/Bohr) is set by the GPW host, not by D4 (~10⁻⁴ Ha/Bohr).

Offline re-pin (human only; never ctest):

```bash
export PATH=/software/sse2/el9_epyc9005/easybuild/prefix/software/dftd4/3.7.0-gfbf-2025b/bin:$PATH
export LD_LIBRARY_PATH=/software/sse2/el9_epyc9005/easybuild/prefix/software/dftd4/3.7.0-gfbf-2025b/lib:$LD_LIBRARY_PATH
# + FlexiBLAS / GCCcore from buildenv-gcc module
cd tools/cp2k_oracle/inputs/gpw-d4-water
dftd4 -f pbe --mbdscale 0.0 --grad pin_grad.txt --json pin.json POSCAR
# remap O O H H H H → O H H O H H before comparing to ATV
```

## Performance

Source: `REFERENCE.md` (queue / handoff: **perf_exempt**).

D4 two-body force evaluation is \(O(N^2 \times N_{\mathrm{images}})\) (same
order as the energy; not \(6N\) energy evaluations) on a 6-atom cell
(micro–milliseconds). Full GPW SCF + electronic forces dominate wall time.
A fair **full QS ENERGY_FORCE** ratio vs CP2K is **unavailable** on Arrhenius
(no DFTD4-linked CP2K).

| Twin | \(T_{\mathrm{oracle}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | --------------------------: | ----------------------: | ----: | ---- | ------ |
| Full QS ENERGY_FORCE | n/a (CP2K lacks DFTD4 on Arrhenius) | ~17.6 (product, OMP=8) | **perf_exempt** | — | Twin input retained for DFTD4-linked hosts |
| D4 gradient pin | n/a (CLI dftd4 offline) | micro–ms (geometry-only) | **perf_exempt** | — | Primary hard gate is accuracy |

**perf_exempt = true.** Reason (from REFERENCE / closed acceptance): primary
deliverable is dftd4 E_disp + gradient pin parity and product wiring; CP2K QS
wall is unavailable (no DFTD4 build). Same policy as `gpw-d4-water` /
`gpw-d4-stress`. Re-measure a strict full-SCF ENERGY_FORCE twin later if
product policy requires \(r = T_{\mathrm{ATV}}/T_{\mathrm{CP2K}} \le 10\) on a
host with libdftd4-linked CP2K.

## Limitations

- **Cell is a water dimer**, not bulk ice Ih — sufficient for periodic D4
  force / EEQ parity; not a structural ice benchmark.
- **Hard gates are D4 energy + −∇E + wiring**, not absolute GPW force
  components vs CP2K QS (DFTD4 missing on pin host).
- **Two-body only** — no ATM / C9 three-body / MBD term (\(s_9 = 0\)).
- **Geometry-only D4 two-body forces** are the analytic gradient of
  \(E_{\mathrm{disp}}\) (pair + CN + EEQ adjoint). ATM (\(s_9 \neq 0\)) still
  uses central FD of the full energy.
- **`d3` / `d3bj` / `d4` / `ts` / `mbd` / `xdm` mutually exclusive** — any two
  true throws in the force path.
- **Multi-SCF FD of full product forces** on this cell is out of regression
  budget; dual-gate lives on cheap H₂ in the unit suite.
- **Oracle split:** GPW pin is dftd4 3.7.0; molecular `dft-d4-h2o-pbe` is
  PySCF + pyscf-dispersion — do not mix geometries / BC when comparing.
- Frame forces are eV/Å; pin comparison must convert to Ha/Bohr
  (`units::kEvPerAngstromInHartreePerBohr`).
- Wall **perf_exempt** for this wave; not a speed optimization target.

## References

- Offline pin: `regressions/gpw/gpw-d4-forces/REFERENCE.md`
- Oracle inputs: `tools/cp2k_oracle/inputs/gpw-d4-forces/`,
  `tools/cp2k_oracle/inputs/gpw-d4-water/`
- Oracle notes: `tools/cp2k_oracle/README.md`
- Roadmap: `docs/roadmap/dft.md` §8 Dispersion
- Related energy pin: [gpw-d4-water](gpw-d4-water.md)
- Related stress product: `regressions/gpw/gpw-d4-stress/`
- Related bare GPW forces: [gpw-forces-water](gpw-forces-water.md)
- Related D3(BJ) energy: [gpw-pbe-d3bj-ice](gpw-pbe-d3bj-ice.md)
- Related molecular D4: [dft-d4-h2o-pbe](dft-d4-h2o-pbe.md)
- GPW narrative chapter: `docs/tutorial/ch09-gpw.md`
- Closed spec (git history): `.claude/specs/gpw-d4-forces/` (deleted at done;
  harness wave 9 / 2026-07-26)
- E. Caldeweyher, C. Bannwarth, S. Grimme, *J. Chem. Phys.* **150**, 154122
  (2019) (DFT-D4)
- J. VandeVondele et al., *Comput. Phys. Commun.* **167**, 103 (2005) (GPW)
- dftd4 3.7.0 (CLI pin backend; same library family CP2K links as DFTD4)
