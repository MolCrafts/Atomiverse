# GPW PBE-D4 analytic cell stress — periodic water dimer

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `gpw-d4-stress` |
| Regression | `regressions/gpw/gpw-d4-stress/` |
| Public API | `cpu::gpw::GPWEvaluator` (`d4`, `compute_stress`, `stress_use_fd=false`); free `gpw_stress` / `gpw_stress_fd` with `GpwDft::d4` |
| Roadmap | `docs/roadmap/dft.md` §8 Dispersion |

## Summary

Productizes **Hellmann–Feynman fixed-\(P\) orthorhombic cell stress** on
**Γ-point GPW PBE** with additive **DFT-D4** (Becke–Johnson two-body, ATM
off) for a **periodic water dimer** (same 12 Å cubic cell as
`gpw-d4-water` / `gpw-pbe-d3bj-ice`). Dispersion is geometry-only; the
analytic path rebuilds \(E_{\mathrm{disp}}^{\mathrm{D4}}(\{\mathbf{R}\},\mathbf{h})\)
at each strained cell while holding the unstrained density matrix \(P\) and
energy-weighted \(W\).

**Oracle split (binding).** Hard \(E_{\mathrm{disp}}\) pin is offline
**dftd4 3.7.0** (Arrhenius CP2K 2026.1 lacks DFTD4). Decisive stress gate is
**internal dual-gate**: product analytic vs multi-SCF strain-FD
(`gpw_stress_fd`) on the same `d4=true` method — not an absolute CP2K
`STRESS_TENSOR` pin on this host. Twin POSCAR / QS input retained under
`tools/cp2k_oracle/inputs/gpw-d4-water/` (+ `inputs/gpw-d4-stress/` notes).

Public surface: bool / stress members on `cpu::gpw::GPWEvaluator` and
`cpu::gpw::GpwDft` (no `*Config` bag, no `make_*` / `build_*`). Frame
`kStress` is eV/Å³; pin channel `last_stress_ha_bohr3()` [Ha/Bohr³].
Product gate: `regressions/gpw/gpw-d4-stress` (`reg_gpw-d4-stress`). Wall
ratio is **perf_exempt** (no fair CP2K DFTD4 QS stress wall).

## Method

### Host + dispersion

Closed-shell **Γ-point Gaussian-and-plane-waves (GPW)** Kohn–Sham
(VandeVondele et al., *Comput. Phys. Commun.* **167**, 103 (2005)) with
PBE XC and GTH PBE PPs. **DFT-D4** (Caldeweyher et al., *J. Chem. Phys.*
**150**, 154122 (2019)) is an additive London correction with BJ damping
and **periodic** lattice sums / EEQ2019 / EN-weighted CNs — same two-body
path as [`gpw-d4-water`](gpw-d4-water.md) (`s_9 = 0`, `--mbdscale 0.0`).

**PBE-D4 parameters** (SAW190103 BJ; dftd4 `pbe-D4`):

| Parameter | Value | Unit |
| --------- | ----: | ---- |
| \(s_6\) | 1.0 | dimensionless |
| \(s_8\) | 0.95948085 | dimensionless |
| \(a_1\) | 0.38574991 | dimensionless |
| \(a_2\) | 4.80688534 | Bohr |

### Analytic fixed-\(P\) cell stress with D4

One SCF at the unstrained cell → density matrix \(P\) and energy-weighted
\(W = \sum_i 2\varepsilon_i c_i c_i^\top\). Then **central FD of the
non-self-consistent Lagrangian** under orthorhombic diagonal strain at
**fixed fractional coordinates**:

\[
E_L[P,W;\varepsilon]
  = \mathrm{Tr}(P\,T) + \mathrm{Tr}(P\,V_{\mathrm{pp}})
  + E_{\mathrm{es}}[\rho_P] + E_{\mathrm{xc}}[\rho_P]
  + E_{\mathrm{disp}}^{\mathrm{D4}}(\{\mathbf{R}\},\mathbf{h})
  - \mathrm{Tr}(W\,S).
\]

The \(-\mathrm{Tr}(W\,S)\) term is the Pulay contribution. D4 is rebuilt
from strained Cartesian positions / cell (geometry-only; no SCF density
dependence). Stress sign matches CP2K `STRESS_TENSOR`:

\[
\sigma_{aa}
=
-\Omega^{-1}
\frac{\partial E_L}{\partial\varepsilon_{aa}},
\qquad a\in\{x,y,z\},
\]

with \(E\) in Hartree and \(\Omega\) in Bohr³. Voigt order
\((\sigma_{xx},\sigma_{yy},\sigma_{zz},\sigma_{xy},\sigma_{xz},\sigma_{yz})\);
product off-diagonals are ~0 on this orthorhombic cell. Default
\(h = 2\times 10^{-3}\) (`stress_strain_h`).

**Oracle twin (multi-SCF FD).** `gpw_stress_fd` fully re-SCFs at \(\pm h\)
per axis (D4 folded via the same `fold_dispersion` path as energy). Used
only for dual-gate residual, never as the product default
(`stress_use_fd=false`).

**Relation to other stress / D4 slugs.**

| Slug | Deliverable |
| ---- | ----------- |
| [`gpw-d4-water`](gpw-d4-water.md) | Periodic D4 **energy** pin only |
| [`gpw-analytic-stress`](gpw-analytic-stress.md) | Analytic fixed-\(P\) **without** D4 product gate |
| [`gpw-stress-water`](gpw-stress-water.md) | Strain-FD / CP2K NUMERICAL H₂O MT (no D4) |
| **`gpw-d4-stress`** | Analytic fixed-\(P\) **+ D4** dual-gate + E_disp pin |

**Key exclusions.**

- Absolute CP2K `STRESS_TENSOR` pin (host lacks DFTD4)
- Three-body ATM / MBD (\(s_9 \neq 0\))
- D4 **nuclear forces** product gate (energy + stress only)
- Shear / triclinic strain; non-orthorhombic Voigt product
- UKS / multi-k / DFT+U stress on this pin geometry
- Molecular `cpu::scf::D4` path changes
- CUDA GPW
- Public `*Config` / `make_*` / live dftd4 or CP2K from ctest

## Implementation

Public surface only (`GPWEvaluator` / `GpwDft` members; Frame in/out).
Regression binds the evaluator into `cpu::Evaluate`:

```cpp
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_evaluator.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/molrs.hpp"

// Frame: 6 atoms, water dimer [Å], cubic 12 Å (REFERENCE.md).
// kinds: O DZVP-MOLOPT-SR-GTH + GTH-PBE-q6; H … + GTH-PBE-q1

cpu::gpw::GPWEvaluator eval({O, H});
eval.cutoff_ry = 200.0;
eval.max_iter = 120;
eval.tol = 1e-6;
eval.mix = 0.4;
eval.xc = cpu::gpw::XcKind::PBE;
eval.nonperiodic = false;
eval.d4 = true;              // exclusive vs d3 / d3bj / ts / xdm / mbd
eval.compute_forces = false;
eval.compute_stress = true;  // product analytic fixed-P (default path)
eval.stress_use_fd = false;  // multi-SCF FD is oracle only
eval.stress_strain_h = 2e-3;

cpu::Evaluate driver;
driver.on(evaluate::Phase::evaluate).add_ref(eval);
driver.build();
driver.run(frame);

const double e_disp = eval.last_result().e_disp;          // [Ha]
const auto& san = eval.last_stress_ha_bohr3();            // [Ha/Bohr³] Voigt
// frame.meta.get(kStress)  // [eV/Å³]

// Dual-gate oracle (same geometry + d4) — product tests only:
// auto sfd = cpu::gpw::gpw_stress_fd(atoms, method, /*h=*/2e-3, …);
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
  cmake --build build -j$(nproc) --target atomiverse reg_gpw-d4-stress
  ./build/regressions/reg_gpw-d4-stress
  # or: cmake --build build --target run_reg_gpw-d4-stress
'
```

| Path | Role |
| ---- | ---- |
| `include/atomiverse/cpu/gpw/gpw_evaluator.h` | `d4`, `compute_stress`, `stress_use_fd`, `last_stress_ha_bohr3` |
| `include/atomiverse/cpu/gpw/gpw_solver.h` | `GpwDft::d4`, `gpw_stress` / `gpw_stress_fd` |
| `src/cpu/gpw/internal/gpw_stress.cpp` | Fixed-\(P\)+Pulay + D4 rebuild; multi-SCF FD |
| `include/atomiverse/cpu/scf/dftd4.h` / tables | Shared D4 parameter tables / energy |
| `tests/cpu/gpw/test_cpu_gpw_stress.cpp` | Unit dual-gate (stress family) |
| `regressions/gpw/gpw-d4-stress/reg_gpw-d4-stress.cpp` | Product gate |
| `regressions/gpw/gpw-d4-stress/REFERENCE.md` | Offline pins + walls |
| `tools/cp2k_oracle/inputs/gpw-d4-stress/` | Pin notes (reuse `gpw-d4-water` POSCAR) |
| `tools/cp2k_oracle/inputs/gpw-d4-water/` | POSCAR + CP2K twin (DFTD4 when linked) |

## Validation (dftd4 + dual-gate)

Source:
[`regressions/gpw/gpw-d4-stress/REFERENCE.md`](../../../regressions/gpw/gpw-d4-stress/REFERENCE.md).
Hard energy oracle: offline **dftd4 3.7.0**. Stress oracle: **internal**
multi-SCF strain-FD on the same method. External-engine / dual-gate pins —
**never** live dftd4 or CP2K from ctest.

**Setup:**

| Item | Value |
| ---- | ----- |
| System | Water dimer (ice-relevant H-bond pair), 6 atoms |
| Geometry [Å] | O (0,0,0); H (0.758602, 0, ±0.504284); O (3,0,0); H (3.758602, 0, ±0.504284) |
| Cell | ABC **12.0 12.0 12.0 Å**, PERIODIC XYZ |
| Basis / PP | `DZVP-MOLOPT-SR-GTH` + `GTH-PBE-q6` (O) / `GTH-PBE-q1` (H) |
| XC + disp | PBE host + DFT-D4 two-body (\(s_9=0\)) |
| Occupations | RKS, 16 valence e⁻ → `n_occ = 8` |
| Grid | CUTOFF **200** Ry |
| Stress step | \(h = 2\times 10^{-3}\) |
| E_disp pin command | `dftd4 -f pbe --mbdscale 0.0 --json pin.json POSCAR` |

| Quantity | Atomiverse / pin | Oracle | Tol | Notes |
| -------- | ---------------: | -----: | --- | ----- |
| \(E_{\mathrm{disp}}\) [Ha] | −0.00099458842745846774 | dftd4 3.7.0 same | \(1\times 10^{-8}\) Ha | Hard gate (shared with `gpw-d4-water`) |
| max\|\(\sigma^{\mathrm{an}}-\sigma^{\mathrm{fd}}\)\| diag | **7.39×10⁻⁷** (first green) | multi-SCF FD twin | \(2\times 10^{-5}\) Ha/Bohr³ | Decisive dual-gate |
| off-diag product \(\sigma\) | ~0 | — | \(1\times 10^{-8}\) Ha/Bohr³ | Orthorhombic Voigt |
| max\|\(\sigma_{\mathrm{diag}}\)\| floor | finite | — | \(> 1\times 10^{-10}\) | Non-trivial stress |
| Frame `kStress` | eV/Å³ from Ha/Bohr³ | unit conversion | \(1\times 10^{-9}\) rel | Product path check |
| \(E_{\mathrm{tot}}\) [Ha] | −34.3313259463 (context) | — | not absolute-parity gated | First green, 11 SCF iters |

**Measured product diagonals (first green, 2026-07-26):**

| Component | Analytic [Ha/Bohr³] |
| --------- | ------------------: |
| \(\sigma_{xx}\) | −2.17585158×10⁻⁵ |
| \(\sigma_{yy}\) | +2.00723219×10⁻⁵ |
| \(\sigma_{zz}\) | +5.82669568×10⁻⁵ |

**Tolerance rationale.** Matching geometry, PBE-D4 tables, and periodic
sums track dftd4 to nano-Hartree on this cell. Analytic vs multi-SCF FD
residual \(\sim 10^{-6}\)–\(10^{-7}\) Ha/Bohr³ is geometry-only D4 +
fixed-\(P\) consistency; gate is **2×10⁻⁵** Ha/Bohr³ (same order as other
GPW stress dual-gates). Absolute CP2K stress unavailable without DFTD4.

Offline re-pin (human only; never ctest):

```bash
export PATH=/software/sse2/el9_epyc9005/easybuild/prefix/software/dftd4/3.7.0-gfbf-2025b/bin:$PATH
export LD_LIBRARY_PATH=/software/sse2/el9_epyc9005/easybuild/prefix/software/dftd4/3.7.0-gfbf-2025b/lib:$LD_LIBRARY_PATH
# + FlexiBLAS / GCCcore from buildenv-gcc module
cd tools/cp2k_oracle/inputs/gpw-d4-water
dftd4 -f pbe --mbdscale 0.0 --json pin.json POSCAR
# Dispersion energy: -9.9458842745847E-04 Eh
```

## Performance

Source: `REFERENCE.md` (queue / handoff: **perf_exempt**).

| Twin | \(T_{\mathrm{oracle}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | --------------------------: | ----------------------: | ----: | ---- | ------ |
| Product analytic (OMP=8, Release) | n/a (no CP2K DFTD4) | **21.4** | **perf_exempt** | — | One SCF + fixed-\(P\) FD |
| Multi-SCF FD twin (OMP=8) | — (internal) | **42.3** | — | — | Dual-gate only; 6 strained SCFs |
| Dual-gate wall (sum) | — | **~64** | — | — | Not a fair external ratio |
| Full QS stress vs CP2K | n/a | — | **perf_exempt** | — | Arrhenius CP2K lacks DFTD4 |

**perf_exempt = true.** Reason (REFERENCE / handoff): no fair CP2K QS
stress wall with DFTD4 on the pin host; dual-gate wall is dominated by
multi-SCF FD, not the product analytic path; primary hard gates are
E_disp pin parity + analytic↔FD residual (same policy as `gpw-d4-water` /
`gpw-dftu-stress`). Re-measure a strict full-SCF twin later if product
policy requires \(r = T_{\mathrm{ATV}}/T_{\mathrm{CP2K}} \le 10\) on a host
with libdftd4-linked CP2K.

## Limitations

- **Water dimer cell**, not bulk ice Ih — lattice-sum / stress consistency
  pin, not a structural ice benchmark.
- **No absolute CP2K stress pin** on Arrhenius (build without DFTD4).
- **Two-body D4 only** — ATM / MBD off (\(s_9 = 0\)).
- **Stress + energy**, not D4 nuclear forces as a product gate.
- **Orthorhombic diagonal only** — shear / triclinic out of scope.
- **Γ-RKS only** on this slug — UKS / multi-k D4 stress not gated here.
- **`d3` / `d3bj` / `d4` / `ts` / `xdm` / `mbd` mutually exclusive** —
  any two true throws at analytic stress.
- **Oracle split:** E_disp = dftd4 3.7.0; stress residual = internal FD —
  do not conflate with molecular `dft-d4-h2o-pbe` (PySCF) geometry/BC.
- Wall **perf_exempt** for this wave; not a speed optimization target.

## References

- Offline pin: `regressions/gpw/gpw-d4-stress/REFERENCE.md`
- Oracle notes: `tools/cp2k_oracle/inputs/gpw-d4-stress/`,
  `tools/cp2k_oracle/inputs/gpw-d4-water/`, `tools/cp2k_oracle/README.md`
- Roadmap: `docs/roadmap/dft.md` §8 Dispersion
- Sibling energy pin: [gpw-d4-water](gpw-d4-water.md)
- Sibling analytic stress (no D4 gate): [gpw-analytic-stress](gpw-analytic-stress.md)
- Sibling strain-FD / CP2K NUMERICAL: [gpw-stress-water](gpw-stress-water.md)
- Molecular D4 (PySCF): [dft-d4-h2o-pbe](dft-d4-h2o-pbe.md)
- GPW narrative chapter: `docs/tutorial/ch09-gpw.md`
- E. Caldeweyher, C. Bannwarth, S. Grimme, *J. Chem. Phys.* **150**, 154122
  (2019) (DFT-D4)
- J. VandeVondele et al., *Comput. Phys. Commun.* **167**, 103 (2005)
  (Quickstep GPW)
- Nielsen & Martin, *Phys. Rev. B* **32**, 3780 (1985) (stress context)
- dftd4 3.7.0 (CLI pin backend; same library family CP2K links as DFTD4)
