# GPW Γ-RKS energy — isolated H₂O / PBE / DZVP-MOLOPT-GTH (Martyna–Tuckerman)

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `gpw-water-pbe-mt` |
| Regression | `regressions/gpw/gpw-water-pbe-mt/` |
| Public API | `cpu::gpw::GPWEvaluator` + `cpu::single_point`; `Kind` + `GaussianBasis` / `GthPotential` lookup ctors |
| Roadmap | narrative twin [ch09-gpw](../ch09-gpw.md) |

## Summary

Γ-point closed-shell **Gaussian-and-plane-waves (GPW)** Kohn–Sham DFT total
energy for an **isolated** water molecule: contracted MOLOPT Gaussians for the
wavefunction, uniform real-space grid + FFT for Hartree and XC, GTH
pseudopotentials, **PBE** XC, density cutoff **400 Ry**. Cluster electrostatics
use the **Martyna–Tuckerman (MT)** Poisson solver (`nonperiodic = true`; CP2K
`PERIODIC NONE` / `PSOLVER MT`). Public path is concrete
`cpu::gpw::GPWEvaluator` (constructor args / members; no `*Config` bag) driven
by `cpu::single_point(frame, eval)`. This is the **first** offline-CP2K-pinned
e2e under `regressions/gpw/`: product gate
`reg_gpw-water-pbe-mt` asserts SCF convergence and
\(|E_{\mathrm{tot}} - E_{\mathrm{CP2K}}| \le 10^{-3}\) Ha. Fair wall vs CP2K is
**`perf_exempt`** (no CP2K binary on the pin host).

## Method

**GPW dual representation** (Quickstep family; Lippert, Hutter, Parrinello
1997/1999). Density matrix \(P\) lives in a small contracted Gaussian basis
\(\{\chi_\mu\}\). Each SCF cycle:

1. Collocate \(\rho(\mathbf{r}) = \sum_{\mu\nu} P_{\mu\nu}\,\chi_\mu(\mathbf{r})\chi_\nu(\mathbf{r})\)
   onto a uniform real-space grid set by the plane-wave cutoff \(E_{\mathrm{cut}}\)
   (here 400 Ry).
2. FFT → reciprocal space; solve Poisson for the Hartree potential; evaluate
   \(v_{\mathrm{xc}}[\rho]\) on the grid — \(\mathcal{O}(N\log N)\).
3. Integrate grid potentials back against the Gaussians to form
   \(V_{\mu\nu}^{\mathrm{Hxc}}\); add kinetic + GTH local/nonlocal PP in the
   AO basis; diagonalize (or OT — not this pin); rebuild \(P\).

**Core treatment.** Norm-conserving **GTH** analytic pseudopotentials remove
core electrons. Valence counts on this system: O **q6**, H **q1** → 8 valence
electrons → \(n_{\mathrm{occ}} = 4\) closed-shell pairs.

**Isolated electrostatics.** A finite cubic cell with ordinary periodic Poisson
would couple the molecule to its images. Martyna–Tuckerman replaces the
reciprocal-space Coulomb kernel so that the electrostatic potential matches
the free-space solution for a charge density localized inside the cell (here
\(L = 10\) Å). Atomiverse exposes this as `nonperiodic = true` on
`GPWEvaluator`.

**What this pin includes**

| Item | Setting |
| ---- | ------- |
| Spin | Closed-shell RKS (Γ only) |
| XC | PBE GGA (`XcKind::PBE`) |
| Basis | DZVP-MOLOPT-GTH (O, H) |
| PP | GTH-PBE-q6 (O), GTH-PBE-q1 (H) |
| Cutoff | 400 Ry |
| SCF | direct \(P\)-mixing \(\alpha=0.3\), `max_iter=200`, `tol=1e-7` |
| BC | MT isolated (`nonperiodic=true`) |

**Key exclusions (non-goals of this slug)**

- Analytic forces / stress / UKS / k-points / D3 / BOMD (other queue slugs:
  `gpw-forces-water`, `gpw-uks-o2-gamma`, `gpw-pbe-d3bj-ice`, …).
- Bit-parity vs CP2K (erf-screened MT tuning is follow-on); e2e band is **1 mHa**.
- Live CP2K (or any third-party) inside ctest.
- MPI multi-rank; OT minimizer (`use_ot`); public Config bags / factories.

## Implementation

Public surface only (`cpu::Frame` + `GPWEvaluator` ctor args / members; lookup
constructors for basis/PP tables — not factories for a parallel world model):

```cpp
#include "atomiverse/cpu/evaluate/single_point.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_evaluator.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/molrs.hpp"

using namespace ATV;

// CP2K getting-started h2o.inp geometry [Å], 10 Å cubic cell.
cpu::Frame frame(3);
const int z[3] = {8, 1, 1};
const double x[3] = {5.0000, 5.7586, 4.2414};
const double y[3] = {5.0000, 5.0000, 5.0000};
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
    /*kinds*/ {O, H}, /*cutoff_ry*/ 400.0, /*max_iter*/ 200, /*tol*/ 1e-7,
    /*mix*/ 0.3, /*xc*/ cpu::gpw::XcKind::PBE, /*nonperiodic*/ true);
cpu::single_point(frame, eval);

const double e_ha = eval.last_energy_hartree();   // [Hartree]
const bool ok = eval.last_result().converged;
// frame.meta.get(kEnergy) is also set [eV]
```

Build and run the regression (CPU login; modules as in `AGENTS.md`):

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
  cmake --build build --target reg_gpw-water-pbe-mt -j$(nproc)
  ./build/regressions/reg_gpw-water-pbe-mt
'
```

| Path | Role |
| ---- | ---- |
| `include/atomiverse/cpu/gpw/gpw_evaluator.h` | Public `Kind`, `GPWEvaluator` |
| `include/atomiverse/cpu/gpw/gaussian_basis.h` | `GaussianBasis(Element, name)` lookup |
| `include/atomiverse/cpu/gpw/gth_potential.h` | `GthPotential(Element, name)` lookup |
| `include/atomiverse/cpu/evaluate/single_point.h` | `single_point(frame, method)` |
| `src/cpu/gpw/` | GPW SCF / collocate / integrate (private) |
| `regressions/gpw/gpw-water-pbe-mt/reg_gpw-water-pbe-mt.cpp` | Product gate |
| `tools/cp2k_oracle/inputs/gpw-water-pbe-mt.inp` | Offline CP2K twin (never ctest) |

## Validation (vs CP2K)

Source:
[`regressions/gpw/gpw-water-pbe-mt/REFERENCE.md`](../../../regressions/gpw/gpw-water-pbe-mt/REFERENCE.md).
Oracle: offline **CP2K Quickstep** getting-started `h2o.inp` family (same
geometry, DZVP-MOLOPT-GTH + GTH-PBE, CUTOFF 400, PERIODIC NONE / PSOLVER MT,
PBE). External-engine pin — **not** a golden of Atomiverse’s own output; **never**
launch CP2K from ctest. Twin input:
`tools/cp2k_oracle/inputs/gpw-water-pbe-mt.inp`. Cross-recorded by
`examples/09_gpw.cpp`, `examples/results/09_gpw.json`, and
[`docs/tutorial/ch09-gpw.md`](../ch09-gpw.md).

**Setup:**

| Item | Value |
| ---- | ----- |
| Molecule | H₂O, charge 0, singlet (closed-shell RKS) |
| Geometry [Å] | O (5.0000, 5.0000, 5.0000); H (5.7586, 5.0000, 5.5043); H (4.2414, 5.0000, 5.5043) |
| Cell | cubic \(L = 10\) Å, non-periodic (MT) |
| Basis | DZVP-MOLOPT-GTH |
| Pseudopotentials | GTH-PBE-q6 (O), GTH-PBE-q1 (H) |
| XC | PBE |
| Density cutoff | 400 Ry |
| Valence | 8 e⁻ → \(n_{\mathrm{occ}} = 4\) |
| SCF | mix 0.3, max_iter 200, tol \(10^{-7}\) |
| Provenance | CP2K getting-started `h2o.inp`; historical engine milestone `6d7f0955` (2026-06-18) |

**CP2K energy decomposition (Hartree; twin):**

| Term | CP2K |
| ---- | ---: |
| self energy | −44.34715500 |
| core Hamiltonian | +13.07138998 |
| Hartree | +18.34339895 |
| XC | −4.27997143 |
| **total** | **−17.21233714458516** |

**Accuracy gate (product):**

| Quantity | Atomiverse (verified 2026-07-23) | CP2K pin | Tol | Notes |
| -------- | -------------------------------: | -------: | --- | ----- |
| \(E_{\mathrm{tot}}\) | **−17.2127580657 Ha** | **−17.21233714458516 Ha** | **\(1\times 10^{-3}\) Ha** | Product gate |
| \(\lvert\Delta E\rvert\) | \(4.209\times 10^{-4}\) Ha | — | < 1 mHa | Known residual ~0.42 mHa (ch09 live table) |
| SCF | converged, 9 iters | — | `converged == true` | Product gate |
| Frame energy | finite `frame.meta.get(kEnergy)` [eV] | — | finite | Product gate |

Tolerance rationale (from REFERENCE): the residual on this minimal
isolated-water stack is ~0.42 mHa; a **1 mHa** e2e band covers it with margin
while still failing gross method / geometry / cutoff mistakes. Sub-mHa / MT
kernel bit-parity is a follow-on, not this regression’s bar.

Offline re-pin (human only; never ctest):

```bash
# when a CP2K binary is available
cp2k.psmp -i tools/cp2k_oracle/inputs/gpw-water-pbe-mt.inp \
          -o /tmp/gpw-water-pbe-mt.out
# record TOTAL ENERGY [Ha] into REFERENCE.md
```

## Performance

Source: [`REFERENCE.md`](../../../regressions/gpw/gpw-water-pbe-mt/REFERENCE.md)
(2026-07-23).

| Item | Value |
| ---- | ----- |
| `perf_exempt` | **true** |
| Reason | No CP2K binary / module on the build host at pin time (`cp2k`, `cp2k.psmp`, `cp2k.sopt` not found; `module avail cp2k` empty). Fair wall twin deferred until offline CP2K is available. |
| \(T_{\mathrm{ATV}}\) (OMP=1, Release) | ~17.7 s (single-point SCF wall on pin host) |
| \(r = T_{\mathrm{ATV}}/T_{\mathrm{CP2K}}\) | **n/a (exempt)** |
| `within_10x` | n/a (exempt) |

. Gate when a twin exists:
\(r = T_{\mathrm{ATV}}/T_{\mathrm{CP2K}} \le 10\) on the same host/core count
(single-rank).

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | -----------------------: | ----------------------: | ----: | ---- | ------ |
| H₂O PBE MT (400 Ry) | — | ~17.7 | **exempt** | n/a | No CP2K binary at pin; remeasure when available |

## Limitations

- **Energy-only foothold** for Γ-RKS + MT isolated BC. Forces:
  `gpw-forces-water`; UKS: `gpw-uks-o2-gamma`; D3(BJ): `gpw-pbe-d3bj-ice`;
  OT vs diag: `ot-minimizer-water`; stress / BOMD still queue
  (`gpw-stress-water`, `gpw-bomd-water`).
- **~0.42 mHa residual** vs CP2K on this stack — within the 1 mHa product band;
  not bit-identical (MT screening / grid details).
- **No k-points** on the Gaussian GPW path in-tree yet (`gpw-kpoints-si`
  pending).
- **CPU only** — `cuda-gpw` is blocked on the CP2K loop.
- **`perf_exempt`**: no fair \(r\) until an offline CP2K binary is timed on the
  same host; do not invent a speed ratio.
- Basis/PP tables are static in-tree headers (`cpu/gpw/basis_data/`); unsupported
  element/name **throws** (no silent fallback).

## References

- Offline pin: `regressions/gpw/gpw-water-pbe-mt/REFERENCE.md`
- Regression consumer: `regressions/gpw/gpw-water-pbe-mt/reg_gpw-water-pbe-mt.cpp`
- Oracle twin: `tools/cp2k_oracle/inputs/gpw-water-pbe-mt.inp`
- Oracle notes: `tools/cp2k_oracle/README.md`
- Narrative chapter: [`docs/tutorial/ch09-gpw.md`](../ch09-gpw.md)
- Example + JSON: `examples/09_gpw.cpp`, `examples/results/09_gpw.json`
- Closed spec (git history): `.claude/specs/gpw-water-pbe-mt/` on
  `wf/gpw-water-pbe-mt` / `dc7ad1bc`
- Public API: `include/atomiverse/cpu/gpw/gpw_evaluator.h`
- G. Lippert, J. Hutter, M. Parrinello, *Mol. Phys.* **92**, 477 (1997);
  *Theor. Chem. Acc.* **103**, 124 (1999) — GPW / Quickstep.
- S. Goedecker, M. Teter, J. Hutter, *Phys. Rev. B* **54**, 1703 (1996) — GTH.
- G. J. Martyna, M. E. Tuckerman, *J. Chem. Phys.* **110**, 2810 (1999) — MT
  Poisson.
- J. VandeVondele et al., *Comput. Phys. Commun.* **167**, 103 (2005) — CP2K
  Quickstep; MOLOPT / DZVP-MOLOPT-GTH table family.
