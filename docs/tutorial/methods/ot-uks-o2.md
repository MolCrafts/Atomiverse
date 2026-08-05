# OT orbital minimizer for Γ-UKS — O₂ / PADE / DZVP-MOLOPT-GTH

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| slug | `ot-uks-o2` |
| Regression | `regressions/gpw/ot-uks-o2/` |
| Public API | `cpu::gpw::GPWEvaluator` (`use_ot`, `n_alpha`/`n_beta`, `XcKind::PADE`, `nonperiodic`); `cpu::single_point` |
| Roadmap | `docs/roadmap/dft.md` — product OT minimizer (`use_ot` RKS + **Γ-UKS**) |
| Queue | CP2K capability loop (`docs/developer/cp2k-method-queue.md`) — done 2026-07-26 |

## Summary

**Orbital Transformation (OT)** on **unrestricted Kohn–Sham (UKS) GPW** at the
**Γ point**: separate α/β occupied manifolds on the Stiefel-type constraint
\(C_\sigma^\top S C_\sigma = I\), spin-resolved **PADE (LSD)** XC, Martyna–Tuckerman
isolated electrostatics. Product surface is the public member pair
`cpu::gpw::GPWEvaluator::use_ot` + `n_alpha` / `n_beta` (no `*Config` bag, no
env required for product code). This page pins **triplet O₂** OT total energy
and Löwdin \(\langle S^2\rangle\) against an offline **CP2K 2026.1** OT twin
(`&OT`, `PRECONDITIONER FULL_SINGLE_INVERSE`, CUTOFF **250 Ry**), and dual-checks
OT vs in-tree UKS DIIS on the same Atomiverse grids.

Public path: construct `GPWEvaluator` with one `Kind` per element (lookup
constructors `GaussianBasis` / `GthPotential`), set members, run
`cpu::single_point(frame, eval)`. Energy on `frame.meta` is **eV**; Hartree and
\(\langle S^2\rangle\) via `last_energy_hartree()` / `last_s2()`. Product gate:
`reg_ot-uks-o2`. Fair OT-only wall ratio
\(r = T_{\mathrm{ATV}}/T_{\mathrm{CP2K}} \approx 4.6\) (OMP=4, energy OT, ≤10×).

Closed-shell OT twin: [ot-minimizer-water](ot-minimizer-water.md). DIIS-only
UKS energy at 400 Ry: [gpw-uks-o2-gamma](gpw-uks-o2-gamma.md). Related
narrative: [Ch. 9 GPW](../ch09-gpw.md).

## Method

**GPW (Quickstep family).** Wavefunction in a contracted Gaussian AO basis;
density collocation on a uniform real-space grid; Hartree via FFT Poisson; XC
on the grid; potential integrated back into the AO basis. Core electrons
removed by analytic **GTH** pseudopotentials. Grid resolution set by the
plane-wave cutoff in Rydberg. Citation: Lippert, Hutter, Parrinello,
*Mol. Phys.* **92**, 477 (1997).

**UKS / LSD.** With \(N_\alpha \neq N_\beta\), build spin densities
\(\rho_\alpha\), \(\rho_\beta\); Hartree from \(\rho=\rho_\alpha+\rho_\beta\);
XC via spin-resolved PADE LSD; separate \(H_\alpha\) and \(H_\beta\).
Occupations are explicit integer members `n_alpha` / `n_beta` (not a
multiplicity bag). Citation: Goedecker, Teter, Hutter, *Phys. Rev. B* **54**,
1703 (1996).

**OT parameterization** (Kühne et al., *J. Chem. Phys.* **152**, 194103 (2020)
§VII C; VandeVondele & Hutter, *J. Chem. Phys.* **118**, 4365 (2003)). For
reference occupied coefficients \(C_0\) with \(C_0^\top S C_0 = I\) and an
\(S\)-tangent step \(x\) (\(C_0^\top S x = 0\)):

\[
C(x) = C_0\,\cos U + x\,U^{-1}\sin U,\qquad
U = (x^\top S x)^{1/2},
\]

which preserves \(C(x)^\top S C(x) = I\) for any such \(x\). **Open-shell** OT
uses orbital occupation **1** per spin channel (\(P_\sigma = C_\sigma C_\sigma^\top\));
closed-shell RKS uses occupation 2 (\(P = 2\,C C^\top\)). Gradient per spin is
consistent with that occupation (see `internal/ot.h`).

**Minimization.** Joint preconditioned CG (PR+) with Armijo line search on the
rebuilt total energy after density/Hartree/XC update. In-tree dense
preconditioners (internal, not public API) follow the CP2K class used by the
twin (`FULL_SINGLE_INVERSE`-style \(M = H + a S\)). Outer SCF rebuilds
\(H_\sigma\) until residual / energy criteria meet `tol`. Same self-consistent
fixed point as UKS DIIS when both converge on identical grids.

**Isolated electrostatics.** `nonperiodic = true` selects **Martyna–Tuckerman**
Poisson (`PERIODIC NONE` / `PSOLVER MT`). The molecule must sit **centered**
in the cell: nonperiodic MT grids cover \([0,L)\); atoms near an edge truncate
Gaussian tails and destroy the energy. Citation: Martyna & Tuckerman,
*J. Chem. Phys.* **110**, 2810 (1999).

**Valence bookkeeping (this pin).** O with GTH-PBE-q6 → 6 valence e⁻ per O →
12 valence e⁻ total. Multiplicity 3 ⇒ \(N_\alpha=7\), \(N_\beta=5\)
(not all-electron \(9/7\)).

**Spin metric.** After convergence, \(\langle S^2\rangle\) is the Löwdin
expectation for a single unrestricted determinant (not spin-projected). Ideal
high-spin value for \(S=1\): \(S(S+1)=2\). Citation: Löwdin, *Phys. Rev.*
**97**, 1509 (1955).

**Key exclusions (binding for this slug).**

- **k-point OT** ○ (Γ only).
- **Sparse / linear-scaling OT preconditioner** ○ (dense \(O(n_b^3)\) inverse).
- **ROKS** OT / spin projection ○.
- **Analytic UKS OT forces / stress / BOMD** ○ on this regression (energy +
  \(\langle S^2\rangle\) only).
- **UKS + PBE-GGA OT** not the pin chemistry (this pin is **PADE LSD**).
- MPI multi-rank OT scaling ○; single-rank fair twin.
- Not all-electron molecular UKS (Gaussian SCF / PySCF loop).

## Implementation

Public surface only (`cpu::Frame` + `GPWEvaluator` members / ctor args; **no**
`*Config` bag, **no** `make_*` / `build_*` factories). Match the regression
(**250 Ry** pin budget; 400 Ry DIIS class is `gpw-uks-o2-gamma`):

```cpp
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/evaluate/single_point.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_evaluator.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/molrs.hpp"

using namespace ATV;

// O₂, R = 1.2075 Å, centered in 12 Å cubic cell (MT requires centering)
cpu::Frame frame(2);
constexpr double L = 12.0;
constexpr double R = 1.2075;
for (int i = 0; i < 2; ++i) {
    frame.span(kAtom_type)[i] = 8;
    frame.span(kAtom_mass)[i] = 15.999;
    frame.span(kAtom_x)[i] = 0.5 * L;
    frame.span(kAtom_y)[i] = 0.5 * L;
}
frame.span(kAtom_z)[0] = 0.5 * L - 0.5 * R;
frame.span(kAtom_z)[1] = 0.5 * L + 0.5 * R;
frame.box = cpu::Box::cubic(L);

cpu::gpw::Kind O{molrs::Element::O,
                 cpu::gpw::GaussianBasis(molrs::Element::O, "DZVP-MOLOPT-GTH"),
                 cpu::gpw::GthPotential(molrs::Element::O, "GTH-PBE-q6")};

cpu::gpw::GPWEvaluator eval(
    /*kinds*/ {O}, /*cutoff_ry*/ 250.0, /*max_iter*/ 200, /*tol*/ 1e-7,
    /*mix*/ 0.4, cpu::gpw::XcKind::PADE, /*nonperiodic*/ true,
    /*compute_forces*/ false, /*fd_step*/ 1e-3, /*d3*/ false,
    /*use_ot*/ true);
// equivalent after construction: eval.use_ot = true;
eval.n_alpha = 7;  // multiplicity 3, 12 valence e⁻
eval.n_beta = 5;

cpu::single_point(frame, eval);

const double e_ha = eval.last_energy_hartree();       // [Ha]
const double e_ev = frame.meta.get(kEnergy);          // [eV]
const double s2 = eval.last_s2();                     // Löwdin ⟨S²⟩
const cpu::gpw::GpwResult& r = eval.last_result();    // converged, iters, …
```

Diagnostic env `ATV_GPW_OT` may still force OT on (OR with the member); product
code and the regression **must not** depend on it (`unsetenv` in the reg binary).

Build (GPW needs FFTW3) and run the regression:

```bash
bash -lc '
  module load buildenv-gcc/2026.03-mpich
  export OPENBLAS_ROOT="$OPENBLASDIR"
  export RUSTUP_TOOLCHAIN=stable-x86_64-unknown-linux-gnu
  export LD_LIBRARY_PATH="$LIBRARY_PATH:$LD_LIBRARY_PATH"
  cmake -B build -DCMAKE_BUILD_TYPE=Release \
        -DATV_BUILD_TESTS=ON -DATV_BUILD_HF=ON \
        -DATV_BUILD_PW=ON -DATV_BUILD_GPW=ON \
        -DATV_BUILD_REGRESSIONS=ON -DATV_BUILD_CUDA=OFF \
        -DBLA_VENDOR=OpenBLAS
  cmake --build build -j$(nproc) --target atomiverse reg_ot-uks-o2
  ./build/regressions/reg_ot-uks-o2
'
```

Relevant sources:

| Piece | Path |
| ----- | ---- |
| Evaluator | `include/atomiverse/cpu/gpw/gpw_evaluator.h` (`use_ot`, `n_alpha`/`n_beta`) |
| Solver (UKS + OT) | `include/atomiverse/cpu/gpw/gpw_solver.h`, `src/cpu/gpw/gpw_solver.cpp` |
| OT manifold / preconds | `include/atomiverse/cpu/gpw/internal/ot.h` |
| PADE LSD grid | `include/atomiverse/cpu/gpw/internal/xc_grid.h` |
| Unit tests (not e2e) | `tests/cpu/gpw/test_cpu_gpw_ot.cpp` |
| Regression | `regressions/gpw/ot-uks-o2/reg_ot-uks-o2.cpp` |
| CP2K twin input | `tools/cp2k_oracle/inputs/ot-uks-o2/o2_uks_ot_pade.inp` |

## Validation (vs CP2K)

Source: [`regressions/gpw/ot-uks-o2/REFERENCE.md`](../../../regressions/gpw/ot-uks-o2/REFERENCE.md)
(offline pin; **never** live CP2K in ctest). Engine: **CP2K 2026.1**
(`cp2k.psmp`, git:5e54ba2), module `CP2K/2026.1` on Arrhenius. Re-pin date:
**2026-07-26**.

| Quantity | CP2K pin | Atomiverse gate | Tol | Notes |
| -------- | -------: | --------------- | --- | ----- |
| \(E_{\mathrm{tot}}\) OT [Ha] | **−31.770029989852723** | \(\lvert E_{\mathrm{ATV,OT}}-E_{\mathrm{CP2K}}\rvert\) | **2.5×10⁻³ Ha** | 250 Ry single-grid ATV vs CP2K multigrid (`NGRIDS 4`, `REL_CUTOFF 40`) |
| \(\langle S^2\rangle\) (single det.) | **2.004666** | \(\lvert\langle S^2\rangle_{\mathrm{ATV}}-\mathrm{pin}\rvert\) | **0.15** | UKS contamination; ideal \(S(S+1)=2\) |
| \(\lvert E_{\mathrm{OT}}-E_{\mathrm{DIIS}}\rvert\) (ATV) | \(\sim 5\times10^{-10}\) Ha | reg asserts | **1×10⁻⁴ Ha** | same grids fixed-point parity |
| Frame energy consistency | \(E_{\mathrm{eV}}/k_{\mathrm{Ha\to eV}} \equiv E_{\mathrm{Ha}}\) | reg asserts | \(10^{-9}\) Ha | `kEnergy` vs `last_energy_hartree()` |
| SCF | converged | `r.converged` | — | `max_iter=200`, `tol=1e-7` |

**Geometry / protocol (must match twin).**

| Item | Value |
| ---- | ----- |
| Molecule | O₂, bond along *z*, **centered** |
| \(R(\mathrm{O–O})\) | **1.2075 Å** |
| Coordinates (Å) | O (6, 6, 5.39625), O (6, 6, 6.60375) |
| Cell | **12 × 12 × 12 Å**, `PERIODIC NONE` |
| Poisson | Martyna–Tuckerman (`PSOLVER MT` / `nonperiodic`) |
| Basis | DZVP-MOLOPT-GTH |
| PP | GTH-PBE-q6 |
| XC | PADE (LDA LSD) |
| Cutoff | **250 Ry** (regression budget) |
| Spin | UKS / LSD, multiplicity 3 → **\(N_\alpha=7\), \(N_\beta=5\)** |
| CP2K SCF | `&OT` `MINIMIZER DIIS`, `PRECONDITIONER FULL_SINGLE_INVERSE`, `OUTER_SCF` |
| ATV SCF | `use_ot = true` (OT) + DIIS dual path for parity |

Key CP2K log lines (from REFERENCE):

```text
  *** SCF run converged in    30 steps ***
  Total energy:                                               -31.77002998985272
  Ideal and single determinant S**2 :                    2.000000       2.004666
 ENERGY| Total FORCE_EVAL ( QS ) energy [hartree]            -31.770029989852723
```

Cross-check: CP2K DIAG at **400 Ry** (centered O₂) was −31.7698035 Ha
(`gpw-uks-o2-gamma`) — same energy class; **250 Ry OT** is the regression twin.

Offline re-run (operator only; not ctest):

```bash
module load buildtool-easybuild/5.2.1-hpca3ef7d197 GCC/14.3.0 MPICH/4.3.2 CP2K/2026.1
export CP2K_DATA_DIR=$EBROOTCP2K/share/cp2k/data
export OMP_NUM_THREADS=4
cd /tmp && ln -sf $CP2K_DATA_DIR/{BASIS_MOLOPT,GTH_POTENTIALS} .
# copy tools/cp2k_oracle/inputs/ot-uks-o2/o2_uks_ot_pade.inp
mpirun -np 1 cp2k.psmp -i o2_uks_ot_pade.inp -o o2_uks_ot_pade.out
```

## Performance

Source: fair wall table in
`regressions/gpw/ot-uks-o2/REFERENCE.md` (2026-08-05). Queue / handoff record:
ratio **~4.6** (fair energy-OT, OMP=4; CP2K-aligned algorithm).
Gate \(r\le 10\) unless `perf_exempt`.

Comparable settings: cutoff 250 Ry, PADE LSD, MT, single rank,
`OMP_NUM_THREADS=4`, `OMP_PROC_BIND=close`.

| Twin | Wall [s] | \(r = T_{\mathrm{ATV}}/T_{\mathrm{CP2K}}\) | ≤10× | Caveat |
| ---- | -------: | ----------------------------------------: | ---- | ------ |
| CP2K OT (`cp2k.psmp`, 1 MPI × 4 OMP) | **8.68** | — | — | primary twin (`CP2K` timer MAXIMUM) |
| Atomiverse OT only | **~40** | **≈ 4.6** | ✓ | 84 energy-OT iters; FULL_SINGLE precond + CG+Armijo |
| Atomiverse DIIS (parity check) | **~14** | — | — | 30 DIIS iters; **not** the fair twin |
| Full regression binary (OT+DIIS) | ≈ 54 | — | — | **not** ratioed against CP2K OT alone |

**Caveats.** (1) Report the **OT-only** ratio; the regression binary runs both
OT and DIIS and must not be ratioed against CP2K OT alone. (2) Remaining gap
is OT outer count (84 vs CP2K 30; product uses CG+Armijo vs CP2K MINIMIZER
DIIS) plus per-iter KS. (3) `within_10x=true`; no optimize pass required for
this foothold.

## Limitations

- **Γ-point UKS OT only** — no k-point OT, no sparse OT, no MPI industrial
  scaling gate on this slug.
- **Grid twin, not bit-identical SCF:** Atomiverse uses a **single** density
  grid; CP2K pin uses multigrid (`NGRIDS 4` / `REL_CUTOFF 40`). Energy
  tolerance is deliberately **2.5×10⁻³ Ha** (milli-Ha class at 250 Ry).
- **OT vs DIIS** in-tree is the tight fixed-point gate (\(\sim 5\times10^{-10}\)
  Ha on identical grids); vs CP2K is energy-class agreement.
- **Forces / stress / BOMD with OT UKS** are out of scope here; closed-shell
  force path is separate (`gpw-forces-water`).
- **Geometry hygiene:** off-center molecules under MT will fail energy parity;
  always center for isolated UKS/RKS pins.
- **ROKS / spin projection** not provided; residual \(\langle S^2\rangle\)
  contamination is accepted (compared to CP2K single-determinant value).
- **Speed:** ~8.5× CP2K on this microcase under OMP=4 fair OT twin — acceptable
  for the product gate, not a claim of production Quickstep parity.
- PW backend has no OT product switch; CUDA OT ○.

## References

- Offline pin + fair wall: [`regressions/gpw/ot-uks-o2/REFERENCE.md`](../../../regressions/gpw/ot-uks-o2/REFERENCE.md)
- Regression consumer: [`reg_ot-uks-o2.cpp`](../../../regressions/gpw/ot-uks-o2/reg_ot-uks-o2.cpp)
- CP2K twin: [`tools/cp2k_oracle/inputs/ot-uks-o2/`](../../../tools/cp2k_oracle/inputs/ot-uks-o2/)
- Oracle contract: [`tools/cp2k_oracle/README.md`](../../../tools/cp2k_oracle/README.md)
- Loop: [`docs/developer/cp2k-method-loop.md`](../../developer/cp2k-method-loop.md),
  queue / handoff under `docs/developer/cp2k-method-*.md`
- Closed-shell OT twin: [`ot-minimizer-water.md`](ot-minimizer-water.md)
- DIIS UKS 400 Ry twin: [`gpw-uks-o2-gamma.md`](gpw-uks-o2-gamma.md)
- Narrative GPW: [`docs/tutorial/ch09-gpw.md`](../ch09-gpw.md)
- Roadmap: [`docs/roadmap/dft.md`](../../roadmap/dft.md)
- J. VandeVondele, J. Hutter, *J. Chem. Phys.* **118**, 4365 (2003) — OT
- T. D. Kühne et al., *J. Chem. Phys.* **152**, 194103 (2020) §VII C — CP2K OT
- G. Lippert, J. Hutter, M. Parrinello, *Mol. Phys.* **92**, 477 (1997) — GPW
- S. Goedecker, M. Teter, J. Hutter, *Phys. Rev. B* **54**, 1703 (1996) — PADE / GTH
- G. J. Martyna, M. E. Tuckerman, *J. Chem. Phys.* **110**, 2810 (1999) — MT Poisson
- P.-O. Löwdin, *Phys. Rev.* **97**, 1509 (1955) — \(\langle S^2\rangle\) unrestricted dets
