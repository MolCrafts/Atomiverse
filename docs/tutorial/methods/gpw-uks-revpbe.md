# Γ-point UKS GPW — O₂ / revPBE / DZVP-MOLOPT-GTH

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `gpw-uks-revpbe` |
| Regression | `regressions/gpw/gpw-uks-revpbe/` |
| Public API | `cpu::gpw::GPWEvaluator` (`n_alpha`/`n_beta`, `XcKind::RevPBE`, `nonperiodic`); `cpu::single_point` |
| Roadmap | `docs/roadmap/dft.md` — Γ-UKS revPBE spin-GGA (`XcKind::RevPBE`) |
| Fair wall | \(r = T_{\mathrm{ATV}}/T_{\mathrm{CP2K}} \approx \mathbf{3.22}\) (≤10×) |

## Summary

**Unrestricted Kohn–Sham (UKS) Gaussian-and-plane-waves (GPW)** at the
**Γ point** with spin-resolved **revPBE GGA** (Zhang–Yang revision of PBE
exchange, \(\kappa=1.245\)). This page pins triplet **O₂** total energy and
Löwdin \(\langle S^2\rangle\) against offline **CP2K 2026.1 Quickstep**
(`LSD`, `MULTIPLICITY 3`, nested `&PBE` / `PARAMETRIZATION revPBE`) with
**Martyna–Tuckerman** isolated electrostatics. A **dual-gate** against
identical-geometry UKS-PBE ensures the functional is activated
(\(|E_{\mathrm{revPBE}}-E_{\mathrm{PBE}}| > 10^{-4}\) Ha).

Public path: construct `cpu::gpw::GPWEvaluator` with one `Kind` per element
(lookup constructors `GaussianBasis` / `GthPotential`), set members
(`n_alpha`, `n_beta`, `cutoff_ry`, `xc = XcKind::RevPBE`, `nonperiodic`, …),
run `cpu::single_point(frame, eval)`. Energy on `frame.meta` is **eV**;
Hartree and \(\langle S^2\rangle\) via `last_energy_hartree()` / `last_s2()`.
Product gate: `reg_gpw-uks-revpbe`.

Related: PADE LSD foothold [gpw-uks-o2-gamma](gpw-uks-o2-gamma.md); closed-shell
revPBE RKS (`reg_gpw-revpbe-si`); UKS-PBE sibling `reg_gpw-uks-pbe`. Narrative
GPW: [Ch. 9](../ch09-gpw.md). Molecular all-electron UKS is a separate SCF /
PySCF-loop concern — not this page.

## Method

**GPW (Quickstep family).** Wavefunction in a contracted Gaussian AO basis;
density collocation on a uniform real-space grid; Hartree via FFT Poisson;
XC on the grid; potential integrated back into the AO basis. Core electrons
removed by analytic **GTH** pseudopotentials. Grid resolution set by the
plane-wave cutoff in Rydberg. Citation: Lippert, Hutter, Parrinello,
*Mol. Phys.* **92**, 477 (1997).

**UKS / spin-GGA.** With \(N_\alpha \neq N_\beta\), build spin densities
\(\rho_\alpha\), \(\rho_\beta\); Hartree from \(\rho=\rho_\alpha+\rho_\beta\);
XC via spin-resolved revPBE on the grid (`xc_grid_revpbe_spin`); diagonalize
separate \(H_\alpha\) and \(H_\beta\). Occupations are explicit integer
members `n_alpha` / `n_beta` (not a multiplicity bag).

**revPBE.** Zhang & Yang (*Phys. Rev. Lett.* **80**, 890 (1998)): same
analytic PBE form with revised exchange enhancement factor
\(\kappa = 1.245\) (PBE uses \(\kappa = 0.804\)); correlation is standard
PBE. At \(\sigma = |\nabla\rho|^2 = 0\), revPBE and PBE share the PW92 LDA
base; at finite gradient they differ. libxc: `GGA_X_PBE_R` + `GGA_C_PBE`.
**CP2K 2026.1 input form (this pin):** nested

```text
&XC_FUNCTIONAL
  &PBE
    PARAMETRIZATION revPBE
  &END PBE
&END XC_FUNCTIONAL
```

Short `XC_FUNCTIONAL revPBE` is **not** a valid enumeration in CP2K 2026.1
(closed-shell RKS pin `gpw-revpbe-si` may use a different stock name depending
on version — always re-pin from the twin input under `tools/cp2k_oracle/`).

**Distinct from RPBE.** Hammer–Hansen–Nørskov RPBE uses a different exchange
form (not a mere \(\kappa\) revision). Do not interchange `XcKind::RevPBE`
and `XcKind::RPBE`.

**Isolated electrostatics.** `nonperiodic = true` selects the
**Martyna–Tuckerman** Poisson twin of CP2K `PERIODIC NONE` / `PSOLVER MT`.
The molecule must sit **centered** in the cell: nonperiodic MT grids cover
\([0,L)\); atoms near an edge truncate Gaussian tails and destroy the
energy. Citation: Martyna & Tuckerman, *J. Chem. Phys.* **110**, 2810 (1999).

**Spin metric.** After convergence, \(\langle S^2\rangle\) is the Löwdin
expectation for a single unrestricted determinant (not spin-projected).
Ideal high-spin value for \(S=1\): \(S(S+1)=2\). Citation: Löwdin,
*Phys. Rev.* **97**, 1509 (1955).

**Valence bookkeeping (this pin).** O with GTH-PBE-q6 → 6 valence e⁻ per O →
12 valence e⁻ total. Multiplicity 3 ⇒ \(N_\alpha=7\), \(N_\beta=5\)
(not all-electron \(9/7\)).

**What this pin includes**

| Item | Setting |
| ---- | ------- |
| Spin | UKS / LSD, multiplicity 3 → **\(N_\alpha=7\), \(N_\beta=5\)** |
| XC | **revPBE** spin-GGA (`XcKind::RevPBE`); PBE twin for dual-gate |
| System | O₂, \(R=1.2075\) Å along *z*, **centered** |
| Coordinates (Å) | O (6, 6, 5.39625), O (6, 6, 6.60375) |
| Cell | **12 × 12 × 12 Å**, `PERIODIC NONE` |
| Poisson | Martyna–Tuckerman (`PSOLVER MT` / `nonperiodic`) |
| Basis | DZVP-MOLOPT-GTH |
| PP | GTH-PBE-q6 |
| Cutoff | **400 Ry** (ATV single density grid; CP2K twin `NGRIDS 4` / `REL_CUTOFF 60`) |
| SCF (ATV) | mix 0.4, `max_iter=200`, `tol=1e-7` |

**Key exclusions (spec scope-out)**

- No **k-point** UKS on this regression (multi-k UKS is a separate queue slug).
- No **RPBE** claim (distinct functional; not this pin).
- No **ROKS** GPW; this page does not gate analytic UKS forces / stress
  (force/stress GGA UKS live on other product paths when present).
- No MPI scaling claim; single-rank pin.
- Not all-electron molecular UKS (Gaussian SCF / PySCF loop).
- Live CP2K never inside ctest — offline pins only.

## Implementation

Public surface only (`cpu::Frame`, concrete evaluator members; **no**
`*Config` bag, **no** `make_*` / `build_*` factories):

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

cpu::gpw::GPWEvaluator eval({O});
eval.cutoff_ry = 400.0;
eval.max_iter = 200;
eval.tol = 1e-7;
eval.mix = 0.4;
eval.xc = cpu::gpw::XcKind::RevPBE;
eval.nonperiodic = true;
eval.n_alpha = 7;  // multiplicity 3, 12 valence e⁻
eval.n_beta = 5;

cpu::single_point(frame, eval);

const double e_ha = eval.last_energy_hartree();       // [Ha]
const double e_ev = frame.meta.get(kEnergy);          // [eV]
const double s2 = eval.last_s2();                     // Löwdin ⟨S²⟩
const cpu::gpw::GpwResult& r = eval.last_result();    // converged, iters, …
```

The product regression also runs an identical-geometry **UKS-PBE** twin and
asserts \(|E_{\mathrm{revPBE}}-E_{\mathrm{PBE}}| > 10^{-4}\) Ha (activation).

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
  cmake --build build -j$(nproc) --target atomiverse reg_gpw-uks-revpbe
  ./build/regressions/reg_gpw-uks-revpbe
  # or: cmake --build build --target run_reg_gpw-uks-revpbe
'
```

Relevant sources:

| Piece | Path |
| ----- | ---- |
| Evaluator | `include/atomiverse/cpu/gpw/gpw_evaluator.h` |
| Solver (UKS SCF) | `include/atomiverse/cpu/gpw/gpw_solver.h`, `src/cpu/gpw/gpw_solver.cpp` |
| revPBE spin-GGA grid | `include/atomiverse/cpu/gpw/internal/xc_grid_gga.h`, `src/cpu/gpw/internal/xc_grid_gga.cpp` |
| Shared pointwise engine | `include/atomiverse/cpu/scf/xc_functional.h` (`XCType::RevPBE`) |
| Unit tests (not e2e) | `tests/cpu/gpw/test_cpu_gpw_uks.cpp`, `test_cpu_gpw_xc_grid_gga.cpp` |
| Regression | `regressions/gpw/gpw-uks-revpbe/reg_gpw-uks-revpbe.cpp` |
| CP2K twin input | `tools/cp2k_oracle/inputs/gpw-uks-revpbe/o2_uks_revpbe.inp` |

## Validation (CP2K)

Source: `regressions/gpw/gpw-uks-revpbe/REFERENCE.md` (offline pin; **never**
live CP2K in ctest). Engine: **CP2K 2026.1** (`cp2k.psmp`), module
`CP2K/2026.1` on Arrhenius. Pin date: **2026-07-26**.

| Quantity | CP2K pin | Atomiverse gate | Tol | Notes |
| -------- | -------: | --------------- | --- | ----- |
| \(E_{\mathrm{tot}}\) [Ha] | **−31.96512377015027** | \(\lvert E_{\mathrm{ATV}}-E_{\mathrm{CP2K}}\rvert\) | **1e-3 Ha** | First-green; ATV single density grid vs CP2K multigrid (`NGRIDS 4`, `REL_CUTOFF 60`) |
| \(\langle S^2\rangle\) (single det.) | **2.005611** | \(\lvert\langle S^2\rangle_{\mathrm{ATV}}-\mathrm{pin}\rvert\) | **0.15** | UKS contamination; ideal \(S(S+1)=2\) |
| Dual-gate vs UKS-PBE | — | \(\lvert E_{\mathrm{revPBE}}-E_{\mathrm{PBE}}\rvert\) | \(> 10^{-4}\) Ha | Hard functional activation |
| Frame energy consistency | \(E_{\mathrm{eV}}/k_{\mathrm{Ha\to eV}} \equiv E_{\mathrm{Ha}}\) | reg asserts | \(10^{-9}\) Ha | `kEnergy` vs `last_energy_hartree()` |
| SCF | converged | `r.converged` | — | ATV `max_iter=200`, `tol=1e-7` |

**Atomiverse first-green (2026-07-26, OMP=4 Release)** from REFERENCE:

| Quantity | Value |
| -------- | ----: |
| \(E_{\mathrm{revPBE}}\) | **−31.965167155386 Ha** |
| \(\lvert\Delta E\rvert\) vs CP2K | \(4.34 \times 10^{-5}\) Ha |
| \(\langle S^2\rangle\) | 2.00561329 |
| \(E_{\mathrm{PBE}}\) twin | −31.928854728432 Ha |
| \(\lvert E_{\mathrm{revPBE}}-E_{\mathrm{PBE}}\rvert\) | \(3.63 \times 10^{-2}\) Ha (\(> 10^{-4}\)) |
| SCF iters (revPBE) | 59 |

Key CP2K log lines (from REFERENCE):

```text
  *** SCF run converged in    15 steps ***
  Total energy:                                               -31.96512377015027
  Ideal and single determinant S**2 :                    2.000000       2.005611
 ENERGY| Total FORCE_EVAL ( QS ) energy [hartree]            -31.965123770150271
 FUNCTIONAL| revPBE, Yingkai Zhang and Weitao Yang, … {spin polarized}
```

Offline re-run (operator only; not ctest):

```bash
module load buildtool-easybuild/5.2.1-hpca3ef7d197 GCC/14.3.0 MPICH/4.3.2 CP2K/2026.1
export CP2K_DATA_DIR=$EBROOTCP2K/share/cp2k/data
export OMP_NUM_THREADS=4
cd /tmp && ln -sf $CP2K_DATA_DIR/{BASIS_MOLOPT,GTH_POTENTIALS} .
# copy tools/cp2k_oracle/inputs/gpw-uks-revpbe/o2_uks_revpbe.inp
cp2k.psmp -i o2_uks_revpbe.inp -o o2_uks_revpbe.out
```

## Performance

Source: same `REFERENCE.md` fair-wall note (2026-07-26) and
the case `REFERENCE.md` (wave 5). Not a ctest gate; optional
offline twin under `tools/cp2k_oracle/`.

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r = T_{\mathrm{ATV}}/T_{\mathrm{CP2K}}\) | ≤10× | Caveat |
| ---- | ------------------------: | ----------------------: | ---------------------------------------: | ---- | ------ |
| O₂ UKS-revPBE / DZVP-MOLOPT-GTH / MT, Γ | **25.8** | **83.1** | **3.22** | ✓ | CP2K: 1 MPI × 4 OMP, multigrid NGRIDS 4; ATV: Release, single density grid, OMP=4; wall is revPBE SCF only |

Gate from closed product: \(r \le 10\) or `perf_exempt`. Here
\(r \approx 3.22\) — product pass without exemption
(`perf_exempt = false`, `within_10x = true`).

## Limitations

- **Grid twin, not bit-identical SCF:** Atomiverse uses a **single** density
  grid; CP2K pin uses multigrid (`NGRIDS 4` / `REL_CUTOFF 60`). Energy
  tolerance is deliberately **1e-3 Ha** for first-green; may tighten when
  multigrid parity is product work.
- **SCF wall:** first-green ATV revPBE SCF ~83 s (OMP=4) on the pin host —
  slower than CP2K multigrid path but within the 10× fair-wall gate. Dual-gate
  product run also evaluates UKS-PBE (extra wall not folded into \(r\)).
- **Γ only** on this page — multi-k collinear UKS / k-forces / k-stress are
  separate CP2K-loop slugs.
- **Geometry hygiene:** off-center molecules under MT will fail energy
  parity; always center for isolated UKS/RKS pins.
- **ROKS / spin projection** not provided; residual \(\langle S^2\rangle\)
  contamination is accepted (compared to CP2K single-determinant value).
- **RPBE ≠ revPBE** — do not document RPBE energy against this pin.
- **No public `*Config` / factories** — parameters are members on
  `GPWEvaluator` / `GpwDft`.

## References

- Offline pin: [`regressions/gpw/gpw-uks-revpbe/REFERENCE.md`](../../../regressions/gpw/gpw-uks-revpbe/REFERENCE.md)
- Regression consumer: [`reg_gpw-uks-revpbe.cpp`](../../../regressions/gpw/gpw-uks-revpbe/reg_gpw-uks-revpbe.cpp)
- CP2K twin: [`tools/cp2k_oracle/inputs/gpw-uks-revpbe/`](../../../tools/cp2k_oracle/inputs/gpw-uks-revpbe/)
- Oracle contract: [`tools/cp2k_oracle/README.md`](../../../tools/cp2k_oracle/README.md)
- Sibling UKS-PADE: [gpw-uks-o2-gamma](gpw-uks-o2-gamma.md)
- Narrative GPW: [`docs/tutorial/ch09-gpw.md`](../ch09-gpw.md)
- Roadmap: [`docs/roadmap/dft.md`](../../roadmap/dft.md)
- G. Lippert, J. Hutter, M. Parrinello, *Mol. Phys.* **92**, 477 (1997) — GPW
- Y. Zhang, W. Yang, *Phys. Rev. Lett.* **80**, 890 (1998) — revPBE
- J. P. Perdew, K. Burke, M. Ernzerhof, *Phys. Rev. Lett.* **77**, 3865 (1996) — PBE base
- G. J. Martyna, M. E. Tuckerman, *J. Chem. Phys.* **110**, 2810 (1999) — MT Poisson
- P.-O. Löwdin, *Phys. Rev.* **97**, 1509 (1955) — \(\langle S^2\rangle\) unrestricted dets
