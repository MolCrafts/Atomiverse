# Γ-point UKS GPW — O₂ / BLYP / DZVP-MOLOPT-GTH

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `gpw-uks-blyp` |
| Regression | `regressions/gpw/gpw-uks-blyp/` |
| Public API | `cpu::gpw::GPWEvaluator` (`n_alpha`/`n_beta`, `XcKind::BLYP`, `nonperiodic`); `cpu::single_point` |
| Roadmap | `docs/roadmap/dft.md` — Γ-UKS BLYP spin-GGA (productized) |
| Fair wall | \(r = T_{\mathrm{ATV}}/T_{\mathrm{CP2K}} \approx \mathbf{2.48}\) (≤10×) |

## Summary

**Unrestricted Kohn–Sham (UKS) Gaussian-and-plane-waves (GPW)** at the
**Γ point** with spin-resolved **BLYP GGA** (Becke 1988 exchange +
Lee–Yang–Parr correlation). This page pins **triplet O₂** total energy and
Löwdin \(\langle S^2\rangle\) against offline **CP2K 2026.1 Quickstep**
(`LSD`, `MULTIPLICITY 3`, `XC_FUNCTIONAL BLYP`) with **Martyna–Tuckerman**
isolated electrostatics.

Public path: construct `cpu::gpw::GPWEvaluator` with one `Kind` per element
(lookup constructors `GaussianBasis` / `GthPotential`), set members
(`n_alpha`, `n_beta`, `cutoff_ry`, `xc = XcKind::BLYP`, `nonperiodic`, …),
run `cpu::single_point(frame, eval)`. Energy on `frame.meta` is **eV**;
Hartree and \(\langle S^2\rangle\) via `last_energy_hartree()` /
`last_s2()`. Product gate: `reg_gpw-uks-blyp` — absolute energy vs CP2K
plus a **dual-gate** \(|E_{\mathrm{BLYP}}-E_{\mathrm{PBE}}| > 10^{-4}\) Ha
on the same geometry (functional activation).

Related: Γ-UKS PADE foothold [`gpw-uks-o2-gamma`](gpw-uks-o2-gamma.md);
Γ-UKS PBE sibling `regressions/gpw/gpw-uks-pbe`; closed-shell BLYP water
`regressions/gpw/gpw-blyp-water`. Narrative: [Ch. 9 GPW](../ch09-gpw.md).
Molecular all-electron UKS/BLYP is a PySCF-loop concern — not this page.

## Method

**GPW (Quickstep family).** Wavefunction in a contracted Gaussian AO basis;
density collocation on a uniform real-space grid; Hartree via FFT Poisson;
XC on the grid; potential integrated back into the AO basis. Core electrons
removed by analytic **GTH** pseudopotentials. Grid resolution set by the
plane-wave cutoff in Rydberg. Citation: Lippert, Hutter, Parrinello,
*Mol. Phys.* **92**, 477 (1997).

**UKS / spin-GGA.** With \(N_\alpha \neq N_\beta\), build spin densities
\(\rho_\alpha\), \(\rho_\beta\); Hartree from \(\rho=\rho_\alpha+\rho_\beta\);
XC via spin-resolved pure GGA on the grid:

\[
E_{\mathrm{xc}}
=
\sum_g w_g\,(\rho_{\alpha,g}+\rho_{\beta,g})\,\varepsilon_{\mathrm{xc}}
(\rho_\alpha,\rho_\beta,\sigma_{\alpha\alpha},\sigma_{\alpha\beta},\sigma_{\beta\beta}),
\]

with spectral \(\nabla\rho_\sigma\) and potentials

\[
v_\alpha = v_{\rho_\alpha} - \nabla\cdot\bigl(
  2 v_{\sigma_{\alpha\alpha}}\nabla\rho_\alpha
  + v_{\sigma_{\alpha\beta}}\nabla\rho_\beta\bigr)
\]

(and the \(\beta\) twin). Occupations are explicit integer members
`n_alpha` / `n_beta` (not a multiplicity bag). Grid entry point:
`xc_grid_blyp_spin` → `xc_grid_gga_spin(..., XCType::BLYP)`.

**BLYP.** Pure semi-local GGA: **B88** exchange (Becke, *Phys. Rev. A*
**38**, 3098 (1988)) + **LYP** correlation (Lee, Yang, Parr, *Phys. Rev. B*
**37**, 785 (1988)). libxc / CP2K names: `GGA_X_B88` + `GGA_C_LYP` /
`XC_FUNCTIONAL BLYP`. No exact exchange
(`hybrid_exchange_fraction(BLYP) = 0`). At \(\sigma=0\) the GGA reduces to
its LDA limit (unit-tested); at finite \(\sigma\) BLYP and PBE differ —
the dual-gate uses that.

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

**Key exclusions (spec scope-out).**

- No **k-point** UKS-BLYP; Γ only (multi-k UKS is separate product work).
- No hybrid B3LYP / exact exchange on the GPW path here.
- No **ROKS** GPW; no analytic-force or stress claim on this regression
  (force/stress UKS paths exist on other slugs).
- No MPI scaling claim; single-rank pin.
- Not all-electron molecular BLYP (Gaussian SCF / PySCF path).

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
eval.xc = cpu::gpw::XcKind::BLYP;
eval.nonperiodic = true;
eval.n_alpha = 7;  // multiplicity 3, 12 valence e⁻
eval.n_beta = 5;

cpu::single_point(frame, eval);

const double e_ha = eval.last_energy_hartree();       // [Ha]
const double e_ev = frame.meta.get(kEnergy);          // [eV]
const double s2 = eval.last_s2();                     // Löwdin ⟨S²⟩
const cpu::gpw::GpwResult& r = eval.last_result();    // converged, iters, …
```

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
  cmake --build build --target reg_gpw-uks-blyp -j$(nproc)
  ./build/regressions/reg_gpw-uks-blyp
'
```

Relevant sources:

| Piece | Path |
| ----- | ---- |
| Evaluator | `include/atomiverse/cpu/gpw/gpw_evaluator.h` |
| Solver (UKS SCF) | `include/atomiverse/cpu/gpw/gpw_solver.h`, `src/cpu/gpw/gpw_solver.cpp` |
| Spin-GGA BLYP grid | `include/atomiverse/cpu/gpw/internal/xc_grid_gga.h` (`xc_grid_blyp_spin`) |
| Unit tests (not e2e) | `tests/cpu/gpw/test_cpu_gpw_uks.cpp`, `test_cpu_gpw_xc_grid_gga.cpp` |
| Regression | `regressions/gpw/gpw-uks-blyp/reg_gpw-uks-blyp.cpp` |
| CP2K twin input | `tools/cp2k_oracle/inputs/gpw-uks-blyp/o2_uks_blyp.inp` |

## Validation (CP2K)

Source: `regressions/gpw/gpw-uks-blyp/REFERENCE.md` (offline pin; **never**
live CP2K in ctest). Engine: **CP2K 2026.1** (`cp2k.psmp`), module
`CP2K/2026.1` on Arrhenius. Re-pin date: **2026-07-26**.

| Quantity | CP2K pin | Atomiverse gate | Tol | Notes |
| -------- | -------: | --------------- | --- | ----- |
| \(E_{\mathrm{tot}}\) [Ha] | **−31.90397685949710** | \(\lvert E_{\mathrm{ATV}}-E_{\mathrm{CP2K}}\rvert\) | **1e-3 Ha** | First-green; ATV single density grid vs CP2K multigrid (`NGRIDS 4`, `REL_CUTOFF 60`) |
| \(\langle S^2\rangle\) (single det.) | **2.006656** | \(\lvert\langle S^2\rangle_{\mathrm{ATV}}-\mathrm{pin}\rvert\) | **0.15** | UKS contamination; ideal \(S(S+1)=2\) |
| Dual-gate vs PBE | — | \(\lvert E_{\mathrm{BLYP}}-E_{\mathrm{PBE}}\rvert\) | \(> 10^{-4}\) Ha | Functional activation on same geometry |
| Frame energy consistency | \(E_{\mathrm{eV}}/k_{\mathrm{Ha\to eV}} \equiv E_{\mathrm{Ha}}\) | reg asserts | \(10^{-9}\) Ha | `kEnergy` vs `last_energy_hartree()` |
| SCF | converged | `r.converged` | — | `max_iter=200`, `tol=1e-7` |

**Atomiverse first-green (2026-07-26, OMP=4 Release)** from REFERENCE:

| Quantity | Value |
| -------- | ----: |
| \(E_{\mathrm{BLYP}}\) | **−31.903662478892** Ha |
| \(\lvert\Delta E\rvert\) vs CP2K | \(3.14\times 10^{-4}\) Ha |
| \(\langle S^2\rangle\) | 2.00658851 |
| \(E_{\mathrm{PBE}}\) twin | −31.928854728464 Ha |
| \(\lvert E_{\mathrm{BLYP}}-E_{\mathrm{PBE}}\rvert\) | \(2.52\times 10^{-2}\) Ha |
| SCF iters (BLYP) | 60 |

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
| XC | **BLYP** (GGA, spin-resolved) |
| Cutoff | 400 Ry |
| Spin | UKS / LSD, multiplicity 3 → **\(N_\alpha=7\), \(N_\beta=5\)** |
| SCF twin notes | CP2K: diagonalization + Broyden α=0.4, `EPS_SCF 1e-6`; ATV: DIIS-style mix 0.4, `tol=1e-7` |

Key CP2K log lines (from REFERENCE / `o2_uks_blyp.key.txt`):

```text
*** SCF run converged in    19 steps ***
Total energy:                                               -31.90397685949710
Ideal and single determinant S**2 :                    2.000000       2.006656
ENERGY| Total FORCE_EVAL ( QS ) energy [hartree]            -31.903976859497096
```

Offline re-run (operator only; not ctest):

```bash
module load buildtool-easybuild/5.2.1-hpca3ef7d197 GCC/14.3.0 MPICH/4.3.2 CP2K/2026.1
export CP2K_DATA_DIR=$EBROOTCP2K/share/cp2k/data
export OMP_NUM_THREADS=4
cd /tmp && ln -sf $CP2K_DATA_DIR/{BASIS_MOLOPT,GTH_POTENTIALS} .
# copy tools/cp2k_oracle/inputs/gpw-uks-blyp/o2_uks_blyp.inp
mpirun -np 1 cp2k.psmp -i o2_uks_blyp.inp -o o2_uks_blyp.out
```

## Performance

Source: same `REFERENCE.md` fair-wall note (2026-07-26); handoff
the case `REFERENCE.md` (wave 4). Not a ctest gate;
optional offline twin under `tools/cp2k_oracle/`.

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r = T_{\mathrm{ATV}}/T_{\mathrm{CP2K}}\) | ≤10× | Caveat |
| ---- | ------------------------: | ----------------------: | ---------------------------------------: | ---- | ------ |
| O₂ UKS-BLYP / DZVP-MOLOPT-GTH / MT, Γ | **31.7** | **78.6** | **2.48** | ✓ | CP2K: 1 MPI × 4 OMP, multigrid NGRIDS 4; ATV: Release, single density grid, OMP=4, 60 BLYP SCF iters |

Gate from closed spec: \(r \le 10\) or `perf_exempt`. Here \(r \approx 2.48\)
— product pass without exemption (`perf_exempt = false`).

## Limitations

- **Grid twin, not bit-identical SCF:** Atomiverse uses a **single** density
  grid; CP2K pin uses multigrid (`NGRIDS 4` / `REL_CUTOFF 60`). Energy
  tolerance is deliberately **1e-3 Ha** for first-green; may tighten when
  multigrid parity is product work.
- **Slower SCF count on ATV:** first-green BLYP path took **60** iterations
  vs CP2K **19** at `EPS_SCF 1e-6` — dominates the fair-wall ratio.
- **Γ only** — periodic k-mesh UKS-BLYP is not claimed here.
- **Geometry hygiene:** off-center molecules under MT will fail energy
  parity; always center for isolated UKS/RKS pins.
- **ROKS / spin projection** not provided; residual \(\langle S^2\rangle\)
  contamination is accepted (compared to CP2K single-determinant value).
- **No hybrid B3LYP** on this GPW surface; hybrid / GAPW / meta-GGA remain
  blocked or out of loop scope.

## References

- Offline pin: [`regressions/gpw/gpw-uks-blyp/REFERENCE.md`](../../../regressions/gpw/gpw-uks-blyp/REFERENCE.md)
- Regression consumer: [`reg_gpw-uks-blyp.cpp`](../../../regressions/gpw/gpw-uks-blyp/reg_gpw-uks-blyp.cpp)
- CP2K twin: [`tools/cp2k_oracle/inputs/gpw-uks-blyp/`](../../../tools/cp2k_oracle/inputs/gpw-uks-blyp/)
- Oracle contract: [`tools/cp2k_oracle/README.md`](../../../tools/cp2k_oracle/README.md)
- Sibling tutorials: [`gpw-uks-o2-gamma`](gpw-uks-o2-gamma.md) (PADE LSD),
  closed-shell water BLYP reg `gpw-blyp-water`
- Narrative GPW: [`docs/tutorial/ch09-gpw.md`](../ch09-gpw.md)
- Roadmap: [`docs/roadmap/dft.md`](../../roadmap/dft.md)
- G. Lippert, J. Hutter, M. Parrinello, *Mol. Phys.* **92**, 477 (1997) — GPW
- A. D. Becke, *Phys. Rev. A* **38**, 3098 (1988) — B88 exchange
- C. Lee, W. Yang, R. G. Parr, *Phys. Rev. B* **37**, 785 (1988) — LYP correlation
- G. J. Martyna, M. E. Tuckerman, *J. Chem. Phys.* **110**, 2810 (1999) — MT Poisson
- P.-O. Löwdin, *Phys. Rev.* **97**, 1509 (1955) — \(\langle S^2\rangle\) unrestricted dets
