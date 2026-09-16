# Γ-point UKS GPW — O₂ / PADE / DZVP-MOLOPT-GTH

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `gpw-uks-o2-gamma` |
| Regression | `regressions/gpw/gpw-uks-o2-gamma/` |
| Public API | `cpu::gpw::GPWEvaluator` (`n_alpha`/`n_beta`, `XcKind::PADE`, `nonperiodic`); `cpu::single_point` |
| Roadmap | `docs/roadmap/dft.md` — Γ-point UKS GPW (PADE LSD foothold) |

## Summary

**Unrestricted Kohn–Sham (UKS) Gaussian-and-plane-waves (GPW)** at the
**Γ point** for open-shell systems: separate α/β densities and KS matrices,
spin-resolved **PADE (LSD)** exchange–correlation, dual-spin DIIS, and
Löwdin \(\langle S^2\rangle\). This page pins the **triplet O₂** energy and
spin contamination against offline **CP2K 2026.1 Quickstep** (`LSD`,
`MULTIPLICITY 3`) with **Martyna–Tuckerman** isolated electrostatics.

Public path: construct `cpu::gpw::GPWEvaluator` with one `Kind` per element
(lookup constructors `GaussianBasis` / `GthPotential`), set members
(`n_alpha`, `n_beta`, `cutoff_ry`, `xc`, `nonperiodic`, …), run
`cpu::single_point(frame, eval)`. Energy on `frame.meta` is **eV**; Hartree
and \(\langle S^2\rangle\) via `last_energy_hartree()` / `last_s2()`. Product
gate: `reg_gpw-uks-o2-gamma`. Fair wall ratio
\(r = T_{\mathrm{ATV}}/T_{\mathrm{CP2K}} \approx 0.98\) (≤10×).

Related narrative: [Ch. 9 GPW](../ch09-gpw.md). Closed-shell MT twin:
`regressions/gpw/gpw-water-pbe-mt`. Molecular all-electron open-shell UKS
is a separate SCF / PySCF-loop concern — not this page.

## Method

**GPW (Quickstep family).** Wavefunction in a contracted Gaussian AO basis;
density collocation on a uniform real-space grid; Hartree via FFT Poisson;
XC on the grid; potential integrated back into the AO basis. Core electrons
removed by analytic **GTH** pseudopotentials. Grid resolution set by the
plane-wave cutoff in Rydberg. Citation: Lippert, Hutter, Parrinello,
*Mol. Phys.* **92**, 477 (1997).

**UKS / LSD.** With \(N_\alpha \neq N_\beta\), build spin densities
\(\rho_\alpha\), \(\rho_\beta\); Hartree from \(\rho=\rho_\alpha+\rho_\beta\);
XC via spin-resolved PADE LSD (`xc_grid_pade_lsd`); diagonalize separate
\(H_\alpha\) and \(H_\beta\). Occupations are explicit integer members
`n_alpha` / `n_beta` (not a multiplicity bag). When both are zero, the
closed-shell `n_occ` path is unchanged.

**PADE LSD.** Goedecker–Teter–Hutter Padé parameterization of the LDA for
spin-polarized densities (same functional family as CP2K
`&XC_FUNCTIONAL PADE` with `LSD`). Citation: Goedecker, Teter, Hutter,
*Phys. Rev. B* **54**, 1703 (1996).

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

- No **k-point** UKS; Γ only.
- No **PBE-GGA UKS** (UKS + `XcKind::PBE` is unsupported for this foothold).
- No **ROKS** GPW; no **analytic UKS forces** / stress on this regression.
- No MPI scaling claim; single-rank pin.
- Not all-electron molecular UKS (Gaussian SCF path).

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
eval.xc = cpu::gpw::XcKind::PADE;
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
  cmake --build build --target reg_gpw-uks-o2-gamma -j$(nproc)
  ./build/regressions/reg_gpw-uks-o2-gamma
'
```

Relevant sources:

| Piece | Path |
| ----- | ---- |
| Evaluator | `include/atomiverse/cpu/gpw/gpw_evaluator.h` |
| Solver (UKS SCF) | `include/atomiverse/cpu/gpw/gpw_solver.h`, `src/cpu/gpw/gpw_solver.cpp` |
| PADE LSD grid | `include/atomiverse/cpu/gpw/internal/xc_grid.h` |
| Unit tests (not e2e) | `tests/cpu/gpw/test_cpu_gpw_uks.cpp` |
| Regression | `regressions/gpw/gpw-uks-o2-gamma/reg_gpw-uks-o2-gamma.cpp` |
| CP2K twin input | `tools/cp2k_oracle/inputs/gpw-uks-o2-gamma/o2_uks_pade.inp` |

## Validation (CP2K)

Source: `regressions/gpw/gpw-uks-o2-gamma/REFERENCE.md` (offline pin; **never**
live CP2K in ctest). Engine: **CP2K 2026.1** (`cp2k.psmp`, git:5e54ba2),
module `CP2K/2026.1` on Arrhenius. Re-pin date: **2026-07-23** (centered
geometry).

| Quantity | CP2K pin | Atomiverse gate | Tol | Notes |
| -------- | -------: | --------------- | --- | ----- |
| \(E_{\mathrm{tot}}\) [Ha] | **−31.76980349931425** | \(\lvert E_{\mathrm{ATV}}-E_{\mathrm{CP2K}}\rvert\) | **1e-3 Ha** | First-green; ATV single density grid vs CP2K multigrid (`NGRIDS 4`, `REL_CUTOFF 60`) |
| \(\langle S^2\rangle\) (single det.) | **2.004659** | \(\lvert\langle S^2\rangle_{\mathrm{ATV}}-\mathrm{pin}\rvert\) | **0.15** | UKS contamination; ideal \(S(S+1)=2\) |
| Integrated spin density (CP2K) | 2.000000 | — | — | log diagnostic only |
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
| Cutoff | 400 Ry |
| Spin | UKS / LSD, multiplicity 3 → **\(N_\alpha=7\), \(N_\beta=5\)** |
| SCF twin notes | CP2K: diagonalization + Broyden α=0.4; ATV: DIIS-style mix 0.4 |

Key CP2K log lines (from REFERENCE):

```text
Total energy:                                               -31.76980349931425
Ideal and single determinant S**2 :                    2.000000       2.004659
ENERGY| Total FORCE_EVAL ( QS ) energy [hartree]            -31.769803499314250
```

Offline re-run (operator only; not ctest):

```bash
module load buildtool-easybuild/5.2.1-hpca3ef7d197 GCC/14.3.0 MPICH/4.3.2 CP2K/2026.1
export CP2K_DATA_DIR=$EBROOTCP2K/share/cp2k/data
cd /tmp && ln -sf $CP2K_DATA_DIR/{BASIS_MOLOPT,GTH_POTENTIALS} .
# copy tools/cp2k_oracle/inputs/gpw-uks-o2-gamma/o2_uks_pade.inp
mpirun -np 1 cp2k.psmp -i o2_uks_pade.inp -o o2_uks_pade.out
```

## Performance

Source: same `REFERENCE.md` fair-wall note (2026-07-23). Not a ctest gate;
optional offline twin under `tools/cp2k_oracle/`.

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r = T_{\mathrm{ATV}}/T_{\mathrm{CP2K}}\) | ≤10× | Caveat |
| ---- | ------------------------: | ----------------------: | ---------------------------------------: | ---- | ------ |
| O₂ UKS-PADE / DZVP-MOLOPT-GTH / MT, Γ | **21.3** | **20.8** | **0.98** | ✓ | CP2K: 1 MPI × 4 OMP, multigrid NGRIDS 4; ATV: Release, single density grid, OMP=4 |

Gate from closed spec: \(r \le 10\) or `perf_exempt`. Here \(r \approx 0.98\)
— product pass without exemption.

## Limitations

- **Grid twin, not bit-identical SCF:** Atomiverse uses a **single** density
  grid; CP2K pin uses multigrid (`NGRIDS 4` / `REL_CUTOFF 60`). Energy
  tolerance is deliberately **1e-3 Ha** for first-green (spec A3); may
  tighten when multigrid parity is product work.
- **UKS + PBE GGA** out of scope; request throws / unsupported path — do not
  document as available.
- **No analytic UKS forces** or stress on this foothold; force regressions
  remain closed-shell (`gpw-forces-water`).
- **Γ only** — periodic k-mesh UKS is roadmap open (plane-wave spin path is
  separate: `pw-spin-polarized-o2`).
- **Geometry hygiene:** off-center molecules under MT will fail energy
  parity; always center for isolated UKS/RKS pins.
- **ROKS / spin projection** not provided; residual \(\langle S^2\rangle\)
  contamination is accepted (compared to CP2K single-determinant value).

## References

- Offline pin: [`regressions/gpw/gpw-uks-o2-gamma/REFERENCE.md`](../../../regressions/gpw/gpw-uks-o2-gamma/REFERENCE.md)
- Regression consumer: [`reg_gpw-uks-o2-gamma.cpp`](../../../regressions/gpw/gpw-uks-o2-gamma/reg_gpw-uks-o2-gamma.cpp)
- CP2K twin: [`tools/cp2k_oracle/inputs/gpw-uks-o2-gamma/`](../../../tools/cp2k_oracle/inputs/gpw-uks-o2-gamma/)
- Oracle contract: [`tools/cp2k_oracle/README.md`](../../../tools/cp2k_oracle/README.md)
- Narrative GPW: [`docs/tutorial/ch09-gpw.md`](../ch09-gpw.md)
- Roadmap: [`docs/roadmap/dft.md`](../../roadmap/dft.md)
- G. Lippert, J. Hutter, M. Parrinello, *Mol. Phys.* **92**, 477 (1997) — GPW
- S. Goedecker, M. Teter, J. Hutter, *Phys. Rev. B* **54**, 1703 (1996) — PADE / GTH
- G. J. Martyna, M. E. Tuckerman, *J. Chem. Phys.* **110**, 2810 (1999) — MT Poisson
- P.-O. Löwdin, *Phys. Rev.* **97**, 1509 (1955) — \(\langle S^2\rangle\) unrestricted dets
