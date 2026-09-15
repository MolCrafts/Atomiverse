# Γ-point UKS GPW multigrid — O₂ / PADE / DZVP-MOLOPT-GTH

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `gpw-uks-multigrid` |
| Regression | `regressions/gpw/gpw-uks-multigrid/` |
| Public API | `cpu::gpw::GPWEvaluator` (`n_alpha`/`n_beta`, `ngrids`, `rel_cutoff`, `XcKind::PADE`, `nonperiodic`); `cpu::single_point` |
| Roadmap | `docs/roadmap/dft.md` — UKS multigrid product collocation |

## Summary

**Unrestricted Kohn–Sham (UKS) Gaussian-and-plane-waves (GPW)** at the
**Γ point** with **CP2K-style multigrid product collocation** (`ngrids > 1`,
`rel_cutoff > 0`). Same open-shell physics as the single-grid foothold
[`gpw-uks-o2-gamma`](gpw-uks-o2-gamma.md) — dual α/β densities and KS matrices,
spin-resolved **PADE (LSD)**, dual-spin DIIS, Löwdin \(\langle S^2\rangle\) —
but each basis function is assigned to a grid level by its most-diffuse
exponent, collocated on that level, prolonged to the fine grid for Hartree+XC,
and the fine potential restricted back for AO integration.

This page pins **triplet O₂** energy and spin contamination against offline
**CP2K 2026.1 Quickstep** (`LSD`, `MULTIPLICITY 3`, `&MGRID NGRIDS 4` /
`REL_CUTOFF 60` / `CUTOFF 400`) with **Martyna–Tuckerman** isolated
electrostatics. Public path: construct `cpu::gpw::GPWEvaluator` with one
`Kind` per element, set members including **`ngrids`** and **`rel_cutoff`**,
run `cpu::single_point(frame, eval)`. Energy on `frame.meta` is **eV**;
Hartree and \(\langle S^2\rangle\) via `last_energy_hartree()` / `last_s2()`.
Product gate: `reg_gpw-uks-multigrid`. Fair wall ratio
\(r = T_{\mathrm{ATV}}/T_{\mathrm{CP2K}} \approx 1.54\) (≤10×).

Related narrative: [Ch. 9 GPW](../ch09-gpw.md). Single-grid UKS twin:
`gpw-uks-o2-gamma`. RKS multigrid machinery is shared
(`internal/multigrid.h`); this regression is the **UKS product path** that
was previously unwired.

## Method

**GPW (Quickstep family).** Wavefunction in a contracted Gaussian AO basis;
density collocation on real-space grids; Hartree via FFT Poisson; XC on the
grid; potential integrated back into the AO basis. Core electrons removed by
analytic **GTH** pseudopotentials. Citation: Lippert, Hutter, Parrinello,
*Mol. Phys.* **92**, 477 (1997).

**Multigrid product collocation (`&MGRID`).** A single uniform grid forces
every Gaussian onto the finest spacing: diffuse functions span huge bounding
boxes (slow) and leave an O(mHa) residual vs CP2K multigrid. CP2K-style
mapping:

1. Build `ngrids` geometrically spaced levels; finest cutoff = `CUTOFF`,
   coarsest ~ `REL_CUTOFF`.
2. Assign each basis function to the **coarsest** level that still resolves
   its most-diffuse exponent.
3. Collocate density contributions per level (compact support on coarse
   levels for diffuse Gaussians).
4. **Prolong** per-level densities to the fine grid; run nonlinear
   Hartree + XC there.
5. **Restrict** the fine potential back to each level and integrate the KS
   matrix.

Inter-level transfer uses **spectral (FFT) prolong / restrict**
(`prolong_fft` / `restrict_fft`) — exact adjoints so the integrated KS
matrix remains the gradient of the collocated energy (variational SCF). For
this pin, ATV uses FFT transfer even under Martyna–Tuckerman
`nonperiodic`: the density lives on a uniform box grid and is well-contained
for centered O₂; real-space trilinear fallback on `periodic=false` left an
O(0.4 Ha) Hartree residual (see REFERENCE implementation note). Default
`ngrids = 1` bypasses multigrid entirely (bit-identical single-grid path).

**UKS / LSD.** With \(N_\alpha \neq N_\beta\), build spin densities
\(\rho_\alpha\), \(\rho_\beta\); Hartree from \(\rho=\rho_\alpha+\rho_\beta\);
XC via spin-resolved PADE LSD; diagonalize separate \(H_\alpha\) and
\(H_\beta\). Occupations are explicit integer members `n_alpha` / `n_beta`.
Product collocation applies independently to dual spin densities when
`ngrids > 1`.

**PADE LSD.** Goedecker–Teter–Hutter Padé parameterization of the LDA for
spin-polarized densities (CP2K `&XC_FUNCTIONAL PADE` + `LSD`). Citation:
Goedecker, Teter, Hutter, *Phys. Rev. B* **54**, 1703 (1996).

**Isolated electrostatics.** `nonperiodic = true` selects
**Martyna–Tuckerman** Poisson (CP2K `PERIODIC NONE` / `PSOLVER MT`). The
molecule must sit **centered** in the cell. Citation: Martyna & Tuckerman,
*J. Chem. Phys.* **110**, 2810 (1999).

**Spin metric.** After convergence, \(\langle S^2\rangle\) is the Löwdin
expectation for a single unrestricted determinant. Ideal high-spin
\(S=1\): \(S(S+1)=2\). Citation: Löwdin, *Phys. Rev.* **97**, 1509 (1955).

**Valence bookkeeping (this pin).** O with GTH-PBE-q6 → 6 valence e⁻ per O →
12 valence e⁻ total. Multiplicity 3 ⇒ \(N_\alpha=7\), \(N_\beta=5\).

**Key exclusions (spec scope-out).**

- No **k-point** UKS multigrid claim on this regression (Γ only).
- No hybrids / meta-GGA / GAPW.
- Multigrid does not change public energy units: Frame still reports **eV**.
- No MPI scaling claim; single-rank pin.
- Not all-electron molecular UKS (Gaussian SCF / PySCF-loop concern).

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
eval.ngrids = 4;        // CP2K &MGRID NGRIDS
eval.rel_cutoff = 60.0; // CP2K REL_CUTOFF [Ry]
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
  cmake --build build --target reg_gpw-uks-multigrid -j$(nproc)
  ./build/regressions/reg_gpw-uks-multigrid
'
```

Relevant sources:

| Piece | Path |
| ----- | ---- |
| Evaluator (`ngrids` / `rel_cutoff`) | `include/atomiverse/cpu/gpw/gpw_evaluator.h` |
| Solver (UKS + multigrid product path) | `include/atomiverse/cpu/gpw/gpw_solver.h`, `src/cpu/gpw/gpw_solver.cpp` |
| Multigrid levels / prolong / restrict | `include/atomiverse/cpu/gpw/internal/multigrid.h` |
| Unit tests (not e2e) | `tests/cpu/gpw/test_cpu_gpw_uks.cpp` (UKS multigrid), `tests/cpu/gpw/test_cpu_gpw_multigrid.cpp` |
| Regression | `regressions/gpw/gpw-uks-multigrid/reg_gpw-uks-multigrid.cpp` |
| CP2K twin input | `tools/cp2k_oracle/inputs/gpw-uks-multigrid/o2_uks_pade_mg.inp` |

## Validation (CP2K)

Source: `regressions/gpw/gpw-uks-multigrid/REFERENCE.md` (offline pin; **never**
live CP2K in ctest). Engine: **CP2K 2026.1** (`cp2k.psmp`, git:5e54ba2),
module `CP2K/2026.1` on Arrhenius. Re-pin date: **2026-07-23** (centered
geometry; energy/S² shared with `gpw-uks-o2-gamma` twin physics). ATV
multigrid product path fair-wall date: **2026-07-26**.

| Quantity | CP2K pin | Atomiverse gate | Tol | Notes |
| -------- | -------: | --------------- | --- | ----- |
| \(E_{\mathrm{tot}}\) [Ha] | **−31.76980349931425** | \(\lvert E_{\mathrm{ATV}}-E_{\mathrm{CP2K}}\rvert\) | **1e-3 Ha** | Both sides NGRIDS 4 / REL_CUTOFF 60 / CUTOFF 400 |
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
| Cutoff | **400 Ry** |
| Multigrid | **NGRIDS 4**, **REL_CUTOFF 60** Ry (ATV + CP2K) |
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
# copy tools/cp2k_oracle/inputs/gpw-uks-multigrid/o2_uks_pade_mg.inp
mpirun -np 1 cp2k.psmp -i o2_uks_pade_mg.inp -o o2_uks_pade_mg.out
```

## Performance

Source: same `REFERENCE.md` fair-wall note (2026-07-26 ATV / 2026-07-23 CP2K).
Not a ctest gate; optional offline twin under `tools/cp2k_oracle/`. Handoff:
the case `REFERENCE.md` ratio **~1.54**.

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r = T_{\mathrm{ATV}}/T_{\mathrm{CP2K}}\) | ≤10× | Caveat |
| ---- | ------------------------: | ----------------------: | ---------------------------------------: | ---- | ------ |
| O₂ UKS-PADE / DZVP-MOLOPT-GTH / MT, Γ, NGRIDS 4 | **21.3** | **32.8** | **1.54** | ✓ | CP2K: 1 MPI × 4 OMP; ATV: Release, `ngrids=4`, `rel_cutoff=60`, OMP=4 |

Gate from closed spec: \(r \le 10\) or `perf_exempt`. Here \(r \approx 1.54\)
— product pass without exemption. Multigrid product path is slower than the
single-grid UKS foothold (\(r \approx 0.98\) on `gpw-uks-o2-gamma`) because
level assignment + prolong/restrict run every SCF density build; still well
inside the 10× wall budget.

## Limitations

- **Grid discretization residual remains:** energy tol is deliberately
  **1e-3 Ha** (shared with the single-grid UKS foothold). Matched NGRIDS
  does not imply bit-identical SCF (different mixers, DIIS vs Broyden,
  grid assignment edge cases).
- **FFT prolong under MT:** spectral transfer is required for energy
  parity with nonperiodic O₂; do not reintroduce pure trilinear on the
  `nonperiodic` product path without re-pinning.
- **Γ only** on this regression — multi-k UKS multigrid is not claimed here.
- **Geometry hygiene:** off-center molecules under MT destroy energy parity;
  always center for isolated UKS pins.
- **ROKS / spin projection** not provided; residual \(\langle S^2\rangle\)
  contamination is accepted (compared to CP2K single-determinant value).
- **Default path still single-grid:** `ngrids = 1` (default) does not engage
  product collocation; set `ngrids` and `rel_cutoff` explicitly to twin CP2K
  multigrid.

## References

- Offline pin: [`regressions/gpw/gpw-uks-multigrid/REFERENCE.md`](../../../regressions/gpw/gpw-uks-multigrid/REFERENCE.md)
- Regression consumer: [`reg_gpw-uks-multigrid.cpp`](../../../regressions/gpw/gpw-uks-multigrid/reg_gpw-uks-multigrid.cpp)
- CP2K twin: [`tools/cp2k_oracle/inputs/gpw-uks-multigrid/`](../../../tools/cp2k_oracle/inputs/gpw-uks-multigrid/)
- Oracle contract: [`tools/cp2k_oracle/README.md`](../../../tools/cp2k_oracle/README.md)
- Single-grid UKS sibling: [`gpw-uks-o2-gamma.md`](gpw-uks-o2-gamma.md)
- Narrative GPW: [`docs/tutorial/ch09-gpw.md`](../ch09-gpw.md)
- Roadmap: [`docs/roadmap/dft.md`](../../roadmap/dft.md)
- G. Lippert, J. Hutter, M. Parrinello, *Mol. Phys.* **92**, 477 (1997) — GPW
- S. Goedecker, M. Teter, J. Hutter, *Phys. Rev. B* **54**, 1703 (1996) — PADE / GTH
- G. J. Martyna, M. E. Tuckerman, *J. Chem. Phys.* **110**, 2810 (1999) — MT Poisson
- P.-O. Löwdin, *Phys. Rev.* **97**, 1509 (1955) — \(\langle S^2\rangle\) unrestricted dets
- CP2K Quickstep `&MGRID` CUTOFF / REL_CUTOFF / NGRIDS multigrid collocation
