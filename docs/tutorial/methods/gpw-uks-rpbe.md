# Γ-point UKS GPW — O₂ / RPBE / DZVP-MOLOPT-GTH

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `gpw-uks-rpbe` |
| Regression | `regressions/gpw/gpw-uks-rpbe/` |
| Public API | `cpu::gpw::GPWEvaluator` (`n_alpha`/`n_beta`, `XcKind::RPBE`, `nonperiodic`); `cpu::single_point` |
| Roadmap | `docs/roadmap/dft.md` — RPBE RKS + Γ-UKS (`reg_gpw-uks-rpbe`) |

## Summary

**Unrestricted Kohn–Sham (UKS) Gaussian-and-plane-waves (GPW)** at the
**Γ point** with spin-resolved **RPBE GGA** (Hammer–Hansen–Nørskov exchange
enhancement + **PBE correlation**). Separate α/β densities and KS matrices,
dual-spin DIIS, Löwdin \(\langle S^2\rangle\). This page pins the **triplet O₂**
total energy and spin contamination against offline **CP2K 2026.1 Quickstep**
(`LSD`, `MULTIPLICITY 3`, libxc-style `&GGA_X_RPBE` + `&GGA_C_PBE`) with
**Martyna–Tuckerman** isolated electrostatics.

**Not revPBE** (Zhang–Yang κ-revision of the rational PBE form — see
`gpw-uks-revpbe` / `gpw-revpbe-si`). Closed-shell RKS RPBE is
`gpw-rpbe-water`.

Public path: construct `cpu::gpw::GPWEvaluator` with one `Kind` per element
(lookup constructors `GaussianBasis` / `GthPotential`), set members
(`n_alpha`, `n_beta`, `cutoff_ry`, `xc = XcKind::RPBE`, `nonperiodic`, …),
run `cpu::single_point(frame, eval)`. Energy on `frame.meta` is **eV**;
Hartree and \(\langle S^2\rangle\) via `last_energy_hartree()` / `last_s2()`.
Product gate: `reg_gpw-uks-rpbe` (CP2K absolute energy + \(\langle S^2\rangle\)
+ **dual-gate** \(|E_{\mathrm{RPBE}}-E_{\mathrm{PBE}}| > 10^{-4}\) Ha). Fair
wall ratio \(r = T_{\mathrm{ATV}}/T_{\mathrm{CP2K}} \approx 2.70\) (≤10×).

Related narrative: [Ch. 9 GPW](../ch09-gpw.md). PADE LSD UKS foothold:
`gpw-uks-o2-gamma`. PBE UKS twin: `gpw-uks-pbe`.

## Method

**GPW (Quickstep family).** Wavefunction in a contracted Gaussian AO basis;
density collocation on a uniform real-space grid; Hartree via FFT Poisson;
XC on the grid; potential integrated back into the AO basis. Core electrons
removed by analytic **GTH** pseudopotentials. Grid resolution set by the
plane-wave cutoff in Rydberg. Citation: Lippert, Hutter, Parrinello,
*Mol. Phys.* **92**, 477 (1997).

**UKS / collinear spin.** With \(N_\alpha \neq N_\beta\), build spin densities
\(\rho_\alpha\), \(\rho_\beta\); Hartree from \(\rho=\rho_\alpha+\rho_\beta\);
XC via spin-resolved GGA on the grid (`xc_grid_rpbe_spin`); diagonalize separate
\(H_\alpha\) and \(H_\beta\). Occupations are explicit integer members
`n_alpha` / `n_beta` (not a multiplicity bag). When both are zero, the
closed-shell `n_occ` path is unchanged.

**RPBE (Hammer–Hansen–Nørskov 1999).** Exchange enhancement is the exponential
form

\[
F_x(s) = 1 + \kappa\bigl(1 - e^{-\mu s^2/\kappa}\bigr)
\]

with the **same** PBE parameters \(\kappa = 0.804\), \(\mu \approx 0.21951\),
and **PBE correlation** (PW92 LDA base + PBE gradient correction). This is
**not** revPBE (which revises \(\kappa\) in the *rational* PBE \(F_x\)).
libxc / CP2K product names: `GGA_X_RPBE` + `GGA_C_PBE`. In CP2K **2026.1**,
the short tokens `XC_FUNCTIONAL RPBE` and `PARAMETRIZATION RPBE` are **not**
valid enumerations — the twin input uses nested libxc subsections.

Atomiverse routes RPBE through the shared GGA grid path already used for PBE /
PBEsol / revPBE / BLYP, with `XcKind::RPBE` selecting the RPBE exchange kernel
and PBE correlation (shared `cpu::scf::XCType::RPBE`).

**Isolated electrostatics.** `nonperiodic = true` selects the
**Martyna–Tuckerman** Poisson twin of CP2K `PERIODIC NONE` / `PSOLVER MT`.
The molecule must sit **centered** in the cell: nonperiodic MT grids cover
\([0,L)\); atoms near an edge truncate Gaussian tails and destroy the energy.
Citation: Martyna & Tuckerman, *J. Chem. Phys.* **110**, 2810 (1999).

**Spin metric.** After convergence, \(\langle S^2\rangle\) is the Löwdin
expectation for a single unrestricted determinant (not spin-projected).
Ideal high-spin value for \(S=1\): \(S(S+1)=2\). Citation: Löwdin,
*Phys. Rev.* **97**, 1509 (1955).

**Valence bookkeeping (this pin).** O with GTH-PBE-q6 → 6 valence e⁻ per O →
12 valence e⁻ total. Multiplicity 3 ⇒ \(N_\alpha=7\), \(N_\beta=5\)
(not all-electron \(9/7\)).

**Dual-gate (activation).** Same geometry with `XcKind::PBE` must differ by
more than \(10^{-4}\) Ha so the RPBE path cannot silently fall back to PBE.
REFERENCE first-green: \(|E_{\mathrm{RPBE}}-E_{\mathrm{PBE}}| \approx
4.46\times 10^{-2}\) Ha.

**Key exclusions (spec scope-out).**

- No **k-point** UKS on this regression (multi-k UKS is `gpw-kpoints-uks` /
  sibling force/stress pins).
- No **analytic UKS RPBE forces** / stress claim on this page (force path
  exists elsewhere; product gate here is energy + \(\langle S^2\rangle\)).
- No MPI scaling claim; single-rank pin.
- Not all-electron molecular UKS (Gaussian SCF / PySCF loop).
- **UGW** RPBE spin XC not productized (`GpwDft::g0w0` still refuses UKS RPBE).

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
eval.xc = cpu::gpw::XcKind::RPBE;  // Hammer–Hansen–Nørskov + PBE-c
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
  cmake --build build -j$(nproc) --target atomiverse reg_gpw-uks-rpbe
  ./build/regressions/reg_gpw-uks-rpbe
'
```

Relevant sources:

| Piece | Path |
| ----- | ---- |
| Evaluator | `include/atomiverse/cpu/gpw/gpw_evaluator.h` |
| Solver (UKS SCF + XcKind) | `include/atomiverse/cpu/gpw/gpw_solver.h`, `src/cpu/gpw/gpw_solver.cpp` |
| RPBE spin GGA grid | `src/cpu/gpw/internal/xc_grid_gga.cpp` (`xc_grid_rpbe_spin`) |
| Regression | `regressions/gpw/gpw-uks-rpbe/reg_gpw-uks-rpbe.cpp` |
| CP2K twin input | `tools/cp2k_oracle/inputs/gpw-uks-rpbe/o2_uks_rpbe.inp` |
| Sibling RKS RPBE | `regressions/gpw/gpw-rpbe-water/` |
| Sibling UKS PBE | `regressions/gpw/gpw-uks-pbe/` |

## Validation (CP2K)

Source: `regressions/gpw/gpw-uks-rpbe/REFERENCE.md` (offline pin; **never**
live CP2K in ctest). Engine: **CP2K 2026.1** (`cp2k.psmp`), modules
`buildtool-easybuild/5.2.1-hpca3ef7d197` + `GCC/14.3.0` + `MPICH/4.3.2` +
`CP2K/2026.1` on Arrhenius. Re-pin date: **2026-07-26**.

| Quantity | CP2K pin | Atomiverse gate | Tol | Notes |
| -------- | -------: | --------------- | --- | ----- |
| \(E_{\mathrm{tot}}\) [Ha] | **−31.97338355451434** | \(\lvert E_{\mathrm{ATV}}-E_{\mathrm{CP2K}}\rvert\) | **1e-3 Ha** | First-green; ATV single density grid vs CP2K multigrid (`NGRIDS 4`, `REL_CUTOFF 60`) |
| \(\langle S^2\rangle\) (single det.) | **2.005630** | \(\lvert\langle S^2\rangle_{\mathrm{ATV}}-\mathrm{pin}\rvert\) | **0.15** | UKS contamination; ideal \(S(S+1)=2\) |
| Dual-gate vs PBE | — | \(\lvert E_{\mathrm{RPBE}}-E_{\mathrm{PBE}}\rvert\) | \(> 10^{-4}\) Ha | Hard activation check |
| Frame energy consistency | \(E_{\mathrm{eV}}/k_{\mathrm{Ha\to eV}} \equiv E_{\mathrm{Ha}}\) | reg asserts | \(10^{-9}\) Ha | `kEnergy` vs `last_energy_hartree()` |
| SCF | converged | `r.converged` | — | `max_iter=200`, `tol=1e-7` |

**Atomiverse first-green (2026-07-26, OMP=4 Release)** from REFERENCE:

| Quantity | Value |
| -------- | ----: |
| \(E_{\mathrm{RPBE}}\) | **−31.973422072918 Ha** |
| \(\lvert\Delta E\rvert\) vs CP2K | \(3.85\times 10^{-5}\) Ha |
| \(\langle S^2\rangle\) | 2.00563766 |
| \(E_{\mathrm{PBE}}\) twin | −31.928854728710 Ha |
| \(\lvert E_{\mathrm{RPBE}}-E_{\mathrm{PBE}}\rvert\) | \(4.46\times 10^{-2}\) Ha |
| SCF iters (RPBE) | 59 |

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
| XC | **RPBE** (`GGA_X_RPBE` + `GGA_C_PBE`, spin-polarized) |
| Cutoff | 400 Ry |
| Spin | UKS / LSD, multiplicity 3 → **\(N_\alpha=7\), \(N_\beta=5\)** |
| SCF twin notes | CP2K: diagonalization + Broyden α=0.4, `EPS_SCF 1e-6`; ATV: DIIS-style mix 0.4, `tol=1e-7` |

Key CP2K log lines (from REFERENCE):

```text
  *** SCF run converged in    15 steps ***
  Total energy:                                               -31.97338355451434
  Ideal and single determinant S**2 :                    2.000000       2.005630
 ENERGY| Total FORCE_EVAL ( QS ) energy [hartree]            -31.973383554514335
 FUNCTIONAL| GGA_X_RPBE: … {scale=1.000, spin-polarized}
 FUNCTIONAL| GGA_C_PBE: … {scale=1.000, spin-polarized}
```

Offline re-run (operator only; not ctest):

```bash
module load buildtool-easybuild/5.2.1-hpca3ef7d197 GCC/14.3.0 MPICH/4.3.2 CP2K/2026.1
export CP2K_DATA_DIR=$EBROOTCP2K/share/cp2k/data
export OMP_NUM_THREADS=4
cd /tmp && ln -sf $CP2K_DATA_DIR/{BASIS_MOLOPT,GTH_POTENTIALS} .
# copy tools/cp2k_oracle/inputs/gpw-uks-rpbe/o2_uks_rpbe.inp
cp2k.psmp -i o2_uks_rpbe.inp -o o2_uks_rpbe.out
```

## Performance

Source: same `REFERENCE.md` fair-wall note (2026-07-26) and
the case `REFERENCE.md` (`ratio ~2.70`). Not a ctest gate;
optional offline twin under `tools/cp2k_oracle/`.

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r = T_{\mathrm{ATV}}/T_{\mathrm{CP2K}}\) | ≤10× | Caveat |
| ---- | ------------------------: | ----------------------: | ---------------------------------------: | ---- | ------ |
| O₂ UKS-RPBE / DZVP-MOLOPT-GTH / MT, Γ | **30.4** | **82.0** | **2.70** | ✓ | CP2K: 1 process × 4 OMP, multigrid NGRIDS 4; ATV: Release, single density grid, OMP=4, 59 SCF iters |

Gate from closed product: \(r \le 10\) or `perf_exempt`. Here
\(r \approx 2.70\) — product pass without exemption (`perf_exempt = false`).

## Limitations

- **Grid twin, not bit-identical SCF:** Atomiverse uses a **single** density
  grid; CP2K pin uses multigrid (`NGRIDS 4` / `REL_CUTOFF 60`). Energy
  tolerance is deliberately **1e-3 Ha** for first-green; first-green
  \(|\Delta E|\) is already \(\sim 4\times 10^{-5}\) Ha.
- **RPBE ≠ revPBE:** do not substitute `XcKind::RevPBE` for this pin or claim
  the same CP2K energy.
- **CP2K input form:** short `XC_FUNCTIONAL RPBE` is invalid in 2026.1 —
  twin requires nested `&GGA_X_RPBE` / `&GGA_C_PBE`.
- **No force/stress e2e** on this slug; energy + \(\langle S^2\rangle\) + dual-gate
  only.
- **Γ only** on this page — multi-k UKS RPBE is not a separate product pin
  (k-UKS uses the same XcKind allow-list when staffed).
- **Geometry hygiene:** off-center molecules under MT fail energy parity;
  always center for isolated UKS/RKS pins.
- **ROKS / spin projection** not provided; residual \(\langle S^2\rangle\)
  contamination is accepted (compared to CP2K single-determinant value).
- **UGW** on UKS-RPBE densities is deferred (`g0w0` error path).

## References

- Offline pin: [`regressions/gpw/gpw-uks-rpbe/REFERENCE.md`](../../../regressions/gpw/gpw-uks-rpbe/REFERENCE.md)
- Regression consumer: [`reg_gpw-uks-rpbe.cpp`](../../../regressions/gpw/gpw-uks-rpbe/reg_gpw-uks-rpbe.cpp)
- CP2K twin: [`tools/cp2k_oracle/inputs/gpw-uks-rpbe/`](../../../tools/cp2k_oracle/inputs/gpw-uks-rpbe/)
- Oracle contract: [`tools/cp2k_oracle/README.md`](../../../tools/cp2k_oracle/README.md)
- Handoff ratio: the case `REFERENCE.md`
- Narrative GPW: [`docs/tutorial/ch09-gpw.md`](../ch09-gpw.md)
- Roadmap: [`docs/roadmap/dft.md`](../../roadmap/dft.md)
- PADE UKS sibling: [`gpw-uks-o2-gamma.md`](gpw-uks-o2-gamma.md)
- B. Hammer, L. B. Hansen, J. K. Nørskov, *Phys. Rev. B* **59**, 7413 (1999) — RPBE
- J. P. Perdew, K. Burke, M. Ernzerhof, *Phys. Rev. Lett.* **77**, 3865 (1996) — PBE correlation
- G. Lippert, J. Hutter, M. Parrinello, *Mol. Phys.* **92**, 477 (1997) — GPW
- G. J. Martyna, M. E. Tuckerman, *J. Chem. Phys.* **110**, 2810 (1999) — MT Poisson
- P.-O. Löwdin, *Phys. Rev.* **97**, 1509 (1955) — \(\langle S^2\rangle\) unrestricted dets
