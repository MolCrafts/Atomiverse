# Multi-k UKS GPW analytic forces — high-spin H₂ / PADE / DZV-GTH-PADE

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Regression | None standing (end-to-end case removed) |
| Public API | `cpu::gpw::GpwDft` (`n_alpha`/`n_beta`, `k_mesh`, `XcKind::PADE`); `GpwDft::forces(frame)` |
| Roadmap | `docs/roadmap/dft.md` — k-point UKS analytic ionic forces |

## Summary

**Collinear unrestricted Kohn–Sham (UKS) Gaussian-and-plane-waves (GPW)** on a
**Monkhorst–Pack k-mesh** with **analytic ionic forces**: one multi-k UKS SCF
followed by one gradient pass over Bloch-phased \(S_k/T_k/V_\mathrm{nl}\), dual
density matrices \(P_\alpha^k/P_\beta^k\), energy-weighted \(W_k^\sigma\), and
spin-resolved \(V_\mathrm{eff}^{\alpha/\beta}\) against complex \(\chi^k\).

This page pins **periodic high-spin H₂** (\(N_\alpha=2\), \(N_\beta=0\)) at
**1×1×2** k-points against frozen first-green Atomiverse numbers (energy +
bond forces + Newton-3 / \(\lvert\sum\mathbf{F}\rvert\)) and the closed-shell
identity multi-k UKS ≡ multi-k RKS. Offline **CP2K** twin input ships under
CP2K oracle (exploratory SZV-MOLOPT basis;
absolute same-basis twin pending DZV-GTH-PADE on the CP2K data path).

Public path: construct `cpu::gpw::GpwDft`, set members (`n_alpha`, `n_beta`,
`k_mesh`, `cutoff_ry`, `xc`, `local_pp_on_grid`, …), `build(frame)`, then
`run(frame)` for energy or `forces(frame)` for energy + forces [eV/Å]. **No** public `*Config`
bag, **no** `make_*` / `build_*` factories. No standing
regression. Performance: **`perf_exempt`** (no same-basis
CP2K fair wall on the pin host).

Related: Γ-UKS energy [`gpw-uks-o2-gamma`](gpw-uks-o2-gamma.md); Γ-UKS forces
`gpw-uks-forces`; multi-k RKS forces
[gpw-kpoint-forces](gpw-kpoint-forces.md); multi-k UKS energy
[gpw-kpoints-uks](gpw-kpoints-uks.md). Narrative: [Ch. 9 GPW](../ch09-gpw.md).

## Method

**GPW (Quickstep family).** Contracted Gaussian AO basis; density collocation on
a uniform real-space grid; Hartree via FFT Poisson; XC on the grid; potential
back-integrated to the AO basis. Core electrons removed by **GTH**
pseudopotentials. Grid resolution set by the plane-wave cutoff [Ry]. Citation:
Lippert, Hutter, Parrinello, *Mol. Phys.* **92**, 477 (1997); forces/algorithm
context: VandeVondele et al., *Comput. Phys. Commun.* **167**, 103 (2005).

**Collinear UKS / PADE LSD.** With \(N_\alpha \neq N_\beta\) (or any
`n_alpha+n_beta > 0`), build spin densities \(\rho_\alpha\), \(\rho_\beta\);
Hartree from \(\rho=\rho_\alpha+\rho_\beta\); XC via spin-resolved PADE LSD;
diagonalize separate \(H_\alpha\) and \(H_\beta\). Occupations are explicit
integer members `n_alpha` / `n_beta` (not a multiplicity bag). When both are
zero, the closed-shell `n_occ` path is unchanged.

**Monkhorst–Pack k-mesh.** Members `k_mesh[3]` (or explicit `k_points` +
weights) select multi-k sampling. Time-reversal folds the mesh: this pin’s
**1×1×2** collapses to a single irreducible \(\kappa_z=\pm\tfrac14\) weight
class (`n_kpts` after TR). Bloch-phased complex matrices and complex \(\chi^k\)
live in the k-point GPW path (`internal/kpoints.h`).

**Analytic ionic forces (multi-k UKS).** At the SCF stationary point,

\[
\mathbf{F}_A = -\frac{\partial E}{\partial \mathbf{R}_A}
=
\mathbf{F}_A^{\mathrm{HF}}
+ \mathbf{F}_A^{\mathrm{Pulay}}
+ \mathbf{F}_A^{\mathrm{grid/core}},
\]

k-summed over occupied Bloch states. Hellmann–Feynman collects explicit
\(\mathbf{R}_A\)-dependence of \(T + V_\mathrm{pp} + V_\mathrm{grid}\) with
dual \(P_k^\sigma\); Pulay uses \(W_k = W_k^\alpha + W_k^\beta\) (occupancy-1 spin
channels) contracted with \(\partial S_k/\partial R_A\); the grid term holds
converged spin-resolved \(V_\mathrm{eff}^{\sigma}\) fixed. Implementation:
`GpwDft::forces(frame)` — one SCF, then the gradient pass at the converged
density.

**Valence bookkeeping (this pin).** H with GTH-PADE-q1 → 1 valence e⁻ per H →
2 valence e⁻ total. Multiplicity 3 ⇒ \(N_\alpha=2\), \(N_\beta=0\).

**Key exclusions (product scope for this slug).**

- No **multi-k UKS cell stress** (residual: `gpw-uks-stress` is Γ-UKS; multi-k
  UKS stress not productized).
- No **same-basis absolute CP2K force twin** in the ≤60 s regression (BASIS_MOLOPT
  lacks DZV-GTH-PADE; oracle input uses SZV-MOLOPT-GTH for documentation only).
- Full **6N finite-difference** force oracle is offline only (too slow for
  ctest; same policy as [gpw-kpoint-forces](gpw-kpoint-forces.md)).
- No hybrid / GAPW / CUDA-GPW; no molecular all-electron UKS (PySCF loop).
- Live CP2K never runs in ctest.

## Implementation

Public surface only (`GpwDft` members + lookup constructors +
`forces(frame)`; **no** `*Config`, **no** factories). Minimal
sequence:

```cpp
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/molrs.hpp"

using namespace ATV;
using namespace ATV::cpu::gpw;

// Compressed H–H = 1.0 Bohr in 12 Bohr cubic cell (PERIODIC XYZ);
// the Frame is in Å, so Bohr values are scaled by kBohrInAngstrom.
constexpr double b2a = units::kBohrInAngstrom;
const double xA[2] = {5.5, 6.5};           // Bohr
cpu::Frame frame(2);
for (int i = 0; i < 2; ++i) {
    frame.span(kAtom_type)[i] = 1;
    frame.span(kAtom_x)[i] = xA[i] * b2a;
    frame.span(kAtom_y)[i] = 6.0 * b2a;
    frame.span(kAtom_z)[i] = 6.0 * b2a;
}
frame.box = cpu::Box::cubic(12.0 * b2a);   // [Å]

GpwDft m{{Kind{molrs::Element::H,
               GaussianBasis(molrs::Element::H, "DZV-GTH-PADE"),
               GthPotential(molrs::Element::H, "GTH-PADE-q1")}}};
m.cutoff_ry = 100.0;
m.n_alpha = 2;                             // multiplicity 3, 2 valence e⁻
m.n_beta  = 0;
m.tol = 1e-8;
m.max_iter = 60;
m.nonperiodic = false;
m.local_pp_on_grid = true;                 // energy and gradient both read it
m.xc = XcKind::PADE;
m.eps_default = 1e-8;
m.k_mesh[0] = 1;
m.k_mesh[1] = 1;
m.k_mesh[2] = 2;                           // MONKHORST-PACK 1×1×2

m.build(frame);
const GpwResult& er = m.forces(frame);     // e_total [Ha], C_occ_k / C_occ_k_beta
// Forces on kAtom_fx/fy/fz [eV/Å]; ÷ units::kHartreePerBohrInEvPerAngstrom → Ha/Bohr
// fx[0] < 0 < fx[1]  (restoring bond); |F[0]+F[1]| ~ machine eps
```

Build (GPW needs FFTW3 + HF/PW) and run the regression:

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
  cmake --build build -j$(nproc) --target atomiverse
'
```

Relevant sources:

| Piece | Path |
| ----- | ---- |
| Method / SCF | `gpw_solver.h`, `src/cpu/gpw/gpw_solver.cpp` |
| Analytic forces | `gpw_forces.h` |
| k-point Bloch | `kpoints.h` |
| Frame-facing evaluator (optional) | `gpw_evaluator.h` |
| CP2K twin input | CP2K oracle |

## Validation (CP2K / product pin)

Source: the pin notes recorded when this page was written (no standing regression; offline pin;
**never** live CP2K in ctest). Primary product numbers are **frozen
Atomiverse first-green** (2026-07-25, Release, OMP=8). CP2K 2026.1 input is a
protocol / documentation twin (SZV-MOLOPT-GTH, not DZV-GTH-PADE).

| Quantity | Pin / gate | Tol | Notes |
| -------- | ---------- | --- | ----- |
| \(E_{\mathrm{tot}}\) [Ha] | **−0.528389371092** | **1e-5 Ha** | k-UKS 1×1×2, 100 Ry, ~6 SCF iters, TR-folded \(n_k\) |
| \(F_{0,x}\) [Ha/Bohr] | **−0.554544062995** | **1e-5** | Restoring; atom 0 at smaller \(x\) |
| \(F_{1,x}\) [Ha/Bohr] | **+0.554544062995** | **1e-5** | Newton-3 pair of \(F_{0,x}\) |
| \(F_{a,y}\), \(F_{a,z}\) | ~0 by symmetry | 1e-6 | Bond along \(x\) |
| \(\lvert\sum_a \mathbf{F}_a\rvert\) | ~\(1.6\times10^{-14}\) | **1e-6 Ha/Bohr** | Translation residual |
| Newton-3 on \(F_x\) | \(\lvert F_{0,x}+F_{1,x}\rvert\) | **1e-6** | Exact pair cancel |
| Closed-shell k-UKS vs k-RKS max\(\lvert\mathrm{d}F\rvert\) | \(7.4\times10^{-7}\) (measured) | **5e-4 Ha/Bohr** | \(N_\alpha=N_\beta=1\) vs `n_occ=1` |
| SCF | `er.converged` | — | `tol=1e-8`, `max_iter=60` |
| Bloch state | `C_occ_k`, `C_occ_k_beta`, `eps_k_beta` size = `n_kpts` | — | Dual-spin k blocks present |

**Geometry / protocol (must match pin).**

| Item | Value |
| ---- | ----- |
| System | High-spin H₂, compressed bond along \(x\) |
| Positions [Bohr] | H1 (5.5, 6.0, 6.0), H2 (6.5, 6.0, 6.0) |
| Cell | Cubic **12 Bohr**, `PERIODIC XYZ` |
| Basis | **DZV-GTH-PADE** (in-tree table) |
| PP | GTH-PADE-q1 |
| XC | PADE (LDA LSD) |
| Cutoff | 100 Ry, NGRIDS 1 (single density grid) |
| k mesh | MONKHORST-PACK **1×1×2** |
| Spin | UKS, **\(N_\alpha=2\), \(N_\beta=0\)** |
| `local_pp_on_grid` | **true** (energy and force both) |
| Electrostatics | Periodic (not MT) |

**CP2K ENERGY_FORCE twin (offline only; different basis).**

```bash
module load buildtool-easybuild/5.2.1-hpca3ef7d197 GCC/14.3.0 MPICH/4.3.2 CP2K/2026.1
export CP2K_DATA_DIR=$EBROOTCP2K/share/cp2k/data
cd /tmp && ln -sf $CP2K_DATA_DIR/{BASIS_MOLOPT,GTH_POTENTIALS} .
# copy <CP2K oracle input for this case>
mpirun -np 1 cp2k.psmp -i h2_uks_kpoint_forces.inp -o h2_uks_kpoint_forces.out
```

Input notes: `LSD`, `MULTIPLICITY 3`, `KPOINTS SCHEME MONKHORST-PACK 1 1 2`,
`CUTOFF 100`, `NGRIDS 1`, basis **SZV-MOLOPT-GTH** (exploratory — not the
product DZV pin). Do not compare absolute \(E\)/\(F\) to Atomiverse without a
same-basis re-pin.

**Offline analytic-vs-FD (related k-RKS path).** On the sibling closed-shell
[gpw-kpoint-forces](gpw-kpoint-forces.md) off-symmetry geometry, max\(\lvert F_{\mathrm{analytic}} -
F_{\mathrm{fd}}\rvert \approx 6\times 10^{-6}\) Ha/Bohr; full 6N FD is outside
the regression budget for this UKS slug as well.

## Performance

Source: pin notes / the case pin notes (wave 3).

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r = T_{\mathrm{ATV}}/T_{\mathrm{CP2K}}\) | ≤10× | Caveat |
| ---- | ------------------------: | ----------------------: | ---------------------------------------: | ---- | ------ |
| Multi-k UKS H₂ forces (DZV-GTH-PADE, 1×1×2) | — | **~1–2** (reg, OMP=8) | **n/a** | n/a | **`perf_exempt`**: no same-basis CP2K fair wall (DZV-GTH-PADE not in BASIS_MOLOPT; no CP2K module on pin host) |

Gate: \(r \le 10\) or `perf_exempt` + reason. This slug is **perf_exempt**.

## Limitations

- **First-green absolute pin is ATV self-consistency**, not a bit-identical
  CP2K DZV twin. Protocol twin exists; absolute energy/force re-pin needs a
  CP2K data path that exposes DZV-GTH-PADE (or an ATV pin rebased on SZV with
  documented grid twin).
- **Grid twin:** single density grid at 100 Ry vs CP2K multigrid options in
  other GPW pins; this slug uses `NGRIDS 1` to stay close to the product path.
- **Multi-k UKS stress** remains a residual (forces productized; stress not).
- XL-BOMD / CPMD fixed-density UKS is out of scope.
- **CUDA-GPW** blocked; plane-wave multi-k UKS is a separate PW surface
  ([pw-uks-kmesh-fe](pw-uks-kmesh-fe.md)).
- **ROKS / spin projection** not provided.

## References

- CP2K twin: CP2K oracle
- Oracle contract: CP2K oracle
- Narrative GPW: [`docs/tutorial/ch09-gpw.md`](../ch09-gpw.md)
- Roadmap: [`docs/roadmap/dft.md`](../../roadmap/dft.md)
- Deps (landed): [gpw-kpoints-uks](gpw-kpoints-uks.md), [gpw-kpoint-forces](gpw-kpoint-forces.md), `gpw-uks-forces`
- G. Lippert, J. Hutter, M. Parrinello, *Mol. Phys.* **92**, 477 (1997) — GPW
- J. VandeVondele et al., *Comput. Phys. Commun.* **167**, 103 (2005) — Quickstep forces
- S. Goedecker, M. Teter, J. Hutter, *Phys. Rev. B* **54**, 1703 (1996) — PADE / GTH
