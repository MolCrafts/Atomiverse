# GPW Γ-RKS revPBE — Si₂ diamond fragment / DZVP-MOLOPT-GTH (Martyna–Tuckerman)

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `gpw-revpbe-si` |
| Regression | `regressions/gpw/gpw-revpbe-si/` |
| Public API | `cpu::gpw::GpwDft` / `GPWEvaluator` with `xc = XcKind::RevPBE`; `GaussianBasis` / `GthPotential` lookup ctors; shared `cpu::scf::XCType::RevPBE` |
| Roadmap | handoff ratio **perf_exempt** |

## Summary

Closed-shell **Gaussian-and-plane-waves (GPW)** Kohn–Sham DFT with the
**revPBE** GGA functional on a two-atom silicon diamond fragment (Γ only,
DZVP-MOLOPT-GTH + GTH-PBE-q4, density cutoff **200 Ry**). revPBE is Zhang &
Yang’s 1998 revision of PBE exchange (same \(\mu\) and correlation \(\beta\),
revised \(\kappa = 1.245\)); it reuses the shared GPW GGA grid path already
used for PBE/PBEsol/BLYP (`evaluate_xc` + spectral \(\nabla\rho\)) with that
exchange parameter. The e2e product gate is a **dual-gate** against
identical-geometry PBE (\(|E_{\mathrm{revPBE}}-E_{\mathrm{PBE}}| > 10^{-4}\) Ha)
plus a soft absolute band on first-green Atomiverse \(E_{\mathrm{revPBE}}\) until
an offline CP2K twin can replace it. Public surface: concrete
`cpu::gpw::GpwDft` members (`xc`, `cutoff_ry`, `nonperiodic`, …); lookup
constructors for basis/PP tables — no public `*Config`, no factories. Fair wall
vs CP2K is **`perf_exempt`** (no CP2K binary on the pin host).

## Method

**GPW dual representation** (Quickstep; Lippert, Hutter, Parrinello 1997/1999).
Density matrix \(P\) lives in contracted Gaussians; each SCF cycle collocates
\(\rho(\mathbf{r})\) on a uniform grid set by \(E_{\mathrm{cut}}\), solves
Hartree + XC on the grid (\(\mathcal{O}(N\log N)\)), and integrates back to the
AO basis with kinetic + GTH local/nonlocal PP before diagonalization and
\(P\)-mixing.

**revPBE** (Zhang & Yang, *Phys. Rev. Lett.* **80**, 890 (1998)). Same analytic
form as PBE with revised exchange enhancement:

| Piece | PBE | revPBE |
| ----- | --: | -----: |
| Exchange \(\mu\) | \(0.21951\) | \(0.21951\) |
| Exchange \(\kappa\) | \(0.804\) | **\(1.245\)** |
| Correlation \(\beta\) | \(0.066725\) | \(0.066725\) |

PW92 LDA correlation base is unchanged. At \(\sigma = |\nabla\rho|^2 = 0\),
revPBE and PBE coincide (LDA limit); at finite \(\sigma\) the larger \(\kappa\)
softens the exchange enhancement and shifts \(E_{\mathrm{xc}}\). libxc / CP2K
names: `GGA_X_PBE_R` + `GGA_C_PBE` / `XC_FUNCTIONAL revPBE`. Atomiverse routes
both through the shared pointwise engine:

\[
E_{\mathrm{xc}}[\rho]
=
\int e_{\mathrm{xc}}(\rho,\sigma)\,d\mathbf{r},
\qquad
\texttt{evaluate\_xc}(\texttt{XCType::RevPBE},\,\rho,\,\sigma,\,\ldots).
\]

On the GPW grid, closed-shell GGA assembly builds spectral \(\nabla\rho\) and
the divergence of the \(v_\sigma\) flux exactly as for `XcKind::PBE`, with
`XcKind::RevPBE` selecting \(\kappa = 1.245\) (`xc_grid_revpbe` →
`xc_grid_gga(..., XCType::RevPBE)`).

**Core / valence.** Norm-conserving **GTH-PBE-q4** on Si (zion 4). Two Si atoms
→ 8 valence electrons → \(n_{\mathrm{occ}} = 4\) closed-shell pairs. Basis:
**DZVP-MOLOPT-GTH** (Si table row transcribed from CP2K `BASIS_MOLOPT` /
`GTH_POTENTIALS` family).

**Boundary conditions.** The product regression uses **isolated** Martyna–Tuckerman
electrostatics (`nonperiodic = true`; CP2K `PERIODIC NONE` / `PSOLVER MT`).
Periodic Γ-only MOLOPT image sums on this host exceed the ≤ 60 s e2e budget;
the dual-gate still proves revPBE activation on the GPW GGA path. The offline
oracle twin input matches the MT / 200 Ry deck for re-pin when a CP2K binary is
available.

**What this pin includes**

| Item | Setting |
| ---- | ------- |
| Spin | Closed-shell RKS (Γ only) |
| XC | **revPBE** GGA (`XcKind::RevPBE`); PBE twin for dual-gate |
| System | Si₂ diamond fragment, \(a = 5.43\) Å cubic cell |
| Positions [Å] | \((0,0,0)\), \((a/4,a/4,a/4) = (1.3575,1.3575,1.3575)\) |
| Basis | DZVP-MOLOPT-GTH (Si) |
| PP | GTH-PBE-q4 (zion 4) |
| Cutoff | **200 Ry** |
| SCF | direct \(P\)-mixing \(\alpha=0.4\), `max_iter=60`, `tol=1e-6` |
| BC | MT isolated (`nonperiodic=true`) |
| Valence | 8 e⁻ → \(n_{\mathrm{occ}} = 4\) |

**Key exclusions (non-goals of this slug)**

- Absolute total-energy digit match to a live CP2K process at pin time (no
  binary on host; do **not** invent CP2K digits from memory).
- UKS + GGA revPBE; k-point GPW (`gpw-kpoints-si`); full bulk conventional
  8-atom Si cell under the 60 s e2e budget.
- Analytic forces / stress / BOMD / D3 / D4 as *this* product’s gate (other
  queue slugs).
- RPBE (Hammer–Hansen–Nørskov) — separate residual `gpw-rpbe-water`.
- Live CP2K (or any third-party) inside ctest.
- Public `*Config` bags / `make_*` / `build_*` factories.

## Implementation

Public surface only (`GpwDft` members + lookup constructors; matches
`reg_gpw-revpbe-si.cpp` — direct `GpwDft` path to stay within the e2e budget):

```cpp
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/molrs.hpp"

using namespace ATV::cpu::gpw;

constexpr double kA2B = 1.8897259885789;
constexpr double a_ang = 5.43;
const double a = a_ang * kA2B;

const GaussianBasis bas(molrs::Element::Si, "DZVP-MOLOPT-GTH");
const GthPotential pot(molrs::Element::Si, "GTH-PBE-q4");
const double pos0[3] = {0.0, 0.0, 0.0};
const double pos1[3] = {a / 4.0, a / 4.0, a / 4.0};

std::vector<GpwAtom> atoms;
atoms.emplace_back(pos0, bas, pot, 14);
atoms.emplace_back(pos1, bas, pot, 14);

GpwDft m;
m.cell[0] = m.cell[1] = m.cell[2] = a;  // Bohr
m.cutoff_ry = 200.0;
m.n_occ = 4;                             // 2 Si × zion 4 → 8 e⁻
m.max_iter = 60;
m.tol = 1e-6;
m.mix = 0.4;
m.xc = XcKind::RevPBE;                   // or XcKind::PBE for dual-gate twin
m.nonperiodic = true;                    // MT isolated Poisson
m.build(atoms);
const GpwResult r = m.run(atoms);        // r.e_total [Ha], r.converged, r.iters
```

Frame-compatible evaluator path (same `xc` / kinds; not the timed e2e path):

```cpp
#include "atomiverse/cpu/evaluate/single_point.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/gpw/gpw_evaluator.h"

// Kind{Element::Si, GaussianBasis(...), GthPotential(...)} per atom type;
// GPWEvaluator(..., /*xc*/ XcKind::RevPBE, /*nonperiodic*/ true);
// cpu::single_point(frame, eval);  // last_energy_hartree() [Ha]
// Public Frame energy remains eV on frame.meta.get(kEnergy).
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
  cmake --build build --target reg_gpw-revpbe-si -j$(nproc)
  ./build/regressions/reg_gpw-revpbe-si
'
```

| Path | Role |
| ---- | ---- |
| `include/atomiverse/cpu/gpw/gpw_solver.h` | `XcKind::{…,RevPBE,…}`, `GpwDft`, `GpwResult` |
| `include/atomiverse/cpu/gpw/gpw_evaluator.h` | Frame-facing `GPWEvaluator` (ctor `xc` arg) |
| `include/atomiverse/cpu/gpw/gaussian_basis.h` | `GaussianBasis(Element, name)` lookup |
| `include/atomiverse/cpu/gpw/gth_potential.h` | `GthPotential(Element, name)` lookup |
| `include/atomiverse/cpu/scf/xc_functional.h` | `XCType::RevPBE`, `evaluate_xc`, `is_gga` |
| `include/atomiverse/cpu/gpw/basis_data/gpw_tables.h` | Si GTH-PBE-q4 + DZVP-MOLOPT-GTH static tables |
| `src/cpu/gpw/internal/xc_grid_gga.cpp` | Closed-shell GGA grid; `xc_grid_revpbe` |
| `src/cpu/scf/xc_functional.cpp` | Pointwise revPBE \(\kappa=1.245\) |
| `regressions/gpw/gpw-revpbe-si/reg_gpw-revpbe-si.cpp` | Product gate (dual-gate + soft absolute) |
| `tools/cp2k_oracle/inputs/gpw-revpbe-si.inp` | Offline CP2K twin (never ctest) |

## Validation (vs CP2K)

Source:
[`regressions/gpw/gpw-revpbe-si/REFERENCE.md`](../../../regressions/gpw/gpw-revpbe-si/REFERENCE.md).
Oracle status at pin time (2026-07-25): **no CP2K binary / module** on the worker
host (`cp2k`, `cp2k.psmp`, `cp2k.sopt` not on PATH; `module avail cp2k` empty).
Absolute CP2K \(E_{\mathrm{tot}}\) digits are therefore **not** invented from
memory and **not** taken from a live third-party process at ctest time. The
product gate uses:

1. **Hard dual-gate** — same geometry / basis / PP / cutoff / MT BC with
   `XcKind::PBE` must shift energy by more than \(10^{-4}\) Ha.
2. **Convergence / finiteness** — revPBE SCF converges; \(E\) finite.
3. **Soft absolute band** — first-green Atomiverse \(E_{\mathrm{revPBE}}\) for
   setup integrity until offline CP2K replaces the pin (cite CP2K version).

Twin input (MT, 200 Ry, revPBE, Si₂ geometry — aligned with the regression):
`tools/cp2k_oracle/inputs/gpw-revpbe-si.inp`. Closed acceptance:
`.claude/specs/gpw-revpbe-si/` (git history on `wf/gpw-revpbe-si` / merge
`e086865c`).

**Setup:**

| Item | Value |
| ---- | ----- |
| System | Si₂ diamond fragment, charge 0, singlet (closed-shell RKS) |
| Lattice | cubic \(a = 5.43\) Å |
| Positions [Å] | Si (0, 0, 0); Si (1.3575, 1.3575, 1.3575) |
| BC | non-periodic MT (`nonperiodic=true`) |
| Basis | DZVP-MOLOPT-GTH |
| Pseudopotential | GTH-PBE-q4 (zion 4) |
| XC | revPBE (PBE twin for dual-gate) |
| Density cutoff | 200 Ry |
| Valence | 8 e⁻ → \(n_{\mathrm{occ}} = 4\) |
| SCF | mix 0.4, max_iter 60, tol \(10^{-6}\) |
| Provenance | Zhang & Yang 1998; tables from CP2K GTH / MOLOPT data family |

**Accuracy gate (product):**

| Quantity | Atomiverse (verified 2026-07-25) | Oracle / pin | Tol | Notes |
| -------- | -------------------------------: | -----------: | --- | ----- |
| \(E_{\mathrm{revPBE}}\) | **−13.0900394799 Ha** | first-green ATV (soft) | **\(1\times 10^{-3}\) Ha** | Until offline CP2K re-pin |
| \(E_{\mathrm{PBE}}\) (twin) | −13.0743742822 Ha | same host twin | — | Dual-gate partner; matches `gpw-pbesol-si` PBE residual |
| \(\lvert E_{\mathrm{revPBE}}-E_{\mathrm{PBE}}\rvert\) | \(1.567\times 10^{-2}\) Ha | floor \(>10^{-4}\) Ha | **hard** | Functional activation |
| SCF (revPBE) | converged, 10 iters | — | `converged == true` | Product gate |
| Finite energy | finite | — | `isfinite` | Product gate |

Unit-level gates (not e2e, under `tests/`): revPBE@\(\sigma=0\) ≡ PBE@\(\sigma=0\)
(LDA limit); at finite \(\sigma\), \(|e_{\mathrm{xc}}|\) differs; Si GTH/MOLOPT
table digits pinned; GGA grid LDA-limit for revPBE.

Offline re-pin (human only; never ctest):

```bash
# when a CP2K binary is available
cp2k.psmp -i tools/cp2k_oracle/inputs/gpw-revpbe-si.inp \
          -o /tmp/gpw-revpbe-si.out
# record TOTAL ENERGY [Ha] into REFERENCE.md; tighten absolute tol
```

## Performance

Source: [`REFERENCE.md`](../../../regressions/gpw/gpw-revpbe-si/REFERENCE.md)
(2026-07-25); queue/handoff **perf_exempt**.

| Item | Value |
| ---- | ----- |
| `perf_exempt` | **true** |
| Reason | No CP2K binary / module on the pin host for a fair wall twin. |
| \(T_{\mathrm{ATV}}\) (both SCFs, OMP=4, Release) | ~24 s total (~14.8 s revPBE + ~9.2 s PBE) |
| \(r = T_{\mathrm{ATV}}/T_{\mathrm{CP2K}}\) | **n/a (exempt)** |
| `within_10x` | n/a (exempt) |

. Gate
when a twin exists: \(r = T_{\mathrm{ATV}}/T_{\mathrm{CP2K}} \le 10\) on the same
host/core count (single-rank).

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | -----------------------: | ----------------------: | ----: | ---- | ------ |
| Si₂ revPBE MT (200 Ry) + PBE dual | — | ~24 (both) | **exempt** | n/a | No CP2K binary at pin; remeasure when available |

## Limitations

- **Dual-gate + soft absolute pin**, not a published CP2K \(E_{\mathrm{tot}}\)
  digit match. Replace the absolute band with an offline twin when CP2K is
  available; dual-gate remains the hard functional-activation criterion.
- **MT isolated** Si₂ foothold, not periodic bulk Si with k-points. Periodic
  MOLOPT Γ sums were budget-hostile on the pin host; k-point GPW is
  `gpw-kpoints-si` (separate product).
- **RKS only** for revPBE on GPW; UKS + GGA revPBE is out of this slug.
- **Energy-only product gate** for this regression — analytic force/stress
  paths accept `XcKind::RevPBE` via the shared GGA grid, but forces/stress are
  not the e2e observable here.
- **Not RPBE.** Hammer–Hansen–Nørskov RPBE remains residual `gpw-rpbe-water`.
- **`perf_exempt`**: no fair \(r\) until offline CP2K is timed on the same
  host; do not invent a speed ratio.
- **CPU only** — CUDA GPW remains blocked on the CP2K loop.
- Basis/PP tables are static in-tree headers; unsupported element/name
  **throws** (no silent fallback).

## References

- Offline pin: `regressions/gpw/gpw-revpbe-si/REFERENCE.md`
- Regression consumer: `regressions/gpw/gpw-revpbe-si/reg_gpw-revpbe-si.cpp`
- Oracle twin: `tools/cp2k_oracle/inputs/gpw-revpbe-si.inp`
- Oracle notes: `tools/cp2k_oracle/README.md`
- Closed spec (git history): `.claude/specs/gpw-revpbe-si/` on
  `wf/gpw-revpbe-si` / merge `e086865c`
- Narrative GPW chapter: [`docs/tutorial/ch09-gpw.md`](../ch09-gpw.md)
- Related method pages: [gpw-pbesol-si](gpw-pbesol-si.md) (PBEsol GGA twin shape),
  [gpw-water-pbe-mt](gpw-water-pbe-mt.md) (PBE/MT energy),
  [pw-pbe-gga-si](pw-pbe-gga-si.md) (PW true PBE on Si)
- Public API: `include/atomiverse/cpu/gpw/gpw_solver.h`,
  `include/atomiverse/cpu/scf/xc_functional.h`
- Y. Zhang, W. Yang, *Phys. Rev. Lett.* **80**, 890 (1998) — revPBE.
- J. P. Perdew, K. Burke, M. Ernzerhof, *Phys. Rev. Lett.* **77**, 3865 (1996)
  — PBE form.
- G. Lippert, J. Hutter, M. Parrinello, *Mol. Phys.* **92**, 477 (1997);
  *Theor. Chem. Acc.* **103**, 124 (1999) — GPW / Quickstep.
- S. Goedecker, M. Teter, J. Hutter, *Phys. Rev. B* **54**, 1703 (1996) — GTH.
- M. Krack, *Theor. Chem. Acc.* **114**, 145 (2005) — GTH tables.
- J. VandeVondele, J. Hutter, *J. Chem. Phys.* **127**, 114105 (2007) — MOLOPT.
- G. J. Martyna, M. E. Tuckerman, *J. Chem. Phys.* **110**, 2810 (1999) — MT
  Poisson.
