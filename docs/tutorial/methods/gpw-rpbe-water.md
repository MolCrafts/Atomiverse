# GPW Γ-RKS RPBE — isolated H₂O / DZVP-MOLOPT-GTH (Martyna–Tuckerman)

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `gpw-rpbe-water` |
| Regression | `regressions/gpw/gpw-rpbe-water/` |
| Public API | `cpu::gpw::GPWEvaluator` + `cpu::single_point` with `xc = XcKind::RPBE`; `Kind` + `GaussianBasis` / `GthPotential` lookup ctors; shared `cpu::scf::XCType::RPBE` |
| Roadmap | handoff ratio **perf_exempt** (dual-gate \|E_RPBE−E_PBE\|); narrative twin [ch09-gpw](../ch09-gpw.md) |

## Summary

Γ-point closed-shell **Gaussian-and-plane-waves (GPW)** Kohn–Sham DFT total
energy for an **isolated** water molecule with the **RPBE** GGA functional
(Hammer, Hansen, Nørskov 1999): contracted MOLOPT Gaussians for the
wavefunction, uniform real-space grid + FFT for Hartree and XC, GTH
pseudopotentials, density cutoff **400 Ry**. Cluster electrostatics use the
**Martyna–Tuckerman (MT)** Poisson solver (`nonperiodic = true`; CP2K
`PERIODIC NONE` / `PSOLVER MT`). Geometry, basis, PP, cutoff, and MT BC match
the PBE foothold [`gpw-water-pbe-mt`](gpw-water-pbe-mt.md) and the BLYP sibling
`gpw-blyp-water` so the only intentional change is the XC functional.

The e2e product gate is a **dual-gate** against identical-geometry PBE
(\(|E_{\mathrm{RPBE}}-E_{\mathrm{PBE}}| > 10^{-4}\) Ha) plus a soft absolute band
on first-green Atomiverse \(E_{\mathrm{RPBE}}\) until an offline CP2K twin can
replace it. Public path: concrete `cpu::gpw::GPWEvaluator` constructor
arguments (no public `*Config` bag, no `make_*` / `build_*` factories) driven by
`cpu::single_point(frame, eval)`. Fair wall vs CP2K is **`perf_exempt`** (no
CP2K binary on the pin host).

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
   AO basis; diagonalize; rebuild \(P\) with direct \(P\)-mixing.

**RPBE** (Hammer, Hansen, Nørskov, *Phys. Rev. B* **59**, 7413 (1999)). Same
\(\kappa = 0.804\) and \(\mu \approx 0.21951\) as PBE exchange, but the
exchange enhancement uses an **exponential** form instead of PBE’s rational
one:

\[
F_x^{\mathrm{RPBE}}(s)
=
1 + \kappa\bigl(1 - e^{-\mu s^2/\kappa}\bigr),
\qquad
s = \frac{|\nabla\rho|}{2\,k_{\mathrm{F}}\,\rho}.
\]

**Correlation is pure PBE** (libxc `GGA_X_RPBE` + `GGA_C_PBE`; CP2K
`XC_FUNCTIONAL RPBE`). This is **not** **revPBE** (Zhang & Yang κ-revision of
the *rational* PBE exchange form — `XcKind::RevPBE` / `reg_gpw-revpbe-si`).

Atomiverse routes RPBE through the shared pointwise engine used by all pure
GGAs on the GPW grid:

\[
E_{\mathrm{xc}}[\rho]
=
\int e_{\mathrm{xc}}(\rho,\sigma)\,d\mathbf{r},
\qquad
\texttt{evaluate\_xc}(\texttt{XCType::RPBE},\,\rho,\,\sigma,\,\ldots).
\]

Closed-shell GGA assembly builds spectral \(\nabla\rho\) and the divergence of
the \(v_\sigma\) flux exactly as for `XcKind::PBE`, with `XcKind::RPBE`
selecting the RPBE exchange kernel (`xc_grid_rpbe` → `xc_grid_gga` with
`XCType::RPBE`).

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
| XC | **RPBE** GGA (`XcKind::RPBE`); PBE twin for dual-gate |
| System | Isolated H₂O, CP2K getting-started geometry |
| Cell | cubic \(L = 10\) Å |
| Positions [Å] | O (5.0000, 5.0000, 5.0000); H (5.7586, 5.0000, 5.5043); H (4.2414, 5.0000, 5.5043) |
| Basis | DZVP-MOLOPT-GTH (O, H) |
| PP | GTH-PBE-q6 (O), GTH-PBE-q1 (H) |
| Cutoff | **400 Ry** |
| SCF | direct \(P\)-mixing \(\alpha=0.3\), `max_iter=200`, `tol=1e-7` |
| BC | MT isolated (`nonperiodic=true`) |
| Valence | 8 e⁻ → \(n_{\mathrm{occ}} = 4\) |

**Key exclusions (non-goals of this slug)**

- Absolute total-energy digit match to a live CP2K process at pin time (no
  binary on host; do **not** invent CP2K digits from memory).
- **revPBE** (different functional; see `gpw-revpbe-si`).
- Analytic forces / stress / UKS / k-points / D3 / D4 / BOMD as *this*
  product’s gate (other queue slugs: `gpw-forces-water`, `gpw-stress-water`,
  `gpw-uks-o2-gamma`, `gpw-pbe-d3bj-ice`, `gpw-bomd-water`, …).
- Live CP2K (or any third-party) inside ctest.
- MPI multi-rank; OT minimizer (`use_ot`); public Config bags / factories.
- CUDA GPW (CPU-only on the CP2K loop).

## Implementation

Public surface only (`cpu::Frame` + `GPWEvaluator` ctor args; lookup
constructors for basis/PP tables — not factories for a parallel world model).
Matches `reg_gpw-rpbe-water.cpp`:

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
    /*mix*/ 0.3, /*xc*/ cpu::gpw::XcKind::RPBE, /*nonperiodic*/ true);
cpu::single_point(frame, eval);

const double e_ha = eval.last_energy_hartree();   // [Hartree]
const bool ok = eval.last_result().converged;
// frame.meta.get(kEnergy) is also set [eV]
```

Dual-gate twin (product regression runs both in one process): same kinds /
cutoff / SCF / MT with `XcKind::PBE` on a second frame; assert
\(|E_{\mathrm{RPBE}}-E_{\mathrm{PBE}}| > 10^{-4}\) Ha.

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
  cmake --build build --target reg_gpw-rpbe-water -j$(nproc)
  ./build/regressions/reg_gpw-rpbe-water
'
```

| Path | Role |
| ---- | ---- |
| `include/atomiverse/cpu/gpw/gpw_solver.h` | `XcKind::{…,RPBE,…}`, `GpwDft`, `GpwResult` |
| `include/atomiverse/cpu/gpw/gpw_evaluator.h` | Frame-facing `GPWEvaluator` (ctor `xc` arg) |
| `include/atomiverse/cpu/gpw/gaussian_basis.h` | `GaussianBasis(Element, name)` lookup |
| `include/atomiverse/cpu/gpw/gth_potential.h` | `GthPotential(Element, name)` lookup |
| `include/atomiverse/cpu/scf/xc_functional.h` | `XCType::RPBE`, `evaluate_xc`, `is_gga` |
| `include/atomiverse/cpu/gpw/internal/xc_grid_gga.h` | `xc_grid_rpbe` / GGA grid energy+potential |
| `src/cpu/gpw/internal/xc_grid_gga.cpp` | Closed-shell GGA grid assembly |
| `regressions/gpw/gpw-rpbe-water/reg_gpw-rpbe-water.cpp` | Product gate (dual-gate + soft absolute) |
| `tools/cp2k_oracle/inputs/gpw-rpbe-water.inp` | Offline CP2K twin (never ctest) |

## Validation (vs CP2K)

Source:
[`regressions/gpw/gpw-rpbe-water/REFERENCE.md`](../../../regressions/gpw/gpw-rpbe-water/REFERENCE.md).
Oracle status at pin time (2026-07-25): **no CP2K binary / module** on the worker
host (`cp2k`, `cp2k.psmp`, `cp2k.sopt` not on PATH; `module avail cp2k` empty).
Absolute CP2K \(E_{\mathrm{tot}}\) digits are therefore **not** invented from
memory and **not** taken from a live third-party process at ctest time. The
product gate uses:

1. **Hard dual-gate** — same geometry / basis / PP / cutoff / MT BC with
   `XcKind::PBE` must shift energy by more than \(10^{-4}\) Ha.
2. **Convergence / finiteness** — RPBE SCF converges; \(E_{\mathrm{RPBE}}\) and
   Frame `kEnergy` [eV] finite.
3. **Soft absolute band** — first-green Atomiverse \(E_{\mathrm{RPBE}}\) for
   setup integrity until offline CP2K replaces the pin (cite CP2K version).

Twin input (MT, 400 Ry, RPBE, same H₂O geometry as the PBE foothold):
`tools/cp2k_oracle/inputs/gpw-rpbe-water.inp`. Closed acceptance:
`.claude/specs/gpw-rpbe-water/` (git history on `wf/gpw-rpbe-water`). PBE twin
residual on this host matches the established
[`gpw-water-pbe-mt`](gpw-water-pbe-mt.md) pin
(\(E_{\mathrm{PBE}} \approx -17.212758\) Ha vs offline CP2K −17.212337 Ha).

**Setup:**

| Item | Value |
| ---- | ----- |
| System | Isolated H₂O, charge 0, singlet (closed-shell RKS) |
| Lattice | cubic \(L = 10\) Å |
| Positions [Å] | O (5.0000, 5.0000, 5.0000); H (5.7586, 5.0000, 5.5043); H (4.2414, 5.0000, 5.5043) |
| BC | non-periodic MT (`nonperiodic=true`) |
| Basis | DZVP-MOLOPT-GTH |
| Pseudopotential | GTH-PBE-q6 (O), GTH-PBE-q1 (H) |
| XC | RPBE (PBE twin for dual-gate) |
| Density cutoff | 400 Ry |
| Valence | 8 e⁻ → \(n_{\mathrm{occ}} = 4\) |
| SCF | mix 0.3, max_iter 200, tol \(10^{-7}\) |
| Provenance | RPBE 1999; tables from CP2K GTH / MOLOPT data family |

**Accuracy gate (product):**

| Quantity | Atomiverse (verified 2026-07-25) | Oracle / pin | Tol | Notes |
| -------- | -------------------------------: | -----------: | --- | ----- |
| \(E_{\mathrm{RPBE}}\) | **−17.2435550328 Ha** | first-green ATV (soft) | **\(1\times 10^{-3}\) Ha** | Until offline CP2K re-pin |
| \(E_{\mathrm{PBE}}\) (twin) | −17.2127580657 Ha | same host twin; vs CP2K −17.212337 Ha on PBE foothold | — | Dual-gate partner |
| \(\lvert E_{\mathrm{RPBE}}-E_{\mathrm{PBE}}\rvert\) | \(3.080\times 10^{-2}\) Ha | floor \(>10^{-4}\) Ha | **hard** | Functional activation |
| SCF (RPBE) | converged, 9 iters | — | `converged == true` | Product gate |
| Finite energy | finite Ha + Frame eV | — | `isfinite` | Product gate |

Offline re-pin (human only; never ctest):

```bash
# when a CP2K binary is available
cp2k.psmp -i tools/cp2k_oracle/inputs/gpw-rpbe-water.inp \
          -o /tmp/gpw-rpbe-water.out
# record TOTAL ENERGY [Ha] into REFERENCE.md; tighten absolute tol
```

## Performance

Source: [`REFERENCE.md`](../../../regressions/gpw/gpw-rpbe-water/REFERENCE.md)
(2026-07-25); queue/handoff **perf_exempt**.

| Item | Value |
| ---- | ----- |
| `perf_exempt` | **true** |
| Reason | No CP2K binary / module on the pin host for a fair wall twin. |
| \(T_{\mathrm{ATV}}\) (RPBE SCF, OMP=4, Release) | ~20.8 s |
| \(T_{\mathrm{ATV}}\) (PBE twin, OMP=4, Release) | ~5.9 s |
| \(r = T_{\mathrm{ATV}}/T_{\mathrm{CP2K}}\) | **n/a (exempt)** |
| `within_10x` | n/a (exempt) |

. Gate when a twin exists:
\(r = T_{\mathrm{ATV}}/T_{\mathrm{CP2K}} \le 10\) on the same host/core count
(single-rank; fair OMP for both).

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | -----------------------: | ----------------------: | ----: | ---- | ------ |
| H₂O RPBE MT (400 Ry) + PBE dual | — | ~20.8 (RPBE) / ~5.9 (PBE) | **exempt** | n/a | No CP2K binary at pin; remeasure when available |

## Limitations

- **Dual-gate + soft absolute pin**, not a published CP2K \(E_{\mathrm{tot}}\)
  digit match. Replace the absolute band with an offline twin when CP2K is
  available; dual-gate remains the hard functional-activation criterion.
- **RPBE ≠ revPBE.** The exponential exchange form is a different functional
  from Zhang–Yang revPBE (`XcKind::RevPBE`); do not cross-cite energies.
- **MT isolated** H₂O energy foothold, not periodic bulk water, forces, or
  AIMD. Forces/stress/BOMD/dispersion live in other slugs.
- **Energy-only product gate** for this regression — the shared GGA grid path
  accepts `XcKind::RPBE` for force assembly, but forces are not the e2e
  observable here (`gpw-forces-water` is the force foothold, PBE/MT water).
- **`perf_exempt`**: no fair \(r\) until offline CP2K is timed on the same
  host; do not invent a speed ratio.
- **CPU only** — CUDA GPW remains blocked on the CP2K loop.
- Basis/PP tables are static in-tree headers; unsupported element/name
  **throws** (no silent fallback).

## References

- Offline pin: `regressions/gpw/gpw-rpbe-water/REFERENCE.md`
- Regression consumer: `regressions/gpw/gpw-rpbe-water/reg_gpw-rpbe-water.cpp`
- Oracle twin: `tools/cp2k_oracle/inputs/gpw-rpbe-water.inp`
- Oracle notes: `tools/cp2k_oracle/README.md`
- Closed spec (git history): `.claude/specs/gpw-rpbe-water/` on
  `wf/gpw-rpbe-water`
- Narrative GPW chapter: [`docs/tutorial/ch09-gpw.md`](../ch09-gpw.md)
- Related method pages: [gpw-water-pbe-mt](gpw-water-pbe-mt.md) (PBE/MT energy
  foothold), [gpw-pbesol-si](gpw-pbesol-si.md) (PBEsol dual-gate pattern),
  [gpw-forces-water](gpw-forces-water.md) (analytic forces)
- Public API: `include/atomiverse/cpu/gpw/gpw_solver.h`,
  `include/atomiverse/cpu/gpw/gpw_evaluator.h`,
  `include/atomiverse/cpu/scf/xc_functional.h`
- B. Hammer, L. B. Hansen, J. K. Nørskov, *Phys. Rev. B* **59**, 7413 (1999)
  — RPBE.
- J. P. Perdew, K. Burke, M. Ernzerhof, *Phys. Rev. Lett.* **77**, 3865 (1996)
  — PBE form (correlation retained; exchange \(\kappa,\mu\) shared).
- Y. Zhang, W. Yang, *Phys. Rev. Lett.* **80**, 890 (1998) — revPBE (distinct).
- G. Lippert, J. Hutter, M. Parrinello, *Mol. Phys.* **92**, 477 (1997);
  *Theor. Chem. Acc.* **103**, 124 (1999) — GPW / Quickstep.
- S. Goedecker, M. Teter, J. Hutter, *Phys. Rev. B* **54**, 1703 (1996) — GTH.
- M. Krack, *Theor. Chem. Acc.* **114**, 145 (2005) — GTH tables.
- J. VandeVondele, J. Hutter, *J. Chem. Phys.* **127**, 114105 (2007) — MOLOPT.
- G. J. Martyna, M. E. Tuckerman, *J. Chem. Phys.* **110**, 2810 (1999) — MT
  Poisson.
