# Multi-k Dudarev DFT+U (GPW UKS) — H₂ toy / PADE / DZV-GTH-PADE

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Regression | None standing (end-to-end case removed) |
| Public API | `cpu::gpw::GpwDft` members `k_mesh` / `k_points` + `k_point_weights`, `k_time_reversal`, `n_alpha`/`n_beta`, `std::vector<PlusU> plus_u` |
| Roadmap | `docs/roadmap/dft.md` — multi-k Dudarev DFT+U SCF ✓ |
| Perf | **`perf_exempt`** (no fair absolute CP2K energy twin on the H₂ toy) |

## Summary

**Dudarev DFT+U on multi-k collinear UKS GPW** closes the product gap between
Γ-only PLUS_U (`gpw-dftu-nio`) and multi-k UKS without U ([gpw-kpoints-uks](gpw-kpoints-uks.md)).
Per-spin Mulliken occupations are **k-weighted Hermitian projections** of
\(P_k^\sigma S_k\); the same Dudarev energy \(E_U\) and potential
\(V^\sigma = U(\tfrac12 I - n^\sigma)\) are added to **every** complex Bloch
Hamiltonian \(H_k^\sigma\).

This page is the **capability foothold**, not a bulk correlated solid pin:
high-spin **H₂** (DZV-GTH-PADE) with PADE LSD, periodic 8 Bohr cell, and
Monkhorst–Pack \(2\times 2\times 2\) (time-reversal → 4 k). Decisive gates are
**internal identities** (Γ-limit complex vs real Γ-UKS+U; U-activation floor;
absolute ATV energy pin). An offline CP2K 2026.1 twin input ships under
CP2K oracle but reports **unphysical
Mulliken charges** on this toy (pathological \(E_U\)) — **not** an absolute
energy twin. Literature-facing PLUS_U numbers remain the Γ NiO molecule pin
(`gpw-dftu-nio`). No standing regression for multi-k SCF energy
(multi-k DFT+U F/σ: `gpw-dftu-kpoint-forces`, `gpw-dftu-kpoint-stress`). Fair wall:
**`perf_exempt`**.

Related: Γ DFT+U — [gpw-dftu-nio](gpw-dftu-nio.md); narrative GPW —
[Ch. 9](../ch09-gpw.md).

## Method

**GPW (Quickstep family).** Wavefunction in contracted Gaussians; density /
Hartree / XC on a uniform real-space grid; GTH pseudopotentials. Grid set by
plane-wave cutoff [Ry]. Citation: Lippert, Hutter, Parrinello, *Mol. Phys.*
**92**, 477 (1997).

**Multi-k UKS / PADE LSD.** Collinear unrestricted path with separate
\(P_k^\alpha\), \(P_k^\beta\) at each sampled \(\mathbf{k}\); spin-resolved
Padé LDA; integer members `n_alpha` / `n_beta`. Monkhorst–Pack mesh uses
CP2K convention
\(\kappa_d = (2r - N_d - 1)/(2N_d)\) (even \(N\) is Γ-avoiding). Optional
time-reversal folds \(\pm\mathbf{k}\) (collinear, no SOC). Explicit
`k_points` + `k_point_weights` (Σ weights = 1) override `k_mesh`. Citation:
Goedecker, Teter, Hutter, *Phys. Rev. B* **54**, 1703 (1996).

**Dudarev DFT+U on k-points (binding).** Per element and angular momentum
\(L\), effective \(U_{\mathrm{eff}}=U-J\) in **eV** on the public surface
(converted to Hartree with `units::kHartreeInEv` inside). Occupations use
symmetrized Mulliken projectors, **k-summed** (CP2K `PLUS_U_METHOD MULLIKEN`):

\[
n^{A\sigma}_{mm'}
  = \sum_k w_k\,\tfrac12\Big[
      (P_k^\sigma S_k)_{mm'} + (S_k P_k^\sigma)_{mm'}
    \Big]_{\text{Herm}}.
\]

Energy (sum over atoms and spin; \(U\) in Hartree):

\[
E_U = \sum_{A\sigma}\frac{U_{\mathrm{eff}}}{2}\Big[\mathrm{Tr}\,n^{A\sigma}
  - \mathrm{Tr}(n^{A\sigma} n^{A\sigma})\Big].
\]

Potential added to **every** spin Fock matrix on that AO subspace:

\[
V^{A\sigma}_{mm'} = U_{\mathrm{eff}}\Big(\tfrac12\delta_{mm'} - n^{A\sigma}_{mm'}\Big)
\quad\Rightarrow\quad
H_k^\sigma \leftarrow H_k^\sigma + V^{A\sigma}.
\]

Matches CP2K simplified rotationally invariant form (Dudarev et al.; CP2K
`U_MINUS_J`). Implementation: `apply_dft_plus_u_k` in
`cpu/gpw/internal/dft_plus_u.h` (not public). Empty `plus_u` leaves multi-k
UKS unchanged. On multi-function shells \(E_U\) can be negative (occupation
matrix eigenvalues outside \([0,1]\)); the regression only requires
\(|E_U|\) non-trivial and total energy pinned.

**Key exclusions (closed-spec scope-out).**

- Bulk AFM rocksalt NiO multi-k PLUS_U absolute twin (follow-on: larger
  basis + fair CP2K energy pin).
- Analytic multi-k DFT+U forces / stress (Γ DFT+U stress is separate:
  `gpw-dftu-stress`).
- Liechtenstein full \(U,J\); OT + multi-k + DFT+U; multigrid / nonperiodic
  k-path; UKS + PBE-GGA DFT+U.
- Fair wall ratio vs CP2K on this H₂ toy (`perf_exempt`).

## Implementation

Public surface: concrete `cpu::gpw::GpwDft` members only (**no** `*Config`
bag, **no** `make_*` / `build_*`). The regression uses the method API
directly (`build(frame)` + `run(frame)`). `plus_u` and the multi-k members
(`k_mesh`, `k_points`, …) all live on `GpwDft`; `GPWEvaluator` binds the
configured method into a driver slot.

```cpp
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/molrs.hpp"

using namespace ATV;
using cpu::gpw::GpwDft;
using cpu::gpw::GpwResult;
using cpu::gpw::PlusU;
using cpu::gpw::XcKind;

// High-spin H2 toy — 8 Bohr cubic cell, DZV-GTH-PADE, periodic
const double L = 8.0 * units::kBohrInAngstrom;   // 8 Bohr [Å]
const double d = 0.7 * units::kBohrInAngstrom;   // ±0.7 Bohr [Å]
cpu::Frame frame(2);
for (int i = 0; i < 2; ++i) {
    frame.span(kAtom_type)[i] = 1;
    frame.span(kAtom_x)[i] = 0.5 * L + (i == 0 ? -d : d);
    frame.span(kAtom_y)[i] = 0.5 * L;
    frame.span(kAtom_z)[i] = 0.5 * L;
}
frame.box = cpu::Box::cubic(L);

GpwDft m{{cpu::gpw::Kind{
    molrs::Element::H,
    cpu::gpw::GaussianBasis(molrs::Element::H, "DZV-GTH-PADE"),
    cpu::gpw::GthPotential(molrs::Element::H, "GTH-PADE-q1")}}};
m.cutoff_ry = 120.0;
m.n_alpha = 2;   // high-spin: both e− α
m.n_beta = 0;
m.max_iter = 120;
m.tol = 1e-7;
m.mix = 0.35;
m.xc = XcKind::PADE;
m.k_mesh[0] = m.k_mesh[1] = m.k_mesh[2] = 2;  // MP 2×2×2
m.k_time_reversal = true;                      // → n_kpts == 4
m.plus_u.push_back(PlusU{molrs::Element::H, /*l=*/0, /*u_minus_j_ev=*/4.0});
m.build(frame);
const GpwResult r = m.run(frame);   // also frame.meta[kEnergy] [eV]

// r.e_total [Ha], r.e_dftu [Ha], r.n_kpts, r.converged
// this regression pins Hartree totals on GpwResult.
```

Γ-limit identity (A1) uses explicit `k_points = {{0,0,0}}` + weight 1 against
real Γ-UKS + same `plus_u` (cutoff 100 Ry). Activation (A2) compares
`E(U=4 eV)` vs `E(U=0)` on the same MP mesh.

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
  cmake --build build -j$(nproc) --target atomiverse
'
```

| Piece | Path |
| ----- | ---- |
| Public `GpwDft::{k_mesh,plus_u,…}` / `GpwResult::{e_dftu,n_kpts}` | `gpw_solver.h` |
| Dudarev multi-k E/V (internal) | `dft_plus_u.h` (`apply_dft_plus_u_k`) |
| SCF wire-up (`run_gpw_scf_kpoints` / `_uks`) | `src/cpu/gpw/gpw_solver.cpp` |
| Unit tests (not e2e) | `tests/cpu/gpw/test_cpu_gpw_dftu.cpp` |
| CP2K twin input | CP2K oracle |

## Validation (vs CP2K + internal identities)

Source: the pin notes recorded when this page was written (no standing regression).
Oracle: offline **CP2K 2026.1** twin **ships** under CP2K oracle but is
**pathology-noted** (not absolute energy gate). **Never** live in ctest. Pin
date: **2026-07-26**.

**Setup (ATV product pin — must match regression):**

| Item | Value |
| ---- | ----- |
| System | H₂ high-spin toy, charge 0, UKS \(N_\alpha=2\), \(N_\beta=0\) |
| Geometry | H at \((0.5L\pm 0.7,\,0.5L,\,0.5L)\) Bohr, \(L=8\) |
| Cell | **8 × 8 × 8 Bohr**, `PERIODIC XYZ` |
| Basis / PP | H `DZV-GTH-PADE` / `GTH-PADE-q1` |
| XC | PADE (LDA LSD) |
| DFT+U | H \(L=0\), \(U-J=4.0\) eV, Mulliken |
| Cutoff | **100 Ry** (A1) / **120 Ry** (A2–A3) |
| k mesh (A2–A3) | MONKHORST-PACK **2×2×2**, `k_time_reversal` → **4** k |
| SCF | DIIS-style mix 0.35, `tol=1e-7`, `max_iter` 100–120 |

**Identity / absolute gates (decisive):**

| Gate | Quantity | Pin / result (2026-07-26) | Tol | Notes |
| ---- | -------- | ------------------------: | --- | ----- |
| A1 | \|E(k=Γ complex+U) − E(real Γ-UKS+U)\| | ≈ \(4\times 10^{-7}\) Ha | **1×10⁻⁶ Ha** | same `plus_u`, cutoff 100 Ry |
| A1 | \|e_dftu(kΓ) − e_dftu(Γ)\| | ≲ \(1\times 10^{-4}\) Ha | **1×10⁻⁴ Ha** | decomposition soft vs total |
| A1 | \|e_dftu\| non-trivial | — | \(>10^{-8}\) Ha | PLUS_U active |
| A2 | \|E(U=4 eV) − E(U=0)\| on MP 2×2×2 | ≈ **9.5×10⁻²** Ha | **≥ 1×10⁻⁴ Ha** | activation floor |
| A3 | \(E_{\mathrm{tot}}\) (MP+TR+U, 120 Ry) | **−0.814170782662** Ha | **1×10⁻³ Ha** | frozen ATV absolute |
| A3 | \(n_{\mathrm{kpts}}\) after TR fold | **4** | exact | MP 2×2×2 |
| A3 | \(E(U=0)\) diagnostic | −0.719059673672 Ha | — | not equality-asserted |
| A3 | e_dftu diagnostic | ≈ −0.177 Ha | \|e\| \(>10^{-8}\) | multi-function s shell |

**Offline CP2K twin (pathology — not absolute energy gate):**

| Item | Value |
| ---- | ----- |
| Input | CP2K oracle |
| Twin settings | same H₂ / DZV / PADE / U=4 eV / MP 2×2×2 / cutoff 120 Ry |
| SCF | converged in **42** steps |
| Reported \(E_U\) | **−158.76078199959053** Ha (unphysical Mulliken) |
| Reported \(E_{\mathrm{tot}}\) | **−157.66686993943307** Ha |
| Wall | ≈ **9.4 s** (OMP=8) |
| Product use | twin retained for method surface; **do not** equality-assert ATV vs these numbers |

Key CP2K log lines (from REFERENCE):

```text
  *** SCF run converged in    42 steps ***
  DFT+U energy:                                              -158.76078199959053
  Total energy:                                              -157.66686993943307
 *** WARNING … DFT+U energy contribution is negative ***
 *** possibly due to unphysical Mulliken charges! ***
```

**Product twin path for literature-facing PLUS_U:** reuse Γ-UKS PLUS_U NiO
molecule (`reg_gpw-dftu-nio`, CP2K \(E = -184.92466\) Ha). Multi-k capability
is proven by A1–A3. A fair multi-k NiO solid twin remains follow-on.

Offline re-run (operator only; not ctest):

```bash
export LD_LIBRARY_PATH=$CP2K_ROOT/lib:$LD_LIBRARY_PATH
export CP2K_DATA_DIR=$CP2K_ROOT/share/cp2k/data
cd /tmp && ln -sf $CP2K_DATA_DIR/{BASIS_SET,GTH_POTENTIALS} .
# copy <CP2K oracle input for this case>
OMP_NUM_THREADS=8 cp2k.psmp -i h2_uks_dftu_kpoints.inp -o h2_uks_dftu_kpoints.out
```

## Performance

Source: fair wall table in
the pin notes recorded when this page was written (no standing regression; 2026-07-26). Handoff:
the case pin notes — **`perf_exempt`**. Same pattern as
[gpw-kpoints-uks](gpw-kpoints-uks.md): decisive gates are internal identities, not
\(T_{\mathrm{ATV}}/T_{\mathrm{CP2K}}\) against a fair absolute energy twin.

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r = T_{\mathrm{ATV}}/T_{\mathrm{CP2K}}\) | ≤10× | Caveat |
| ---- | ------------------------: | ----------------------: | ---------------------------------------: | ---- | ------ |
| H₂ UKS+U MP 2×2×2 | ≈ 9.4 | ≈ **49** (A1–A3 suite) | **n/a** | **perf_exempt** | CP2K energy/Mulliken pathological; ATV wall is full identity suite, not single SCF |

Product pass via exemption (closed-spec perf gate). Ratio does not forecast
bulk NiO multi-k PLUS_U wall or multigrid parity.

## Limitations

- **Toy H₂ foothold, not bulk correlated oxide:** multi-k PLUS_U **SCF energy
  path** is productized; chemistry claims need solid AFM NiO (or equivalent)
  with fair CP2K absolute twin (follow-on).
- **CP2K H₂ twin is pathology-only:** unphysical Mulliken ⇒ cannot tighten
  absolute \(\lvert\Delta E\rvert\) vs CP2K on this system; use A1–A3.
- **No multi-k DFT+U forces / stress** on this regression; stress PLUS_U is
  Γ-scoped (`gpw-dftu-stress`).
- **`GPWEvaluator`** is not the regression entry point — `GpwDft` carries
  `k_mesh` / `k_points` / `plus_u` and is run on the Frame directly.
- **OT + multi-k + DFT+U** unsupported; multigrid / nonperiodic k-path throw.
- **UKS + GGA DFT+U** out of scope (PADE LSD only here).
- **Liechtenstein full \(U,J\)** and PAW/GAPW projectors out of scope.
- **e_dftu sign:** multi-function shells can yield \(E_U < 0\); gates assert
  finiteness / non-triviality / total energy, not \(E_U > 0\).

## References

- CP2K twin: CP2K oracle
- Oracle contract: CP2K oracle
- Γ DFT+U product twin: [gpw-dftu-nio](gpw-dftu-nio.md),
  case `gpw-dftu-nio` pin
- Narrative GPW: [`docs/tutorial/ch09-gpw.md`](../ch09-gpw.md)
- Roadmap: [`docs/roadmap/dft.md`](../../roadmap/dft.md)
- V. I. Anisimov, J. Zaanen, O. K. Andersen, *Phys. Rev. B* **44**, 943 (1991) — DFT+U
- S. L. Dudarev et al., *Phys. Rev. B* **57**, 1505 (1998) — simplified rotationally invariant \(U\)
- G. Lippert, J. Hutter, M. Parrinello, *Mol. Phys.* **92**, 477 (1997) — GPW
- S. Goedecker, M. Teter, J. Hutter, *Phys. Rev. B* **54**, 1703 (1996) — PADE / GTH
