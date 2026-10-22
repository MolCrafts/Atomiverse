# Multi-k UKS OT + Dudarev DFT+U — high-spin H₂ / PADE / DZV-GTH-PADE

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Regression | None standing (end-to-end case removed) |
| Public API | `cpu::gpw::GpwDft` members `use_ot`, `n_alpha`/`n_beta`, `k_mesh` / `k_points`, `plus_u` |
| Roadmap | `docs/roadmap/dft.md` § OT — multi-k UKS OT+U ✓ |

## Summary

**Multi-k collinear UKS GPW** with the **Orbital Transformation (OT)** SCF
minimizer and **Dudarev DFT+U** (`PLUS_U`, Mulliken occupation) is the product
intersection of three previously separate footholds: multi-k UKS OT
([ot-uks-kpoints](ot-uks-kpoints.md)), multi-k RKS OT+U ([ot-dftu-kpoints](ot-dftu-kpoints.md)), and Γ-UKS OT+U
(`ot-dftu`). This page pins that combo on a **high-spin H₂** microcase
(DZV-GTH-PADE, PADE LSD, \(U-J=4\) eV on H \(L=0\)) against offline **CP2K
2026.1** logistics and in-tree OT≡DIIS fixed-point identities.

Product path: construct `cpu::gpw::GpwDft`, set UKS occupations + `use_ot` +
`k_mesh`/`k_points` + `plus_u`, `build(frame)` / `run(frame)`. Method energy
channel is **Hartree** on `GpwResult::e_total` / `e_dftu`; `run(frame)` also
writes the public Frame energy in eV (`kEnergy`). No standing
regression. Fair wall: **`perf_exempt`** — CP2K 2026.1 hard-aborts OT
with k-points, so no OT wall twin exists; decisive gates are internal
OT≡DIIS multi-k UKS+U identities (same pattern as [ot-uks-kpoints](ot-uks-kpoints.md) /
[ot-dftu-kpoints](ot-dftu-kpoints.md)).

Related: closed-shell Γ OT [ot-minimizer-water](ot-minimizer-water.md);
Γ-UKS DFT+U DIIS [gpw-dftu-nio](gpw-dftu-nio.md); Γ UKS
[gpw-uks-o2-gamma](gpw-uks-o2-gamma.md); narrative GPW [Ch. 9](../ch09-gpw.md).

## Method

**GPW (Quickstep family).** Wavefunction in contracted Gaussians; density /
Hartree / XC on a uniform real-space grid; GTH pseudopotentials. Grid set by
plane-wave cutoff [Ry]. Citation: Lippert, Hutter, Parrinello, *Mol. Phys.*
**92**, 477 (1997).

**Collinear UKS.** Separate \(P^\alpha\), \(P^\beta\); spin-resolved Padé LDA;
integer members `n_alpha` / `n_beta` (not a multiplicity bag). This pin:
\(N_\alpha=2\), \(N_\beta=0\) (high-spin H₂, two valence e⁻ on GTH-q1 H).
Citation: Goedecker, Teter, Hutter, *Phys. Rev. B* **54**, 1703 (1996).

**Multi-k sampling.** Monkhorst–Pack `k_mesh` (CP2K
`SCHEME MONKHORST-PACK`) or explicit `k_points` + `k_point_weights`
(Σ weights = 1). Time-reversal folding via `k_time_reversal` (default true).
Even \(N\) is Γ-avoiding shifted mesh. Product path is **periodic**,
**single-grid**, `local_pp_on_grid` only — multigrid / nonperiodic throw at
`build()`.

**OT parameterization** (VandeVondele & Hutter, *J. Chem. Phys.* **118**, 4365
(2003); Kühne et al., *J. Chem. Phys.* **152**, 194103 (2020) §VII C). For
reference occupied \(C_0\) with \(C_0^\dagger S C_0 = I\) and an \(S\)-tangent
step \(x\):

\[
C(x) = C_0\,\cos U + x\,U^{-1}\sin U,\qquad
U = (x^\dagger S x)^{1/2}.
\]

On multi-k UKS the OT manifold is applied **per** \((k,\sigma)\) with the
complex Hermitian \((H_k^\sigma, S_k)\).

**OUTER_SCF + fixed-H OT (DFT+U product path).** With non-empty `plus_u` and
integer occupations, multi-k UKS OT+U uses a **CP2K-style OUTER_SCF**: rebuild
\(H_k^\sigma\) (including Dudarev \(V_U^\sigma\)) from the k-summed spin
densities, then **fixed-H OT** band minimize on each local \((k,\sigma)\)
(`ot_minimize_fixed_k`). Same self-consistent fixed point as multi-k UKS
DIIS+U when both converge on identical grids. Without `plus_u`, multi-k UKS
OT uses the product dual-spin OT loop of [ot-uks-kpoints](ot-uks-kpoints.md).

**Dudarev DFT+U (binding).** Per element and angular momentum \(L\), effective
\(U_{\mathrm{eff}}=U-J\) in **eV** on the public surface (converted to
Hartree inside). Symmetrized Mulliken occupation on the AO subspace of atom
\(A\) with \(\ell=L\) (CP2K `PLUS_U_METHOD MULLIKEN`):

\[
n^{A\sigma}_{mm'} = \tfrac12\big[(P^\sigma S)_{mm'} + (S P^\sigma)_{mm'}\big].
\]

Energy and potential ( \(U\) in Hartree; sum over atoms and spin):

\[
E_U = \sum_{A\sigma}\frac{U_{\mathrm{eff}}}{2}\Big[\mathrm{Tr}\,n^{A\sigma}
  - \mathrm{Tr}(n^{A\sigma} n^{A\sigma})\Big],
\qquad
V^{A\sigma}_{mm'} = U_{\mathrm{eff}}\Big(\tfrac12\delta_{mm'} - n^{A\sigma}_{mm'}\Big).
\]

Matches CP2K simplified rotationally invariant form (Dudarev et al.;
`U_MINUS_J`). Implementation: `cpu/gpw/internal/dft_plus_u.h` (not public).
Empty `plus_u` leaves the multi-k UKS OT path unchanged.

**Oracle note (CP2K 2026.1).** Upstream aborts with
`OT not possible with kpoint calculations` (`qs_scf_initialization.F`).
Absolute CP2K **OT+k+UKS** digits **do not exist** on this oracle version.
The offline twin is multi-k UKS **DIIS/DIAGONALIZATION** + PLUS_U (same
geometry / U / spin / k-mesh) — a **fixed-point** twin of ATV multi-k UKS
OT+U, **not** a wall twin for OT. Decisive acceptance is internal
OT≡DIIS multi-k UKS+U (A1–A2) plus activation / absolute first-green pin
(A3).

**Key exclusions (binding from closed product scope).**

- Multi-k **UKS OT + DFT+U + metallic smearing** ○ (residual;
  [ot-dftu-smearing](ot-dftu-smearing.md) covers RKS OT+U+smear; [ot-uks-smearing](ot-uks-smearing.md) covers UKS
  OT+smear without U).
- Liechtenstein full \(U,J\); PAW/GAPW projectors.
- Multigrid / nonperiodic multi-k OT+U (throws at `build()`).
- Analytic DFT+U forces / stress on this combo.
- Fair OT wall vs CP2K OT+k (impossible on CP2K 2026.1).
- Chemistry-quality correlated solids (NiO AFM etc.) — this is a microcase
  identity gate, not a materials pin (see `gpw-dftu-nio` for Γ molecule U).

## Implementation

Public surface only (`cpu::gpw::GpwDft` members; **no** `*Config` bag, **no**
`make_*` / `build_*`). Match the regression (Frame in Å; the Bohr cell and
offsets are converted with `units::kBohrInAngstrom`):

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

// High-spin H₂, 8 Bohr cubic cell, ±0.7 Bohr from centre along x
constexpr double b2a = units::kBohrInAngstrom;  // Frame is Å
const double L = 8.0 * b2a;                     // 8 Bohr
cpu::Frame frame(2);
for (int i = 0; i < 2; ++i) {
    frame.span(kAtom_type)[i] = 1;
    frame.span(kAtom_x)[i] = 0.5 * L + (i == 0 ? -0.7 : 0.7) * b2a;
    frame.span(kAtom_y)[i] = 0.5 * L;
    frame.span(kAtom_z)[i] = 0.5 * L;
}
frame.box = cpu::Box::cubic(L);

const cpu::gpw::Kind H{molrs::Element::H,
                       cpu::gpw::GaussianBasis(molrs::Element::H, "DZV-GTH-PADE"),
                       cpu::gpw::GthPotential(molrs::Element::H, "GTH-PADE-q1")};

// A3-class product: MP 2×1×1, time-reversal, OT + Dudarev U
GpwDft m{{H}, 100.0, XcKind::PADE};
m.n_alpha = 2;
m.n_beta = 0;
m.max_iter = 200;
m.tol = 1e-7;
m.mix = 0.35;                 // DIIS twin only; ignored when use_ot
m.k_mesh[0] = 2;
m.k_mesh[1] = 1;
m.k_mesh[2] = 1;
m.k_time_reversal = true;
m.use_ot = true;
m.plus_u.push_back(PlusU{molrs::Element::H, /*l=*/0, /*u_minus_j_ev=*/4.0});

m.build(frame);
const GpwResult& r = m.run(frame);
// r.e_total [Ha], r.e_dftu [Ha], r.converged, r.n_kpts, r.s2
```

Explicit single-k twin (A2 class — κ = (¼,0,0)):

```cpp
GpwDft ot = /* base UKS + U as above */;
ot.use_ot = true;
ot.k_points.push_back({0.25, 0.0, 0.0});
ot.k_point_weights.push_back(1.0);
// k_points non-empty overrides k_mesh
```

Build and run the regression:

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
| Public `GpwDft::{use_ot,plus_u,k_mesh,n_alpha,…}` | `gpw_solver.h` |
| Public `PlusU` / `GpwResult::{e_dftu,s2,n_kpts}` | same |
| OUTER_SCF multi-k UKS OT+U | `src/cpu/gpw/gpw_solver.cpp` |
| OT internals (`ot_minimize_fixed_k`, …) | `ot.h` |
| Dudarev E/V (internal) | `dft_plus_u.h` |
| Unit tests (not e2e) | `tests/cpu/gpw/test_cpu_gpw_ot.cpp` (Γ-UKS OT+U) |
| Offline CP2K DIIS multi-k+U twin | CP2K oracle |

**Note on `GPWEvaluator`:** the driver component binds the same configured
`GpwDft` (`GPWEvaluator(std::move(m))`), so `use_ot`, `n_alpha`/`n_beta`,
`plus_u` and `k_mesh` / `k_points` are all set on the method; the product gate
itself calls `GpwDft` directly (as in the regression).

## Validation (vs CP2K / internal identities)

Source: the pin notes recorded when this page was written (no standing regression).
Oracle: offline **CP2K 2026.1** (`cp2k.psmp`); **never** live in ctest.
Pin date: **2026-07-26**.

**Setup (must match twin / regression):**

| Item | Value |
| ---- | ----- |
| System | high-spin H₂, charge 0, UKS \(N_\alpha=2\), \(N_\beta=0\) |
| Cell | **8 Bohr** cubic, `PERIODIC XYZ` |
| Bond | ±0.7 Bohr from cell centre along *x* |
| Basis / PP | H `DZV-GTH-PADE` / `GTH-PADE-q1` |
| XC | PADE (LDA LSD) |
| DFT+U | H \(L=0\), \(U-J=4.0\) eV, `PLUS_U_METHOD MULLIKEN` |
| Cutoff | **100 Ry**, single grid (`NGRIDS 1`) |
| k (A1) | explicit Γ (complex multi-k UKS path at Γ limit) |
| k (A2) | explicit κ = (¼, 0, 0), weight 1 |
| k (A3) | MP **2×1×1**, time-reversal |
| ATV SCF | `use_ot=true`, `tol=1e-7`, `max_iter=200` (OT); DIIS twin `mix=0.35` |
| CP2K twin | DIAGONALIZATION + DIRECT_P_MIXING ALPHA 0.35 (OT+k impossible) |

**Pinned quantities (first green, REFERENCE):**

| Quantity | Atomiverse / pin | Twin | Tol | Notes |
| -------- | ---------------: | ---- | --- | ----- |
| A1 \(E\) OT+U @ Γ | −0.7802796 Ha | DIIS+U −0.7802792 Ha | **1×10⁻⁴ Ha** | multi-k UKS path at Γ |
| A1 \(\lvert\Delta E\rvert\) OT−DIIS | ≈ \(4\times10^{-7}\) | — | ≤ 1×10⁻⁴ | identity |
| A1 \(e_{\mathrm{dftu}}\) | non-trivial; OT≡DIIS | — | 1×10⁻³ Ha on \(e_U\) | |
| A2 \(E\) OT+U (κ=¼) | **−0.802825557 Ha** | DIIS+U −0.802825142 Ha | **1×10⁻⁴ Ha** | 32 OT outer / 66 DIIS |
| A2 \(\lvert\Delta E\rvert\) | ≈ \(4.1\times10^{-7}\) Ha | — | ≤ 1×10⁻⁴ | identity |
| A2 \(e_{\mathrm{dftu}}\), \(\langle S^2\rangle\) | OT ≡ DIIS | — | \(e_U\) 1×10⁻³; \(S^2\) 1×10⁻⁶ | |
| A3 \(E(U=0)\) OT MP 2×1×1 | −0.706134617 Ha | — | — | no-U baseline |
| A3 \(E(U=4)\) OT MP 2×1×1 | **−0.802825557 Ha** | frozen first-green | **1×10⁻³ Ha** | absolute pin class |
| A3 \(e_{\mathrm{dftu}}\) | ≈ −0.1803 Ha | — | \(\lvert e_U\rvert>10^{-8}\) | diagnostic |
| A3 \(\lvert E(U)-E(0)\rvert\) | ≈ 0.0967 Ha | — | ≥ **1×10⁻⁴ Ha** | activation floor |
| SCF | `r.converged` | — | — | A1–A3 |

Gates A1–A2 are **decisive** (internal exactness). A3 freezes the ATV
absolute for MP 2×1×1 OT+U at first green and asserts U activates the energy.

Offline CP2K DIIS multi-k+U twin (operator only; not ctest):

```bash
# module load … CP2K/2026.1; export CP2K_DATA_DIR=…/share/cp2k/data
cd /tmp && ln -sf $CP2K_DATA_DIR/{BASIS_SET,GTH_POTENTIALS} .
cp2k.psmp -i <CP2K oracle input for this case> \
          -o h2_uks_dftu_k.out
# OT+k aborts on CP2K 2026.1; this input is DIAGONALIZATION multi-k UKS+U
```

## Performance

Source: fair wall table in
the pin notes recorded when this page was written (no standing regression; 2026-07-26). Handoff:
the case pin notes → **`perf_exempt`**.

| Item | Value |
| ---- | ----- |
| `perf_exempt` | **true** |
| Reason | CP2K 2026.1 has **no** OT+kpoint path (hard abort); DIIS multi-k twin is not a fair OT wall comparison |
| \(r = T_{\mathrm{ATV}}/T_{\mathrm{CP2K}}\) | n/a |
| `within_10x` | n/a (exempt) |
| ATV wall (A1–A3, OMP=8, Release) | **≈ 0.48 s** |

Comparable settings: cutoff 100 Ry, PADE+U, single rank. Ratio gate does not
apply; do not invent a CP2K OT wall from the DIIS twin.

## Limitations

- **Identity-first pin, not CP2K OT digit match:** no absolute OT+k energy from
  CP2K 2026.1; acceptance is OT≡DIIS multi-k UKS+U + activation + frozen ATV
  absolute (A3).
- **Microcase only:** high-spin H₂ / DZV / 100 Ry — not a materials DFT+U
  claim. Solid AFM NiO remains a separate follow-on class (see
  `gpw-dftu-nio` molecule foothold).
- **No UKS OT+U+smear** product combo (`use_ot` + `plus_u` + metallic
  `smearing` on multi-k UKS throws / residual).
- **No analytic DFT+U forces/stress** on this path; forces stay on DIIS / other
  GPW regressions.
- **Single-grid periodic only** for multi-k; multigrid and MT nonperiodic
  multi-k OT+U are out of scope.
- **`GPWEvaluator`** is not the regression surface — the multi-k product
  gate calls `GpwDft` directly.
- **Dense OT preconditioner** cost does not forecast sparse / linear-scaling
  OT industrial workloads.
- Prior product pins remain the entry points for partial combos: multi-k UKS
  OT without U ([ot-uks-kpoints](ot-uks-kpoints.md)), multi-k RKS OT+U ([ot-dftu-kpoints](ot-dftu-kpoints.md)),
  Γ-UKS OT+U (`ot-dftu`).

## References

- CP2K DIIS multi-k+U twin: CP2K oracle
- Oracle contract: CP2K oracle
- Roadmap: [`docs/roadmap/dft.md`](../../roadmap/dft.md)
- Narrative GPW: [`docs/tutorial/ch09-gpw.md`](../ch09-gpw.md)
- Related handbook: [ot-minimizer-water](ot-minimizer-water.md),
  [gpw-dftu-nio](gpw-dftu-nio.md), [gpw-uks-o2-gamma](gpw-uks-o2-gamma.md)
- J. VandeVondele, J. Hutter, *J. Chem. Phys.* **118**, 4365 (2003) — OT
- T. D. Kühne et al., *J. Chem. Phys.* **152**, 194103 (2020) §VII C — CP2K OT
- S. L. Dudarev et al., *Phys. Rev. B* **57**, 1505 (1998) — simplified \(U\)
- G. Lippert, J. Hutter, M. Parrinello, *Mol. Phys.* **92**, 477 (1997) — GPW
- S. Goedecker, M. Teter, J. Hutter, *Phys. Rev. B* **54**, 1703 (1996) — PADE / GTH
