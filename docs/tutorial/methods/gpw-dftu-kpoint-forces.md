# Multi-k GPW analytic ionic forces with Dudarev DFT+U — H₂ toy + NiO Γ pin

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `gpw-dftu-kpoint-forces` |
| Public API | `cpu::gpw::GpwDft` members (`k_mesh`, `k_points`/`k_point_weights`, `k_time_reversal`, `plus_u`, `n_alpha`/`n_beta`, `local_pp_on_grid`) + `GpwDft::forces(frame)` / `gpw_forces_fd` |
| Roadmap | `docs/roadmap/dft.md` — multi-k DFT+U analytic forces (`reg_gpw-dftu-kpoint-forces`) |
| Oracle | **CP2K 2026.1** (offline pin, CP2K oracle) |

## Summary

**Analytic ionic forces for multi-k collinear UKS GPW with Dudarev DFT+U.**
The Dudarev occupation is built from the geometry-dependent overlap
\(S_k\), so DFT+U contributes an **occupation-matrix force term** on top of
the ordinary Hellmann–Feynman + Pulay pieces. This delivery folds that term
into the existing Pulay \(W\!\cdot\!dS_k\) contraction on both the real Γ and
complex multi-k paths, and (in the same change) fixed the variational
\(\partial E_U/\partial P\) scatter so the SCF fixed point is stationary for
the reported \(E_U[n(P,S)]\) — the precondition for any analytic-vs-FD dual
gate. Decisive validation is a full-6N multi-SCF finite-difference dual gate
on a multi-k H₂ high-spin toy (agreement 6.4×10⁻⁶ Ha/Bohr); the external
anchor is a CP2K 2026.1 `ENERGY_FORCE` pin on a periodic Γ NiO d-shell
system. Product gate: `reg_gpw-dftu-kpoint-forces`.

Related: multi-k DFT+U SCF — [gpw-dftu-kpoints](gpw-dftu-kpoints.md);
+U stress — [gpw-dftu-stress](gpw-dftu-stress.md),
[gpw-dftu-kpoint-stress](gpw-dftu-kpoint-stress.md); Γ +U pin —
[gpw-dftu-nio](gpw-dftu-nio.md); plain k forces —
[gpw-kpoint-forces](gpw-kpoint-forces.md).

## Method

**Occupation.** Per correlated (atom, ℓ) block, spin σ, k-summed symmetrized
Mulliken projection (CP2K `PLUS_U_METHOD MULLIKEN` family):

\[
n^\sigma = \sum_k w_k\,\tfrac12\,\mathrm{Herm}\big[(P_k^\sigma S_k)
  + (S_k P_k^\sigma)\big]\Big|_{\text{block}},
\qquad
E_U = \sum_{A\sigma}\tfrac{U_{\mathrm{eff}}}{2}
  \big[\mathrm{Tr}\,n - \mathrm{Tr}(n\,n)\big].
\]

**Occupation-matrix force term.** Since \(n\) depends on geometry through
\(S_k\),

\[
\frac{dE_U}{dR} = \sum_\sigma\sum_k w_k\,
  \mathrm{Re\,Tr}\big[M_k^\sigma\,\tfrac{dS_k}{dR}\big],
\qquad
M_k^\sigma = \tfrac12\big(\hat V^\sigma P_k^\sigma
  + P_k^\sigma \hat V^\sigma\big),
\]

folded as \(W_k \leftarrow W_k - M_k\) into the Pulay energy-weighted-density
contraction (Γ and multi-k alike; `internal/dft_plus_u.h`,
`subtract_dftu_pulay[_k]`). This mirrors CP2K's own `matrix_w` correction for
its geometry-dependent Mulliken-charges +U (`src/dft_plus_u.F`).

**Variational scatter fix (2026-07-31, this delivery).** The exact
\(\partial E_U/\partial P\) for this occupation is
\(\tfrac12(\hat V S + S \hat V)\); the previous bare \(\hat V\) scatter left
the SCF non-stationary and made analytic vs multi-SCF FD forces disagree at
O(1) Ha/Bohr (and was the ~9×10⁻⁴ Ha/Bohr³ residual in the +U stress dual
gates). CP2K applies the same S-weighted form. All +U SCF fixed points moved;
sibling +U regressions were re-pinned in the same change.

**Projector-model divergence (surfaced, out of scope).** CP2K
`PLUS_U_METHOD MULLIKEN` projects on the on-site **net** AO block
\(q_{ab} = P_{ab}S_{ab}\) (elementwise; geometry-independent ⇒ no occupation
force term), while ATV projects on the symmetrized **gross** product
\(\tfrac12(PS+SP)\) restricted to the block (geometry-dependent). The +U
force *contribution* therefore differs by model, not by bug — quantified in
the regression's REFERENCE and confirmed against ATV's own FD. Aligning the
projector (or adding CP2K's as a second method) is a census decision.

**Not included:** Liechtenstein full \(U,J\); OT + multi-k + DFT+U; GGA +U;
multi-k CP2K force twin (CP2K's k-point diagonalization does not converge on
the twin system — documented below).

## Implementation

Public surface only — concrete `cpu::gpw::GpwDft` members (no `*Config`
bags, no factories), Frame in / Frame out: `forces(frame)` for the analytic
product path and the free `gpw_forces_fd` oracle.

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

// High-spin H2, compressed 1.0 Bohr bond, 12 Bohr cubic cell (Frame is Å)
const double b2a = units::kBohrInAngstrom;
const double L = 12.0 * b2a;
cpu::Frame frame(2);
for (int i = 0; i < 2; ++i) {
    frame.span(kAtom_type)[i] = 1;
    frame.span(kAtom_y)[i] = 0.5 * L;
    frame.span(kAtom_z)[i] = 0.5 * L;
}
frame.span(kAtom_x)[0] = 0.5 * L - 0.5 * b2a;
frame.span(kAtom_x)[1] = 0.5 * L + 0.5 * b2a;
frame.box = cpu::Box::cubic(L);

GpwDft m{{Kind{molrs::Element::H,
               GaussianBasis(molrs::Element::H, "DZV-GTH-PADE"),
               GthPotential(molrs::Element::H, "GTH-PADE-q1")}}, 100.0, XcKind::PADE};
m.n_alpha = 2;  m.n_beta = 0;      // high-spin UKS
m.tol = 1e-8;   m.mix = 0.35;  m.max_iter = 120;
m.local_pp_on_grid = true;
m.k_mesh[0] = 1; m.k_mesh[1] = 1; m.k_mesh[2] = 2;   // MP 1×1×2, TR-folded
m.k_time_reversal = true;
m.plus_u.push_back(PlusU{molrs::Element::H, /*l=*/0, /*u_minus_j_ev=*/2.0});

m.build(frame);
m.forces(frame);                                   // kAtom_fx/fy/fz [eV/Å]
const auto F_fd = gpw_forces_fd(frame, m, /*h=*/1e-3 * b2a);  // FD oracle [eV/Å]
```

U−J = 2 eV deliberately: at 4 eV this DZV s-shell toy sits in the
strong-overlap regime (CP2K itself warns "unphysical Mulliken charges") and
the displaced multi-SCF FD oracle hops between SCF branches — no dual gate
holds on a multi-valued surface. At 2 eV the SCF is single-branch.

Build (GPW needs FFTW3) and run:

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
  cmake --build build --target reg_gpw-dftu-kpoint-forces -j$(nproc)
  reg_gpw-dftu-kpoint-forces
'
```

| Piece | Path |
| ----- | ---- |
| +U energy/potential + W-fold (`subtract_dftu_pulay[_k]`) | `dft_plus_u.h` |
| Force assembly (`GpwDft::forces` / `gpw_forces_fd`) | `gpw_solver.h`, `internal/gpw_forces.h` |
| Public `GpwDft` members | `gpw_solver.h` |
| CP2K twin input | CP2K oracle |

## Validation (vs CP2K + internal dual gates)

Source: case `gpw-dftu-kpoint-forces` pin.
Oracle: offline **CP2K 2026.1** (`cp2k.psmp`, OMP=8), pin date **2026-07-31**.
Never live in ctest.

**System A — multi-k dual gate (decisive).** H₂ high-spin (Nα=2, Nβ=0),
1.0 Bohr bond along x, 12 Bohr cubic periodic cell, DZV-GTH-PADE /
GTH-PADE-q1, PADE LSD, 100 Ry, MP 1×1×2 (TR-folded), H L=0 U−J = 2 eV:

| Gate | Quantity | Measured (2026-07-31) | Tol |
| ---- | -------- | --------------------: | --- |
| A1 | E(k-UKS+U 1×1×2, 100 Ry) | −0.611940650579 Ha (e_dftu −0.10209345) | payload sanity |
| A2 | max\|F_an − F_fd\| (full-6N `gpw_forces_fd`, h=1e-3) | **6.4×10⁻⁶ Ha/Bohr** | 5×10⁻⁵ |
| A3 | \|ΣF\| / Newton-3 on the bond axis | 1.1×10⁻¹⁴ | 1×10⁻⁶ |
| A4 | max\|F(U) − F(U=0)\| (U-term activation) | 5.4×10⁻² Ha/Bohr | ≥ 1×10⁻⁴ |
| A5 | max\|F_kΓ − F_Γ\| (explicit k=Γ ≡ real Γ fold) | 6.0×10⁻⁸ Ha/Bohr | 5×10⁻⁶ |

Pre-fold control: without the W-fold, A2 reads ~1.6 Ha/Bohr (wrong sign) —
the occupation-matrix term is decisive, four orders above the gate residual.

**System B — CP2K `ENERGY_FORCE` pin.** Periodic Γ NiO stretched molecule
(10 Å cubic PBC box, R = 1.8 Å), SZV-GTH-PADE(-q18) / GTH-PADE-q18/q6, PADE
LSD, Nα=13 / Nβ=11, PLUS_U Mulliken Ni d U−J = 6 eV. CP2K output lines
(OT FULL_SINGLE_INVERSE, multigrid 250 Ry / REL 40 / NGRIDS 4, wall 12.84 s):

```text
  *** SCF run converged in    14 steps ***
  DFT+U energy:                                                 0.04952309241684
 ENERGY| Total FORCE_EVAL ( QS ) energy [hartree]           -184.929424563910374
 FORCES|      1  3.17871282E-07 -1.91561775E-06 -8.35837089E-03   8.35837111E-03
 FORCES|      2 -3.55176272E-08  2.19071254E-06 -7.27092413E-03   7.27092447E-03
```

| Check | Status | Value | Tol |
| ----- | ------ | ----: | --- |
| \|E_ATV − E_CP2K\| | **asserted** | 1.18×10⁻² Ha | 5×10⁻² (single grid vs multigrid+OT) |
| transverse F ≈ 0 (axial molecule) | **asserted** | ~1×10⁻⁸ Ha/Bohr | 5×10⁻⁴ |
| ΔF_z(Ni−O) vs CP2K | **diagnostic** | ATV +8.97×10⁻² vs CP2K −1.09×10⁻³ | projector-model diff (see Method) |
| Γ d-shell FD, tol 1e-8, h=2×10⁻³ (offline) | diagnostic | \|F_an − F_fd\| = **3.7×10⁻⁴ Ha/Bohr** | not in ctest budget |

Multi-k CP2K twin attempts (`nio_pbc_kpoint_force.inp`, MP 1×1×2, diag +
Broyden, MAX_SCF up to 600): SCF oscillates and never converges — the
degenerate minority-spin d manifold defeats fixed-multiplicity
diagonalization, and CP2K has no OT with k-points. Multi-k force correctness
is therefore gated by the internal A2 full-6N dual gate + the A5 Γ-limit
identity, with the Γ d-shell pin as the external anchor.

## Performance

Source: case `gpw-dftu-kpoint-forces` pin (2026-07-31,
Release, OMP=8).

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r = T_{\mathrm{ATV}}/T_{\mathrm{CP2K}}\) | ≤10× | Caveat |
| ---- | ------------------------: | ----------------------: | ---------------------------------------: | ---- | ------ |
| NiO Γ +U single point + analytic forces (B) | 12.84 | ≈ 13 | **≈ 1.0** | ✓ | ATV single grid 250 Ry vs CP2K multigrid+OT |
| Full regression A1–A5 + B (incl. full-6N FD) | — | 38.6 | n/a | — | FD oracle cost, not a product path |

## Limitations

- **Projector model differs from CP2K** on the +U force contribution (gross
  symmetrized vs net elementwise Mulliken); force twin is diagnostic, energy
  twin asserted. Alignment/second method is a census follow-on.
- **Toy dual-gate system:** multi-k gate is H₂ HS; the d-shell force check is
  Γ-only (offline FD at tol 1e-8). A converged multi-k CP2K force twin does
  not exist (CP2K k-point SCF non-convergent on the twin).
- **U−J = 2 eV on the toy** — the 4 eV strong-overlap branch structure defeats
  any FD dual gate (multi-valued SCF surface).
- **No OT + multi-k + DFT+U; PADE LSD only** (no GGA +U); Liechtenstein full
  \(U,J\) out of scope.
- Force entry points live under `internal/gpw_forces.h`; the Frame-level
  evaluator surface for multi-k +U forces is follow-on.

## References

- Offline pin: case `gpw-dftu-kpoint-forces` pin
- Regression consumer: `reg_gpw-dftu-kpoint-forces.cpp`
- Oracle contract: CP2K oracle
- Siblings: [gpw-dftu-kpoints](gpw-dftu-kpoints.md), [gpw-dftu-kpoint-stress](gpw-dftu-kpoint-stress.md), [gpw-dftu-nio](gpw-dftu-nio.md), [gpw-kpoint-forces](gpw-kpoint-forces.md)
- S. L. Dudarev et al., *Phys. Rev. B* **57**, 1505 (1998) — U−J functional
- VandeVondele, Krack, Mohamed, Parrinello, Chassaing, Hutter, *Comput. Phys. Commun.* **167**, 103 (2005) — Quickstep GPW forces
- CP2K 2026.1 `src/dft_plus_u.F` — S-weighted KS contribution and `matrix_w` correction
