# Multi-k collinear UKS OT — high-spin H₂ / GPW PADE

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `ot-uks-kpoints` |
| Regression | `regressions/gpw/ot-uks-kpoints/` |
| Public API | `cpu::gpw::GpwDft` members `use_ot`, `n_alpha`/`n_beta`, `k_mesh` / `k_points`+`k_point_weights` |
| Roadmap | `docs/roadmap/dft.md` §10 GPW — multi-k UKS OT ✓ (`reg_ot-uks-kpoints`) |

## Summary

**Multi-k collinear unrestricted Kohn–Sham (UKS) Orbital Transformation (OT)**
extends the GPW OT minimizer from closed-shell multi-k RKS (`ot-kpoints`) and
Γ-point UKS OT (`ot-uks-o2`) to the **complex Bloch path with dual spin
manifolds**. Occupied \(C_k^\alpha\) / \(C_k^\beta\) are optimized on the
Stiefel-type manifold with occupation 1 per spin channel; joint Polak–Ribière+
CG and Armijo line search on the rebuilt total energy share the same fixed
point as multi-k UKS DIIS on identical grids.

Product surface: set `use_ot = true`, open-shell fillings `n_alpha` / `n_beta`,
and either explicit `k_points`+`k_point_weights` or a Monkhorst–Pack `k_mesh`
(optional `k_time_reversal`). No public `*Config` bag, no factories. Product
gate: `regressions/gpw/ot-uks-kpoints` (`reg_ot-uks-kpoints`).

**Oracle note (binding).** CP2K **2026.1** aborts OT with k-points at
`qs_scf_initialization` (`OT not possible with kpoint calculations`). There is
**no** fair CP2K OT+KPOINTS+UKS energy or wall twin on this release. Validation
is **internal exactness** (OT↔DIIS, real-Γ↔k-path Γ) plus a frozen first-green
absolute band; `perf_exempt = true`. Offline input
`tools/cp2k_oracle/inputs/ot-uks-kpoints/` documents the DIAG multi-k UKS twin
only.

## Method

**Setting.** Collinear UKS Gaussian-and-plane-waves (GPW) Kohn–Sham DFT with
spin-resolved **PADE LSD**, GTH pseudopotentials, contracted Gaussian AOs, and
**periodic** orthorhombic cell. Brillouin-zone sampling via explicit fractional
**k** or Monkhorst–Pack mesh; complex Bloch \(H_k,S_k,C_k\) per spin. Method
channel energy is Hartree (`GpwResult::e_total`); public Frame energy remains
eV when the evaluator path is used.

**OT parameterization** (VandeVondele & Hutter, *J. Chem. Phys.* **118**, 4365
(2003); Kühne et al., *J. Chem. Phys.* **152**, 194103 (2020) §VII C). For
reference occupied coefficients \(C_0\) with \(C_0^\dagger S C_0 = I\) and an
\(S\)-tangent step \(x\) (\(C_0^\dagger S x = 0\)):

\[
C(x) = C_0\,\cos U + x\,U^{-1}\sin U,\qquad
U = (x^\dagger S x)^{1/2}.
\]

**UKS multi-k specialization.** Each k-point and each spin channel carries its
own occupied block. Open-shell occupation is **1** (not 2 as in closed-shell
RKS OT). Gradients are formed per spin manifold; CG is joint over the
concatenated spin×k OT variables with Armijo on the **rebuilt total energy**
(same outer Hamiltonian rebuild schedule as multi-k UKS DIIS). For a gapped
high-spin configuration that both converge, OT and DIIS meet at the same
self-consistent fixed point on identical grids.

**Γ identity.** Explicit k-list `{(0,0,0)}` with weight 1 on the complex path
must recover the real Γ UKS OT energy (regression gate A1). Multi-k RKS OT
without spin is `reg_ot-kpoints`; Γ UKS OT against a live CP2K OT twin is
`reg_ot-uks-o2`.

**Key exclusions.**

- No CP2K OT+KPOINTS energy/wall twin on CP2K 2026.1 (engine rejects the combo).
- No metallic FD/MP **smearing** on this slug (see `ot-smearing` for RKS OT+FD).
- No Dudarev DFT+U (see `ot-dftu-kpoints` for multi-k OT+U).
- No multi-k UKS OT **forces / stress / BOMD** product gate.
- No MPI multi-rank OT scaling claim; single-rank identity suite.
- Dense \(O(n_b^3)\) OT preconditioner — sparse / linear-scaling OT out of
  scope.
- PW backend has no multi-k UKS OT product switch.

## Implementation

Public surface only (`cpu::gpw::GpwDft` members; no `*Config`, no factories).
Match the regression (compact high-spin H₂, 8 Bohr cell, 100 Ry, PADE):

```cpp
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/molrs.hpp"

using namespace ATV;

const double L = 8.0; // Bohr, cubic periodic
const cpu::gpw::GaussianBasis bH(molrs::Element::H, "DZV-GTH-PADE");
const cpu::gpw::GthPotential pH(molrs::Element::H, "GTH-PADE-q1");
const double r0[3] = {0.5 * L - 0.7, 0.5 * L, 0.5 * L};
const double r1[3] = {0.5 * L + 0.7, 0.5 * L, 0.5 * L};
std::vector<cpu::gpw::GpwAtom> atoms;
atoms.emplace_back(r0, bH, pH, 1);
atoms.emplace_back(r1, bH, pH, 1);

// Multi-k collinear UKS OT at κ = (1/4, 0, 0)
cpu::gpw::GpwDft ot;
ot.cell[0] = ot.cell[1] = ot.cell[2] = L;
ot.cutoff_ry = 100.0;
ot.n_alpha = 2;   // high-spin H₂: Nα=2, Nβ=0
ot.n_beta = 0;
ot.n_occ = 0;     // UKS path when n_alpha+n_beta > 0
ot.max_iter = 200;
ot.tol = 1e-8;
ot.xc = cpu::gpw::XcKind::PADE;
ot.eps_default = 1e-7;
ot.use_ot = true;
ot.k_points.push_back({0.25, 0.0, 0.0});
ot.k_point_weights.push_back(1.0);

ot.build(atoms);
const auto& res = ot.run(atoms);
// res.e_total [Hartree]; res.converged; res.s2; res.n_kpts; res.k_frac
```

Monkhorst–Pack alternative (regression A3):

```cpp
cpu::gpw::GpwDft ot_mp;
// ... same base UKS fields as above ...
ot_mp.use_ot = true;
ot_mp.k_mesh[0] = 2;
ot_mp.k_mesh[1] = 1;
ot_mp.k_mesh[2] = 1;
ot_mp.k_time_reversal = true;
```

DIIS twin: same members with `use_ot = false` (default). Explicit Γ identity:
`k_points = {{0,0,0}}`, `k_point_weights = {1}` vs real Γ (`k_mesh` left at
`{1,1,1}` / empty `k_points`).

Build and run the regression:

```bash
bash -lc '
  module load buildenv-gcc/2026.03-mpich
  export OPENBLAS_ROOT=$OPENBLASDIR
  export RUSTUP_TOOLCHAIN=stable-x86_64-unknown-linux-gnu
  export LD_LIBRARY_PATH=$LIBRARY_PATH:$LD_LIBRARY_PATH
  cmake -B build -DCMAKE_BUILD_TYPE=Release \
        -DATV_BUILD_TESTS=ON -DATV_BUILD_HF=ON \
        -DATV_BUILD_PW=ON -DATV_BUILD_GPW=ON \
        -DATV_BUILD_REGRESSIONS=ON -DATV_BUILD_CUDA=OFF \
        -DBLA_VENDOR=OpenBLAS
  cmake --build build --target reg_ot-uks-kpoints -j$(nproc)
  ./build/regressions/reg_ot-uks-kpoints
'
```

| Path | Role |
| ---- | ---- |
| `include/atomiverse/cpu/gpw/gpw_solver.h` | Public `GpwDft`: `use_ot`, UKS, k-mesh |
| `include/atomiverse/cpu/gpw/internal/ot.h` | OT manifold / preconds (internal) |
| `tests/cpu/gpw/` | Unit coverage for OT / UKS / k paths (as applicable) |
| `regressions/gpw/ot-uks-kpoints/` | E2E product gate + `REFERENCE.md` |
| `tools/cp2k_oracle/inputs/ot-uks-kpoints/h2_uks_ot_kpoints.inp` | Offline CP2K DIAG multi-k UKS doc twin |

## Validation (vs CP2K)

Source: [`regressions/gpw/ot-uks-kpoints/REFERENCE.md`](../../../regressions/gpw/ot-uks-kpoints/REFERENCE.md).
Oracle: offline **CP2K 2026.1** documentation twin only; **never** live in ctest.

**Setup:**

| Item | Value |
| ---- | ----- |
| System | High-spin H₂, periodic, \(N_\alpha=2\), \(N_\beta=0\) |
| Geometry | ±0.7 Bohr from cell centre along *x* (compact DZV class) |
| Cell | **8 Bohr** cubic, `PERIODIC XYZ` |
| Basis / PP | DZV-GTH-PADE / GTH-PADE-q1 |
| XC | PADE LSD |
| Cutoff | **100 Ry**, single grid |
| SCF | `max_iter=120` (DIIS) / `200` (OT), `tol=1e-8` |
| k (A2) | GENERAL κ = (¼, 0, 0), w = 1 |
| k (A3) | Monkhorst–Pack 2×1×1 with time-reversal fold |
| CP2K OT+KPOINTS | **Rejected** on 2026.1 (`OT not possible with kpoint calculations`) |

**Hard-coded pins [Hartree] (first green, 2026-07-26):**

| Quantity | Atomiverse / pin | CP2K 2026.1 | Tol | Notes |
| -------- | ---------------: | ----------: | --- | ----- |
| multi-k UKS OT \(E\) (κ=¼) | **−0.7061343842** | n/a (OT+k abort) | **1×10⁻³** vs frozen | A4 frozen first-green band |
| multi-k UKS DIIS (same grids) | −0.7061343842 | DIAG twin unreliable* | — | 6 DIIS iters |
| \(\lvert E_{\mathrm{OT}}-E_{\mathrm{DIIS}}\rvert\) | \(4.8\times10^{-12}\) | — | ≤ **1×10⁻⁴** | A2 fixed-point |
| real Γ UKS OT | −0.6687052609 | — | — | real Γ path |
| k-path Γ UKS OT | −0.6687052608 | — | ≤ **1×10⁻⁶** vs real Γ | A1 |
| MP 2×1×1 OT ≡ DIIS | (gate A3) | — | ≤ **1×10⁻⁴** | TR mesh |

\*Compact high-spin H₂ multi-k UKS DIAG on CP2K did not converge reliably
(outer-SCF energy oscillation); not used as a digit pin. Absolute energy
stability is the **frozen first-green** band until a future CP2K release
offers OT+KPOINTS.

**Regression gates:**

| Gate | Check | Tol |
| ---- | ----- | ---: |
| A1 | real Γ UKS OT ≡ k-path Γ UKS OT | 1e-6 Ha |
| A2 | multi-k UKS OT ≡ multi-k UKS DIIS (κ=¼); \(S^2\) match | 1e-4 Ha; 1e-6 on \(S^2\) |
| A3 | MP 2×1×1 OT ≡ DIIS | 1e-4 Ha |
| A4 | \(\lvert E_{\mathrm{OT}} - (-0.7061343842)\rvert\) | 1e-3 Ha |

Offline DIAG documentation twin (host with CP2K; **not** ctest):

```bash
# CP2K 2026.1 rejects OT+KPOINTS; input is DIAG multi-k UKS only
cp2k.psmp -i tools/cp2k_oracle/inputs/ot-uks-kpoints/h2_uks_ot_kpoints.inp \
          -o h2_uks_ot_kpoints.out
```

## Performance

Source: `regressions/gpw/ot-uks-kpoints/REFERENCE.md` and
the case `REFERENCE.md` (wave 8).

| Item | Value |
| ---- | ----- |
| `perf_exempt` | **true** |
| Reason | CP2K 2026.1 rejects OT+KPOINTS; no fair OT wall twin |
| \(r = T_{\mathrm{ATV}}/T_{\mathrm{CP2K}}\) | **n/a** |
| `within_10x` | n/a (exempt) |

Identity suite only; do not invent a ratio against CP2K DIAG or against
unrelated OT pins (`ot-minimizer-water`, `ot-uks-o2`).

## Limitations

- **Capability beyond offline oracle OT surface:** multi-k UKS OT is product in
  ATV while CP2K 2026.1 still forbids OT with k-points. Absolute digits are a
  frozen first-green band, not a cited CP2K OT energy.
- Compact high-spin H₂ is a **toy** multi-k UKS identity system (same class as
  `gpw-kpoints-uks`); not a materials-grade k-mesh convergence study.
- No forces, stress, AIMD, smearing, or DFT+U on this slug (siblings:
  `gpw-kpoint-uks-forces`, `gpw-kpoint-uks-stress`, `ot-smearing`,
  `ot-dftu-kpoints`).
- Dense OT preconditioner cost; no sparse OT, no MPI industrial scaling gate.
- PW / CUDA multi-k UKS OT are out of scope for the CP2K GPW product path.

## References

- Offline pin + gates: `regressions/gpw/ot-uks-kpoints/REFERENCE.md`
- Regression binary: `regressions/gpw/ot-uks-kpoints/reg_ot-uks-kpoints.cpp`
- CP2K oracle input: `tools/cp2k_oracle/inputs/ot-uks-kpoints/h2_uks_ot_kpoints.inp`
- Oracle rules: `tools/cp2k_oracle/README.md` (offline only; never in ctest)
- Related handbooks: [ot-minimizer-water](ot-minimizer-water.md) (Γ RKS OT),
  `regressions/gpw/ot-kpoints` (multi-k RKS OT),
  `regressions/gpw/ot-uks-o2` (Γ UKS OT + live CP2K OT pin),
  [gpw-uks-o2-gamma](gpw-uks-o2-gamma.md) (Γ UKS DIIS)
- Roadmap: `docs/roadmap/dft.md` §10 (multi-k UKS OT ✓)
- VandeVondele, J.; Hutter, J. *J. Chem. Phys.* **118**, 4365 (2003),
  doi:10.1063/1.1543154
- Kühne, T. D. et al. *J. Chem. Phys.* **152**, 194103 (2020) §VII C,
  doi:10.1063/5.0007045
