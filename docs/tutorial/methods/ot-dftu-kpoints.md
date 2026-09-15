# Multi-k OT + Dudarev DFT+U — H₂ / GPW PADE

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `ot-dftu-kpoints` |
| Regression | `regressions/gpw/ot-dftu-kpoints/` |
| Public API | `cpu::gpw::GpwDft` members `use_ot`, `k_mesh` / `k_points`, `plus_u` |
| Roadmap | `docs/roadmap/dft.md` §10 GPW — multi-k OT+DFT+U ✓ (`reg_ot-dftu-kpoints`) |

## Summary

**Orbital Transformation (OT)** SCF on the **complex multi-k** GPW path,
combined with **Dudarev DFT+U** (`plus_u` / CP2K `PLUS_U` Mulliken), is a
product combo that neither Γ-only OT+U (`reg_ot-dftu`) nor multi-k DIIS+U
(`reg_gpw-dftu-kpoints`) covers alone. Outer SCF rebuilds \(H_k\) (including
\(V_U\)) from the BZ-summed density; each local \(k\) then runs a **fixed-H**
OT band minimize (`ot_minimize_fixed_k`) — CP2K-style OUTER_SCF for the OT+U
case. Decisive acceptance is **internal fixed-point parity** OT ≡ DIIS under
the same multi-k grids and \(U\); absolute first-green energy is frozen for
the MP \(2\times2\times2\) pin.

Public surface: concrete `cpu::gpw::GpwDft` members (no `*Config`, no
factories). Product gate: `reg_ot-dftu-kpoints`. Fair wall vs CP2K OT is
**`perf_exempt`** — CP2K **2026.1** hard-aborts OT with k-points
(`OT not possible with kpoint calculations`).

Related: Γ OT water — [ot-minimizer-water](ot-minimizer-water.md); Γ DFT+U —
[gpw-dftu-nio](gpw-dftu-nio.md); narrative GPW — [Ch. 9](../ch09-gpw.md).

## Method

**GPW (Quickstep family).** Wavefunction in contracted Gaussians; density /
Hartree / XC on a uniform real-space grid; GTH pseudopotentials. Citation:
Lippert, Hutter, Parrinello, *Mol. Phys.* **92**, 477 (1997).

**Multi-k RKS.** Monkhorst–Pack mesh (`k_mesh`) or explicit fractional
`k_points` + weights. Complex Bloch matrices per \(k\); k-summed density;
time-reversal folding optional (`k_time_reversal`). Even \(N\) is Γ-avoiding
shifted (CP2K `SCHEME MONKHORST-PACK` convention).

**OT parameterization** (VandeVondele & Hutter, *J. Chem. Phys.* **118**, 4365
(2003); Kühne et al., *J. Chem. Phys.* **152**, 194103 (2020) §VII C). For
reference occupied \(C_0\) with \(C_0^\top S C_0 = I\) and \(S\)-tangent step
\(x\):

\[
C(x) = C_0\,\cos U + x\,U^{-1}\sin U,\qquad
U = (x^\top S x)^{1/2}.
\]

**OT + multi-k + PLUS_U (product path).** When `use_ot && plus_u` non-empty
on multi-k:

1. Build density \(\rho\) from the current \(\{C_k\}\) (k-weighted).
2. Rebuild every \(H_k\) from \(\rho\), including Dudarev \(V_U\) on the
   Mulliken subspace.
3. **Fixed-H** OT minimize on each local \(k\) (no Armijo one-step on this
   combo path).
4. Outer residual / energy criteria until `tol` or `max_iter`.

Without PLUS_U, multi-k OT may still use the one-step-per-outer Armijo path
(see `reg_ot-kpoints`). Both routes share the same energy assembly
(`build_k_energy`) as multi-k DIIS.

**Dudarev DFT+U.** Per element and angular momentum \(L\), \(U_{\mathrm{eff}}=U-J\)
in **eV** on the public surface (Hartree internally). Symmetrized Mulliken
occupation (CP2K `PLUS_U_METHOD MULLIKEN`):

\[
n^{A\sigma}_{mm'} = \tfrac12\big[(P^\sigma S)_{mm'} + (S P^\sigma)_{mm'}\big],
\qquad
E_U = \sum_{A\sigma}\frac{U_{\mathrm{eff}}}{2}\Big[\mathrm{Tr}\,n^{A\sigma}
  - \mathrm{Tr}(n^{A\sigma} n^{A\sigma})\Big],
\]

\[
V^{A\sigma}_{mm'} = U_{\mathrm{eff}}\Big(\tfrac12\delta_{mm'} - n^{A\sigma}_{mm'}\Big).
\]

On multi-k, occupations and \(E_U\) are BZ-averaged consistently with the
density path. Citation: Dudarev et al., *Phys. Rev. B* **57**, 1505 (1998).

**CP2K twin role (binding).** CP2K 2026.1 **cannot** run OT+kpoints (abort in
`qs_scf_initialization.F`). The offline input
`tools/cp2k_oracle/inputs/ot-dftu-kpoints/h2_ot_dftu_kpoints.inp` is the
**DIAGONALIZATION + multi-k PLUS_U** twin — same geometry, mesh, \(U\), and
fixed point as ATV OT+U, **not** a fair OT wall twin.

**Key exclusions.**

- Multi-k **UKS** OT + DFT+U (throws / not wired; see `ot-uks-kpoints` for
  multi-k UKS OT without U).
- Absolute CP2K OT+k energy digits (oracle version has no OT+k path).
- Analytic multi-k OT+U forces / stress as a fair twin claim.
- Sparse / linear-scaling OT preconditioner.

## Implementation

Public surface only: concrete `cpu::gpw::GpwDft` members (`use_ot`,
`k_mesh` / `k_points`, `plus_u`). No `*Config` bag, no `make_*` / `build_*`.
Regression matches this sequence (cell / coords in **Bohr** on the solver
path):

```cpp
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/molrs.hpp"

using namespace ATV;
using cpu::gpw::GpwDft;
using cpu::gpw::GpwResult;
using cpu::gpw::PlusU;
using cpu::gpw::XcKind;

// H2 toy: 8 Bohr cubic, ±0.7 Bohr from centre along x (as reg A3)
const double L = 8.0;  // Bohr
const cpu::gpw::GaussianBasis bH(molrs::Element::H, "DZV-GTH-PADE");
const cpu::gpw::GthPotential pH(molrs::Element::H, "GTH-PADE-q1");
const double r0[3] = {0.5 * L - 0.7, 0.5 * L, 0.5 * L};
const double r1[3] = {0.5 * L + 0.7, 0.5 * L, 0.5 * L};
std::vector<cpu::gpw::GpwAtom> atoms;
atoms.emplace_back(r0, bH, pH, 1);
atoms.emplace_back(r1, bH, pH, 1);

GpwDft m;
m.cell[0] = m.cell[1] = m.cell[2] = L;
m.cutoff_ry = 100.0;
m.n_occ = 1;
m.max_iter = 200;
m.tol = 1e-7;
m.mix = 0.35;                         // DIIS twin path; OT uses OUTER_SCF
m.xc = XcKind::PADE;
m.k_mesh[0] = m.k_mesh[1] = m.k_mesh[2] = 2;
m.k_time_reversal = true;             // TR-fold → 4 k for 2×2×2
m.use_ot = true;
m.plus_u.push_back(PlusU{molrs::Element::H, /*l=*/0, /*u_minus_j_ev=*/4.0});

m.build(atoms);
const GpwResult r = m.run(atoms);
// r.e_total [Ha], r.e_dftu [Ha], r.converged, r.n_kpts, r.iters
```

Explicit single-k (gates A1/A2) instead of `k_mesh`:

```cpp
m.k_points = {{0.25, 0.0, 0.0}};   // or {{0,0,0}} for Γ
m.k_point_weights = {1.0};        // must sum to 1
// leave k_mesh default; non-empty k_points overrides mesh
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
  cmake --build build -j$(nproc) --target atomiverse reg_ot-dftu-kpoints
  ./build/regressions/reg_ot-dftu-kpoints
'
```

| Path | Role |
| ---- | ---- |
| `include/atomiverse/cpu/gpw/gpw_solver.h` | Public `GpwDft::{use_ot,k_mesh,k_points,plus_u}` |
| `include/atomiverse/cpu/gpw/internal/ot.h` | OT manifold / fixed-k minimize (internal) |
| `include/atomiverse/cpu/gpw/internal/dft_plus_u.h` | Dudarev E/V (internal) |
| `src/cpu/gpw/gpw_solver.cpp` | Multi-k OUTER_SCF OT+U wire-up |
| `tests/cpu/gpw/test_cpu_gpw_ot.cpp` | Unit footholds (e2e pin is the regression) |
| `regressions/gpw/ot-dftu-kpoints/` | E2E product gate + `REFERENCE.md` |
| `tools/cp2k_oracle/inputs/ot-dftu-kpoints/` | Offline CP2K **DIIS** multi-k PLUS_U twin |

## Validation (vs CP2K)

Source:
[`regressions/gpw/ot-dftu-kpoints/REFERENCE.md`](../../../regressions/gpw/ot-dftu-kpoints/REFERENCE.md).
Oracle: offline **CP2K 2026.1** (`cp2k.psmp`); **never** live in ctest.
Pin date: **2026-07-26**.

**Setup (must match twin / regression):**

| Item | Value |
| ---- | ----- |
| System | H₂, charge 0, closed-shell RKS (\(n_{\mathrm{occ}}=1\)) |
| Cell | **8 Bohr** cubic, `PERIODIC XYZ` |
| Geometry (Bohr) | H (3.3, 4.0, 4.0), H (4.7, 4.0, 4.0) |
| Basis / PP | `DZV-GTH-PADE` / `GTH-PADE-q1` |
| XC | PADE |
| Cutoff | **100 Ry**, single density grid (`NGRIDS 1` on twin) |
| DFT+U | H \(L=0\), \(U-J=4.0\) eV, Mulliken |
| k (A1) | explicit Γ, weight 1 |
| k (A2) | explicit \(\kappa=(1/4,0,0)\), weight 1 |
| k (A3) | MP \(2\times2\times2\), time-reversal → **4** k |
| ATV SCF | OT OUTER_SCF vs DIIS twin (`mix=0.35`, `tol=1e-7`) |
| CP2K twin SCF | DIAGONALIZATION + DIRECT_P_MIXING (no OT — version abort) |

**Pinned quantities (first-green ATV, 2026-07-26):**

| Quantity | Atomiverse / pin | Twin / notes | Tol | Gate |
| -------- | ---------------: | ------------ | --- | ---- |
| A1 \(E_{\mathrm{OT+U}}\) (Γ multi-k path) | −1.0867759 Ha | DIIS+U −1.0867757 Ha | \(\lvert\Delta\rvert\le 10^{-4}\) Ha | OT ≡ DIIS |
| A1 \(e_{\mathrm{dftu}}\) | non-trivial; OT vs DIIS | — | \(\lvert\Delta e_U\rvert\le 10^{-3}\) Ha | A1 |
| A2 \(E_{\mathrm{OT+U}}\) (\(\kappa=\frac14\)) | **−1.081998145** Ha | DIIS+U −1.081997919 Ha (\(\lvert\Delta\rvert\sim 2.3\times10^{-7}\)) | \(10^{-4}\) Ha | OT ≡ DIIS |
| A3 \(E(U=0)\), OT MP\(2^3\) | −1.102850837 Ha | activation baseline | — | diagnostic |
| A3 \(E(U=4)\), OT MP\(2^3\) | **−1.077477969** Ha | frozen first-green | \(10^{-3}\) Ha | absolute pin |
| A3 \(e_{\mathrm{dftu}}\) | \(\approx -0.0023\) Ha | — | \(\lvert e_U\rvert > 10^{-8}\) | non-trivial U |
| A3 \(\lvert E(U)-E(0)\rvert\) | \(\approx 0.0254\) Ha | — | \(\ge 10^{-4}\) Ha | activation |
| A3 \(n_{\mathrm{kpts}}\) | 4 | TR-folded \(2\times2\times2\) | exact | mesh |

**Decisive policy:** A1–A2 OT ≡ DIIS identities are authoritative. A3 freezes
the absolute multi-k OT+U energy for regression drift detection. CP2K does
**not** supply OT+k absolute digits on 2026.1; the DIIS multi-k PLUS_U input
documents the fair **fixed-point** twin chemistry only.

Offline oracle sketch (operator host with CP2K; not ctest):

```bash
export LD_LIBRARY_PATH=$CP2K_ROOT/lib:$LD_LIBRARY_PATH
export CP2K_DATA_DIR=$CP2K_ROOT/share/cp2k/data
cd /tmp && ln -sf $CP2K_DATA_DIR/{BASIS_SET,GTH_POTENTIALS} .
# copy tools/cp2k_oracle/inputs/ot-dftu-kpoints/h2_ot_dftu_kpoints.inp
OMP_NUM_THREADS=8 cp2k.psmp -i h2_ot_dftu_kpoints.inp -o diis_k_u.out
# Expect DIAG path; OT+k would abort on CP2K 2026.1
```

## Performance

Source: `regressions/gpw/ot-dftu-kpoints/REFERENCE.md` and
the case `REFERENCE.md` (wave 8). Gate: **`perf_exempt`**.

| Twin | Wall [s] | \(r = T_{\mathrm{ATV}}/T_{\mathrm{CP2K}}\) | ≤10× | Caveat |
| ---- | -------: | ----------------------------------------: | ---- | ------ |
| CP2K OT + multi-k + U | n/a | n/a | exempt | CP2K 2026.1 aborts OT with k-points |
| CP2K DIIS multi-k + U | (offline twin) | **not** ratioed | — | fixed-point twin only, not OT wall |
| ATV A1–A3 (Release) | ≈ **15–25** | n/a (exempt) | — | e2e ≤60 s budget |

**Why exempt.** Fair OT wall requires a CP2K OT+k twin. That path does not
exist on the pinned oracle version. DIIS multi-k wall is a different
algorithm and must not be ratioed against ATV OT. Same exemption pattern as
`ot-kpoints` / `ot-uks-kpoints`.

## Limitations

- **Closed-shell multi-k only** for OT+DFT+U. Multi-k UKS OT+U is out of
  scope (`gpw_solver` throws / not wired on that combo).
- **No absolute CP2K OT+k pin** until an oracle version supports OT with
  k-points; drift control uses frozen ATV absolute + OT≡DIIS.
- **Toy H \(L=0\) U** on DZV PADE — chemistry foothold for the SCF combo, not
  a correlated-solid pin (contrast Ni \(d\) on [gpw-dftu-nio](gpw-dftu-nio.md)).
- **Single density grid, 100 Ry** — milli-Ha class absolute vs a future
  multigrid CP2K OT twin is not claimed.
- **Forces / stress** with multi-k OT+U are not this regression’s product
  gate (see `gpw-dftu-kpoint-stress` for multi-k DIIS+U stress).
- **PW / CUDA** have no multi-k OT+U product switch.
- **Speed:** `perf_exempt`; do not advertise a ≤10× ratio for this slug.

## References

- Offline pin + gates:
  [`regressions/gpw/ot-dftu-kpoints/REFERENCE.md`](../../../regressions/gpw/ot-dftu-kpoints/REFERENCE.md)
- Regression consumer:
  [`reg_ot-dftu-kpoints.cpp`](../../../regressions/gpw/ot-dftu-kpoints/reg_ot-dftu-kpoints.cpp)
- CP2K DIIS multi-k PLUS_U twin:
  [`tools/cp2k_oracle/inputs/ot-dftu-kpoints/`](../../../tools/cp2k_oracle/inputs/ot-dftu-kpoints/)
- Oracle contract: [`tools/cp2k_oracle/README.md`](../../../tools/cp2k_oracle/README.md)
- Roadmap: [`docs/roadmap/dft.md`](../../roadmap/dft.md) §10 (multi-k OT+DFT+U ✓)
- Related pages: [ot-minimizer-water](ot-minimizer-water.md),
  [gpw-dftu-nio](gpw-dftu-nio.md), [Ch. 9 GPW](../ch09-gpw.md)
- Closed spec (git history): `.claude/specs/ot-dftu-kpoints/` (deleted at done)
- J. VandeVondele, J. Hutter, *J. Chem. Phys.* **118**, 4365 (2003) — OT
- T. D. Kühne et al., *J. Chem. Phys.* **152**, 194103 (2020) §VII C — CP2K OT
- S. L. Dudarev et al., *Phys. Rev. B* **57**, 1505 (1998) — simplified \(U\)
- G. Lippert, J. Hutter, M. Parrinello, *Mol. Phys.* **92**, 477 (1997) — GPW
