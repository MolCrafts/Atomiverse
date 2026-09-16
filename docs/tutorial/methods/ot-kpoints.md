# Multi-k Orbital Transformation (OT) SCF — compact H₂ / GPW PADE

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `ot-kpoints` |
| Regression | `regressions/gpw/ot-kpoints/` |
| Public API | `cpu::gpw::GpwDft` members `use_ot`, `k_mesh` / `k_points` + `k_point_weights` |
| Roadmap | `docs/roadmap/dft.md` §10 GPW — OT multi-k ✓ (`reg_ot-kpoints`) |
| Oracle | offline CP2K (`tools/cp2k_oracle/inputs/ot-kpoints/`); **never** live in ctest |

## Summary

**Orbital Transformation (OT)** on the **complex multi-k** GPW path optimizes
occupied Bloch coefficients \(C_k\) on the Stiefel-type manifold
\(C_k^\dagger S_k C_k = I\) without a dense generalized eigensolve + Pulay DIIS
at every outer SCF step. Product surface: public members
`cpu::gpw::GpwDft::use_ot` (default `false`) **combined with** either an explicit
list `k_points` / `k_point_weights` or a Monkhorst–Pack mesh `k_mesh[3]`
(optionally time-reversal folded via `k_time_reversal`). No public `*Config`
bag, no factories.

This page pins the **multi-k OT total energy** for a compact periodic H₂
(DZV-GTH-PADE / GTH-PADE-q1 / PADE / 100 Ry) against **in-tree multi-k DIIS**
on identical grids (exactness identities), with a frozen first-green absolute
band until an offline CP2K OT+KPOINTS energy is available. Product gate:
`regressions/gpw/ot-kpoints` (`reg_ot-kpoints`). Fair wall vs CP2K:
**`perf_exempt`** (no CP2K module/binary on the pin host).

Related product OT surfaces (separate slugs): Γ RKS water
[`ot-minimizer-water`](ot-minimizer-water.md); Γ UKS O₂ `reg_ot-uks-o2`;
OT+DFT+U Γ `reg_ot-dftu`.

## Method

**Setting.** Closed-shell multi-k Gaussian-and-plane-waves (GPW) Kohn–Sham DFT
with Padé LDA, contracted DZV Gaussians, GTH pseudopotentials, and full 3D
periodicity. Wavefunction in the Gaussian AO basis; density / Hartree / XC on a
uniform real-space grid with plane-wave cutoff \(E_{\mathrm{cut}}\) [Ry]. Method
channel reports total energy in **Hartree** (`GpwResult::e_total`); public Frame
energy (when routed through an evaluator) remains eV on `kEnergy`.

**k sampling.** Two product constructions:

1. **Explicit list** — `k_points` as fractional reciprocal coordinates
   (CP2K `&KPOINTS SCHEME GENERAL` / `&KPOINT` with `UNIT B_VECTOR`) plus
   `k_point_weights` with \(\sum_w = 1\) (no implicit renormalization).
2. **Monkhorst–Pack mesh** — `k_mesh = {N_x,N_y,N_z}` with CP2K convention
   \(\kappa_d = (2r - N_d - 1)/(2 N_d)\) (even \(N\) is Γ-avoiding). Optional
   time-reversal fold (`k_time_reversal = true`, default) doubles weights on
   retained partners.

`{1,1,1}` (or a single explicit Γ) keeps the real Γ path; multi-k uses complex
Bloch \(H_k,S_k,C_k\).

**OT parameterization** (VandeVondele & Hutter, *J. Chem. Phys.* **118**, 4365
(2003); Kühne et al., *J. Chem. Phys.* **152**, 194103 (2020) §VII C). For
reference occupied coefficients \(C_0\) with \(C_0^\dagger S C_0 = I\) and an
\(S\)-tangent step \(x\) (\(C_0^\dagger S x = 0\)):

\[
C(x) = C_0\,\cos U + x\,U^{-1}\sin U,\qquad
U = (x^\dagger S x)^{1/2},
\]

which preserves \(C(x)^\dagger S C(x) = I\). Closed-shell band-energy gradient
consistent with \(P = 2\,C C^\dagger\):

\[
g = (I - S C_0 C_0^\dagger)\,(4 H C_0).
\]

On multi-k the same manifold is applied **per k** to complex \(C_k\) with the
corresponding Hermitian \(H_k,S_k\); the outer SCF rebuilds \(H_k\) from the
**k-summed** density and takes a joint preconditioned CG step with Armijo line
search on the total energy until residual / energy criteria meet `tol`. Same
self-consistent fixed point as multi-k DIIS for a gapped closed-shell ground
state on identical grids.

**DIIS twin path.** `use_ot = false` (default): Hermitian generalized
diagonalization of \((H_k,S_k)\) plus density mixing. Regression gates A2/A3
require \(\lvert E_{\mathrm{OT}}-E_{\mathrm{DIIS}}\rvert \le 10^{-4}\) Ha.

**Key exclusions (binding).**

- Multi-k **UKS** OT — fail-fast at `build()` (A4); Γ UKS OT is a separate product
  (`reg_ot-uks-o2`).
- Multi-k OT + **DFT+U** — not wired; throws (use DIIS multi-k DFT+U or Γ OT+U).
- Multi-k OT + **metallic smearing** — integer occupations only; throws.
- Sparse / linear-scaling OT preconditioner ○ (dense inverse cost).
- MPI multi-rank OT industrial scaling gate ○.
- Absolute CP2K digit pin — deferred until offline CP2K is available on a pin
  host (input already shipped; A5 holds a frozen first-green band).

## Implementation

Public surface only (`cpu::gpw::GpwDft` members; no `*Config`, no factories).
Match the regression (compact H₂, 8 Bohr cell, 100 Ry, PADE):

```cpp
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/molrs.hpp"

using namespace ATV;
using cpu::gpw::GpwDft;

const double L = 8.0; // Bohr
const cpu::gpw::GaussianBasis bH(molrs::Element::H, "DZV-GTH-PADE");
const cpu::gpw::GthPotential pH(molrs::Element::H, "GTH-PADE-q1");
std::vector<cpu::gpw::GpwAtom> atoms;
const double r0[3] = {0.5 * L - 0.7, 0.5 * L, 0.5 * L};
const double r1[3] = {0.5 * L + 0.7, 0.5 * L, 0.5 * L};
atoms.emplace_back(r0, bH, pH, 1);
atoms.emplace_back(r1, bH, pH, 1);

// Explicit single k = (1/4, 0, 0) — CP2K GENERAL twin
GpwDft ot;
ot.cell[0] = ot.cell[1] = ot.cell[2] = L;
ot.cutoff_ry = 100.0;
ot.n_occ = 1;
ot.max_iter = 200;
ot.tol = 1e-8;
ot.xc = cpu::gpw::XcKind::PADE;
ot.eps_default = 1e-7;
ot.use_ot = true;
ot.k_points.push_back({0.25, 0.0, 0.0});
ot.k_point_weights.push_back(1.0);
ot.build(atoms);
const auto& ro = ot.run(atoms);
// ro.e_total [Hartree]; ro.converged; ro.n_kpts; ro.k_frac

// Monkhorst–Pack 2×1×1 with time-reversal (alternate construction)
GpwDft mp;
mp.cell[0] = mp.cell[1] = mp.cell[2] = L;
mp.cutoff_ry = 100.0;
mp.n_occ = 1;
mp.max_iter = 200;
mp.tol = 1e-8;
mp.xc = cpu::gpw::XcKind::PADE;
mp.use_ot = true;
mp.k_mesh[0] = 2;
mp.k_mesh[1] = 1;
mp.k_mesh[2] = 1;
mp.k_time_reversal = true;
mp.build(atoms);
const auto& rm = mp.run(atoms);
```

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
  cmake --build build -j$(nproc) --target atomiverse reg_ot-kpoints
  ./build/regressions/reg_ot-kpoints
'
```

| Path | Role |
| ---- | ---- |
| `include/atomiverse/cpu/gpw/gpw_solver.h` | Public `GpwDft::{use_ot,k_mesh,k_points,…}` |
| `include/atomiverse/cpu/gpw/internal/ot.h` | Real OT primitives (internal) |
| `src/cpu/gpw/gpw_solver.cpp` | Multi-k OT SCF branch (complex \(C_k\)) |
| `regressions/gpw/ot-kpoints/` | E2E product gate + `REFERENCE.md` |
| `tools/cp2k_oracle/inputs/ot-kpoints/h2_ot_kpoints.inp` | Offline CP2K OT+KPOINTS twin |

## Validation (vs CP2K / internal exactness)

Source: [`regressions/gpw/ot-kpoints/REFERENCE.md`](../../../regressions/gpw/ot-kpoints/REFERENCE.md).
Oracle: offline **CP2K** (`tools/cp2k_oracle/`); **never** live in ctest.

**Setup:**

| Item | Value |
| ---- | ----- |
| System | Periodic H₂, charge 0, closed shell, \(n_{\mathrm{occ}}=1\) |
| Geometry [Bohr] | H at \((L/2 \pm 0.7,\, L/2,\, L/2)\), \(L=8\) |
| Cell | 8 Bohr cubic, `PERIODIC XYZ` |
| Basis / PP | DZV-GTH-PADE; GTH-PADE-q1 (H) |
| XC | PADE (Padé LDA) |
| Cutoff | **100 Ry**, single grid (`NGRIDS 1`) |
| SCF | `max_iter` up to 200, `tol=1e-8`; OT vs DIIS dual path |
| k (A2 / CP2K twin) | explicit \(\kappa=(1/4,0,0)\), weight 1 |
| k (A3) | MP \(2\times1\times1\) + time-reversal |
| CP2K OT settings | `&OT` `MINIMIZER DIIS`, `PRECONDITIONER FULL_SINGLE_INVERSE`, outer SCF |

**Pinned / measured energies [Hartree] (first green 2026-07-26):**

| Quantity | Atomiverse / pin | CP2K | Tol | Notes |
| -------- | ---------------: | ---: | --- | ----- |
| \(E_{\mathrm{tot}}\) multi-k OT (\(\kappa=1/4\)) | **−1.1080204833** | *pending* | **1×10⁻³** (A5) | frozen first-green band; 13 OT iters |
| \(E_{\mathrm{tot}}\) multi-k DIIS (same grids) | −1.1080204833 | — | — | 5 DIIS iters |
| \(\lvert E_{\mathrm{OT}}-E_{\mathrm{DIIS}}\rvert\) (A2) | \(6.2\times10^{-11}\) | — | ≤ 1×10⁻⁴ | identity gate |
| real Γ OT | −1.1123413990 | — | — | real-path baseline |
| k-path Γ OT | −1.1123413989 | — | ≤ 1×10⁻⁶ vs real (A1) | explicit \(\Gamma\) via k-list |

**Gates (product):**

| Gate | Check | Tol |
| ---- | ----- | ---: |
| A1 | real Γ OT ≡ k-path Γ OT | 1e-6 Ha |
| A2 | multi-k OT ≡ multi-k DIIS (\(\kappa=1/4\)) | 1e-4 Ha |
| A3 | MP \(2\times1\times1\) OT ≡ DIIS | 1e-4 Ha |
| A4 | multi-k UKS OT throws at `build()` | — |
| A5 | absolute vs frozen first-green | 1e-3 Ha |

**Offline CP2K status (2026-07-26):** pin host had no CP2K module/binary
(`module avail cp2k` empty). Absolute CP2K digits are **not invented**. Replace
A5 with a cited CP2K OT+KPOINTS energy when available:

```bash
# host with CP2K module only — never ctest
module load ... CP2K/...
export CP2K_DATA_DIR=.../share/cp2k/data
cp2k.psmp -i tools/cp2k_oracle/inputs/ot-kpoints/h2_ot_kpoints.inp -o ot_k.out
# ENERGY| Total FORCE_EVAL ( QS ) energy [hartree]  <pin>
```

Identity gates A1–A4 remain decisive regardless of absolute CP2K digits.

## Performance

Source: `regressions/gpw/ot-kpoints/REFERENCE.md` and
the case `REFERENCE.md` (wave 7). Queue record:
**`perf_exempt`**.

| Twin | Wall [s] | \(r = T_{\mathrm{ATV}}/T_{\mathrm{CP2K}}\) | ≤10× | Caveat |
| ---- | -------: | ----------------------------------------: | ---- | ------ |
| CP2K OT+KPOINTS | n/a | n/a | n/a | no binary on pin host |
| ATV multi-k OT | n/a | **perf_exempt** | n/a | fair wall deferred |

**Caveats.** (1) Do not fabricate a ratio without a matched offline CP2K twin
wall. (2) Dense OT preconditioner + complex multi-k matrices dominate this
microcase; ratio (when measured) will not forecast sparse/linear-scaling OT.
(3) The regression binary runs several OT+DIIS constructions (A1–A5) and is
**not** a single-path fair twin until split.

## Limitations

- Closed-shell multi-k RKS OT only; multi-k UKS OT, multi-k OT+DFT+U, and
  OT+smearing are fail-fast (product exclusions).
- Absolute energy class vs CP2K is a **frozen first-green** band (1e-3 Ha) until
  offline CP2K re-pin; sub-µHa multigrid parity is not claimed at 100 Ry.
- Dense \(O(n_b^3)\) preconditioner inverse — sparse OT out of scope.
- Forces / stress under multi-k OT are not gated by this slug (use DIIS multi-k
  force/stress product rows: `gpw-kpoint-forces`, `gpw-kpoint-stress`).
- PW backend has no OT product switch (`docs/roadmap/dft.md` OT column: GPW ✓,
  PW ○, CUDA ○).
- Fair wall vs CP2K is **perf_exempt** on this pin host.

## References

- Offline pin + gates: `regressions/gpw/ot-kpoints/REFERENCE.md`
- Regression binary: `regressions/gpw/ot-kpoints/reg_ot-kpoints.cpp`
- CP2K oracle input: `tools/cp2k_oracle/inputs/ot-kpoints/h2_ot_kpoints.inp`
- Closed spec (git history): `.claude/specs/ot-kpoints/` (deleted at done)
- Sibling Γ OT handbook: [`ot-minimizer-water`](ot-minimizer-water.md)
- Narrative GPW chapter: `docs/tutorial/ch09-gpw.md`
- Roadmap: `docs/roadmap/dft.md` §10 (OT multi-k ✓ on GPW)
- VandeVondele, J.; Hutter, J. *J. Chem. Phys.* **118**, 4365 (2003),
  doi:10.1063/1.1543154 (original OT)
- Kühne, T. D. et al. *J. Chem. Phys.* **152**, 194103 (2020) §VII C,
  doi:10.1063/5.0007045 (CP2K / formulation followed in-tree)
