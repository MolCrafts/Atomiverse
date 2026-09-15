# GPW analytic forces at k-points — H₂ / PADE / DZV-GTH-PADE (periodic, MP 1×1×2)

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `gpw-kpoint-forces` |
| Regression | `regressions/gpw/gpw-kpoint-forces/` |
| Public API | `cpu::gpw::GpwDft` (`k_mesh`); `gpw_forces_analytic` (RKS Bloch path) |
| Roadmap / queue | `docs/roadmap/dft.md` — multi-k forces; queue **done** 2026-07-25 (`perf_exempt`) |

## Summary

Closed-shell **periodic Gaussian-and-plane-waves (GPW)** Kohn–Sham DFT with
**Monkhorst–Pack k-point sampling** and **analytic ionic forces** on a
compressed H₂ bond. One SCF on the complex Bloch path builds
\(S_k\), \(T_k\), nonlocal PP, and grid \(\chi^k\); one gradient pass
assembles Hellmann–Feynman + Pulay terms k-summed with weights, reusing
occupied MO coefficients from `GpwResult::C_occ_k`.

Public surface: construct `cpu::gpw::GpwDft`, set members (`k_mesh`,
`cutoff_ry`, `xc`, `local_pp_on_grid`, …), `build` / `run` for energy, then
`gpw_forces_analytic(atoms, cfg)` for forces [Ha/Bohr]. No public `*Config`,
no `make_*` / `build_*` factories. Product gate:
`reg_gpw-kpoint-forces`. Decisive correctness is **analytic vs finite-difference**
identity on the k-path energy (max residual \(5.98\times 10^{-6}\) Ha/Bohr);
absolute first-green energy/force digits freeze the regression. Fair wall vs
CP2K is **`perf_exempt`** (no same-basis CP2K twin: in-tree `DZV-GTH-PADE` is
not in CP2K `BASIS_MOLOPT`).

Related: Γ forces [`gpw-forces-water`](gpw-forces-water.md); k-mesh energy
`gpw-kpoints-si`; k-stress `gpw-kpoint-stress`; narrative [Ch. 9 GPW](../ch09-gpw.md).

## Method

**GPW Kohn–Sham (Quickstep)** (Lippert, Hutter, Parrinello 1997/1999;
VandeVondele et al., *Comput. Phys. Commun.* **167**, 103 (2005)). Density
matrix in contracted Gaussians; collocate \(\rho\) on a uniform grid set by
\(E_{\mathrm{cut}}\); FFT Hartree + grid XC; integrate back to AO; diagonalize
and mix until residual \(\le\) `tol`. Cores via GTH PP; product forces keep
`local_pp_on_grid = true` so energy and gradient share the collocated local
short-range PP.

**k-point RKS.** Non-Γ sampling uses Monkhorst–Pack (CP2K
`&KPOINTS SCHEME MONKHORST-PACK`):

\[
\kappa_d = \frac{2r - N_d - 1}{2 N_d},\qquad r=1,\ldots,N_d
\]

so even \(N_d\) is a Γ-avoiding shifted mesh. Optional **time-reversal** fold
(`k_time_reversal = true`, product default) maps \(\pm\mathbf{k}\) pairs and
doubles the retained weight. Explicit fractional lists (`k_points` +
`k_point_weights` summing to 1) override the mesh. Matrices are complex and
Bloch-phased; density and forces are Brillouin-zone sums

\[
E = \sum_{\mathbf{k}} w_{\mathbf{k}} E_{\mathbf{k}},\qquad
\mathbf{F}_A = \sum_{\mathbf{k}} w_{\mathbf{k}} \mathbf{F}_A(\mathbf{k}).
\]

This pin: `k_mesh = {1,1,2}` → after TR, one independent \(\kappa_z = \pm 1/4\)
class with total weight 1.

**Analytic ionic forces (k-RKS product path):**

\[
\mathbf{F}_A
=
-\frac{\partial E}{\partial \mathbf{R}_A}
=
\mathbf{F}_A^{\mathrm{HF}}
+ \mathbf{F}_A^{\mathrm{Pulay}}
+ \mathbf{F}_A^{\mathrm{grid/core}},
\]

with per-\(\mathbf{k}\) density \(P_k\) and energy-weighted \(W_k\) rebuilt from
`C_occ_k` after SCF. Hellmann–Feynman collects explicit \(\mathbf{R}_A\)
dependence of \(T\), nonlocal GTH, and grid/core terms; Pulay is
\(W\cdot\partial S/\partial R_A\). AO center derivatives use the standard
Gaussian raising/lowering identity; operator-center derivatives follow
translational invariance. At SCF stationarity this is \(-\nabla_{\mathbf{R}}E\)
of the **same** multi-k energy functional (proved by central FD, not by
live CP2K digit match).

**Twin settings (binding for the regression):**

| Knob | Value |
| ---- | ----- |
| System | Periodic H₂, bond compressed along \(x\) |
| Positions [Bohr] | H1 \((5.5, 6.0, 6.0)\), H2 \((6.5, 6.0, 6.0)\) |
| Cell | Cubic 12 Bohr, `PERIODIC XYZ` (`nonperiodic = false`) |
| XC | PADE LDA (`XcKind::PADE`) |
| Basis | DZV-GTH-PADE (H) — in-tree static table |
| PP | GTH-PADE-q1 |
| Cutoff | 100 Ry (single grid) |
| k mesh | MONKHORST-PACK \(1\times 1\times 2\), TR on |
| SCF | `tol` \(1\times 10^{-8}\), `max_iter` 60, `n_occ = 1` |
| `local_pp_on_grid` | **true** |
| Observable | \(E_{\mathrm{tot}}\) [Ha]; \(\mathbf{F}\) [Ha/Bohr] |

**Key exclusions (non-goals of this slug)**

- k-point **UKS** forces (k-UKS energy is `gpw-kpoints-uks`; forces follow-on).
- Cell **stress** at k (`gpw-kpoint-stress`); Γ MT water forces
  (`gpw-forces-water`).
- Hybrid / meta-GGA / GAPW / OT / DFT+U / multigrid on the k-force path
  (`build()` rejects OT, DFT+U, multigrid, nonperiodic for multi-k).
- Absolute same-basis CP2K ENERGY_FORCE digit match (CP2K `BASIS_MOLOPT`
  lacks DZV-GTH-PADE; exploratory MOLOPT pins differ by basis/geometry).
- Live CP2K (or any third-party) inside ctest.
- Full 6N FD inside the ≤ 60 s e2e binary (offline FD only).
- Public `*Config` bags / factory constructors.

## Implementation

Public / product surface used by the regression (`GpwDft` members +
lookup constructors + `gpw_forces_analytic`; no `*Config`):

```cpp
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/cpu/gpw/internal/gpw_forces.h"
#include "atomiverse/molrs.hpp"

using namespace ATV::cpu::gpw;

const auto basis = GaussianBasis(molrs::Element::H, "DZV-GTH-PADE");
const auto pot = GthPotential(molrs::Element::H, "GTH-PADE-q1");

const double rA[3] = {5.5, 6.0, 6.0};
const double rB[3] = {6.5, 6.0, 6.0};  // 1.0 Bohr compressed bond
std::vector<GpwAtom> atoms = {GpwAtom(rA, basis, pot, 1),
                              GpwAtom(rB, basis, pot, 1)};

GpwDft m;
m.cell[0] = m.cell[1] = m.cell[2] = 12.0;  // Bohr
m.cutoff_ry = 100.0;
m.n_occ = 1;
m.tol = 1e-8;
m.max_iter = 60;
m.nonperiodic = false;
m.local_pp_on_grid = true;
m.xc = XcKind::PADE;
m.eps_default = 1e-8;
m.k_mesh[0] = 1;
m.k_mesh[1] = 1;
m.k_mesh[2] = 2;

m.build(atoms);
const GpwResult er = m.run(atoms);  // er.e_total [Ha], er.n_kpts, er.C_occ_k

// Forces: one SCF + one gradient (or pass a matching cfg after energy)
const auto F = gpw_forces_analytic(atoms, /*cfg with same members*/ m_copy);
// F[a][0..2] in Ha/Bohr; Frame path (if used) still reports eV/Å on kAtom_f*
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
  cmake --build build -j$(nproc) --target atomiverse reg_gpw-kpoint-forces
  ./build/regressions/reg_gpw-kpoint-forces
'
```

| Path | Role |
| ---- | ---- |
| `include/atomiverse/cpu/gpw/gpw_solver.h` | `GpwDft` (`k_mesh`, `k_points`, …), `GpwResult` (`C_occ_k`) |
| `include/atomiverse/cpu/gpw/internal/gpw_forces.h` | `gpw_forces_analytic` / `gpw_forces_from_scf` |
| `include/atomiverse/cpu/gpw/internal/kpoints.h` | Monkhorst–Pack, Bloch matrices, complex \(\chi^k\) |
| `include/atomiverse/cpu/gpw/gaussian_basis.h` | `GaussianBasis(Element, name)` lookup |
| `include/atomiverse/cpu/gpw/gth_potential.h` | `GthPotential(Element, name)` lookup |
| `src/cpu/gpw/gpw_solver.cpp` | k-point RKS SCF branch |
| `src/cpu/gpw/internal/` | Force assembly (Bloch-phased derivatives) |
| `regressions/gpw/gpw-kpoint-forces/reg_gpw-kpoint-forces.cpp` | Product gate |
| `regressions/gpw/gpw-kpoint-forces/REFERENCE.md` | Offline pins + FD identity |
| `tools/cp2k_oracle/inputs/gpw-kpoint-forces.inp` | CP2K ENERGY_FORCE twin (offline only) |
| `tests/cpu/gpw/test_cpu_gpw_forces.cpp` | Unit k-force / Γ-identity (not e2e) |

## Validation (vs CP2K / FD)

Source:
[`regressions/gpw/gpw-kpoint-forces/REFERENCE.md`](../../../regressions/gpw/gpw-kpoint-forces/REFERENCE.md).
Oracle rules: offline pins only (`tools/cp2k_oracle/README.md`); **never** live
CP2K from ctest. Absolute same-basis CP2K digits are **not** the product gate
for this slug (basis mismatch); decisive gate is analytic \(\equiv\) FD of the
k-path energy, plus frozen first-green ATV bands and Newton-3 / symmetry.

**Setup:**

| Item | Value |
| ---- | ----- |
| Engine (ATV) | GPW RKS + analytic k-forces (`gpw_forces_analytic`) |
| System | H₂, charge 0, singlet, \(n_{\mathrm{occ}}=1\) |
| Lattice | cubic \(a = 12\) Bohr, periodic |
| Positions [Bohr] | H (5.5, 6.0, 6.0); H (6.5, 6.0, 6.0) |
| Basis / PP | DZV-GTH-PADE / GTH-PADE-q1 |
| XC | PADE |
| Density cutoff | 100 Ry |
| k sampling | MP \(1\times 1\times 2\), time-reversal on |
| Provenance | Quickstep forces (VandeVondele et al. 2005); GTH/MOLOPT family tables |

**Offline correctness (decisive):**

| Check | Result | Notes |
| ----- | -----: | ----- |
| max\|F_analytic − F_fd\| | **\(5.98\times 10^{-6}\)** Ha/Bohr | Related off-symmetry geom.; \(h=10^{-3}\) Bohr; 120 Ry; OMP=8 |
| Explicit-Γ vs real Γ forces | **\(1.1\times 10^{-10}\)** Ha/Bohr | Complex k machinery at \(\kappa=0\) vs `assemble_gpw_forces` |
| \|∑F\| (product geom.) | \(\sim 3\times 10^{-14}\) Ha/Bohr | Exact Newton-3 on compressed bond |

**Atomiverse first-green pins (2026-07-25, product geometry):**

| Quantity | Atomiverse pin | Tol (gate) | Notes |
| -------- | -------------: | ---------- | ----- |
| \(E_{\mathrm{tot}}\) [Ha] | **−1.057153829426** | \(1\times 10^{-5}\) Ha | 8 SCF iters; soft absolute freeze |
| \(F_{0x}\) [Ha/Bohr] | **−0.23541914472** | \(1\times 10^{-5}\) | Restoring (H1 pulls toward H2) |
| \(F_{1x}\) [Ha/Bohr] | **+0.23541914472** | \(1\times 10^{-5}\) | Newton-3 partner |
| \(F_y, F_z\) | \(\sim 0\) | \(\le 10^{-6}\) | Bond along \(x\) |
| \|∑F\| [Ha/Bohr] | \(\le 10^{-6}\) gate | product residual \(\sim 10^{-14}\) | A4 |
| SCF | `converged == true` | hard | A1 |

**CP2K ENERGY_FORCE (exploratory, not absolute gate):**

| Field | Value |
| ----- | ----- |
| Engine | CP2K 2026.1 (Arrhenius offline) |
| Input | `tools/cp2k_oracle/inputs/gpw-kpoint-forces.inp` |
| Note | Twin uses DZVP-MOLOPT-GTH when DZV-GTH-PADE is absent from `BASIS_MOLOPT`; exploratory SZV-MOLOPT pin differs in basis/geometry (\(E\approx -1.026\) Ha, \|F_x\|\(\approx 0.18\) Ha/Bohr). Same-basis absolute twin pending CP2K data that exposes DZV-GTH-PADE. |

Tolerance rationale: FD residual \(\sim 6\times 10^{-6}\) Ha/Bohr sets the
scientific claim; regression force/energy tols \(10^{-5}\) freeze host
reproducibility of the first-green band without claiming chemical absolute
match to a different-basis CP2K deck.

## Performance

Source: [`REFERENCE.md`](../../../regressions/gpw/gpw-kpoint-forces/REFERENCE.md);
handoff / queue **perf_exempt** (2026-07-25).

| Item | Value |
| ---- | ----- |
| `perf_exempt` | **true** |
| Reason | No same-basis CP2K fair wall (DZV-GTH-PADE not in BASIS_MOLOPT). |
| \(T_{\mathrm{ATV}}\) (regression, OMP=8, Release) | ~1 s |
| \(r = T_{\mathrm{ATV}}/T_{\mathrm{CP2K}}\) | **n/a (exempt)** |
| `within_10x` | n/a (exempt) |

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | -----------------------: | ----------------------: | ----: | ---- | ------ |
| H₂ k-force PADE MP 1×1×2 | — | ~1 | **exempt** | n/a | Re-measure when same-basis CP2K data path exists |

Gate when a twin exists: \(r = T_{\mathrm{ATV}}/T_{\mathrm{CP2K}} \le 10\) on the
same host/core count (single-rank). Do not invent a speed ratio.

## Limitations

- **RKS k-forces only** — k-UKS forces and k-stress are separate slugs /
  follow-ons; Γ UKS forces live under `gpw-uks-forces`.
- **Soft absolute energy/force freeze**, not published CP2K \(E\)/\(F\) digit
  match on DZV-GTH-PADE. Decisive proof is analytic \(\equiv\) FD of the
  k-path functional.
- **Tiny H₂ foothold** (100 Ry, \(1\times 1\times 2\)) for ≤ 60 s e2e — not a
  bulk solid force campaign.
- **`perf_exempt`**: no fair \(r\) until a same-basis offline CP2K twin is
  timed on the same host.
- k-path **rejects** nonperiodic MT, OT, DFT+U, and multigrid at `build()`
  (periodic single-grid `local_pp_on_grid` only).
- **CPU only** — CUDA GPW remains out of the CP2K loop scope.
- Frame public forces (evaluator path) are eV/Å; this regression compares
  Ha/Bohr from `gpw_forces_analytic` directly.
- Basis/PP tables are static in-tree headers; unsupported element/name
  **throws** (no silent fallback).

## References

- Offline pin: `regressions/gpw/gpw-kpoint-forces/REFERENCE.md`
- Regression consumer: `regressions/gpw/gpw-kpoint-forces/reg_gpw-kpoint-forces.cpp`
- Oracle twin: `tools/cp2k_oracle/inputs/gpw-kpoint-forces.inp`
- Oracle notes: `tools/cp2k_oracle/README.md`
- Spec history: `.claude/specs/gpw-kpoint-forces/` (closed at done; git on
  `wf/gpw-kpoint-forces`)
- Narrative GPW: [`docs/tutorial/ch09-gpw.md`](../ch09-gpw.md)
- Related method pages: [gpw-forces-water](gpw-forces-water.md) (Γ MT forces),
  [gpw-analytic-stress](gpw-analytic-stress.md) / [gpw-stress-water](gpw-stress-water.md)
  (cell stress), `gpw-kpoints-si` / `gpw-kpoint-stress` regressions
- Public API: `include/atomiverse/cpu/gpw/gpw_solver.h`,
  `include/atomiverse/cpu/gpw/internal/gpw_forces.h`
- J. VandeVondele, M. Krack, F. Mohamed, M. Parrinello, T. Chassaing,
  J. Hutter, *Comput. Phys. Commun.* **167**, 103 (2005) — Quickstep GPW
  forces.
- G. Lippert, J. Hutter, M. Parrinello, *Mol. Phys.* **92**, 477 (1997);
  *Theor. Chem. Acc.* **103**, 124 (1999) — GPW.
- S. Goedecker, M. Teter, J. Hutter, *Phys. Rev. B* **54**, 1703 (1996) — GTH /
  PADE.
- H. J. Monkhorst, J. D. Pack, *Phys. Rev. B* **13**, 5188 (1976) —
  Monkhorst–Pack meshes.
