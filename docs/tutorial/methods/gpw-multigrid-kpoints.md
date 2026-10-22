# Multi-k GPW RKS with REL_CUTOFF multigrid — diamond Si / PBE / DZVP-MOLOPT-GTH

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `gpw-multigrid-kpoints` |
| Public API | `cpu::gpw::GpwDft` (`k_mesh`, `k_time_reversal`, `ngrids`, `rel_cutoff`, `cutoff_ry`, `eps_default`); lookup ctors `GaussianBasis` / `GthPotential` |
| Roadmap | `docs/roadmap/dft.md` — k-point GPW × multigrid product collocation |

## Summary

**Closed-shell GPW Kohn–Sham DFT (PBE) with Monkhorst–Pack k-point sampling
and CP2K-style `CUTOFF`/`REL_CUTOFF` multigrid product collocation** — the
combination the k-point branch used to reject at `GpwDft::build`
(`"multigrid (ngrids > 1)"`). Γ-point multigrid
([`gpw-uks-multigrid`](gpw-uks-multigrid.md)) and single-grid k-point GPW
(`gpw-kpoints-si`) already shipped separately; this page opens the product of
the two: each Gaussian pair product is assigned to a grid level by its
most-diffuse exponent, and under Bloch sampling its ket-image shift becomes a
phase index, so per-level densities are built from a k-weighted real-space
stack, prolonged spectrally to the fine grid for Poisson + XC, and the fine
potential restricted back per level into per-cell stacks phase-summed into
Hermitian per-k KS matrices.

This page pins **diamond Si (4-atom BCT cell), MP 2×2×2 time-reversal-folded,
CUTOFF 150 Ry, NGRIDS 4, REL_CUTOFF 60 Ry** against offline **CP2K 2026.1
Quickstep**. Measured ATV energy is within **4.73e-7 Ha** of the pin (gate
tolerance 1e-3 Ha). Fair wall ratio
\(r = T_{\mathrm{ATV}}/T_{\mathrm{CP2K}} \approx 1.6\) (≤10×). Product gate:
`reg_gpw-multigrid-kpoints`.

Related: [Ch. 9 GPW](../ch09-gpw.md); Γ multigrid siblings
`gpw-uks-multigrid`; single-grid k-point sibling `gpw-kpoints-si` and
[`gpw-kpoint-forces`](gpw-kpoint-forces.md).

## Method

**GPW (Quickstep family).** AO wavefunction in contracted Gaussians; density
collocated on real-space grids; Hartree via FFT Poisson; XC on the grid;
potential integrated back into the AO basis; core electrons via analytic GTH
pseudopotentials. Citation: Lippert, Hutter, Parrinello, *Mol. Phys.* **92**,
477 (1997).

**Multigrid product collocation (`&MGRID`).** `ngrids` geometrically spaced
levels; finest = `CUTOFF`, level assignment governed by `REL_CUTOFF`: each
primitive-pair product goes to the coarsest level that still resolves its
exponent. Per-level densities are **prolonged spectrally (FFT)** to the fine
grid; Hartree + XC run there; the fine potential is **restricted** (exact
adjoint) back to each level for KS integration — so the KS matrix stays the
gradient of the collocated energy (variational SCF).

**Bloch sampling over the same product list.** The geometry-fixed
primitive-pair product list is the one the Γ multigrid uses; under a k mesh,
each product's periodic ket-image shift **t** becomes a phase index:

- density weight per product from the real-space stack
  \(D(t) = \sum_k w_k \,\mathrm{Re}\!\left[P^k e^{-2\pi i\,\kappa\cdot t}\right]\)
  (Γ limit: \(D(t) = P\));
- integrated potential lands in real per-cell stacks \(V_t\), phase-summed
  per k into the Hermitian grid KS matrix
  \(V^k = \sum_t e^{+2\pi i\,\kappa\cdot t}\, V_t\).

**Time reversal.** `k_time_reversal = true` folds the MP 2×2×2 mesh (8 k) to
4 k with doubled weights; the full-mesh vs folded identity was measured
machine-exact at bring-up (|Δ| ≤ 3e-14 Ha; see REFERENCE).

**Product screening is mass-aware.** Screening whole products at
`eps_default` as a *value* threshold discards integrated mass up to
\((\pi/p)^{3/2}\approx 70\times\) that value per product, and a periodic cell
holds thousands of image pairs at threshold — on this system the density
integrated to 15.910 of 16 e⁻ and the SCF sat 4.1e-2 Ha off single grid.
Fixed by the derived screen margin `kProductMassMargin = 1e-4`
(`internal/multigrid.cpp`, Γ and k paths alike).

**Key exclusions (spec scope-out).**

- RKS Γ/k closed shell only on this regression — no UKS-multigrid-k claim.
- Nonlocal vdW (`vdw_df` / `vdw_df2` / `rvv10`) on the k path is rejected
  fail-fast at `build()` (found silently dropped while opening this gate).
- No hybrids / meta-GGA / GAPW; no k-point forces/stress claim on this page.
- Single-rank pin; no MPI scaling claim.

## Implementation

Public surface only (concrete method members; **no** `*Config` bag, **no**
`make_*` factories). Minimal call sequence, matching the regression's main
run:

```cpp
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/molrs.hpp"

using namespace ATV;

// Diamond Si, BCT 4-atom setting: a = 5.43 Å,
// cell = (a/√2, a/√2, a), fractional (0,0,0) (½,½,½) (½,0,¼) (0,½,¾)
const double a = 5.43;                         // Å
const double L[3] = {a / std::sqrt(2.0), a / std::sqrt(2.0), a};
const double frac[4][3] = {{0, 0, 0}, {0.5, 0.5, 0.5}, {0.5, 0, 0.25}, {0, 0.5, 0.75}};
cpu::Frame frame(4);
for (int i = 0; i < 4; ++i) {
    frame.span(kAtom_type)[i] = 14;            // Si
    frame.span(kAtom_x)[i] = frac[i][0] * L[0];
    frame.span(kAtom_y)[i] = frac[i][1] * L[1];
    frame.span(kAtom_z)[i] = frac[i][2] * L[2];
}
frame.box = cpu::Box::orthorhombic(L[0], L[1], L[2]);

cpu::gpw::GpwDft m{{cpu::gpw::Kind{
    molrs::Element::Si,
    cpu::gpw::GaussianBasis(molrs::Element::Si, "DZVP-MOLOPT-GTH"),
    cpu::gpw::GthPotential(molrs::Element::Si, "GTH-PBE-q4")}}, 150.0, cpu::gpw::XcKind::PBE};  // CP2K &MGRID CUTOFF
m.ngrids = 4;          // CP2K NGRIDS
m.rel_cutoff = 60.0;   // CP2K REL_CUTOFF [Ry]
m.n_occ = 8;           // 16 valence e⁻, closed shell
m.max_iter = 60;
m.tol = 1e-8;
m.eps_default = 1e-7;
m.k_mesh[0] = m.k_mesh[1] = m.k_mesh[2] = 2;   // MONKHORST-PACK 2 2 2
m.k_time_reversal = true;                      // 4 k, doubled weights

m.build(frame);
const auto& r = m.run(frame);   // r.e_total [Ha], r.converged, r.iters,
                                // r.k_frac / r.k_weights / r.eps_k
```

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
  cmake --build build --target reg_gpw-multigrid-kpoints -j$(nproc)
  reg_gpw-multigrid-kpoints
'
```

Relevant sources:

| Piece | Path |
| ----- | ---- |
| Concrete method (`k_mesh` / `ngrids` / `rel_cutoff`) | `gpw_solver.h`, `src/cpu/gpw/gpw_solver.cpp` |
| Multigrid levels / D(t) stacks / prolong / restrict | `multigrid.h`, `src/cpu/gpw/internal/multigrid.cpp` |
| Unit tests (mechanisms, not e2e) | `tests/cpu/gpw/test_cpu_gpw_multigrid.cpp` (`GpwMultigridCells.*`: Γ-limit, D/V_t phase plumbing, per-cell adjointness, TR folding) |
| CP2K twin input | CP2K oracle |

## Validation (CP2K)

Source: case `gpw-multigrid-kpoints` pin (offline pin;
**never** live CP2K in ctest). Engine: **CP2K 2026.1** (`cp2k.psmp`), module
`CP2K/2026.1` on Arrhenius. Pin date: **2026-07-31**.

| Quantity | CP2K pin | Atomiverse gate | Tol | Notes |
| -------- | -------: | --------------- | --- | ----- |
| \(E_{\mathrm{tot}}\) [Ha], NGRIDS 4 | **−15.802416412503845** | M5 \(\lvert E_{\mathrm{ATV}}-E_{\mathrm{CP2K}}\rvert\) | **1e-3 Ha** | measured **4.73e-7 Ha**; ATV −15.8024168854 Ha, 7 DIIS iters |
| \(E_{\mathrm{tot}}\) [Ha], NGRIDS 1 (context) | −15.802416412655166 | — | — | CP2K's own mg↔sg gap 1.5e-10 Ha: multigrid = cost optimization, not physics |
| mg vs sg (ATV, 40 Ry / REL 20) | — | M3 \(\lvert E_{\mathrm{mg}}-E_{\mathrm{sg}}\rvert\) | 1e-3 Ha | measured 1.13e-5 Ha |
| SCF | converged | M2 / M4 `r.converged` + finite + k payload shapes | — | 4 TR k-points, weights ¼ |

**Geometry / protocol (must match twin).**

| Item | Value |
| ---- | ----- |
| Crystal | diamond Si, BCT 4-atom cell (orthorhombic-compatible fcc setting) |
| Lattice | a = 5.43 Å (10.26120 Bohr); cell = (a/√2, a/√2, a) |
| Atoms (fractional) | (0,0,0), (½,½,½), (½,0,¼), (0,½,¾) |
| Basis / PP | DZVP-MOLOPT-GTH / GTH-PBE-q4 (16 valence e⁻ → n_occ = 8) |
| XC | PBE |
| Multigrid | **CUTOFF 150 Ry, NGRIDS 4, REL_CUTOFF 60 Ry** (ATV + CP2K) |
| k mesh | MONKHORST-PACK 2×2×2, TR-folded → 4 k, w = ¼ |
| SCF | DIIS depth 6, tol 1e-8 Ha, `EPS_DEFAULT 1e-7` |

Key CP2K log line (from REFERENCE):

```text
ENERGY| Total FORCE_EVAL ( QS ) energy [hartree]     -15.802416412503845
```

(SCF converged in 13 steps on the CP2K side.)

Bring-up exactness identities (measured once, then unit-pinned):

| Identity | Measured |
| --- | --- |
| Γ-limit: explicit k = (0,0,0) mg k-path vs Γ mg product path (95 Ry) | 2.0e-14 Ha |
| Time reversal: full 8k vs TR-folded 4k (95 / 60 / 40 Ry mg) | ≤ 2.8e-14 Ha |
| mg vs sg at 150 Ry / REL 60 | 2.28e-5 Ha |

Offline re-run (operator only; not ctest):

```bash
module load buildtool-easybuild/5.2.1-hpca3ef7d197 GCC/14.3.0 MPICH/4.3.2 CP2K/2026.1
export CP2K_DATA_DIR=$EBROOTCP2K/share/cp2k/data
ln -sf $CP2K_DATA_DIR/{BASIS_MOLOPT,GTH_POTENTIALS} .
# copy <CP2K oracle input for this case>
OMP_NUM_THREADS=4 mpirun -np 1 cp2k.psmp -i si_kpoints_mg.inp -o si_kpoints_mg.out
```

## Performance

Source: same pin notes (2026-07-31). Not a ctest gate; offline twin
under CP2K oracle.

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r = T_{\mathrm{ATV}}/T_{\mathrm{CP2K}}\) | ≤10× | Caveat |
| ---- | ------------------------: | -----------------------: | ---------------------------------------: | ---- | ------ |
| Si BCT4 PBE, MP 2×2×2 TR, 150 Ry, NGRIDS 4 | **23.6** | **38.6** | **1.6** | ✓ | CP2K: 1 rank × OMP 4; ATV: Release, OMP 8, main MP run alone; whole regression (3 SCFs) ≈ 55 s at OMP 8 |

Gate: \(r \le 10\) or `perf_exempt`. Here \(r \approx 1.6\) — pass without
exemption.

## Limitations

- **eps-ladder residual:** the mg↔sg gap is the REL_CUTOFF product-screen
  residual (+3.5e-5 Ha at `eps_default 1e-7`, +2.6e-6 at 1e-9), mHa-class
  budget — not CP2K's 1e-10-class internal gap. Tightening `eps_default`
  converges it.
- **Do not weaken the mass-aware screen:** reverting `kProductMassMargin`
  to a raw value threshold silently drops ~0.09 e⁻ of density in periodic
  cells (4e-2 Ha error) while all MT-isolated multigrid regressions stay
  green.
- **Nonlocal vdW on the k path** is rejected at `build()` — previously it
  silently ran semilocal-only. Re-open only with its own pin.
- **RKS / Γ+k energy only** here: k-point multigrid forces/stress and UKS×k
  multigrid are separate queue rows.
- Full-SCF TR / Γ-limit twins are **not** re-run in the regression (≈15 s
  per multigrid SCF; ≤60 s budget goes to the pin) — mechanisms live in
  `tests/cpu/gpw/test_cpu_gpw_multigrid.cpp`.

## References

- Offline pin: case `gpw-multigrid-kpoints` pin
- Regression consumer: `reg_gpw-multigrid-kpoints.cpp`
- CP2K twin: CP2K oracle
- Oracle contract: CP2K oracle
- Siblings: [`gpw-uks-multigrid.md`](gpw-uks-multigrid.md) (Γ multigrid),
  `gpw-kpoints-si` regression, [`gpw-kpoint-forces.md`](gpw-kpoint-forces.md)
- Narrative GPW: [`docs/tutorial/ch09-gpw.md`](../ch09-gpw.md)
- G. Lippert, J. Hutter, M. Parrinello, *Mol. Phys.* **92**, 477 (1997) — GPW
- H. J. Monkhorst, J. D. Pack, *Phys. Rev. B* **13**, 5188 (1976) — MP k sampling
- J. VandeVondele, J. Hutter, *J. Chem. Phys.* **127**, 114105 (2007) — MOLOPT
- CP2K Quickstep `&MGRID` CUTOFF / REL_CUTOFF / NGRIDS; `&KPOINTS SCHEME MONKHORST-PACK`
