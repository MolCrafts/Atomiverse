# Multi-k GPW OT + Dudarev DFT+U + Fermi–Dirac smearing — H₂ / PADE / DZV-GTH

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Regression | None standing (end-to-end case removed) |
| Public API | `cpu::gpw::GpwDft` members `use_ot`, `plus_u`, `smearing`, `degauss` [Ry], `extra_bands`, `k_points` / `k_point_weights` |
| Roadmap | `docs/roadmap/dft.md` §8/§10 — multi-k RKS **OT+DFT+U+smear** ✓ |

## Summary

This page is the product foothold for the **triple combo** on closed-shell
**multi-k GPW**: Orbital Transformation (OT) SCF, Dudarev DFT+U (`PLUS_U` /
Mulliken), and **Fermi–Dirac** metallic occupations with Mermin free energy
\(F = E_{\mathrm{KS}} - TS\). Public surface is members on the concrete
method `cpu::gpw::GpwDft` — **no** `*Config` bag, **no** `make_*` / `build_*`
factories.

Product path: with `use_ot` + non-empty `plus_u` + `smearing != none`, ATV uses
the free-energy PR+ Armijo OT manifold on the **full AO spectrum** with
fractional band occupations \(f\); \(V_U\) is folded into each \(H_k\) via the
shared multi-k energy builder (`build_k_energy`). The same fixed point is
reached by multi-k DIIS + U + FD on identical grids.

**Decisive acceptance** is internal OT ≡ DIIS free-energy identity (not
absolute CP2K OT+k digits). CP2K **2026.1** hard-aborts OT with k-points and
does not productize OT+SMEAR; the offline twin is multi-k **DIIS** +
FERMI_DIRAC + PLUS_U under CP2K oracle.
Fair wall vs CP2K OT: **`perf_exempt`**.

Related handbook pages: OT alone — [ot-minimizer-water](ot-minimizer-water.md);
DFT+U alone — [gpw-dftu-nio](gpw-dftu-nio.md); PW metallic smear —
[pw-smearing-al](pw-smearing-al.md); narrative GPW — [Ch. 9](../ch09-gpw.md).

## Method

**GPW (Quickstep family).** Wavefunction in contracted Gaussians; density /
Hartree / XC on a uniform real-space grid; GTH pseudopotentials. Grid set by
plane-wave cutoff [Ry]. Citation: Lippert, Hutter, Parrinello, *Mol. Phys.*
**92**, 477 (1997).

**Setting (this pin).** Closed-shell multi-k RKS, PADE LDA, single density
grid, **periodic** cubic cell (not Martyna–Tuckerman). One explicit general
k-point \(\kappa = (1/4,0,0)\) with weight 1. Toy H₂ (2 valence e⁻) is a
numerical combo fixture — not a metal chemistry claim.

**Orbital Transformation (OT).** Occupied (and, under smear, fractionally
occupied) MO coefficients are optimized on the Stiefel-type manifold
\(C^\dagger S C = I\) instead of a dense generalized eigensolve each outer
step. For multi-k OT **without** smear, product code uses OUTER_SCF + fixed-H
OT per \(k\). For **U + smear**, product code uses free-energy PR+ Armijo OT
with fractional occupations over the full AO spectrum. Citation: VandeVondele
& Hutter, *J. Chem. Phys.* **118**, 4365 (2003); Kühne et al., *J. Chem. Phys.*
**152**, 194103 (2020) §VII C.

**Dudarev DFT+U.** Per element and angular momentum \(L\), effective
\(U_{\mathrm{eff}}=U-J\) in **eV** on the public surface (converted to Hartree
with `units::kHartreeInEv` inside). Symmetrized Mulliken occupation on the AO
subspace of atom \(A\) with \(\ell=L\) (CP2K `PLUS_U_METHOD MULLIKEN`):

\[
n^{A\sigma}_{mm'} = \tfrac12\big[(P^\sigma S)_{mm'} + (S P^\sigma)_{mm'}\big].
\]

Energy (sum over atoms / spin; \(U\) in Hartree):

\[
E_U = \sum_{A\sigma}\frac{U_{\mathrm{eff}}}{2}\Big[\mathrm{Tr}\,n^{A\sigma}
  - \mathrm{Tr}(n^{A\sigma} n^{A\sigma})\Big].
\]

Potential on that subspace:

\[
V^{A\sigma}_{mm'} = U_{\mathrm{eff}}\Big(\tfrac12\delta_{mm'} - n^{A\sigma}_{mm'}\Big).
\]

On the multi-k path \(V_U\) enters each \(H_k\); \(E_U\) is BZ-consistent with
the fixed-\(P\) path. Citation: Dudarev et al., *Phys. Rev. B* **57**, 1505
(1998). Implementation: `cpu/gpw/internal/dft_plus_u.h` (not public).

**Fermi–Dirac smearing (Mermin free energy).** With reduced energy
\(x = (\varepsilon-\mu)/\sigma\) and \(\sigma =\) `degauss` converted to the
SCF energy unit:

\[
f(x) = \frac{1}{1+e^{x}}\quad\text{(spinless)},\qquad
f_b\in[0,2]\ \text{(closed-shell band form)}.
\]

\(\mu\) is chosen so \(\sum_{k,b} w_k f_b(\varepsilon_{kb},\mu,\sigma) = N_e\).
`extra_bands` is API parity with CP2K `ADDED_MOS` / PW `extra_bands` (GPW
already diagonalizes the full AO spectrum on the DIIS path). Reported total:

\[
F = E_{\mathrm{KS}} - TS,\qquad
\texttt{GpwResult::e\_total} = F,\quad
\texttt{e\_entropy} = -TS,\quad
\texttt{e\_fermi} = \mu.
\]

Citation: Mermin, *Phys. Rev.* **137**, A1441 (1965).

**DIIS twin path.** `use_ot = false` (default): generalized diagonalization of
\((H_k,S_k)\) + density mixing (`mix`) with the same `plus_u` + FD occupations.
Same self-consistent free-energy fixed point when both converge on identical
grids (gate A2: \(\lvert F_{\mathrm{OT}}-F_{\mathrm{DIIS}}\rvert\le 10^{-4}\) Ha).

**Key exclusions (binding from closed product path / REFERENCE).**

- Multi-k **UKS** OT + DFT+U + smear (throws at `build`; residual; A7).
- Absolute energy twin vs CP2K OT+k (CP2K 2026.1 aborts
  `OT not possible with kpoint calculations`).
- Fair OT wall vs CP2K OT (no OT+k / OT+SMEAR product combo in CP2K 2026.1) —
  **`perf_exempt`**.
- Liechtenstein full \(U,J\); GAPW/PAW projectors; forces/stress under the
  triple combo as a pin; MPI industrial scaling gate.
- CUDA GPW OT / DFT+U / smear.

## Implementation

Public surface only: construct `cpu::gpw::GpwDft`, set members, `build` /
`run` on a `cpu::Frame` (the path used for the recorded values). Header:
`gpw_solver.h`. No `*Config`, no factories.
Inside a Driver the same configured `GpwDft` is bound through
`cpu::gpw::GPWEvaluator(std::move(dft))`; the recorded values call the
method directly.

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
using cpu::gpw::Smearing;

// H2 toy, 8 Bohr cubic, bond ±0.7 Bohr from cell centre along x
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

GpwDft m{{H}};
m.cutoff_ry = 100.0;
m.n_occ = 1;                 // closed-shell RKS, Ne = 2
m.max_iter = 220;
m.tol = 1e-7;
m.mix = 0.35;                // DIIS path; ignored under free-energy OT+smear
m.xc = cpu::gpw::XcKind::PADE;
m.k_points.push_back({0.25, 0.0, 0.0});
m.k_point_weights.push_back(1.0);
m.smearing = Smearing::fermi_dirac;
m.degauss = 0.02;            // Ry (CP2K Te ≈ 3157.75 K)
m.extra_bands = 2;
m.plus_u.push_back({molrs::Element::H, /*l=*/0, /*u_minus_j_ev=*/4.0});
m.use_ot = true;             // product OT + U + FD multi-k RKS

m.build(frame);
const GpwResult& r = m.run(frame);   // also writes frame.meta[kEnergy] [eV]
// r.e_total [Ha] = F (Mermin free energy)
// r.e_dftu [Ha], r.e_fermi [Ha], r.e_entropy [Ha], r.n_elec_occ ≈ 2
// r.converged, r.iters
```

Build configuration:

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
'
```

| Path | Role |
| ---- | ---- |
| `gpw_solver.h` | Public `GpwDft` / `PlusU` / `Smearing` / `GpwResult` |
| `ot.h` | OT parameterization / free-energy PR+ (internal) |
| `dft_plus_u.h` | Dudarev E/V (internal) |
| `src/cpu/gpw/gpw_solver.cpp` | Multi-k OT+U+smear SCF wire-up |
| `tests/cpu/gpw/test_cpu_gpw_ot.cpp` | Unit: OT+U+FD fixed-point notes (not e2e) |
| CP2K oracle | Offline CP2K multi-k DIIS+U+FD twin |

## Validation (vs CP2K)

Source: values recorded when this page was written (no standing regression).
Oracle: offline **CP2K 2026.1** (`cp2k.psmp`); **never** live in ctest. Pin
date: **2026-07-26**.

### System (must match twin + regression)

| Item | Value |
| ---- | ----- |
| System | H₂, charge 0, closed-shell RKS |
| Cell | **8 Bohr** cubic, `PERIODIC XYZ` |
| Coordinates (Bohr) | H (3.3, 4.0, 4.0), H (4.7, 4.0, 4.0) — bond ±0.7 from centre |
| Basis / PP | `DZV-GTH-PADE` / `GTH-PADE-q1` (H) |
| XC | PADE |
| Cutoff | **100 Ry**, single grid (`NGRIDS 1` twin) |
| \(n_{\mathrm{occ}}\) | 1 (\(N_e = 2\)) |
| DFT+U | H \(L=0\), \(U-J=4.0\) eV, Mulliken |
| k | general \(\kappa=(0.25,0,0)\), weight 1 |
| Smear | Fermi–Dirac, `degauss = 0.02` Ry, `extra_bands = 2` |
| OT (ATV) | `use_ot = true` |
| SCF | `tol=1e-7`, `max_iter` 160–220, mix 0.35 (DIIS twin) |

### Accuracy table (hard-coded pins)

Decisive gates are **internal** OT ≡ DIIS free-energy identities on identical
ATV grids. Absolute CP2K OT+k free energy **does not exist** in 2026.1.

| Quantity | Atomiverse / pin | Twin / notes | Tol | Gate |
| -------- | ---------------: | ------------ | --- | ---- |
| \(F_{\mathrm{OT}}\) (U=4, FD) | **−1.081998158198 Ha** | frozen first-green (2026-07-26) | \(10^{-3}\) Ha | A6 |
| \(F_{\mathrm{DIIS}}\) same | −1.081997918765 Ha | same grids as OT | — | A1 |
| \(\lvert F_{\mathrm{OT}}-F_{\mathrm{DIIS}}\rvert\) | **2.4×10⁻⁷ Ha** | fixed-point parity | \(\le 10^{-4}\) Ha | A2 |
| \(E_U\) (OT) | ≈ −8.57×10⁻⁴ Ha | non-trivial; OT vs DIIS \(E_U\) | \(\le 10^{-3}\) Ha | A3 |
| \(N_e=\sum w f\) | **2.0** | valence count | \(10^{-5}\) | A4 |
| \(F(U=0)\) OT+FD | −1.108020857737 Ha | activation baseline | — | A5 |
| \(\lvert F(U)-F(0)\rvert\) | ≈ **0.026 Ha** | U activation floor | \(\ge 10^{-4}\) Ha | A5 |
| Multi-k UKS OT+U+FD | throws at `build` | residual combo | must throw | A7 |
| OT / DIIS iters | 11 / 42 | diagnostic | — | — |

Offline CP2K twin documents the **DIIS** multi-k product counterpart of the
ATV free-energy fixed point (not a fair OT wall twin):

| Item | Value |
| ---- | ----- |
| Code | **CP2K 2026.1** (`cp2k.psmp`) |
| Input | CP2K oracle |
| Method | Quickstep GPW, PADE, `PLUS_U_METHOD MULLIKEN` |
| SCF | `&DIAGONALIZATION` + `&SMEAR METHOD FERMI_DIRAC` + `ADDED_MOS 2` |
| \(T_e\) | 3157.75 K (\(\approx 0.02\) Ry \(k_B T\)) |
| OT+k | **aborts** (`qs_scf_initialization.F`: OT not possible with kpoints) |
| OT+SMEAR | not a CP2K product combo (fixed integer OT occupations) |

```bash
# Operator-only re-pin (never ctest)
export CP2K_DATA_DIR=.../share/cp2k/data
cp2k.psmp -i <CP2K oracle input for this case> \
          -o /tmp/cp2k_oracle_ot_dftu_smear.out
```

## Performance

Source: values recorded when this page was written (no standing regression) +
the case pin notes (wave 9). Gate \(r\le 10\) unless
`perf_exempt`.

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | ------------------------: | ----------------------: | ----: | ---- | ------ |
| Fair OT vs CP2K OT+k+U+smear | n/a | **~0.169** (OT-only Release, OMP=4) | **n/a** | **perf_exempt** | CP2K has no OT+k / OT+SMEAR product path |
| Full recorded check A0–A7 | n/a | **≈ 0.4** (Release) | — | — | includes DIIS twin + activation + UKS throw |

**Exemption reason (binding):** CP2K 2026.1 hard-aborts OT with k-points and
does not combine product OT with SMEAR; the DIIS multi-k twin is not a fair
OT wall comparison. Decisive product gates are the free-energy identities
(same pattern as [ot-dftu-kpoints](ot-dftu-kpoints.md) / [ot-smearing](ot-smearing.md)). Handoff records
**`perf_exempt`**.

## Limitations

- **Combo numerical fixture, not solid-metal chemistry.** H₂ DZV / 100 Ry /
  one general \(k\) exercises the OT+U+FD free-energy path; it is not a
  production metallic DFT+U claim.
- **Absolute energy is ATV-frozen (A6), not CP2K OT-matched.** CP2K absolute
  OT+k digits do not exist in 2026.1; the twin only documents DIIS+U+FD
  settings.
- **Multi-k UKS OT + U + smear** remains out of product (A7 throws). Sibling
  landings cover UKS OT+smear ([ot-uks-smearing](ot-uks-smearing.md)) and multi-k UKS OT+U integer
  ([ot-uks-dftu](ot-uks-dftu.md)) separately.
- **Integer multi-k OT+U** (no smear) uses OUTER_SCF + fixed-H OT
  ([ot-dftu-kpoints](ot-dftu-kpoints.md)); free-energy PR+ Armijo is the U+smear path — do not
  conflate the two algorithm sketches.
- **No forces / stress / BOMD** acceptance on this triple combo.
- **Single-rank** microcase; no MPI OT scaling claim.
- **Speed:** `perf_exempt` — do not invent a ratio against the DIIS twin.

## References

- Offline pin + walls: values recorded on this page (no standing regression)
- CP2K twin:
  CP2K oracle
- Oracle contract: CP2K oracle
- Roadmap: [`docs/roadmap/dft.md`](../../roadmap/dft.md)
- Related pages: [ot-minimizer-water](ot-minimizer-water.md),
  [gpw-dftu-nio](gpw-dftu-nio.md), [pw-smearing-al](pw-smearing-al.md)
- Narrative GPW: [`docs/tutorial/ch09-gpw.md`](../ch09-gpw.md)
- J. VandeVondele, J. Hutter, *J. Chem. Phys.* **118**, 4365 (2003) — OT
- S. L. Dudarev et al., *Phys. Rev. B* **57**, 1505 (1998) — simplified \(U\)
- N. D. Mermin, *Phys. Rev.* **137**, A1441 (1965) — free-energy DFT
- G. Lippert, J. Hutter, M. Parrinello, *Mol. Phys.* **92**, 477 (1997) — GPW
- T. D. Kühne et al., *J. Chem. Phys.* **152**, 194103 (2020) §VII C — CP2K OT
