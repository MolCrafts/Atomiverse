# OT + Dudarev DFT+U (Γ-UKS GPW) — NiO molecule / PADE / SZV

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `ot-dftu` |
| Regression | `regressions/gpw/ot-dftu/` |
| Public API | `cpu::gpw::GPWEvaluator` members `use_ot`, `plus_u`, `n_alpha`/`n_beta`; `cpu::single_point` |
| Roadmap | `docs/roadmap/dft.md` §10 GPW — OT + DFT+U ✓ (`reg_ot-dftu`) |

## Summary

**Orbital Transformation (OT)** SCF combined with **Dudarev DFT+U** on **Γ-point
unrestricted GPW** is the product path when a correlated open-shell system needs
both a diagonalization-free occupied-manifold minimizer and a Hubbard \(U_{\mathrm{eff}}\)
correction. Separately, OT (`reg_ot-uks-o2`, `reg_ot-minimizer-water`) and
DFT+U (`reg_gpw-dftu-nio`) were already green; this slug closes the **combo**:
`use_ot=true` with non-empty `plus_u` on UKS no longer throws, and energy /
spin match the offline CP2K OT+`PLUS_U` twin.

Public surface: construct `cpu::gpw::GPWEvaluator` with `Kind` rows (lookup
`GaussianBasis` / `GthPotential`), set `use_ot = true`, UKS occupations, and
`plus_u.push_back({Ni, 2, 6.0})` with \(U-J\) in **eV**, then
`cpu::single_point(frame, eval)`. Total energy on Frame is **eV** (`kEnergy`);
Hartree total, DFT+U contribution, \(\langle S^2\rangle\), and Mulliken spin via
`last_energy_hartree()` / `last_result().{e_dftu,s2,mulliken_spin}`. Product
gate: `reg_ot-dftu`. Fair wall ratio
\(r = T_{\mathrm{ATV}}/T_{\mathrm{CP2K}} \approx 0.26\) (≤10×).

Related: OT alone — [ot-minimizer-water](ot-minimizer-water.md); DFT+U DIIS
foothold — [gpw-dftu-nio](gpw-dftu-nio.md); narrative GPW — [Ch. 9](../ch09-gpw.md).

## Method

**GPW (Quickstep family).** Wavefunction in contracted Gaussians; density /
Hartree / XC on a uniform real-space grid; GTH pseudopotentials. Grid set by
plane-wave cutoff [Ry]. Citation: Lippert, Hutter, Parrinello, *Mol. Phys.*
**92**, 477 (1997).

**UKS / PADE LSD.** Separate \(P^\alpha\), \(P^\beta\); spin-resolved Padé LDA;
integer members `n_alpha` / `n_beta`. For this pin: 24 valence e⁻ (Ni GTH-q18 +
O GTH-q6), multiplicity 3 ⇒ \(N_\alpha=13\), \(N_\beta=11\). Citation:
Goedecker, Teter, Hutter, *Phys. Rev. B* **54**, 1703 (1996).

**Isolated electrostatics.** `nonperiodic = true` ⇔ CP2K `PERIODIC NONE` /
`PSOLVER MT`. Geometry must be **centered** in the cell. Citation: Martyna &
Tuckerman, *J. Chem. Phys.* **110**, 2810 (1999).

**OT parameterization** (same product manifold as RKS/UKS OT). Occupied
coefficients \(C\) stay on \(C^\top S C = I\) without a dense generalized
eigensolve each outer step. Preconditioned CG / DIIS-style OT steps rebuild
\(H\) from the collocated density until residual / energy criteria meet `tol`.
CP2K twin: `&OT` `PRECONDITIONER FULL_SINGLE_INVERSE`, `MINIMIZER DIIS`, plus
`OUTER_SCF`. Citations: VandeVondele & Hutter, *J. Chem. Phys.* **118**, 4365
(2003); Kühne et al., *J. Chem. Phys.* **152**, 194103 (2020) §VII C.

**Dudarev DFT+U (binding).** Per element and angular momentum \(L\), effective
\(U_{\mathrm{eff}}\) in eV on the public surface (converted to Hartree with
`units::kHartreeInEv` inside). Occupation on the AO subspace of atom \(A\) with
\(\ell=L\) (symmetrized Mulliken, CP2K `PLUS_U_METHOD MULLIKEN`):

\[
n^{A\sigma}_{mm'} = \tfrac12\big[(P^\sigma S)_{mm'} + (S P^\sigma)_{mm'}\big].
\]

Energy (sum over atoms and spin channels; \(U\) in Hartree):

\[
E_U = \sum_{A\sigma}\frac{U_{\mathrm{eff}}}{2}\Big[\mathrm{Tr}\,n^{A\sigma}
  - \mathrm{Tr}(n^{A\sigma} n^{A\sigma})\Big].
\]

Potential added to the spin Fock matrix on that subspace:

\[
V^{A\sigma}_{mm'} = U_{\mathrm{eff}}\Big(\tfrac12\delta_{mm'} - n^{A\sigma}_{mm'}\Big).
\]

Matches CP2K simplified rotationally invariant form (Dudarev et al.; CP2K
`U_MINUS_J`). Implementation lives in `cpu/gpw/internal/dft_plus_u.h` (not
public). On the OT path, \(V_U\) is folded into the effective \(H\) that drives
the manifold gradient — same \(E_U\) formula as the DIIS path.

**Combo product claim.** Before this slug, non-empty `plus_u` on the UKS OT
path threw; DIIS+U and OT-without-U each worked. After productization, OT+U and
DIIS+U converge to the same fixed point within **1×10⁻³ Ha** on identical
single-grid settings (regression dual check).

**Key exclusions.**

- Liechtenstein full \(U,J\) tensor; orbital-dependent U ramping.
- Analytic DFT+U forces / stress on the OT path (separate DIIS stress / force
  rows: `gpw-dftu-stress`, `gpw-forces-water`).
- Multi-k OT + DFT+U combo (multi-k DFT+U DIIS ✓ via `gpw-dftu-kpoints`; multi-k
  OT rejects or is separate `ot-kpoints` scope).
- Solid rocksalt AFM NiO product pin; GAPW / PAW projectors; MPI OT scaling.
- PW backend OT+U (GPW only).

## Implementation

Public surface only (`cpu::Frame` + evaluator members; **no** `*Config` bag,
**no** `make_*` / `build_*`):

```cpp
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/evaluate/single_point.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_evaluator.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/molrs.hpp"

using namespace ATV;

// NiO molecule, R = 1.668 Å along z, centered in 10 Å cubic cell (MT)
constexpr double L = 10.0;
constexpr double R = 1.668;
cpu::Frame frame(2);
frame.span(kAtom_type)[0] = 28;
frame.span(kAtom_type)[1] = 8;
frame.span(kAtom_mass)[0] = 58.693;
frame.span(kAtom_mass)[1] = 15.999;
frame.span(kAtom_x)[0] = 0.5 * L;
frame.span(kAtom_y)[0] = 0.5 * L;
frame.span(kAtom_z)[0] = 0.5 * L - 0.5 * R;
frame.span(kAtom_x)[1] = 0.5 * L;
frame.span(kAtom_y)[1] = 0.5 * L;
frame.span(kAtom_z)[1] = 0.5 * L + 0.5 * R;
frame.box = cpu::Box::cubic(L);

cpu::gpw::Kind Ni{molrs::Element::Ni,
                  cpu::gpw::GaussianBasis(molrs::Element::Ni, "SZV-GTH-PADE-q18"),
                  cpu::gpw::GthPotential(molrs::Element::Ni, "GTH-PADE-q18")};
cpu::gpw::Kind O{molrs::Element::O,
                 cpu::gpw::GaussianBasis(molrs::Element::O, "SZV-GTH-PADE"),
                 cpu::gpw::GthPotential(molrs::Element::O, "GTH-PADE-q6")};

cpu::gpw::GPWEvaluator eval({Ni, O});
eval.cutoff_ry = 250.0;
eval.max_iter = 300;
eval.tol = 1e-5;
eval.mix = 0.35;              // DIIS path only; ignored when use_ot
eval.xc = cpu::gpw::XcKind::PADE;
eval.nonperiodic = true;
eval.use_ot = true;           // product OT path (no ATV_GPW_OT env required)
eval.n_alpha = 13;            // multiplicity 3, 24 valence e⁻
eval.n_beta = 11;
eval.plus_u.push_back({molrs::Element::Ni, /*l=*/2, /*u_minus_j_ev=*/6.0});

cpu::single_point(frame, eval);

const double e_ha = eval.last_energy_hartree();          // [Ha]
const double e_ev = frame.meta.get(kEnergy);             // [eV]
const cpu::gpw::GpwResult& r = eval.last_result();
const double e_u = r.e_dftu;                             // [Ha]
const double s2 = eval.last_s2();                        // Löwdin ⟨S²⟩
const double m_Ni = r.mulliken_spin.empty() ? 0.0 : r.mulliken_spin[0];
```

Diagnostic env `ATV_GPW_OT` may still force OT on (OR with the member); product
code and the regression **must not** depend on it (`unsetenv` in the reg).

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
  cmake --build build --target reg_ot-dftu -j$(nproc)
  ./build/regressions/reg_ot-dftu
'
```

| Piece | Path |
| ----- | ---- |
| Public `use_ot` / `plus_u` | `include/atomiverse/cpu/gpw/gpw_evaluator.h` |
| Public `PlusU` / `GpwResult::{e_dftu,mulliken_spin}` | `include/atomiverse/cpu/gpw/gpw_solver.h` |
| OT parameterization (internal) | `include/atomiverse/cpu/gpw/internal/ot.h` |
| Dudarev E/V (internal) | `include/atomiverse/cpu/gpw/internal/dft_plus_u.h` |
| SCF wire-up (OT + \(V_U\)) | `src/cpu/gpw/gpw_solver.cpp` |
| Regression | `regressions/gpw/ot-dftu/reg_ot-dftu.cpp` |
| CP2K twin input | `tools/cp2k_oracle/inputs/ot-dftu/nio_mol_pade_u_ot.inp` |

## Validation (vs CP2K)

Source: [`regressions/gpw/ot-dftu/REFERENCE.md`](../../../regressions/gpw/ot-dftu/REFERENCE.md).
Oracle: offline **CP2K 2026.1** (`cp2k.psmp`); **never** live in ctest. Pin date:
**2026-07-26**. Twin of `gpw-dftu-nio` chemistry with OT as the product path
on both sides (CP2K always used `&OT` for this NiO mol pin).

**Setup (must match twin):**

| Item | Value |
| ---- | ----- |
| System | NiO **molecule**, charge 0, UKS multiplicity 3 |
| Bond | \(R(\mathrm{Ni–O})=\) **1.668 Å** along *z*, **centered** |
| Coordinates (Å) | Ni (5, 5, 4.166), O (5, 5, 5.834) |
| Cell | **10 × 10 × 10 Å**, `PERIODIC NONE` |
| Poisson | Martyna–Tuckerman (`PSOLVER MT` / `nonperiodic`) |
| Basis | Ni `SZV-GTH-PADE-q18`, O `SZV-GTH-PADE` |
| PP | Ni `GTH-PADE-q18` (18e), O `GTH-PADE-q6` (6e) → **24** valence e⁻ |
| XC | PADE (LDA LSD) |
| DFT+U | Ni \(L=2\), \(U-J=6.0\) eV, `PLUS_U_METHOD MULLIKEN` |
| Cutoff | **250 Ry** (ATV single density grid; CP2K `NGRIDS 4` / `REL_CUTOFF 40`) |
| Spin | \(N_\alpha=13\), \(N_\beta=11\) |
| SCF twin | CP2K: OT `FULL_SINGLE_INVERSE` + OUTER_SCF; ATV: `use_ot=true`, `tol=1e-5`, `max_iter=300` |

**Pinned quantities:**

| Quantity | CP2K pin | Atomiverse (REFERENCE) | Tol (regression) | Notes |
| -------- | -------: | ---------------------: | ---------------- | ----- |
| \(E_{\mathrm{tot}}\) [Ha] | **−184.92466076474310** | ≈ −184.94575 | **5×10⁻² Ha** | single-grid vs multigrid+OT |
| \(\lvert\Delta E\rvert\) vs CP2K | — | ≈ **2.1×10⁻²** Ha | ≤ 5×10⁻² | within gate |
| \(\lvert E_{\mathrm{OT+U}}-E_{\mathrm{DIIS+U}}\rvert\) (ATV) | — | ≈ **3.8×10⁻⁴** Ha | **1×10⁻³ Ha** | fixed-point parity |
| DFT+U energy [Ha] | **0.05230865222344** | \(>10^{-4}\) (asserted) | diagnostic | `r.e_dftu`; not equality-asserted |
| \(\langle S^2\rangle\) (single det.) | **2.019198** | ≈ 2.015 | **0.25** | ideal \(S(S+1)=2\) |
| Mulliken \(m_{\mathrm{Ni}}\) [e] | **1.177632** | ≈ 1.00 | **0.5** and \(m_{\mathrm{Ni}}>0.3\) | spin localization |
| Frame energy consistency | — | \(E_{\mathrm{eV}}/k_{\mathrm{Ha\to eV}}\equiv E_{\mathrm{Ha}}\) | \(10^{-9}\) Ha | reg assert |
| SCF | converged | `r.converged` | — | ~23 OT iters (ATV, REFERENCE) |

Key CP2K log lines (from REFERENCE):

```text
  DFT+U energy:                                                 0.05230865222344
  Ideal and single determinant S**2 :                    2.000000       2.019198
 ENERGY| Total FORCE_EVAL ( QS ) energy [hartree]           -184.924660764743095
```

Offline re-run (operator only; not ctest):

```bash
export LD_LIBRARY_PATH=$CP2K_ROOT/lib:$LD_LIBRARY_PATH
export CP2K_DATA_DIR=$CP2K_ROOT/share/cp2k/data
cd /tmp && ln -sf $CP2K_DATA_DIR/{BASIS_SET,GTH_POTENTIALS} .
# copy tools/cp2k_oracle/inputs/ot-dftu/nio_mol_pade_u_ot.inp
OMP_NUM_THREADS=8 cp2k.psmp -i nio_mol_pade_u_ot.inp -o ot_dftu.out
```

## Performance

Source: fair wall table in `regressions/gpw/ot-dftu/REFERENCE.md` (2026-07-26).
Handoff / queue: ratio **~0.26**. Gate \(r\le 10\) unless `perf_exempt`.

Comparable settings: cutoff 250 Ry, PADE+U, MT, single rank, `OMP_NUM_THREADS=8`.

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r = T_{\mathrm{ATV}}/T_{\mathrm{CP2K}}\) | ≤10× | Caveat |
| ---- | ------------------------: | ----------------------: | ---------------------------------------: | ---- | ------ |
| NiO mol OT+PADE+U / SZV / MT, Γ | **13.65** | **3.58** (OT+U) | **0.26** | ✓ | CP2K multigrid+OT; ATV Release single density grid |
| ATV DIIS+U (same binary) | — | **3.03** | — | — | dual-path check only; not the fair twin |

**Caveats.** (1) Report the **OT-only** fair ratio vs CP2K OT+`PLUS_U`; the
regression also runs DIIS+U for fixed-point parity and must not be ratioed
against CP2K alone as a summed wall. (2) Ratio is for this SZV molecule
microcase; it does not forecast solid NiO, multigrid parity, or sparse OT.
(3) Product pass without exemption (`within_10x=true`).

## Limitations

- **Molecule foothold, not solid AFM:** periodic rocksalt AFM NiO remains out
  of this regression (same as `gpw-dftu-nio`); needs larger basis / k-points
  for a production solid pin.
- **Grid / multigrid twin, not bit-identical:** ATV single density grid vs CP2K
  multigrid; energy tolerance is deliberately **5×10⁻² Ha**. Milli-Ha class is
  not claimed at 250 Ry.
- **SZV-only pin:** minimal PADE GTH basis; not a chemistry-quality NiO
  electronic-structure claim.
- **No analytic DFT+U forces / stress on OT** in this slug; force and stress
  product paths are separate DIIS-oriented rows.
- **UKS + PBE-GGA** unsupported for GPW UKS on this path; DFT+U here is PADE
  LSD only.
- **Γ only for the combo:** multi-k OT and multi-k DFT+U exist as separate
  product rows; OT+U×k is not this pin.
- **Liechtenstein full \(U,J\)** and PAW/GAPW projectors out of scope.
- **Geometry hygiene:** off-center molecules under MT destroy energy parity;
  always center for isolated pins.

## References

- Offline pin + fair wall: [`regressions/gpw/ot-dftu/REFERENCE.md`](../../../regressions/gpw/ot-dftu/REFERENCE.md)
- Regression consumer: [`reg_ot-dftu.cpp`](../../../regressions/gpw/ot-dftu/reg_ot-dftu.cpp)
- CP2K twin: [`tools/cp2k_oracle/inputs/ot-dftu/`](../../../tools/cp2k_oracle/inputs/ot-dftu/)
- Oracle contract: [`tools/cp2k_oracle/README.md`](../../../tools/cp2k_oracle/README.md)
- Related DIIS+U foothold: [`gpw-dftu-nio.md`](gpw-dftu-nio.md) /
  [`regressions/gpw/gpw-dftu-nio/`](../../../regressions/gpw/gpw-dftu-nio/)
- Related OT RKS water: [`ot-minimizer-water.md`](ot-minimizer-water.md)
- Narrative GPW: [`docs/tutorial/ch09-gpw.md`](../ch09-gpw.md)
- Roadmap: [`docs/roadmap/dft.md`](../../roadmap/dft.md)
- J. VandeVondele, J. Hutter, *J. Chem. Phys.* **118**, 4365 (2003) — OT
- S. L. Dudarev et al., *Phys. Rev. B* **57**, 1505 (1998) — simplified \(U\)
- V. I. Anisimov, J. Zaanen, O. K. Andersen, *Phys. Rev. B* **44**, 943 (1991) — DFT+U
- G. Lippert, J. Hutter, M. Parrinello, *Mol. Phys.* **92**, 477 (1997) — GPW
- S. Goedecker, M. Teter, J. Hutter, *Phys. Rev. B* **54**, 1703 (1996) — PADE / GTH
- G. J. Martyna, M. E. Tuckerman, *J. Chem. Phys.* **110**, 2810 (1999) — MT Poisson
- T. D. Kühne et al., *J. Chem. Phys.* **152**, 194103 (2020) §VII C — CP2K OT formulation
