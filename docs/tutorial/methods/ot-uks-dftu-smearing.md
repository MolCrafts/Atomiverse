# Multi-k UKS OT + Dudarev DFT+U + metallic smearing — H₂ / PADE / DZV-GTH

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `ot-uks-dftu-smearing` |
| Regression | `regressions/gpw/ot-uks-dftu-smearing/` |
| Public API | `cpu::gpw::GpwDft` members `use_ot`, `plus_u`, `smearing`, `degauss`, `extra_bands`, `n_alpha`/`n_beta`, `k_points` / `k_mesh` |
| Roadmap | `docs/roadmap/dft.md` — multi-k **UKS OT+U+smear** (`reg_ot-uks-dftu-smearing`) ✓ |

## Summary

This page documents the **full product stack** on multi-k collinear **UKS**
GPW: **Orbital Transformation (OT)** SCF, **Dudarev DFT+U** (`plus_u`), and
**Fermi–Dirac** metallic occupations (Mermin free energy
\(F = E_{\mathrm{KS}} - TS\)). Integer multi-k UKS OT+U keeps OUTER_SCF +
fixed-\(H\) OT; the **U + smear** path uses free-energy PR+ Armijo OT on the
full AO spectrum per spin, with a **shared** chemical potential \(\mu\) over
\(\alpha+\beta\) (\(f_\sigma\in[0,1]\)) and \(V_U^\sigma\) folded into
\(H_k^\sigma\). Same self-consistent fixed point as multi-k UKS DIIS + U + FD.

Public surface is concrete `cpu::gpw::GpwDft` **members** only — no `*Config`
bag, no `make_*` / `build_*` factories. Textbook pin: equal-spin **H₂**
(DZV-GTH-PADE / GTH-PADE-q1), PADE LSD, single density grid at 100 Ry,
explicit κ = (¼,0,0), \(U-J=4.0\) eV on H \(L=0\), FD `degauss = 0.02` Ry.
Product gate: `reg_ot-uks-dftu-smearing`. Fair wall twin: **none** — CP2K
2026.1 aborts OT+k and product OT+SMEAR; status **`perf_exempt`**.

Related combos: RKS OT+U+smear — `ot-dftu-smearing`; UKS OT+smear —
`ot-uks-smearing`; UKS OT+U integer — `ot-uks-dftu`; Γ DFT+U foothold —
[gpw-dftu-nio](gpw-dftu-nio.md); closed-shell OT —
[ot-minimizer-water](ot-minimizer-water.md); metallic PW smear —
[pw-smearing-al](pw-smearing-al.md). Narrative GPW: [Ch. 9](../ch09-gpw.md).

## Method

**GPW (Quickstep family).** Wavefunction in contracted Gaussians; density /
Hartree / XC on a uniform real-space grid; GTH pseudopotentials. Grid set by
plane-wave cutoff [Ry]. Citation: Lippert, Hutter, Parrinello, *Mol. Phys.*
**92**, 477 (1997).

**Multi-k collinear UKS.** Explicit or mesh \(k\)-points; complex Bloch
matrices \(H_k^\sigma\), \(S_k\); spin-resolved density and Padé LDA XC.
Members `n_alpha` / `n_beta` seed the integer occupations when smearing is
off; under FD they only bias the **initial** density. Equal-spin seed
(\(n_\alpha=n_\beta=1\) for H₂) keeps DIIS and free-energy OT on the same
Mermin minimum — a high-spin integer seed (\(n_\alpha=2\), \(n_\beta=0\))
can leave DIIS metastable while OT finds the lower closed-shell-like
minimum (REFERENCE seed note).

**Orbital Transformation (OT).** Diagonalization-free SCF: orbitals
optimized on the \(S\)-orthonormal manifold (VandeVondele & Hutter 2003;
Kühne et al. 2020 §VII C). Member `use_ot = true`. Integer multi-k UKS
OT+U: OUTER_SCF + fixed-\(H\) OT subspace. **OT + metallic smear:**
free-energy PR+ with Armijo line search over the **full** AO spectrum per
spin channel (not a fixed occupied block).

**Dudarev DFT+U.** Per element and angular momentum \(L\), effective
\(U_{\mathrm{eff}}=U-J\) in **eV** on the public surface (converted to
Hartree with `units::kHartreeInEv` inside). Symmetrized Mulliken occupation
on the AO subspace of atom \(A\) with \(\ell=L\) (CP2K `PLUS_U_METHOD
MULLIKEN`):

\[
n^{A\sigma}_{mm'} = \tfrac12\big[(P^\sigma S)_{mm'} + (S P^\sigma)_{mm'}\big].
\]

Energy and potential (sum over atoms and spin; \(U\) in Hartree):

\[
E_U = \sum_{A\sigma}\frac{U_{\mathrm{eff}}}{2}\Big[\mathrm{Tr}\,n^{A\sigma}
  - \mathrm{Tr}(n^{A\sigma} n^{A\sigma})\Big],
\qquad
V^{A\sigma}_{mm'} = U_{\mathrm{eff}}\Big(\tfrac12\delta_{mm'} - n^{A\sigma}_{mm'}\Big).
\]

On the multi-k path, \(V_U^\sigma\) is folded into \(H_k^\sigma\) inside
`build_uks_k_energy`. Implementation: `cpu/gpw/internal/dft_plus_u.h`
(not public). Empty `plus_u` leaves the legacy UKS path unchanged.

**Metallic Fermi–Dirac occupations.** Shared \(\mu\) over \(\alpha+\beta\)
with spin-orbital fillings \(f_\sigma\in[0,1]\); electron count
\(N_e = \sum_{k,b,\sigma} w_k f_{kb}^\sigma\). Width `degauss` [Ry]
(\(=k_B T_e\)); `extra_bands` adds empty states above the integer seed.
Reported total is the **Mermin free energy**
\(F = E_{\mathrm{KS}} - TS\); `GpwResult::{e_fermi,e_entropy,n_elec_occ}`
expose \(\mu\), \(-TS\), and \(N_e\).

**CP2K product gap (binding).** CP2K 2026.1 **aborts** OT with k-points
(`OT not possible with kpoint calculations` in `qs_scf_initialization.F`)
and product OT uses fixed integer occupations (no OT+SMEAR). Absolute
CP2K OT+k free-energy digits **do not exist**. The offline twin is
**multi-k UKS DIIS + FERMI_DIRAC + PLUS_U** — algorithm / fixed-point
documentation, not a fair OT wall twin. Decisive product gate:
**in-tree** \(\lvert F_{\mathrm{OT}}-F_{\mathrm{DIIS}}\rvert\) plus a frozen
first-green absolute.

**Key exclusions (this slug).**

- Fair OT wall vs CP2K (impossible under CP2K 2026.1 product limits).
- Liechtenstein full \(U,J\); GAPW / PAW projectors.
- Forces / stress / BOMD under UKS OT+U+smear as acceptance pins.
- CUDA OT / GPW UKS OT on device.
- High-spin seed as the energy pin (equal-spin only).

## Implementation

Public surface only: construct `cpu::gpw::GpwDft`, set members, `build` /
`run` on `GpwAtom` rows (regression path). Frame + `GPWEvaluator` also
exposes `use_ot` / `plus_u` / spin / k / smear members for outer Drivers;
this pin uses the solver directly to match the regression. **No**
`*Config`, **no** factories.

```cpp
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/molrs.hpp"

using namespace ATV;
using cpu::gpw::GpwAtom;
using cpu::gpw::GpwDft;
using cpu::gpw::GpwResult;

// H₂ in 8 Bohr cubic cell; ±0.7 Bohr from centre along x (REFERENCE)
const double L = 8.0;
const cpu::gpw::GaussianBasis bH(molrs::Element::H, "DZV-GTH-PADE");
const cpu::gpw::GthPotential pH(molrs::Element::H, "GTH-PADE-q1");
const double r0[3] = {0.5 * L - 0.7, 0.5 * L, 0.5 * L};
const double r1[3] = {0.5 * L + 0.7, 0.5 * L, 0.5 * L};
std::vector<GpwAtom> atoms;
atoms.emplace_back(r0, bH, pH, 1);
atoms.emplace_back(r1, bH, pH, 1);

GpwDft m;
m.cell[0] = m.cell[1] = m.cell[2] = L;
m.cutoff_ry = 100.0;
m.n_occ = 0;
m.n_alpha = 1;   // equal-spin seed under FD
m.n_beta = 1;
m.max_iter = 220;
m.tol = 1e-7;
m.mix = 0.35;    // DIIS path only; ignored when use_ot
m.xc = cpu::gpw::XcKind::PADE;
m.k_points.push_back({0.25, 0.0, 0.0});
m.k_point_weights.push_back(1.0);
m.smearing = cpu::gpw::Smearing::fermi_dirac;
m.degauss = 0.02;   // Ry
m.extra_bands = 2;
m.plus_u.push_back({molrs::Element::H, /*l=*/0, /*u_minus_j_ev=*/4.0});
m.use_ot = true;

m.build(atoms);
const GpwResult& r = m.run(atoms);
// r.e_total [Ha] = Mermin free energy F
// r.e_dftu [Ha], r.n_elec_occ ≈ 2, r.e_fermi, r.e_entropy, r.converged
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
  cmake --build build -j$(nproc) --target atomiverse reg_ot-uks-dftu-smearing
  ./build/regressions/reg_ot-uks-dftu-smearing
'
```

| Piece | Path |
| ----- | ---- |
| Public `GpwDft` / `PlusU` / `Smearing` / `GpwResult` | `include/atomiverse/cpu/gpw/gpw_solver.h` |
| Evaluator members (`use_ot`, `plus_u`, …) | `include/atomiverse/cpu/gpw/gpw_evaluator.h` |
| OT parameterization (internal) | `include/atomiverse/cpu/gpw/internal/ot.h` |
| Dudarev E/V (internal) | `include/atomiverse/cpu/gpw/internal/dft_plus_u.h` |
| SCF wire-up | `src/cpu/gpw/gpw_solver.cpp` |
| Unit (not e2e): OT / U / smear pieces | `tests/cpu/gpw/test_cpu_gpw_ot.cpp` |
| Regression | `regressions/gpw/ot-uks-dftu-smearing/` |
| CP2K DIIS twin | `tools/cp2k_oracle/inputs/ot-uks-dftu-smearing/` |

## Validation (vs CP2K)

Source:
[`regressions/gpw/ot-uks-dftu-smearing/REFERENCE.md`](../../../regressions/gpw/ot-uks-dftu-smearing/REFERENCE.md).
Oracle: offline **CP2K 2026.1** (`cp2k.psmp`); **never** live in ctest. Pin
date: **2026-07-26**. Closed gates A0–A6.

**Setup (must match twin / regression):**

| Item | Value |
| ---- | ----- |
| System | H₂, charge 0, UKS equal-spin seed under smear |
| Geometry [Bohr] | H (3.3, 4.0, 4.0); H (4.7, 4.0, 4.0) — ±0.7 from centre |
| Cell | **8 × 8 × 8 Bohr**, `PERIODIC XYZ` |
| Basis / PP | `DZV-GTH-PADE` / `GTH-PADE-q1` |
| XC | PADE (LSD) |
| Cutoff | **100 Ry**, single density grid (`NGRIDS 1` twin) |
| Spin seed | \(n_\alpha=1\), \(n_\beta=1\) (CP2K `MULTIPLICITY 1`) |
| DFT+U | H \(L=0\), \(U-J=4.0\) eV, `PLUS_U_METHOD MULLIKEN` |
| k | GENERAL κ = (0.25, 0, 0), weight 1 |
| Smear | Fermi–Dirac, `degauss = 0.02` Ry (\(T_e = 3157.75\) K), `extra_bands = 2` / `ADDED_MOS 2` |
| OT (ATV) | `use_ot = true`; DIIS twin `use_ot = false` |
| SCF | ATV: `tol=1e-7`, `max_iter=160` (DIIS) / `220` (OT), mix 0.35 |

**Pinned quantities (first green, 2026-07-26):**

| Quantity | Atomiverse / pin | Tol (regression) | Notes |
| -------- | ---------------: | ---------------- | ----- |
| \(F_{\mathrm{OT}}\) (U=4, FD) [Ha] | **−1.081998158198** | **1×10⁻³** (A6 frozen) | primary absolute |
| \(F_{\mathrm{DIIS}}\) same [Ha] | −1.081997968639 | — | same grids |
| \(\lvert F_{\mathrm{OT}}-F_{\mathrm{DIIS}}\rvert\) | **1.9×10⁻⁷** | ≤ **1×10⁻⁴** (A2) | decisive fixed-point |
| \(E_U\) (OT) [Ha] | ≈ −8.57×10⁻⁴ | non-trivial; OT vs DIIS ≤ 1×10⁻³ (A3) | `r.e_dftu` |
| \(N_e\) | **2.0** | 1×10⁻⁵ (A4) | valence count |
| \(F(U=0)\) OT+FD [Ha] | −1.108020857737 | — | activation baseline |
| \(\lvert F(U)-F(0)\rvert\) | ≈ 0.026 | ≥ **1×10⁻⁴** (A5) | U activation floor |
| OT / DIIS iters | 11 / 43 | converged (A1) | REFERENCE |

**Offline CP2K twin (DIIS only — algorithm documentation):**

| Item | Value |
| ---- | ----- |
| Input | `tools/cp2k_oracle/inputs/ot-uks-dftu-smearing/h2_uks_ot_dftu_smear.inp` |
| Method | Quickstep GPW, LSD PADE, multi-k UKS, PLUS_U, FERMI_DIRAC |
| SCF | `DIAGONALIZATION` + `DIRECT_P_MIXING` ALPHA 0.35 — **not** OT |
| Absolute \(E\) | **not** asserted vs ATV in ctest (grid / SCF twin class; A6 freezes ATV) |

Offline re-run (operator only; not ctest):

```bash
export CP2K_DATA_DIR=.../share/cp2k/data
# host with CP2K 2026.1; link BASIS_SET + GTH_POTENTIALS into cwd
cp2k.psmp -i tools/cp2k_oracle/inputs/ot-uks-dftu-smearing/h2_uks_ot_dftu_smear.inp \
          -o h2_uks_ot_dftu_smear.out
# ENERGY| Total FORCE_EVAL ( QS ) energy [hartree]  …
```

Attempting `&OT` under the same multi-k + SMEAR setup aborts on CP2K 2026.1
(see twin input header comments).

## Performance

Source: REFERENCE performance section + handoff
the case `REFERENCE.md` row `ot-uks-dftu-smearing`. Gate:
\(r = T_{\mathrm{ATV}}/T_{\mathrm{CP2K}} \le 10\) **or** `perf_exempt`.

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | ------------------------: | ----------------------: | ----: | ---- | ------ |
| Fair OT wall | **n/a** | OT-only ≈ **0.325** (OMP=4, Release) | — | **perf_exempt** | CP2K aborts OT+k and OT+SMEAR |

**Why exempt.** There is no product CP2K binary that runs multi-k OT (with
or without SMEAR). The DIIS twin is a fixed-point document, not a fair
algorithm wall. Product gate is internal OT ≡ DIIS free energy (A2), same
class as `ot-dftu-smearing` / `ot-uks-dftu`. Wall above is indicative only
(single microcase; not a multi-system standing suite).

## Limitations

- **Microcase H₂ toy**, not a correlated solid (NiO rocksalt, Fe, …).
  Chemistry claims need larger cells, k-meshes, and realistic \(U\) on \(d\).
- **CP2K has no OT+k / OT+SMEAR digits** in 2026.1 — absolute free energy is
  ATV-frozen (A6); twin is DIIS multi-k UKS + U + FD only.
- **Equal-spin seed required** for OT≡DIIS on this pin; high-spin integer
  seeds can split DIIS vs free-energy OT minima (not an OT bug).
- **Single density grid @ 100 Ry** — not multigrid parity with industrial
  Quickstep defaults.
- **Forces / stress / AIMD** under this combo are not acceptance pins here.
- **CUDA** OT / UKS OT+U+smear out of scope.
- **Liechtenstein full \(U,J\)**, PAW/GAPW, and orbital-dependent U ramping
  out of scope (same as `gpw-dftu-nio`).
- **Speed:** `perf_exempt`; do not invent a fair \(r\) from the DIIS twin.

## References

- Offline pin + walls:
  [`regressions/gpw/ot-uks-dftu-smearing/REFERENCE.md`](../../../regressions/gpw/ot-uks-dftu-smearing/REFERENCE.md)
- Regression consumer:
  [`reg_ot-uks-dftu-smearing.cpp`](../../../regressions/gpw/ot-uks-dftu-smearing/reg_ot-uks-dftu-smearing.cpp)
- CP2K DIIS twin:
  [`tools/cp2k_oracle/inputs/ot-uks-dftu-smearing/`](../../../tools/cp2k_oracle/inputs/ot-uks-dftu-smearing/)
- Oracle contract:
  [`tools/cp2k_oracle/README.md`](../../../tools/cp2k_oracle/README.md)
- Handoff (`perf_exempt`):
  the case `REFERENCE.md`
- Roadmap: [`docs/roadmap/dft.md`](../../roadmap/dft.md)
- Narrative GPW: [`docs/tutorial/ch09-gpw.md`](../ch09-gpw.md)
- Related method pages: [ot-minimizer-water](ot-minimizer-water.md),
  [gpw-dftu-nio](gpw-dftu-nio.md), [pw-smearing-al](pw-smearing-al.md),
  [gpw-uks-o2-gamma](gpw-uks-o2-gamma.md)
- Public header: `include/atomiverse/cpu/gpw/gpw_solver.h`
- J. VandeVondele, J. Hutter, *J. Chem. Phys.* **118**, 4365 (2003) — OT
- T. D. Kühne et al., *J. Chem. Phys.* **152**, 194103 (2020) §VII C — CP2K OT
- S. L. Dudarev et al., *Phys. Rev. B* **57**, 1505 (1998) — simplified \(U\)
- N. D. Mermin, *Phys. Rev.* **137**, A1441 (1965) — free energy
- G. Lippert, J. Hutter, M. Parrinello, *Mol. Phys.* **92**, 477 (1997) — GPW
- S. Goedecker, M. Teter, J. Hutter, *Phys. Rev. B* **54**, 1703 (1996) — PADE / GTH
