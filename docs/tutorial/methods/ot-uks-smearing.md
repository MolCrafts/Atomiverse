# UKS OT + metallic Fermi–Dirac smearing — BCC Na / multi-k GPW

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `ot-uks-smearing` |
| Regression | `regressions/gpw/ot-uks-smearing/` |
| Public API | `cpu::gpw::GpwDft` members `use_ot`, `n_alpha`/`n_beta`, `smearing`, `degauss` [Ry], `extra_bands`, `k_mesh` |
| Roadmap | `docs/roadmap/dft.md` §10 GPW — OT+metallic smear **UKS** ✓ (`reg_ot-uks-smearing`) |
## Summary

Collinear **UKS** Gaussian-and-plane-waves (GPW) DFT with **Orbital
Transformation (OT)** SCF **and** **Fermi–Dirac** metallic occupation
smearing on a multi-k mesh. Occupations are fractional spin-orbitals
\(f_{\sigma kb}\in[0,1]\) with a shared chemical potential \(\mu\); the SCF
reports the **Mermin free energy** \(F = E_{\mathrm{KS}} - TS\). Product
surface is public members on the concrete method `cpu::gpw::GpwDft` — **no**
`*Config` bag, **no** factories. Textbook pin: **BCC Na** conventional
cubic (\(a=4.225\) Å, 2 atoms), PADE/LSD, GTH-PBE-q9, DZVP-MOLOPT-SR-GTH,
MP \(2\times2\times2\), `degauss = 0.02` Ry. Product gate:
`regressions/gpw/ot-uks-smearing` (`reg_ot-uks-smearing`). Fair OT-only wall
ratio \(r = T_{\mathrm{ATV}}/T_{\mathrm{CP2K}} \approx 0.12\) (OMP=4, ≪10×).
CP2K 2026.1 hard-aborts OT with k-points, so the offline twin is **DIIS +
LSD + FERMI_DIRAC** only; product accuracy is **internal exactness**
(OT+FD ≡ DIIS+FD) plus a frozen free-energy pin.

## Method

**Setting.** Multi-k collinear spin-polarized GPW Kohn–Sham DFT with Padé
LSD, contracted short-range MOLOPT Gaussians, GTH pseudopotentials, and
periodic XYZ electrostatics. Wavefunction in the Gaussian AO basis;
density / Hartree / XC on a uniform real-space grid with plane-wave cutoff
\(E_{\mathrm{cut}}\) [Ry]. Method channel energy is Hartree
(`GpwResult::e_total`); public Frame energy remains eV when the evaluator
path is used. This regression exercises the concrete method `GpwDft`
directly.

**UKS occupations under smear.** Seed integers `n_alpha` / `n_beta` set the
cold-start ferro/AFM density only. Under `Smearing::fermi_dirac` (or
Methfessel–Paxton), magnetization is **free**: a single \(\mu\) is chosen
so

\[
N_{\mathrm{occ}}
  = \sum_{\sigma\in\{\alpha,\beta\}} \sum_{k,b} w_k\,
    f_{\sigma kb}(\varepsilon_{\sigma kb},\mu,\sigma_{\mathrm{smear}})
  = Z_{\mathrm{val}}\cdot N_{\mathrm{atoms}}
\]

with spin-orbital fillings \(f\in[0,1]\) (`spin_degen = 1` via the shared
`cpu::pw::internal` occupation fill). For BCC Na with GTH-q9,
\(N_e = 18\). BCC Na is non-magnetic in the ground state — the ferro seed
(\(n_\alpha=10\), \(n_\beta=8\)) typically collapses under smear (same class
as `gpw-uks-smearing` / `ot-smearing`).

**Fermi–Dirac free energy.** With reduced energy
\(x = (\varepsilon-\mu)/\sigma\) and \(\sigma =\) `degauss` (converted to Ha
inside SCF):

\[
f(x) = \frac{1}{1+e^{x}},\qquad
s = -\bigl[f\ln f + (1-f)\ln(1-f)\bigr],
\]

\[
F = E_{\mathrm{KS}} - TS,\qquad
E_{\mathrm{entropy}} = -\sigma\sum_{\sigma kb} w_k\, s_{\sigma kb}.
\]

`GpwResult::e_total` is \(F\) when smearing is on; `e_fermi` is \(\mu\)
[Ha]; `e_entropy` is the \(-TS\) piece; `n_elec_occ` is the filled count.

**OT parameterization (spin-polarized, fractional occupations).** Closed-shell
OT (VandeVondele & Hutter 2003; Kühne et al. 2020 §VII C) is the Stiefel
manifold \(C^\top S C = I\). Under metallic smear the OT manifold expands
to the full independent AO spectrum (extra virtuals via `extra_bands`);
the projected band-energy gradient uses **non-uniform** spinor occupations
\(f\):

\[
g_\sigma = (I - S C_\sigma C_\sigma^\dagger)\,(H_\sigma C_\sigma f_\sigma)
\]

(internal; product authors set `use_ot = true` only). Preconditioned CG on
the OT manifold replaces dense \((H,S)\) diagonalization + Pulay DIIS each
outer SCF step. Outer loop rebuilds \(H_\sigma\) from collocated
\(\rho_\alpha,\rho_\beta\) until residual / free-energy criteria meet `tol`.

**DIIS twin path.** `use_ot = false` (default): generalized diagonalization
of \((H_\sigma,S)\) plus density mixing (`mix`). Same self-consistent
fixed point for a metallic collinear ground state when both converge on
identical grids — the product identity gate.

**Relation to sibling product paths.**

| Slug | Spin | SCF | Smear | k-mesh |
| ---- | ---- | --- | ----- | ------ |
| `ot-minimizer-water` | RKS | OT | none | Γ |
| `ot-smearing` | RKS | OT | FD/MP | multi-k |
| `gpw-uks-smearing` | UKS | DIIS | FD/MP | multi-k |
| `ot-uks-kpoints` | UKS | OT | none (integer) | multi-k |
| **`ot-uks-smearing`** | **UKS** | **OT** | **FD/MP** | **multi-k** |

**Key exclusions.**

- Absolute free-energy match to CP2K (grid / `EPS_DEFAULT` / REL_CUTOFF
  differ by ~0.6 Ha) — **not** a product gate.
- CP2K OT + k-points (hard abort in CP2K 2026.1) — no fair OT energy/wall twin.
- Multi-k UKS OT + DFT+U + smear (`ot-uks-dftu` is integer multi-k UKS OT+U;
  `ot-dftu-smearing` is RKS OT+U+smear).
- Cold / Marzari–Vanderbilt smear; non-collinear / SOC; sparse OT.
- Forces / stress / BOMD under UKS OT + smear (separate queue rows if needed).
- CUDA GPW OT/smear; MPI multi-rank industrial scaling gate on this slug.

## Implementation

Public surface only: construct `cpu::gpw::GpwDft`, assign members, `build` /
`run` on `std::vector<GpwAtom>`. **No** `*Config`, **no** `make_*` /
`build_*` factories. Matches the regression:

```cpp
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/molrs.hpp"

using namespace ATV;
using cpu::gpw::GpwDft;
using cpu::gpw::Smearing;

constexpr double kA2B = 1.0 / 0.52917720859; // Å → Bohr
const double a_ang = 4.225;
const double a = a_ang * kA2B;

// BCC conventional cubic: 2 Na at (0,0,0) and (½,½,½) [Bohr]
const cpu::gpw::GaussianBasis bNa(molrs::Element::Na, "DZVP-MOLOPT-SR-GTH");
const cpu::gpw::GthPotential pNa(molrs::Element::Na, "GTH-PBE-q9");
const double frac[2][3] = {{0.0, 0.0, 0.0}, {0.5, 0.5, 0.5}};
std::vector<cpu::gpw::GpwAtom> atoms;
for (int i = 0; i < 2; ++i) {
    const double pos[3] = {frac[i][0] * a, frac[i][1] * a, frac[i][2] * a};
    atoms.emplace_back(pos, bNa, pNa, /*Z=*/11);
}

GpwDft m;
m.cell[0] = m.cell[1] = m.cell[2] = a;
m.cutoff_ry = 150.0;
m.n_occ = 0;
m.n_alpha = 10; // ferro seed; free under smear
m.n_beta = 8;
m.max_iter = 200;
m.tol = 1e-6;
m.eps_default = 1e-7;
m.xc = cpu::gpw::XcKind::PADE;
m.diis_depth = 6;
m.mix = 0.4;
m.k_mesh[0] = m.k_mesh[1] = m.k_mesh[2] = 2;
m.k_time_reversal = true;
m.smearing = Smearing::fermi_dirac;
m.degauss = 0.02; // Ry
m.extra_bands = 4;
m.use_ot = true;

m.build(atoms);
const auto& r = m.run(atoms);
// r.e_total [Ha] free energy F; r.e_fermi [Ha]; r.n_elec_occ ≈ 18;
// r.e_entropy [Ha]; r.converged; r.n_kpts > 1
```

Build and run the regression (scoped — never bare full-tree build for this
slug alone):

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
  cmake --build build -j$(nproc) --target atomiverse reg_ot-uks-smearing
  ./build/regressions/reg_ot-uks-smearing
  # or: cmake --build build --target run_reg_ot-uks-smearing
'
```

| Path | Role |
| ---- | ---- |
| `include/atomiverse/cpu/gpw/gpw_solver.h` | Public `GpwDft::{use_ot,smearing,degauss,n_alpha,n_beta,k_mesh,…}` |
| `include/atomiverse/cpu/gpw/internal/ot.h` | OT parameterization / fractional-occ gradient (internal) |
| `regressions/gpw/ot-uks-smearing/` | E2E product gate + `REFERENCE.md` |
| `tools/cp2k_oracle/inputs/ot-uks-smearing/` | Offline DIIS+LSD+FD twin + README |
| `regressions/gpw/ot-smearing/` | Closed-shell RKS OT+FD sibling |
| `regressions/gpw/gpw-uks-smearing/` | Multi-k UKS DIIS+FD sibling |

## Validation (vs CP2K / internal exactness)

Source: [`regressions/gpw/ot-uks-smearing/REFERENCE.md`](../../../regressions/gpw/ot-uks-smearing/REFERENCE.md).
Oracle: offline **CP2K 2026.1** for wall class + DIIS LSD twin input; **never**
live in ctest. Product free-energy accuracy is **ATV-internal** (OT ≡ DIIS)
plus a frozen first-green pin.

**Setup:**

| Item | Value |
| ---- | ----- |
| System | BCC Na conventional cubic, charge 0, collinear UKS seed |
| Geometry [Å] | \(a = 4.225\); Na (0,0,0), Na (2.1125, 2.1125, 2.1125) |
| Cell | cubic, `PERIODIC XYZ` |
| Basis / PP | DZVP-MOLOPT-SR-GTH; GTH-PBE-q9 (\(Z_{\mathrm{val}}=9\) → \(N_e=18\)) |
| XC | PADE (LSD) |
| Cutoff (pin) | **150 Ry** |
| k-mesh | Monkhorst–Pack \(2\times2\times2\), time-reversal fold |
| Spin seed | \(n_\alpha=10\), \(n_\beta=8\) (ferro; free under smear) |
| Smearing | Fermi–Dirac, `degauss = 0.02` Ry, `extra_bands = 4` |
| SCF | `max_iter=120` (DIIS) / `200` (OT), `tol=1e-6`, `mix=0.4` |
| CP2K twin | DIIS + LSD + FERMI_DIRAC only (OT+kpoints aborted by CP2K) |

**Product gates (from regression):**

| Gate | Check | Tol |
| ---- | ----- | ---: |
| A1 | `build` accepts UKS OT + FD | — |
| A2 | OT+FD and DIIS+FD SCF converge; multi-k active | — |
| A3 | \(\lvert F_{\mathrm{OT}} - F_{\mathrm{DIIS}}\rvert\) | \(10^{-4}\) Ha |
| A4 | \(N_{\mathrm{occ}} = 18\) | \(10^{-5}\) |
| A5 | frozen free-energy pin (first green) | \(10^{-3}\) Ha |
| A6 | finite \(E_F\), entropy term | — |

**Pinned free energies [Hartree] (first green 2026-07-26):**

| Quantity | Atomiverse / pin | CP2K 2026.1 | Tol | Notes |
| -------- | ---------------: | ----------: | --- | ----- |
| \(F\) OT | **−92.1263304935** | — | \(10^{-3}\) Ha | frozen first-green pin |
| \(F\) DIIS twin (ATV) | −92.1263306080 | — | — | same grids as OT |
| \(\lvert F_{\mathrm{OT}}-F_{\mathrm{DIIS}}\rvert\) | **1.15×10⁻⁷** | — | ≤ \(10^{-4}\) | measured identity |
| \(N_{\mathrm{occ}}\) | **18.0** | — | \(10^{-5}\) | electron count |
| OT iters | 12 | — | — | first green |

Absolute free energy is **not** compared to CP2K (grid / `EPS_DEFAULT` /
rel-cutoff class differ by ~0.6 Ha). Electron count + OT≡DIIS identity are
the product accuracy gates. Same system class as `reg_gpw-uks-smearing` /
`reg_ot-smearing`.

Offline oracle sketch (host with CP2K module; **not** ctest):

```bash
module load ... CP2K/2026.1
export CP2K_DATA_DIR=.../share/cp2k/data
cp2k.psmp -i tools/cp2k_oracle/inputs/ot-uks-smearing/na_bcc_uks_fermi_dirac.inp \
          -o na_uks_fd.out
# twin = DIAG/DIIS + LSD + SMEAR; OT + KPOINTS is hard-aborted by CP2K
```

## Performance

Source: fair wall table in
`regressions/gpw/ot-uks-smearing/REFERENCE.md` and
the case `REFERENCE.md` (wave 9, 2026-07-26). Queue record:
ratio **~0.12**. Gate \(r\le 10\).

Comparable settings: cutoff 150 Ry, PADE, multi-k \(2^3\), single rank,
`OMP_NUM_THREADS=4`. CP2K wall class is the DIIS LSD twin (~89.5 s, same
class as `gpw-uks-smearing` / RKS FD).

| Twin | Wall [s] | \(r = T_{\mathrm{ATV}}/T_{\mathrm{CP2K}}\) | ≤10× | Caveat |
| ---- | -------: | ----------------------------------------: | ---- | ------ |
| CP2K DIIS+LSD+FD twin | **~89.5** | — | — | no OT+kpoints twin exists |
| ATV OT+FD only | **~10.3** | **~0.115** | ✓ | fair OT-only wall |
| ATV OT+DIIS (full reg binary) | higher | — | — | **not** a fair twin vs CP2K alone |

**Caveats.** (1) Report the **OT-only** ratio; the regression binary also runs
DIIS for the identity gate — do not ratio the full binary wall against CP2K.
(2) CP2K cannot run OT+kpoints, so the wall twin is algorithm-class DIIS+FD
(same system), not OT-vs-OT. (3) Dense OT preconditioner cost does not forecast
sparse/linear-scaling OT. (4) `within_10x=true`; no optimize pass required.

## Limitations

- Multi-k collinear UKS GPW only; no non-collinear / SOC, no CUDA, no MPI
  industrial scaling gate on this slug.
- Absolute energy vs CP2K is **not** a product claim (~0.6 Ha grid class
  offset). Use OT≡DIIS + \(N_e\) + frozen pin.
- CP2K product surface does not offer OT + k-points; fair OT energy twin is
  impossible — document DIIS twin wall class only.
- Multi-k UKS OT + DFT+U + smear remains out of scope (see `ot-uks-dftu`,
  `ot-dftu-smearing` for the partial combos).
- Forces / stress / BOMD with UKS OT + smear are not gated here.
- BCC Na is a soft metallic non-magnet; ferro seed collapses — do not treat
  residual magnetization as a product observable on this pin.
- Speed: ~0.12× CP2K on this microcase under OMP=4 — strong for the wave-9
  gate, not a claim of production Quickstep parity on large systems.

## References

- Offline pin + fair wall: `regressions/gpw/ot-uks-smearing/REFERENCE.md`
- Regression binary: `regressions/gpw/ot-uks-smearing/reg_ot-uks-smearing.cpp`
- Closed spec (git history): `.claude/specs/ot-uks-smearing/` (deleted at done)
- CP2K oracle inputs: `tools/cp2k_oracle/inputs/ot-uks-smearing/`
- Sibling tutorials: [ot-minimizer-water](ot-minimizer-water.md) (Γ RKS OT),
  [pw-smearing-al](pw-smearing-al.md) (PW FD metals)
- Roadmap: `docs/roadmap/dft.md` §10 (UKS OT+smear ✓ on GPW)
- VandeVondele, J.; Hutter, J. *J. Chem. Phys.* **118**, 4365 (2003),
  doi:10.1063/1.1543154 (original OT)
- Kühne, T. D. et al. *J. Chem. Phys.* **152**, 194103 (2020) §VII C,
  doi:10.1063/5.0007045 (CP2K / OT formulation)
- Mermin, N. D. *Phys. Rev.* **137**, A1441 (1965) (finite-\(T\) free energy)
