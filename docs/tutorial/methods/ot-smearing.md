# OT + metallic Fermi–Dirac smearing — BCC Na / multi-k GPW

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `ot-smearing` |
| Regression | `regressions/gpw/ot-smearing/` |
| Public API | `cpu::gpw::GpwDft` members `use_ot`, `smearing`, `degauss` [Ry], `extra_bands`, `mp_order`; `GpwResult::{e_total,e_fermi,n_elec_occ,e_entropy}` |
| Roadmap | [dft.md](../../roadmap/dft.md) §10 — OT + metallic smear ✓ (`reg_ot-smearing`) |
## Summary

**Orbital Transformation (OT)** optimizes occupied MO coefficients on the
Stiefel-type manifold \(C^\dagger S C = I\) without a dense generalized
eigensolve each outer SCF step. **Metallic occupation smearing** replaces fixed
integer fillings with fractional band occupations \(f_b(\varepsilon,\mu)\) and
reports the **Mermin free energy** \(F = E_{\mathrm{KS}} - TS\). This product
slug co-enables both on **multi-k closed-shell RKS GPW**: OT manifold size
grows to \(n_{\mathrm{occ}} + \max(\texttt{extra\_bands},1)\), occupations are
FD/MP (shared with the PW path), and the projected OT gradient uses
non-uniform band occupations. Public surface is members on
`cpu::gpw::GpwDft` — **no** `*Config` bag, **no** `make_*` factory. Textbook
pin: **BCC Na** conventional cell, PADE / GTH-PBE-q9 / DZVP-MOLOPT-SR-GTH,
Fermi–Dirac `degauss = 0.02` Ry, Monkhorst–Pack \(2\times2\times2\). Product
gate: free energy **OT+FD ≡ DIIS+FD**, electron count \(N_e=18\), frozen
\(F\) pin. Offline CP2K 2026.1 documents the **DIIS + FERMI_DIRAC** twin
(stock CP2K OT does not combine with SMEAR). Regression:
`reg_ot-smearing`. Fair OT-only wall ratio
\(r = T_{\mathrm{ATV}}/T_{\mathrm{CP2K}} \approx 0.20\) (≤10×).

## Method

**Setting.** Spin-unpolarized multi-k Gaussian-and-plane-waves (GPW)
Kohn–Sham DFT: PADE (LDA), contracted MOLOPT Gaussians, GTH pseudopotentials,
periodic single-grid Poisson. Wavefunction in the Gaussian AO basis;
density / Hartree / XC on a uniform real-space grid with plane-wave cutoff
\(E_{\mathrm{cut}}\) [Ry]. Internal SCF energies are Hartree
(`GpwResult::e_total`); public Frame energy remains eV when the evaluator
path is used.

**OT parameterization** (same manifold as [ot-minimizer-water](ot-minimizer-water.md);
multi-k complex form in `internal/ot.h`). For reference coefficients \(C_0\)
with \(C_0^\dagger S C_0 = I\) and \(S\)-tangent step \(x\):

\[
C(x) = C_0\,\cos U + x\,U^{-1}\sin U,\qquad
U = (x^\dagger S x)^{1/2}.
\]

**Metallic occupations (closed-shell).** With reduced energy
\(x = (\varepsilon-\mu)/\sigma\) and \(\sigma =\) `degauss` (Ry in the public
API; converted to Ha inside the SCF):

| Kind | Spinless \(f(x)\) | Public enum |
| ---- | ----------------- | ----------- |
| Fixed integer | step fill of `n_occ` | `Smearing::none` |
| Fermi–Dirac | \(1/(1+e^{x})\) → band \(f_b\in[0,2]\) | `Smearing::fermi_dirac` |
| Methfessel–Paxton order \(N\) | MP series of order `mp_order` | `Smearing::methfessel_paxton` |

Shared chemical potential \(\mu\) enforces

\[
N_{\mathrm{elec}}
  = \sum_{k,b} w_k\, f_b(\varepsilon_{kb},\mu,\sigma)
  = Z_{\mathrm{val}}\cdot N_{\mathrm{atoms}}
\]

(for BCC Na with GTH-PBE-q9, \(N_e = 18\)). `extra_bands` adds empty MOs
above \(n_{\mathrm{occ}}\) (CP2K `ADDED_MOS` / PW parity).

**OT + fractional occupations.** The active OT subspace is
\(n_{\mathrm{occ}} + \max(\texttt{extra\_bands},1)\). With non-uniform band
occupations \(f_i\in[0,2]\) and density \(P = C\,\mathrm{diag}(f)\,C^\dagger\),
the projected band-energy gradient is

\[
g = (I - S C C^\dagger)\,(2 H C f)
\]

(column \(i\) of \(H C\) scaled by \(f_i\); `ot_gradient_k(..., f)`). Outer SCF
rebuilds \(H\) from the collocated density until residual / energy criteria
meet `tol`. `GpwResult.e_total` is the free energy \(F\) when smearing is on;
`e_fermi` is \(\mu\) [Ha]; `e_entropy` is the \(-TS\) piece; `n_elec_occ` is
\(\sum w f\).

**DIIS twin path.** `use_ot = false` with the same `smearing` / `degauss` /
`extra_bands` / k-mesh: generalized diagonalization + density mixing. Same
self-consistent free-energy fixed point for a metallic closed-shell ground
state when both converge on identical grids (product identity gate).

**Key exclusions (binding).**

- **UKS OT + smear** — `build` throws (multi-k UKS OT with integer occupations
  is productized under `ot-uks-kpoints`; UKS OT + FD/MP remains residual).
- OT + DFT+U + smear on the multi-k path — not this pin (`ot-dftu-kpoints`
  is integer / separate combo).
- Sparse / linear-scaling OT preconditioner ○ (dense inverse cost).
- Stock CP2K OT+SMEAR product twin — CP2K keeps fixed integer OT occupations;
  offline twin is DIIS + FERMI_DIRAC.
- CUDA GPW OT / smear; cold / Marzari–Vanderbilt smear enum.
- Forces / stress / BOMD under OT+smear as an acceptance pin.

## Implementation

Public surface only: construct `cpu::gpw::GpwDft`, set members, `build` /
`run` on GTH atoms (regression path). No `*Config`, no factories. Headers:
`include/atomiverse/cpu/gpw/gpw_solver.h` (API + `Smearing` enum);
OT internals in `include/atomiverse/cpu/gpw/internal/ot.h` (not a user type).

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

// BCC conventional cubic: 2 Na at (0,0,0) and (½,½,½)
const cpu::gpw::GaussianBasis bNa(molrs::Element::Na, "DZVP-MOLOPT-SR-GTH");
const cpu::gpw::GthPotential pNa(molrs::Element::Na, "GTH-PBE-q9");
const double frac[2][3] = {{0.0, 0.0, 0.0}, {0.5, 0.5, 0.5}};
std::vector<cpu::gpw::GpwAtom> atoms;
for (int i = 0; i < 2; ++i) {
    const double pos[3] = {frac[i][0] * a, frac[i][1] * a, frac[i][2] * a};
    atoms.emplace_back(pos, bNa, pNa, 11);
}

GpwDft ot;
ot.cell[0] = ot.cell[1] = ot.cell[2] = a;
ot.cutoff_ry = 150.0;
ot.n_occ = 9;            // Ne/2 for GTH-PBE-q9 × 2
ot.max_iter = 200;
ot.tol = 1e-6;
ot.eps_default = 1e-7;
ot.xc = cpu::gpw::XcKind::PADE;
ot.diis_depth = 6;
ot.mix = 0.4;
ot.k_mesh[0] = ot.k_mesh[1] = ot.k_mesh[2] = 2;
ot.k_time_reversal = true;
ot.smearing = Smearing::fermi_dirac;
ot.degauss = 0.02;       // Ry
ot.extra_bands = 4;
ot.use_ot = true;

ot.build(atoms);
const auto& r = ot.run(atoms);
// r.e_total [Ha] free energy F; r.e_fermi [Ha]; r.n_elec_occ ≈ 18;
// r.e_entropy [Ha] (−TS); r.converged, r.iters; r.n_kpts > 1
```

DIIS twin: same members with `use_ot = false` (and a milder `max_iter` is
fine for the identity gate). Multi-k UKS OT + FD still rejects at `build`
(`n_alpha`/`n_beta` + `use_ot` + `smearing != none`).

```bash
# configure (login shell + Lmod — see AGENTS.md)
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
  cmake --build build --target reg_ot-smearing -j$(nproc)
  ./build/regressions/reg_ot-smearing
'
```

| Path | Role |
| ---- | ---- |
| `include/atomiverse/cpu/gpw/gpw_solver.h` | Public `use_ot`, `Smearing`, `degauss`, `extra_bands` |
| `include/atomiverse/cpu/gpw/internal/ot.h` | Multi-k OT + fractional gradient (internal) |
| `src/cpu/gpw/gpw_solver.cpp` | Build gates (RKS OT+smear OK; UKS OT+smear throws) |
| `regressions/gpw/ot-smearing/` | E2E product gate + `REFERENCE.md` |
| `tools/cp2k_oracle/inputs/ot-smearing/na_ot_fermi_dirac.inp` | Offline CP2K DIIS+FD twin |
| Related: `reg_gpw-metallic-smearing` | DIIS-only metallic smear (same system class) |
| Related: `reg_ot-minimizer-water` | Γ OT, fixed integer occupations |

## Validation (vs CP2K)

Source:
[`regressions/gpw/ot-smearing/REFERENCE.md`](../../../regressions/gpw/ot-smearing/REFERENCE.md)
(offline pins only; **never** live CP2K in ctest).

### System (Atomiverse pin)

| Setting | Value |
| ------- | ----- |
| Geometry | BCC Na conventional cubic, \(a = 4.225\) Å, 2 atoms |
| Basis / PP | `DZVP-MOLOPT-SR-GTH`; `GTH-PBE-q9` (\(Z_{\mathrm{val}}=9\) → \(N_e=18\)) |
| XC | PADE (LDA) |
| Density cutoff | 150 Ry |
| k-mesh | Monkhorst–Pack \(2\times2\times2\), time-reversal fold |
| Smearing | Fermi–Dirac, `degauss = 0.02` Ry |
| OT | `use_ot = true`, `extra_bands = 4` |
| DIIS twin | same without `use_ot` |
| SCF | `tol=1e-6`, `eps_default=1e-7`, `mix=0.4`, OT `max_iter=200` |
| Pin date | first green 2026-07-26 |

### Product gates

| Gate | Check | Tol |
| ---- | ----- | ---: |
| A1 | `build` accepts RKS OT + FD | — |
| A2 | OT+FD SCF converges; finite \(E_F\), entropy; multi-k active | — |
| A3 | \(\lvert F_{\mathrm{OT}} - F_{\mathrm{DIIS}}\rvert\) | \(10^{-4}\) Ha |
| A4 | \(N_{\mathrm{occ}} = 18\) | \(10^{-5}\) |
| A5 | frozen free-energy pin (OT) | \(10^{-3}\) Ha |
| A6 | multi-k UKS OT + FD smear throws at `build` | — |

### Hard-coded free-energy pin

| Quantity | Atomiverse / pin | Tol | Notes |
| -------- | ---------------: | --- | ----- |
| Free energy \(F\) (OT) | **−92.1263304935 Ha** | **\(10^{-3}\) Ha** | primary frozen pin (A5) |
| Free energy \(F\) (DIIS twin) | −92.1263306084 Ha | — | same grids as OT |
| \(\lvert F_{\mathrm{OT}}-F_{\mathrm{DIIS}}\rvert\) | **1.15×10⁻⁷ Ha** | ≤ \(10^{-4}\) Ha | identity gate (A3) |
| \(N_{\mathrm{occ}}=\sum w f\) | **18.0** | \(10^{-5}\) | valence count |
| OT iters (first green) | 12 | — | diagnostic |

Absolute free energy is **not** compared to CP2K as a product gate: grid /
`EPS_DEFAULT` / rel-cutoff defaults still leave ~0.6 Ha class offset.
Electron count + OT≡DIIS identity + frozen ATV pin are the acceptance
surface. Same system class as `reg_gpw-metallic-smearing`.

### Offline CP2K twin (algorithm / SMEAR documentation)

| Item | Value |
| ---- | ----- |
| Code | **CP2K 2026.1** (same class as metallic-smearing) |
| Input | `tools/cp2k_oracle/inputs/ot-smearing/na_ot_fermi_dirac.inp` |
| Method | Quickstep GPW, PADE, GTH-PBE-q9 + DZVP-MOLOPT-SR-GTH |
| Cell | BCC Na conventional \(a=4.225\) Å |
| SCF | DIIS / Broyden mix; **not** `&OT` (CP2K OT ↛ SMEAR) |
| SMEAR | FERMI_DIRAC, \(T_e = 3157.75\) K (\(\approx 0.02\) Ry \(k_B T\)) |
| ADDED_MOS | 4 |
| k-points | Monkhorst–Pack \(2\times2\times2\) |
| Example \(E_{\mathrm{tot}}\) | **−92.76593904 Ha** (not ATV pin) |

Oracle path: `tools/cp2k_oracle/README.md` (offline only).

Offline sketch (host with CP2K module; not ctest):

```bash
module load ... CP2K/2026.1
export CP2K_DATA_DIR=.../share/cp2k/data
cp2k.psmp -i tools/cp2k_oracle/inputs/ot-smearing/na_ot_fermi_dirac.inp -o ot.out
# ENERGY| Total FORCE_EVAL ( QS ) energy [hartree]  ~ -92.76593904
```

## Performance

Source: fair wall table in
`regressions/gpw/ot-smearing/REFERENCE.md` (2026-07-26). Handoff /
queue record: ratio **~0.20**. Gate \(r\le 10\) unless `perf_exempt`.

Comparable settings: cutoff 150 Ry, PADE, multi-k \(2^3\), single rank,
`OMP_NUM_THREADS=4`.

| Twin | Wall [s] | \(r = T_{\mathrm{ATV}}/T_{\mathrm{CP2K}}\) | ≤10× | Caveat |
| ---- | -------: | ----------------------------------------: | ---- | ------ |
| CP2K DIIS+FD (`na_ot_fermi_dirac.inp`) | **~89.5** | — | — | same class as metallic-smearing twin |
| ATV OT+FD only (fair) | **~18.2** | **~0.20** | ✓ | primary ratio |
| ATV OT+DIIS (full regression binary) | longer | — | — | **not** a fair twin — identity gate also runs DIIS |

\[
r = T_{\mathrm{atv}} / T_{\mathrm{cp2k}} \approx 18.2 / 89.5 \approx 0.20 \ll 10
\]

`within_10x = true`. **Caveats.** (1) Report the **OT-only** wall; the
regression binary also runs the DIIS twin for identity and must not be
ratioed against CP2K alone without splitting. (2) CP2K twin is DIIS+SMEAR,
not OT+SMEAR — ratio is dual-gate throughput, not a same-algorithm OT
race. (3) Dense OT preconditioner cost on small cells does not forecast
sparse/linear-scaling OT. (4) **ctest never launches CP2K.**

## Limitations

- **Closed-shell multi-k RKS only** for OT+smear. Multi-k UKS OT + FD/MP
  remains fail-fast (`A6`); use `gpw-uks-smearing` (DIIS UKS smear) or
  `ot-uks-kpoints` (UKS OT, integer occupations).
- **Absolute energy is ATV-frozen, not CP2K-matched.** Product gates are
  OT≡DIIS free-energy identity, \(N_e\), and the frozen Ha pin; CP2K
  documents SMEAR settings / algorithm only (~0.6 Ha class offset).
- **Stock CP2K has no OT+SMEAR product twin** — offline input is DIIS +
  FERMI_DIRAC by design.
- **No cold / Marzari–Vanderbilt** public enum; e2e gate is Fermi–Dirac
  (MP available via members + unit tests).
- **OT + DFT+U + smear** multi-k combo not this pin (see `ot-dftu` /
  `ot-dftu-kpoints`).
- **Forces / stress / BOMD** under OT+smear are not acceptance pins here.
- **CUDA GPW** OT/smear out of scope.
- **Speed.** \(r\approx 0.20\) on the documented dual; no multi-system
  standing CP2K perf suite in ctest.

## References

- Offline pin + fair wall: `regressions/gpw/ot-smearing/REFERENCE.md`
- Regression binary: `regressions/gpw/ot-smearing/reg_ot-smearing.cpp`
- CP2K twin input: `tools/cp2k_oracle/inputs/ot-smearing/na_ot_fermi_dirac.inp`
- Oracle notes: `tools/cp2k_oracle/README.md`
- Handoff ratio **~0.20**: the case `REFERENCE.md`
- Roadmap: `docs/roadmap/dft.md` §10 (OT + metallic smear ✓)
- Public header: `include/atomiverse/cpu/gpw/gpw_solver.h`
- Internal OT (fractional gradient): `include/atomiverse/cpu/gpw/internal/ot.h`
- Related tutorials: [ot-minimizer-water](ot-minimizer-water.md) (Γ OT,
  fixed occ); [pw-smearing-al](pw-smearing-al.md) (PW metallic FD)
- Related regressions: `reg_gpw-metallic-smearing`, `reg_ot-kpoints`,
  `reg_ot-uks-kpoints`, `reg_gpw-uks-smearing`
- Literature: VandeVondele & Hutter, *J. Chem. Phys.* **118**, 4365 (2003)
  (OT); Kühne et al., *J. Chem. Phys.* **152**, 194103 (2020) §VII C;
  Mermin, *Phys. Rev.* **137**, A1441 (1965); Methfessel & Paxton,
  *Phys. Rev. B* **40**, 3616 (1989)
- Narrative GPW background: [ch09-gpw](../ch09-gpw.md)
