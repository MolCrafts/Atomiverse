# Periodic Γ-point unrestricted direct RPA (URPA@UKS) — sc-Si₂ / DZVP-MOLOPT-GTH

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `periodic-urpa-si` |
| Regression | `regressions/gpw/periodic-urpa-si/` |
| Public API | `cpu::gpw::GpwDft::direct_rpa(atoms)` → `cpu::gpw::RpaResult` after Γ **UKS** `run()` (`cpu/gpw/rpa.h`, `gpw_solver.h`) |
| Roadmap | handoff **perf_exempt**; roadmap §10 Periodic RPA residual prose (URPA@UKS) |

## Summary

**Unrestricted direct RPA (URPA)** correlation on top of a converged **Γ-point
UKS GPW PBE** reference with closed-shell occupations \(n_\alpha = n_\beta = 4\)
on artificial sc-Si₂ (cubic \(a = 5.43\) Å, DZVP-MOLOPT-GTH + GTH-PBE-q4,
density cutoff **30 Ry**, periodic XYZ). Screening is **spin-conserving dRPA**
over α→α ⊕ β→β transitions (Coulomb kernel only; spins explicit — no
closed-shell factor 2). Occupied–virtual MO ERIs \((ia|jb)\) come from the
product **RI-GPW** path (same machinery as `periodic-rpa-si` / UMP2); correlation
is the **Furche plasmon formula** on the composite OV spectrum. When
\(C^\alpha \approx C^\beta\), **URPA ≡ RKS dRPA** — the product gate dual-gates
against the closed-shell first-green \(E_c\). Public path: concrete
`GpwDft` members → `build` → `run` → `direct_rpa` — no public `*Config`, no
factories. Fair wall vs CP2K is **`perf_exempt`** (CP2K RI-RPA ≠ plasmon
algorithm; no fair twin required on the pin host).

## Method

**Level.** Direct random-phase approximation as a **post-KS correlation energy**
on GPW orbitals (periodic analogue of molecular `RHF::direct_rpa`, open-shell
channel). Not TDHF/TDDFT Casida excitation spectra, not RPAX/SOSEX, not
frequency-integrated ACFDT product paths.

**Reference.** Γ-point **UKS-PBE** (`n_alpha`, `n_beta`). The regression deck uses
closed-shell occupations (\(n_\alpha = n_\beta = 4\), multiplicity 1) so that
the unrestricted path reduces to closed-shell dRPA when orbitals match — a
**setup-integrity dual-gate**, not a high-spin open-shell pin. Multi-k URPA is
a follow-on (`direct_rpa` throws on UKS + `n_kpts > 0`).

**Kernel (spin-conserving, Coulomb only).** Same-spin OV tensors
\(V^{\alpha\alpha}\), \(V^{\beta\beta}\) and cross-spin \(V^{\alpha\beta}\) from
`mo_eri_ovov{,_ri}` / `mo_eri_ovov_ab{,_ri}`. Casida blocks over the composite
OV space \(\alpha{\to}\alpha \oplus \beta{\to}\beta\) with **no** closed-shell
factor 2; spins are explicit. Kernel is Coulomb-only
(\(\alpha_{\mathrm{hf}} = 0\)).

**Correlation.** Furche plasmon formula (exact spectral reduction; no
imaginary-frequency quadrature):

\[
E_c = \tfrac12\Bigl(\sum_s \Omega_s - \mathrm{Tr}\,A\Bigr),
\]

with \(\Omega_s > 0\) the positive eigenvalues of the Hermitian dRPA reduction.
Implemented in `internal::direct_rpa_uhf`. Total energy is **dRPA@KS**:

\[
E_{\mathrm{tot}} = E_{\mathrm{SCF}} + E_c
\]

(no EXX replacement — periodic exact exchange remains blocked).

**\((ia|jb)\) route.** Product RI-GPW when `mp2_ri = true` and the cell is
periodic; else Poisson-per-pair foothold. The regression cross-checks
\(|E_c^{\mathrm{RI}} - E_c^{\mathrm{can}}| \le 10^{-8}\) Ha. Virtual space
capped per spin by `mp2_n_virt` (0 = all kept virtuals).

**Observables (`RpaResult`)**

| Symbol | Field | Unit | Notes |
| ------ | ----- | ---- | ----- |
| \(E_c\) | `e_corr` | Ha | typically \(< 0\) |
| \(E_{\mathrm{SCF}}\) | `e_scf` | Ha | last UKS total |
| \(E_{\mathrm{tot}}\) | `e_total` | Ha | `e_scf + e_corr` |
| \(\mathrm{Tr}\,A\) | `tr_A` | Ha | composite α⊕β (≈ 2 × RKS when α=β) |
| \(n_{\mathrm{occ}}\) | `n_occ` | — | spin-orbital \(n_\alpha+n_\beta\) |
| \(n_{\mathrm{virt}}\) | `n_virt` | — | spin-orbital after cap |
| \(\Omega_s\) | `omega_ha` | Ha | composite OV length \(n_{\mathrm{ov}}^\alpha+n_{\mathrm{ov}}^\beta\) |
| RI flag | `ri` | bool | product RI-GPW path |

Internal SCF/RPA pins are reported in **Hartree** (cited). Public Frame energy
for evaluator paths remains **eV** (`kEnergy`) — this regression drives
`GpwDft` directly and asserts Ha fields on `RpaResult`.

**What this pin includes**

| Item | Setting |
| ---- | ------- |
| Crystal | artificial sc-Si₂ (same deck as `periodic-rpa-si` / `periodic-ump2-si`) |
| Cell | cubic **a = 5.43 Å**, periodic XYZ, **Γ only** |
| Positions | Si (0,0,0), Si (a/4,a/4,a/4) |
| Basis / PP | **DZVP-MOLOPT-GTH** / **GTH-PBE-q4** |
| XC / spin | **PBE**, UKS \(n_\alpha = n_\beta = 4\) |
| Cutoff | **30 Ry** |
| SCF | tol \(10^{-6}\), mix 0.3, level_shift 0.3 Ha, max_iter 80 |
| RPA | all kept virtuals per spin; `mp2_ri = true` (product + Poisson dual) |

**Key exclusions (non-goals of this slug)**

- Multi-k URPA / finite-\(q\) unrestricted screening (follow-on).
- Absolute digit match to CP2K **frequency-integrated** RI-RPA + EXX total
  (algorithmically different; do **not** invent CP2K \(E_c\) as a plasmon twin).
- High-spin open-shell URPA (product dual-gates closed-shell occupations).
- RPAX / SOSEX / hybrid screening kernels; OT references (no canonical virtuals).
- Live CP2K inside ctest; public `*Config` / `make_*` / `build_*`.

## Implementation

Public surface only (`GpwDft` members + lookup constructors; matches
`reg_periodic-urpa-si.cpp`):

```cpp
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/cpu/gpw/rpa.h"
#include "atomiverse/molrs.hpp"

using namespace ATV::cpu::gpw;

constexpr double kA2B = 1.8897259885789;
constexpr double a_bohr = 5.43 * kA2B;

const GaussianBasis bas(molrs::Element::Si, "DZVP-MOLOPT-GTH");
const GthPotential pot(molrs::Element::Si, "GTH-PBE-q4");
const double pos0[3] = {0.0, 0.0, 0.0};
const double pos1[3] = {a_bohr / 4.0, a_bohr / 4.0, a_bohr / 4.0};

std::vector<GpwAtom> atoms;
atoms.emplace_back(pos0, bas, pot, 14);
atoms.emplace_back(pos1, bas, pot, 14);

GpwDft m;
m.cell[0] = m.cell[1] = m.cell[2] = a_bohr;  // Bohr
m.cutoff_ry = 30.0;
m.n_alpha = 4;
m.n_beta = 4;               // UKS closed-shell occupations
m.max_iter = 80;
m.tol = 1e-6;
m.mix = 0.3;
m.level_shift = 0.3;        // small-gap sc-Si₂ aid (eps unshifted for RPA)
m.xc = XcKind::PBE;
m.mp2_ri = true;            // product RI-GPW (ia|jb)
m.build(atoms);
const GpwResult& scf = m.run(atoms);  // scf.converged, scf.e_total [Ha]

m.mp2_n_virt = 0;           // 0 = all kept virtuals per spin
const RpaResult& r = m.direct_rpa(atoms);
// r.e_corr, r.e_total [Ha]; r.omega_ha (composite α⊕β OV); r.ri == true
```

Guards (throw, never silent): `logic_error` before `build()` / converged
`run()` or missing β orbitals; `invalid_argument` for multi-k UKS, `use_ot`,
empty virtual channels, or vanishing HOMO–LUMO gap (< 1e-6 Ha);
`runtime_error` if the dRPA reduction is ill-conditioned.

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
  cmake --build build --target reg_periodic-urpa-si -j$(nproc)
  ./build/regressions/reg_periodic-urpa-si
'
```

| Path | Role |
| ---- | ---- |
| `include/atomiverse/cpu/gpw/gpw_solver.h` | `GpwDft::direct_rpa`, UKS members |
| `include/atomiverse/cpu/gpw/rpa.h` | `RpaResult` public value type |
| `src/cpu/gpw/rpa.cpp` | Γ URPA dispatch → `internal::direct_rpa_uhf` |
| `include/atomiverse/cpu/gpw/gaussian_basis.h` | `GaussianBasis(Element, name)` |
| `include/atomiverse/cpu/gpw/gth_potential.h` | `GthPotential(Element, name)` |
| `regressions/gpw/periodic-urpa-si/reg_periodic-urpa-si.cpp` | Product gate |
| `tools/cp2k_oracle/inputs/periodic-urpa-si.inp` | Offline CP2K twin (never ctest) |

## Validation (vs CP2K)

Source:
[`regressions/gpw/periodic-urpa-si/REFERENCE.md`](../../../regressions/gpw/periodic-urpa-si/REFERENCE.md).

**Oracle status.** Input twin:
`tools/cp2k_oracle/inputs/periodic-urpa-si.inp` (UKS MULT 1 + `&RI_RPA`
MINIMAX). CP2K’s default path is **frequency-integrated RI-RPA with Gaussian
RI_AUX and EXX total-energy assembly**, not the exact plasmon formula on PW-RI
\((ia|jb)\). Absolute CP2K \(E_c\) digits are **not invented** as a hard twin of
the plasmon value. Pin strategy (binding; mirrors `periodic-rpa-si` /
`periodic-ump2-si`):

1. **Hard internal gates** — SCF converged; \(E_c < 0\); all \(\Omega > 0\);
   \(E_{\mathrm{tot}} = E_{\mathrm{SCF}} + E_c\); RI ≡ Poisson ≤ \(10^{-8}\) Ha.
2. **Dual-gate band** vs RKS dRPA first-green on the same deck
   (\(|E_c^{\mathrm{URPA}} - E_c^{\mathrm{RKS}}| \le 2\times 10^{-3}\) Ha).
3. Offline re-pin when a fair plasmon-equivalent CP2K URPA path exists —
   replace the band with cited correlation digits (CP2K version required).

**Setup:**

| Item | Value |
| ---- | ----- |
| System | sc-Si₂, charge 0, UKS \(n_\alpha=n_\beta=4\) (MULT 1) |
| Lattice | cubic \(a = 5.43\) Å, periodic XYZ, Γ only |
| Positions [Å] | Si (0, 0, 0); Si (1.3575, 1.3575, 1.3575) |
| Basis / PP | DZVP-MOLOPT-GTH / GTH-PBE-q4 |
| XC | PBE |
| Density cutoff | 30 Ry |
| SCF | mix 0.3, level_shift 0.3 Ha, max_iter 80, tol \(10^{-6}\) |
| RPA | RI-GPW + Poisson cross-check; all virtuals |

**Accuracy gate (product):**

| Quantity | Atomiverse / pin | Oracle / twin | Tol | Notes |
| -------- | ---------------: | ------------: | --- | ----- |
| \(E_c\) (URPA first-green, 2026-07-26) | **−0.2658538251 Ha** | dual-gate vs RKS | — | Δ vs RKS ≈ \(1.5\times 10^{-7}\) Ha |
| \(E_c\) (RKS dRPA first-green, `periodic-rpa-si`) | **−0.2658539776 Ha** | closed-shell twin | **\(2\times 10^{-3}\) Ha** band | regression hard pin |
| \(E_{\mathrm{SCF}}\) (UKS-PBE) | **−7.5955445033 Ha** | matches RKS ~\(4\times 10^{-9}\) | soft | REFERENCE diagnostic |
| \(\mathrm{Tr}\,A\) (URPA composite) | 136.534225 | ≈ 2 × RKS Tr A | soft | α⊕β OV |
| \(n_{\mathrm{occ}}\) / \(n_{\mathrm{virt}}\) / \(n_\Omega\) | 8 / 44 / 176 | spin-orbital composite | hard | product gate |
| RI vs Poisson \(E_c\) | \(|\Delta| \le 10^{-8}\) Ha | internal | **hard** | `mp2_ri` on/off |
| SCF | converged | — | `converged == true` | product gate |
| \(\Omega_s\) | all \(> 0\) | — | hard | every root |

Unit tier (`tests/cpu/gpw/test_cpu_gpw_rpa.cpp`): `direct_rpa_uhf` \(E_c\)
dual-gates `direct_rpa_rhf` on identical α=β tensors; UKS unconverged guard.

Offline re-pin (human only; **never** ctest):

```bash
# when a CP2K binary is available — diagnostic only (algorithm differs)
cp2k.psmp -i tools/cp2k_oracle/inputs/periodic-urpa-si.inp \
          -o /tmp/periodic-urpa-si.out
# record RI-RPA correlation only as near-twin diagnostic; do not invent
# plasmon-equivalent digits without a documented algorithm match
```

## Performance

Source: [`REFERENCE.md`](../../../regressions/gpw/periodic-urpa-si/REFERENCE.md);
queue/handoff **perf_exempt** (wave 7, 2026-07-26).

| Item | Value |
| ---- | ----- |
| `perf_exempt` | **true** |
| Reason | CP2K RI-RPA (frequency + EXX path) ≠ plasmon dRPA algorithm; no fair wall twin; no CP2K binary required on pin host. |
| \(T_{\mathrm{ATV}}\) (8 OpenMP, EPYC login) | SCF ≈ 14 s + RI-URPA ≈ 0.7 s + Poisson cross-check ≈ 2.7 s → **~18 s** total |
| Wall budget | ≤ 60 s (e2e regression) |
| \(r = T_{\mathrm{ATV}}/T_{\mathrm{CP2K}}\) | **n/a (exempt)** |
| `within_10x` | n/a (exempt) |

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | -----------------------: | ----------------------: | ----: | ---- | ------ |
| sc-Si₂ URPA@UKS PBE (30 Ry, Γ, RI) | — | ~18 (SCF+RI+Poisson) | **exempt** | n/a | Algorithm mismatch vs CP2K RI-RPA; remeasure only if plasmon-equivalent twin is defined |

Campaign records the same: ratio **perf_exempt**.

## Limitations

- **Dual-gate + internal gates**, not a published CP2K plasmon \(E_c\) digit
  match. CP2K `&RI_RPA` MINIMAX + HF(EXX) total is a different energy
  definition — treat offline prints as diagnostics only until algorithms align.
- **Γ UKS only.** Multi-k / finite-\(q\) URPA throws; unrestricted k-point
  screening is a follow-on.
- **Closed-shell occupations** on the product deck (\(n_\alpha=n_\beta\)).
  High-spin open-shell URPA physics is not the e2e gate (unit dual-gate uses
  α=β tensors).
- **dRPA@KS total** — no EXX replacement; periodic hybrid remains blocked.
- **`perf_exempt`**: do not invent a fair \(r\); wall target is the ≤ 60 s
  regression budget.
- **CPU only** — CUDA GPW remains blocked on the CP2K loop.
- Basis/PP tables are static in-tree headers; unsupported element/name
  **throws** (no silent fallback).
- Artificial sc-Si₂ at 30 Ry is a **method foothold**, not production bulk Si
  (diamond Si₈ / production cutoffs are separate decks).

## References

- Offline pin: `regressions/gpw/periodic-urpa-si/REFERENCE.md`
- Regression consumer: `regressions/gpw/periodic-urpa-si/reg_periodic-urpa-si.cpp`
- Oracle twin: `tools/cp2k_oracle/inputs/periodic-urpa-si.inp`
- Oracle notes: `tools/cp2k_oracle/README.md`
- Roadmap: `docs/roadmap/dft.md` § Periodic RPA / URPA@UKS
- Related method pages: [periodic-mp2-si](periodic-mp2-si.md) (Γ MP2 foothold),
  [direct-rpa-h2-sto3g](direct-rpa-h2-sto3g.md) (molecular plasmon dRPA),
  [gpw-uks-o2-gamma](gpw-uks-o2-gamma.md) (Γ UKS energy)
- Public API: `include/atomiverse/cpu/gpw/gpw_solver.h`,
  `include/atomiverse/cpu/gpw/rpa.h`
- F. Furche, *J. Chem. Phys.* **114**, 5982 (2001) — plasmon RPA correlation.
- X. Ren et al., *New J. Phys.* **14**, 053020 (2012) — RPA in condensed matter.
- M. Del Ben, J. Hutter, J. VandeVondele, *J. Chem. Theory Comput.* **8**, 4177
  (2012) — GPW \((ia|jb)\) / WF correlation route.
- J. P. Perdew, K. Burke, M. Ernzerhof, *Phys. Rev. Lett.* **77**, 3865 (1996)
  — PBE.
- G. Lippert, J. Hutter, M. Parrinello, *Mol. Phys.* **92**, 477 (1997);
  *Theor. Chem. Acc.* **103**, 124 (1999) — GPW / Quickstep.
- S. Goedecker, M. Teter, J. Hutter, *Phys. Rev. B* **54**, 1703 (1996) — GTH.
- J. VandeVondele, J. Hutter, *J. Chem. Phys.* **127**, 114105 (2007) — MOLOPT.
