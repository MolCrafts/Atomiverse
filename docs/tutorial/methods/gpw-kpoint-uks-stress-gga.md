# Multi-k UKS GPW analytic cell stress — H₂ / RevPBE·RPBE·BLYP (GGA)

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `gpw-kpoint-uks-stress-gga` |
| Regression | `regressions/gpw/gpw-kpoint-uks-stress-gga/` |
| Public API | `cpu::gpw::GpwDft` + `cpu::gpw::gpw_stress` / `gpw_stress_fd`; members `n_alpha`/`n_beta`, `k_mesh`, `xc` (`RevPBE` \| `RPBE` \| `BLYP`) |
| Roadmap | [dft.md](../../roadmap/dft.md) § Periodic DFT / GPW — multi-k UKS stress for pure GGA |
| Queue | CP2K capability loop — harness wave 11 done (2026-07-26) |

## Summary

**Collinear unrestricted Kohn–Sham (UKS) Gaussian-and-plane-waves (GPW)** with
**Monkhorst–Pack k-point sampling** and **Hellmann–Feynman fixed-\(P_k^\sigma\)
analytic cell stress** for pure **spin-GGA** exchange–correlation: **RevPBE**,
**RPBE**, and **BLYP**. Product path is the same fixed-density-matrix strain
theorem as closed-shell / PADE multi-k UKS stress
([`gpw-kpoint-uks-stress`](../../../regressions/gpw/gpw-kpoint-uks-stress/REFERENCE.md)),
extended so the non-self-consistent Lagrangian rebuilds **spin GGA XC** under
orthorhombic diagonal strain.

Textbook twin: high-spin H₂ (multiplicity 3, \(N_\alpha=2\), \(N_\beta=0\)),
compressed bond, cubic 12 Bohr cell, **1×1×2** MP mesh, **DZV-GTH-PADE** /
**GTH-PADE-q1**, cutoff 100 Ry. Decisive gate is **analytic vs multi-SCF
strain-FD** on the same method (not a live CP2K absolute stress pin). Soft
energy pins and an **activation** check \(\lvert E_{\mathrm{RevPBE}}-E_{\mathrm{PBE}}\rvert\)
ensure GGA is actually live. Public energy on Frame remains **eV**; pin
channels and stress use **Ha / Ha·Bohr⁻³**.

Product gate: `reg_gpw-kpoint-uks-stress-gga`. Fair wall vs same-basis CP2K
NUMERICAL stress is **`perf_exempt`** (DZV-GTH-PADE not in `BASIS_MOLOPT`;
documentation SZV twins only). Oracle contract: offline pins under
`tools/cp2k_oracle/` — **never** live CP2K in ctest.

Related narrative: [Ch. 9 GPW](../ch09-gpw.md). Predecessor stress pages:
[`gpw-analytic-stress`](gpw-analytic-stress.md) (Γ RKS fixed-\(P\)),
[`gpw-stress-water`](gpw-stress-water.md) (Γ strain-FD / CP2K NUMERICAL).

## Method

**GPW (Quickstep family).** Wavefunction in a contracted Gaussian AO basis;
density collocation on a uniform real-space grid; Hartree via FFT Poisson; XC
on the grid; potential integrated back into the AO KS matrix. Core electrons
removed by analytic **GTH** pseudopotentials. Grid resolution set by the
plane-wave cutoff in Rydberg. Citation: Lippert, Hutter, Parrinello,
*Mol. Phys.* **92**, 477 (1997); VandeVondele et al., *Comput. Phys. Commun.*
**167**, 103 (2005).

**UKS + k-points.** With \(N_\alpha \neq N_\beta\), spin densities
\(\rho_\alpha\), \(\rho_\beta\) drive separate KS matrices; occupations are
explicit integer members `n_alpha` / `n_beta`. A **Monkhorst–Pack** mesh
(`k_mesh`) yields Bloch MOs \(C_{\mathrm{occ},k}^\sigma\), per-k density and
energy-weighted density matrices \(P_k^\sigma\), \(W_k^\sigma\) (occupancy 1
per spin channel here), and BZ-averaged total energy. Time-reversal folds the
1×1×2 mesh to \(\kappa_z=\pm 1/4\) with weight 1 (`n_kpts` reported on the
result may be the folded count).

**Spin-GGA XC.** Pure GGAs use \(\rho^\sigma\) and \(\lvert\nabla\rho^\sigma\rvert\)
(and total-density gradient terms per functional) on the collocation grid.
This product expands multi-k UKS stress beyond PADE LSD / PBE-class
footholds to **RevPBE**, **RPBE**, and **BLYP** (`XcKind`). Hybrids and
meta-GGA remain out of scope.

**Hellmann–Feynman fixed-\(P_k^\sigma\) analytic cell stress.** After one
multi-k UKS SCF:

1. Rebuild \(P_k^\sigma\), \(W_k^\sigma\) from occupied Bloch MOs and band
   energies.
2. Central finite difference of the **non-self-consistent** Lagrangian
   \[
   E_L\bigl[\{P_k^\sigma\},\{W_k^\sigma\};\varepsilon\bigr]
   \]
   under orthorhombic diagonal strain \(\varepsilon\) at **fixed fractional
   coordinates**: rebuild \(S_k\), \(T_k\), \(V_{\mathrm{nl}}^k\), grid,
   Poisson, and **spin GGA XC**; do **not** re-converge SCF.
3. Stress (CP2K `STRESS_TENSOR` sign; \(E\) Hartree, \(\Omega\) Bohr³):
   \[
   \sigma_{aa}
   =
   -\Omega^{-1}
   \frac{\partial E_L}{\partial\varepsilon_{aa}},
   \qquad a\in\{x,y,z\}.
   \]

Default strain step \(h = 2\times 10^{-3}\). Voigt order
\((\sigma_{xx},\sigma_{yy},\sigma_{zz},\sigma_{xy},\sigma_{xz},\sigma_{yz})\);
product off-diagonals must be \(\approx 0\) for orthorhombic v1.

**Dual-gate oracle.** Multi-SCF strain-FD (`gpw_stress_fd`) on the **same**
UKS+k+GGA method is the decisive twin (measured residuals \(\sim 10^{-10}\)–
\(10^{-9}\) Ha/Bohr³). CP2K `STRESS_TENSOR NUMERICAL` inputs exist as
**documentation** twins (SZV-MOLOPT-GTH, not DZV-GTH-PADE); absolute
same-basis CP2K stress is pending.

**Activation.** On the same geometry/mesh, \(\lvert E_{\mathrm{RevPBE}}-E_{\mathrm{PBE}}\rvert\)
must exceed \(1\times 10^{-4}\) Ha so the product cannot pass on a
PBE-mislabelled path.

**Key exclusions**

- Closed-form term tables without fixed-\(P_k\) samples; shear / triclinic
  strain (off-diag product zeros).
- Hybrid / meta-GGA / GAPW / PAW / CUDA GPW.
- Live CP2K from ctest; public `*Config` bags / `make_*` factories.
- Same-basis CP2K fair NUMERICAL wall (see Performance).
- Molecular post-HF (PySCF loop) and classical electrochem MD (MetalWalls).

## Implementation

Public surface only: `GpwAtom` + lookup constructors
(`GaussianBasis` / `GthPotential`), `GpwDft` members, free `gpw_stress` /
`gpw_stress_fd`. **No** public `*Config`, **no** `make_*` / `build_*`
factories. Match the regression:

```cpp
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/molrs.hpp"

using namespace ATV::cpu::gpw;

const double L = 12.0;  // Bohr
const GaussianBasis bH(molrs::Element::H, "DZV-GTH-PADE");
const GthPotential pH(molrs::Element::H, "GTH-PADE-q1");
const double r0[3] = {0.5 * L - 0.5, 0.5 * L, 0.5 * L};
const double r1[3] = {0.5 * L + 0.5, 0.5 * L, 0.5 * L};
std::vector<GpwAtom> atoms{
    GpwAtom(r0, bH, pH, 1),
    GpwAtom(r1, bH, pH, 1),
};

auto make_uks = [&](XcKind xc) {
    GpwDft m;
    m.cell[0] = m.cell[1] = m.cell[2] = L;
    m.cutoff_ry = 100.0;
    m.n_alpha = 2;
    m.n_beta = 0;
    m.max_iter = 60;
    m.tol = 1e-8;
    m.xc = xc;
    m.local_pp_on_grid = true;
    m.k_mesh[0] = 1;
    m.k_mesh[1] = 1;
    m.k_mesh[2] = 2;
    return m;
};

// Energy soft pin path
GpwDft energy = make_uks(XcKind::RevPBE);
energy.build(atoms);
const auto& er = energy.run(atoms);  // er.e_total [Ha]

// Analytic product stress vs multi-SCF FD dual-gate
const double h = 2e-3;
const auto san = gpw_stress(atoms, make_uks(XcKind::RevPBE), h);   // Voigt Ha/Bohr³
const auto sfd = gpw_stress_fd(atoms, make_uks(XcKind::RevPBE), h);
```

`GPWEvaluator` can also request product stress on a Frame path
(`compute_stress`, `stress_use_fd=false` default, `last_stress_ha_bohr3()`,
Frame `kStress` in eV/Å³) — this regression exercises the free `gpw_stress`
API on `GpwDft` with explicit multi-k UKS + GGA members.

Build and run (scoped targets only):

```bash
bash -lc '
  module load buildenv-gcc/2026.03-mpich
  export OPENBLAS_ROOT="$OPENBLASDIR"
  export RUSTUP_TOOLCHAIN=stable-x86_64-unknown-linux-gnu
  export LD_LIBRARY_PATH="$LIBRARY_PATH:$LD_LIBRARY_PATH"
  cmake -B build -DCMAKE_BUILD_TYPE=Release \
        -DATV_BUILD_TESTS=ON -DATV_BUILD_HF=ON -DATV_BUILD_PW=ON \
        -DATV_BUILD_GPW=ON -DATV_BUILD_REGRESSIONS=ON \
        -DATV_BUILD_CUDA=OFF -DBLA_VENDOR=OpenBLAS
  cmake --build build -j$(nproc) --target atomiverse reg_gpw-kpoint-uks-stress-gga
  ./build/regressions/reg_gpw-kpoint-uks-stress-gga
  # or: cmake --build build --target run_reg_gpw-kpoint-uks-stress-gga
'
```

| Path | Role |
| ---- | ---- |
| `include/atomiverse/cpu/gpw/gpw_solver.h` | `GpwDft`, `gpw_stress`, `gpw_stress_fd` |
| `include/atomiverse/cpu/gpw/gpw_evaluator.h` | Frame-facing stress members (product path) |
| `src/cpu/gpw/internal/gpw_stress.cpp` | Fixed-\(P_k^\sigma\)+Pulay + multi-SCF FD |
| `include/atomiverse/cpu/gpw/gaussian_basis.h` | `GaussianBasis(Element, "DZV-GTH-PADE")` |
| `include/atomiverse/cpu/gpw/gth_potential.h` | `GthPotential(Element, "GTH-PADE-q1")` |
| `regressions/gpw/gpw-kpoint-uks-stress-gga/reg_gpw-kpoint-uks-stress-gga.cpp` | Product dual-gate + activation |
| `regressions/gpw/gpw-kpoint-uks-stress-gga/REFERENCE.md` | Hard-coded pins + tolerances |
| `tools/cp2k_oracle/inputs/gpw-kpoint-uks-stress-gga/` | CP2K NUMERICAL documentation twins |

## Validation (vs CP2K / dual-gate)

Source:
[`regressions/gpw/gpw-kpoint-uks-stress-gga/REFERENCE.md`](../../../regressions/gpw/gpw-kpoint-uks-stress-gga/REFERENCE.md).
Oracle: offline **CP2K** documentation twins + **ATV dual-gate** (analytic vs
`gpw_stress_fd`). Hard-coded pins in the regression binary — **never** live
CP2K from ctest.

### Twin settings (binding)

| Knob | Value |
| ---- | ----- |
| System | High-spin H₂ (M = 3), compressed H–H = 1.0 Bohr along \(x\) |
| Positions [Bohr] | H1 (5.5, 6.0, 6.0), H2 (6.5, 6.0, 6.0) |
| Cell | Cubic **12 Bohr**, periodic XYZ |
| Basis | **DZV-GTH-PADE** (H) — in-tree table |
| PP | **GTH-PADE-q1** |
| XC product | **RevPBE / RPBE / BLYP** (spin GGA) |
| XC activation | **PBE** on same geometry |
| Cutoff | 100 Ry, `local_pp_on_grid = true` |
| k mesh | MONKHORST-PACK **1×1×2** (TR → \(\kappa_z=\pm 1/4\), \(w=1\)) |
| Spin | \(N_\alpha=2\), \(N_\beta=0\) |
| Stress step | \(h = 2\times 10^{-3}\) |
| Observable | Voigt \(\sigma\) [Ha/Bohr³]; \(E_{\mathrm{tot}}\) [Ha] soft |

### Dual-gate + soft energy pins (decisive; first green 2026-07-26)

| Quantity | Pin / measured | Tol (ATV gate) | Notes |
| -------- | -------------: | -------------- | ----- |
| \(E\) (k-UKS PBE 1×1×2, 100 Ry) | **−0.566633887639 Ha** | activation baseline | Same geometry as GGA cases |
| \(E\) (k-UKS **RevPBE**) | **−0.574211377589 Ha** | \|E − pin\| ≤ **1×10⁻³** Ha | Soft ATV first-green pin; 6 iters |
| \(E\) (k-UKS **RPBE**) | **−0.575044480519 Ha** | same | Soft pin |
| \(E\) (k-UKS **BLYP**) | **−0.558884091671 Ha** | same | Soft pin |
| \(\lvert E_{\mathrm{RevPBE}}-E_{\mathrm{PBE}}\rvert\) | **7.577×10⁻³ Ha** | ≥ **1×10⁻⁴** Ha | Activation (GGA live) |
| max\|\(\sigma_{\mathrm{an}}-\sigma_{\mathrm{fd}}\)\| RevPBE | **4.63×10⁻¹⁰** Ha/Bohr³ | ≤ **1×10⁻⁴** | Decisive dual-gate |
| max\|\(\sigma_{\mathrm{an}}-\sigma_{\mathrm{fd}}\)\| RPBE | **5.60×10⁻¹⁰** Ha/Bohr³ | same | |
| max\|\(\sigma_{\mathrm{an}}-\sigma_{\mathrm{fd}}\)\| BLYP | **1.54×10⁻⁹** Ha/Bohr³ | same | |
| Off-diagonal product | ~0 | ≤ **1×10⁻⁸** | Orthorhombic v1 |
| max\|\(\sigma_{\mathrm{diag}}\)\| floor | nonzero | \> **1×10⁻⁸** Ha/Bohr³ | Avoid trivial zero stress |

**RevPBE analytic stress (representative):**

| Component | analytic [Ha/Bohr³] |
| --------- | ------------------: |
| \(\sigma_{xx}\) | **4.22391784×10⁻⁴** |
| \(\sigma_{yy}\) | **−7.94777145×10⁻⁵** |
| \(\sigma_{zz}\) | **−7.92623369×10⁻⁵** |

### CP2K NUMERICAL twins (documentation only)

| Field | Value |
| ----- | ----- |
| Engine | CP2K 2026.1 (when available) |
| Inputs | `tools/cp2k_oracle/inputs/gpw-kpoint-uks-stress-gga/h2_uks_kpoint_stress_{revpbe,rpbe,blyp}.inp` |
| Mode | `RUN_TYPE ENERGY_FORCE`, `STRESS_TENSOR NUMERICAL`, `LSD`, `MULTIPLICITY 3`, MP 1 1 2 |
| XC | RevPBE (`PBE` `PARAMETRIZATION REVPBE`), RPBE, BLYP |
| Basis / PP in twin | **SZV-MOLOPT-GTH** / GTH-PBE-q1 — **not** DZV-GTH-PADE |
| Status | Absolute same-basis twin **pending**; pin host had no CP2K module at first green |

Offline re-run sketch (operator only; **not** ctest):

```bash
# module load ... CP2K/2026.1 when available
export CP2K_DATA_DIR=.../share/cp2k/data
cp2k.psmp -i tools/cp2k_oracle/inputs/gpw-kpoint-uks-stress-gga/h2_uks_kpoint_stress_revpbe.inp \
          -o /tmp/cp2k_oracle_h2_uks_kpoint_stress_revpbe.out
# STRESS| Numerical stress tensor [bar]  — convert with
# 1 Ha/Bohr³ = 2.94210107994716×10⁸ bar  (same table as gpw-stress-water)
```

**Tolerance rationale.** Dual-gate residual \(\sim 10^{-10}\) is far inside
\(10^{-4}\) Ha/Bohr³ (absorbs SCF / grid / strain-step noise while forcing
correct analytic path). Soft energy pins \(10^{-3}\) Ha match sibling
k-UKS force/stress footholds. Activation floor separates GGA from PBE.

## Performance

Source: `REFERENCE.md` + handoff wave 11 (the case `REFERENCE.md`).

| Item | Value |
| ---- | ----- |
| `perf_exempt` | **true** |
| Reason | No same-basis CP2K fair NUMERICAL stress wall (DZV-GTH-PADE not in `BASIS_MOLOPT`; no CP2K module on pin host) |
| \(r = T_{\mathrm{ATV}}/T_{\mathrm{CP2K}}\) | n/a (exempt) |
| `within_10x` | n/a (exempt) |
| \(T_{\mathrm{ATV}}\) (regression dual-gate, OMP=8, Release) | **~2.7 s** |

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | ------------------------: | -----------------------: | ----: | ---- | ------ |
| Multi-k UKS GGA stress (RevPBE/RPBE/BLYP dual-gate) | — | **~2.7** | n/a | exempt | Absolute CP2K fair wall deferred until same-basis data path |

Gate policy: \(r \le 10\) **or** `perf_exempt`. Here exempt by design; when
CP2K + DZV-class basis are available, replace with a single-rank fair wall
and target \(r \le 10\).

## Limitations

- **Dual-gate product, not absolute CP2K stress pin.** Documentation twins
  use SZV-MOLOPT-GTH; ATV product uses DZV-GTH-PADE. Do not claim bar-level
  CP2K stress parity until a same-basis NUMERICAL pin lands.
- **Orthorhombic diagonal only** — no triclinic shear; off-diagonal Voigt
  components are product zeros.
- **Three pure GGAs** (RevPBE, RPBE, BLYP) on multi-k UKS stress — not a
  blanket “all GGA” claim; hybrids / meta-GGA blocked elsewhere in the queue.
- **High-spin H₂ textbook geometry** — not bulk solids, metals with smearing,
  DFT+U stress (`gpw-dftu-kpoint-stress`), or dispersion stress (D3/D4).
- **`perf_exempt`** until same-basis CP2K wall exists; regression wall ~2.7 s
  is diagnostic only.
- Public Frame energy is eV; pin compare uses Hartree and Ha/Bohr³ stress.
- Unit tests under `tests/cpu/gpw/` only for pieces; e2e is exclusively
  `reg_gpw-kpoint-uks-stress-gga`.
- CPU only — no CUDA GPW.

## References

- Offline pin + dual-gate numbers:
  [`regressions/gpw/gpw-kpoint-uks-stress-gga/REFERENCE.md`](../../../regressions/gpw/gpw-kpoint-uks-stress-gga/REFERENCE.md)
- Regression consumer:
  [`reg_gpw-kpoint-uks-stress-gga.cpp`](../../../regressions/gpw/gpw-kpoint-uks-stress-gga/reg_gpw-kpoint-uks-stress-gga.cpp)
- CP2K documentation twins:
  [`tools/cp2k_oracle/inputs/gpw-kpoint-uks-stress-gga/`](../../../tools/cp2k_oracle/inputs/gpw-kpoint-uks-stress-gga/)
- Oracle contract: [`tools/cp2k_oracle/README.md`](../../../tools/cp2k_oracle/README.md)
- Predecessor PADE multi-k UKS stress:
  [`regressions/gpw/gpw-kpoint-uks-stress/REFERENCE.md`](../../../regressions/gpw/gpw-kpoint-uks-stress/REFERENCE.md)
- Γ analytic stress narrative: [`gpw-analytic-stress.md`](gpw-analytic-stress.md)
- Narrative GPW: [`docs/tutorial/ch09-gpw.md`](../ch09-gpw.md)
- Roadmap: [`docs/roadmap/dft.md`](../../roadmap/dft.md)
- Spec history (closed, deleted at done): git `wf/gpw-kpoint-uks-stress-gga`
  (`.claude/specs/gpw-kpoint-uks-stress-gga/`)
- G. Lippert, J. Hutter, M. Parrinello, *Mol. Phys.* **92**, 477 (1997) — GPW
- J. VandeVondele et al., *Comput. Phys. Commun.* **167**, 103 (2005) — Quickstep
- O. H. Nielsen, R. M. Martin, *Phys. Rev. B* **32**, 3780 (1985) — fixed-\(P\) cell stress
- Y. Zhang, W. Yang, *Phys. Rev. Lett.* **80**, 890 (1998) — RevPBE
- B. Hammer, L. B. Hansen, J. K. Nørskov, *Phys. Rev. B* **59**, 7413 (1999) — RPBE
- A. D. Becke, *Phys. Rev. A* **38**, 3098 (1988); C. Lee, W. Yang, R. G. Parr,
  *Phys. Rev. B* **37**, 785 (1988) — BLYP
