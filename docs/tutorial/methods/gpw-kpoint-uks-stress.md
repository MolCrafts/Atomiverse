# Multi-k UKS GPW analytic cell stress — high-spin H₂ / PADE / DZV-GTH-PADE

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `gpw-kpoint-uks-stress` |
| Regression | `regressions/gpw/gpw-kpoint-uks-stress/` |
| Public API | `cpu::gpw::gpw_stress` / `gpw_stress_fd` on `GpwDft` (`n_alpha`/`n_beta` + `k_mesh`); multi-k collinear UKS fixed-\(\{P_k^\sigma,W_k^\sigma\}\) |
| Related narrative | [Ch. 9 GPW](../ch09-gpw.md) |
| Queue | CP2K capability loop — wave 4 done (2026-07-26) |
| Roadmap | `docs/roadmap/dft.md` — multi-k UKS analytic cell stress |

## Summary

**Collinear unrestricted Kohn–Sham (UKS) Gaussian-and-plane-waves (GPW)**
with **Monkhorst–Pack k-point sampling** and **Hellmann–Feynman fixed-\(P_k^\sigma\)
analytic cell stress**. Product path: one multi-k UKS SCF rebuilds occupied
Bloch MOs and band energies per spin → density matrices \(P_k^\sigma\) and energy-
weighted \(W_k^\sigma\) (occupancy 1 per spin channel) → central finite difference
of the **non-self-consistent** Lagrangian under orthorhombic diagonal strain
at fixed fractional coordinates. Multi-SCF strain-FD (`gpw_stress_fd`) remains
the dual-gate oracle on the same UKS+k method.

Closed-shell identity: multi-k UKS with \(N_\alpha=N_\beta=1\) matches multi-k
RKS stress. Twin system is **high-spin H₂** (\(N_\alpha=2\), \(N_\beta=0\)) in a
periodic 12 Bohr cubic cell, PADE LSD, in-tree DZV-GTH-PADE / GTH-PADE-q1,
mesh \(1\times1\times2\). Product gate:
`regressions/gpw/gpw-kpoint-uks-stress` (`reg_gpw-kpoint-uks-stress`).
Fair wall **`perf_exempt`** (no same-basis CP2K NUMERICAL stress wall on pin
host — BASIS_MOLOPT lacks DZV-GTH-PADE; no CP2K module at first green).

Public surface: free `cpu::gpw::gpw_stress(atoms, method, h)` with members on
`GpwDft` — **no** public `*Config` bag, **no** `make_*` / `build_*` factories.
Internal energy/stress in Ha / Ha·Bohr⁻³; Frame energy still eV when routed
through Evaluate (this regression exercises the solver free path directly).

Deps closed earlier: `gpw-kpoints-uks`, `gpw-kpoint-stress`, `gpw-uks-stress`,
`gpw-kpoint-uks-forces`. Sibling force pin reuses the same geometry and soft
energy.

## Method

**GPW Kohn–Sham (Quickstep family)** (VandeVondele et al., *Comput. Phys.
Commun.* **167**, 103 (2005)): density matrix in a contracted Gaussian AO
basis; collocate \(\rho\) onto a real-space grid; FFT Poisson + grid XC;
integrate potentials back to the AO KS matrix. Core electrons removed by
**GTH** pseudopotentials. Citation lineage: Lippert, Hutter, Parrinello,
*Mol. Phys.* **92**, 477 (1997).

**Multi-k collinear UKS.** With \(N_\alpha \neq N_\beta\) and
`k_mesh ≠ {1,1,1}` (or equivalent explicit mesh), build spin densities from
Bloch-phased density matrices \(P_k^\alpha\), \(P_k^\beta\); Hartree from total
\(\rho=\rho_\alpha+\rho_\beta\); spin-resolved **PADE LSD** on the grid;
diagonalize separate \(H_k^\alpha\), \(H_k^\beta\). Time-reversal folds the
\(1\times1\times2\) MP mesh to a single irreducible \(\kappa_z=\pm 1/4\)
point with weight 1 in the product pin.

**Analytic fixed-\(P_k^\sigma\) cell stress (product).** After one multi-k
UKS SCF:

1. Collect occupied Bloch MOs `C_occ_k` / `C_occ_k_beta` and band energies
   `eps_k` / `eps_k_beta` from `GpwResult`.
2. Rebuild \(P_k^\sigma=\sum_i^{\mathrm{occ}} c_{ik}^\sigma (c_{ik}^\sigma)^\dagger\)
   and \(W_k^\sigma=\sum_i^{\mathrm{occ}} \varepsilon_{ik}^\sigma\,
   c_{ik}^\sigma (c_{ik}^\sigma)^\dagger\) (occupancy 1 per spin).
3. Apply dimensionless diagonal strain \(\pm h\) at **fixed fractional
   coordinates**; recompute the **non-self-consistent** Lagrangian
   \(E_L[\{P_k^\sigma\},\{W_k^\sigma\};\varepsilon]\) by rebuilding
   \(S_k\), \(T_k\), \(V_{\mathrm{nl}}^k\), grid, Poisson, and spin XC —
   **without** re-converging SCF.
4. Form

\[
\sigma_{aa}
=
-\Omega^{-1}
\frac{\partial E_L}{\partial\varepsilon_{aa}}
\approx
-\Omega^{-1}
\frac{E_L(+h_a)-E_L(-h_a)}{2h},
\qquad a\in\{x,y,z\},
\]

with \(E_L\) in Hartree and \(\Omega\) in Bohr³ (CP2K `STRESS_TENSOR` sign).
The \(-\mathrm{Tr}(W S)\) channel supplies the Pulay contribution (Nielsen &
Martin, *Phys. Rev. B* **32**, 3780 (1985); GPW analogue of plane-wave
`pw_stress`).

**Oracle multi-SCF twin.** `gpw_stress_fd` fully re-converges UKS SCF at each
strained cell. Dual-gate residual on this pin is
\(\max_a\lvert\sigma_a^{\mathrm{an}}-\sigma_a^{\mathrm{fd}}\rvert
\sim 3.9\times 10^{-10}\) Ha/Bohr³ (gate \(1\times 10^{-4}\)).

**Twin settings (binding):**

| Knob | Value |
| ---- | ----- |
| System | High-spin H₂ (\(M=3\)), compressed bond along \(x\) |
| Positions [Bohr] | H1 (5.5, 6.0, 6.0), H2 (6.5, 6.0, 6.0) |
| Cell | Cubic **12 Bohr**, `PERIODIC XYZ` |
| Basis | **DZV-GTH-PADE** (H) — in-tree table |
| PP | GTH-PADE-q1 |
| XC | PADE LSD (`XcKind::PADE`) |
| Grid | CUTOFF **100 Ry**, `local_pp_on_grid = true` |
| k mesh | MONKHORST-PACK **1×1×2** (TR → \(\kappa_z=\pm 1/4\), \(w=1\)) |
| Spin | \(N_\alpha=2\), \(N_\beta=0\) (2 valence e⁻ on H GTH-q1 pair) |
| SCF | `tol = 1\times 10^{-8}`, `max_iter = 60` |
| Stress step | \(h = 2\times 10^{-3}\) |
| Observable | Voigt \(\sigma\) [Ha/Bohr³]; \(E_{\mathrm{tot}}\) [Ha] soft pin |

**Valence bookkeeping.** Each H with GTH-PADE-q1 contributes 1 valence e⁻ →
2 valence e⁻ total. Multiplicity 3 ⇒ \(N_\alpha=2\), \(N_\beta=0\) (not
all-electron).

**Key exclusions (spec / product scope-out):**

- Triclinic / off-diagonal shear stress — orthorhombic v1; off-diagonal
  Voigt components are **exactly 0** (gate \(\le 10^{-8}\)).
- Multi-k **DFT+U** stress (`gpw-dftu-stress` is a separate Γ/mol path).
- Hybrid / meta-GGA / GAPW / PAW; CUDA GPW.
- Absolute same-basis CP2K NUMERICAL stress digits — BASIS_MOLOPT twin uses
  **SZV-MOLOPT-GTH**, not DZV-GTH-PADE (documentation twin only).
- Public `*Config` bags / `make_*` factories; live CP2K from ctest.
- Frame Evaluate wrapper is optional; product regression uses solver free
  functions (same numbers as evaluator path when wired).

## Implementation

Public surface only (`GpwAtom` / `GpwDft` members, free `gpw_stress` /
`gpw_stress_fd`; **no** `*Config`, **no** factories). Match the regression:

```cpp
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/molrs.hpp"

using namespace ATV::cpu::gpw;

const double L = 12.0; // Bohr
const GaussianBasis bH(molrs::Element::H, "DZV-GTH-PADE");
const GthPotential pH(molrs::Element::H, "GTH-PADE-q1");
const double r0[3] = {0.5 * L - 0.5, 0.5 * L, 0.5 * L};
const double r1[3] = {0.5 * L + 0.5, 0.5 * L, 0.5 * L};
const std::vector<GpwAtom> atoms{
    GpwAtom(r0, bH, pH, /*Z_valence=*/1),
    GpwAtom(r1, bH, pH, 1)};

GpwDft method;
method.cell[0] = method.cell[1] = method.cell[2] = L;
method.cutoff_ry = 100.0;
method.n_alpha = 2;
method.n_beta = 0;
method.max_iter = 60;
method.tol = 1e-8;
method.xc = XcKind::PADE;
method.local_pp_on_grid = true;
method.k_mesh[0] = 1;
method.k_mesh[1] = 1;
method.k_mesh[2] = 2;

method.build(atoms);
const GpwResult& er = method.run(atoms);
// er.e_total [Ha]; er.C_occ_k / C_occ_k_beta for fixed-P rebuild

const double h = 2e-3;
const auto san = gpw_stress(atoms, method, h);     // product analytic
const auto sfd = gpw_stress_fd(atoms, method, h);  // multi-SCF oracle
// Voigt: san[0..2] = σ_xx,yy,zz [Ha/Bohr³]; san[3..5] shear (0)
```

Build and run the regression (GPW needs HF + PW + FFTW3):

```bash
bash -lc '
  module load buildenv-gcc/2026.03-mpich
  export OPENBLAS_ROOT="$OPENBLASDIR"
  export RUSTUP_TOOLCHAIN=stable-x86_64-unknown-linux-gnu
  export LD_LIBRARY_PATH="$LIBRARY_PATH:$LD_LIBRARY_PATH"
  cmake -B build -DCMAKE_BUILD_TYPE=Release \
        -DATV_BUILD_TESTS=ON -DATV_BUILD_REGRESSIONS=ON \
        -DATV_BUILD_HF=ON -DATV_BUILD_PW=ON -DATV_BUILD_GPW=ON \
        -DATV_BUILD_CUDA=OFF -DBLA_VENDOR=OpenBLAS
  cmake --build build -j$(nproc) --target reg_gpw-kpoint-uks-stress
  ./build/regressions/reg_gpw-kpoint-uks-stress
'
```

| Path | Role |
| ---- | ---- |
| `include/atomiverse/cpu/gpw/gpw_solver.h` | `GpwDft`, `gpw_stress`, `gpw_stress_fd`; multi-k UKS contract |
| `include/atomiverse/cpu/gpw/internal/gpw_stress.h` | Private fixed-\(P_k^\sigma\) + multi-SCF FD |
| `src/cpu/gpw/internal/gpw_stress.cpp` | Orthorhombic analytic / FD implementation |
| `include/atomiverse/cpu/gpw/gpw_evaluator.h` | Frame Evaluate path (`compute_stress`; k-UKS members) |
| `regressions/gpw/gpw-kpoint-uks-stress/reg_gpw-kpoint-uks-stress.cpp` | Product dual-gate + identity |
| `regressions/gpw/gpw-kpoint-uks-stress/REFERENCE.md` | Hard-coded pins + `perf_exempt` |
| `tools/cp2k_oracle/inputs/gpw-kpoint-uks-stress/h2_uks_kpoint_stress.inp` | CP2K NUMERICAL documentation twin (SZV) |

## Validation (vs CP2K / dual-gate)

Source:
[`regressions/gpw/gpw-kpoint-uks-stress/REFERENCE.md`](../../../regressions/gpw/gpw-kpoint-uks-stress/REFERENCE.md).
Oracle policy: offline **CP2K 2026.1** documentation twin when the module is
available; **never** live CP2K from ctest. Decisive product gates are
**internal dual-gate** (analytic vs multi-SCF FD) and **closed-shell
identity** (k-UKS ≡ k-RKS). Soft energy pin matches sibling
`gpw-kpoint-uks-forces` (same geometry/method).

**Setup:**

| Item | Value |
| ---- | ----- |
| Engine | CP2K 2026.1 Quickstep GPW (when available) |
| Twin input | `tools/cp2k_oracle/inputs/gpw-kpoint-uks-stress/h2_uks_kpoint_stress.inp` |
| Twin basis note | CP2K `BASIS_MOLOPT` → **SZV-MOLOPT-GTH**; ATV product → **DZV-GTH-PADE** |
| Pin host (first green) | No CP2K module — absolute NUMERICAL stress digits **not invented** |
| Atomiverse path | `gpw_stress` + `gpw_stress_fd`, \(h=2\times 10^{-3}\) |

### Accuracy table (hard-coded pins)

| Quantity | Pin / measured | Tol (ATV gate) | Notes |
| -------- | -------------: | -------------- | ----- |
| \(E_{\mathrm{tot}}\) [Ha] | **−0.528389371092** | \(\lvert\Delta E\rvert\le 1\times 10^{-3}\) | Soft; 6 SCF iters; \(n_k=1\) TR-folded |
| \(\sigma_{xx}\) analytic [Ha/Bohr³] | **4.26434770×10⁻⁴** | dual-gate vs FD | Product fixed-\(P_k^\sigma\) |
| \(\sigma_{yy}\) analytic [Ha/Bohr³] | **−8.07875807×10⁻⁵** | dual-gate vs FD | Orthorhombic diagonal |
| \(\sigma_{zz}\) analytic [Ha/Bohr³] | **−8.05687961×10⁻⁵** | dual-gate vs FD | Nearly \(\sigma_{yy}\) (cubic cell, bond \(\parallel x\)) |
| \(\max_a\lvert\sigma_a^{\mathrm{an}}-\sigma_a^{\mathrm{fd}}\rvert\) | **3.86×10⁻¹⁰** | \(\le 1\times 10^{-4}\) | Decisive dual-gate |
| Closed-shell k-UKS vs k-RKS \(\max\lvert\mathrm{d}\sigma\rvert\) | **2.35×10⁻¹⁰** | \(\le 5\times 10^{-4}\) | \(N_\alpha=N_\beta=1\) ≡ `n_occ=1` |
| Off-diagonal \(\sigma_{xy,xz,yz}\) | **exactly 0** (product) | \(\le 1\times 10^{-8}\) | Orthorhombic v1 |
| \(\max_a\lvert\sigma_a\rvert\) floor | **> 1×10⁻⁸** | magnitude sanity | Compressed bond → finite \(\sigma_{xx}\) |

**Tolerance rationale.** Dual-gate \(10^{-4}\) Ha/Bohr³ is generous vs measured
\(\sim 10^{-10}\) residual and absorbs SCF residual / grid differences while
requiring correct order and sign. Soft energy \(10^{-3}\) Ha matches the
sibling k-UKS forces twin class. Absolute CP2K NUMERICAL Voigt stress is
**deferred** until a same-basis (DZV-GTH-PADE) data path exists on the pin
host — do not invent SZV vs DZV absolute stress digits.

Offline oracle sketch (host with CP2K module; not ctest):

```bash
module load buildtool-easybuild/5.2.1-hpca3ef7d197 GCC/14.3.0 MPICH/4.3.2 CP2K/2026.1
export CP2K_DATA_DIR=$EBROOTCP2K/share/cp2k/data
cd /tmp && ln -sf $CP2K_DATA_DIR/{BASIS_MOLOPT,GTH_POTENTIALS} .
# copy tools/cp2k_oracle/inputs/gpw-kpoint-uks-stress/h2_uks_kpoint_stress.inp
mpirun -np 1 cp2k.psmp -i h2_uks_kpoint_stress.inp -o h2_uks_kpoint_stress.out
# STRESS_TENSOR NUMERICAL; basis SZV-MOLOPT-GTH — documentation twin only
```

## Performance

Source: `REFERENCE.md` + handoff (the case `REFERENCE.md`).
**`perf_exempt = true`**.

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r = T_{\mathrm{ATV}}/T_{\mathrm{CP2K}}\) | ≤10× | Caveat |
| ---- | ------------------------: | ----------------------: | ---------------------------------------: | ---- | ------ |
| H₂ multi-k UKS stress (dual-gate regression) | n/a | **~1.0** (OMP=8, Release) | **n/a (`perf_exempt`)** | n/a | No same-basis CP2K fair NUMERICAL wall |

**Exemption reason (binding):** DZV-GTH-PADE is an **in-tree** Atomiverse
table; CP2K `BASIS_MOLOPT` on the pin host does not expose it. The shipped
CP2K input uses SZV-MOLOPT-GTH for documentation only. No CP2K module was
available at first green (2026-07-26). When a same-basis twin wall exists,
record single-rank walls for a product-only analytic pass and replace this
section (target \(r\le 10\)).

Regression wall (~1 s) covers SCF + analytic stress + multi-SCF FD + closed-
shell identity — within the ≤ 60 s e2e budget.

## Limitations

- **Orthorhombic diagonal only** — no triclinic shear; off-diagonal Voigt
  components are hard zeros.
- **Dual-gate primary, CP2K absolute secondary** — absolute NUMERICAL stress
  vs CP2K awaits same-basis data; energy soft pin is shared with
  `gpw-kpoint-uks-forces`.
- **Cheap high-spin H₂ pin** — not bulk magnetic solid stress; larger k-meshes
  and production cutoffs are out of this regression budget.
- **PADE LSD only on this pin** — multi-k UKS + PBE-GGA stress is a separate
  capability surface if requested later (Γ-UKS PBE stress: `gpw-uks-stress`).
- **Multi-k DFT+U stress** not implemented (`gpw_stress` docs).
- **Cost** — analytic path is one SCF + 6 fixed-\(P\) samples; FD oracle is
  6+ re-converged SCFs (regression only).
- Unit tests under `tests/cpu/gpw/` only; e2e CP2K-class parity is exclusively
  the regression binary.

## References

- Offline pin + fair wall: [`regressions/gpw/gpw-kpoint-uks-stress/REFERENCE.md`](../../../regressions/gpw/gpw-kpoint-uks-stress/REFERENCE.md)
- Regression consumer: [`reg_gpw-kpoint-uks-stress.cpp`](../../../regressions/gpw/gpw-kpoint-uks-stress/reg_gpw-kpoint-uks-stress.cpp)
- CP2K twin: [`tools/cp2k_oracle/inputs/gpw-kpoint-uks-stress/`](../../../tools/cp2k_oracle/inputs/gpw-kpoint-uks-stress/)
- Oracle contract: [`tools/cp2k_oracle/README.md`](../../../tools/cp2k_oracle/README.md)
- Related pins: `gpw-kpoint-uks-forces`, `gpw-uks-stress`, `gpw-kpoint-stress`,
  `gpw-kpoints-uks`, `gpw-analytic-stress`, `gpw-stress-water`
- Narrative GPW: `docs/tutorial/ch09-gpw.md`
- Roadmap: `docs/roadmap/dft.md`
- Spec history (closed, deleted at done): git `wf/gpw-kpoint-uks-stress` /
  `.claude/specs/gpw-kpoint-uks-stress/` (deleted on merge)
- J. VandeVondele et al., *Comput. Phys. Commun.* **167**, 103 (2005) — Quickstep
- O. H. Nielsen, R. M. Martin, *Phys. Rev. B* **32**, 3780 (1985) — fixed-\(P\) cell theorem
- S. Goedecker, M. Teter, J. Hutter, *Phys. Rev. B* **54**, 1703 (1996) — PADE / GTH
- G. Lippert, J. Hutter, M. Parrinello, *Mol. Phys.* **92**, 477 (1997) — GPW
