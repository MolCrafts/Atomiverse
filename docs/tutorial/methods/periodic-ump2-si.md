# Periodic Γ-point GPW UMP2 — sc-Si₂ / DZVP-MOLOPT-GTH (UKS PBE reference)

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `periodic-ump2-si` |
| Regression | `regressions/gpw/periodic-ump2-si/` |
| Public API | `cpu::gpw::GpwDft::mp2(atoms)` → `cpu::gpw::Mp2Result` on a **UKS** reference (`n_alpha`/`n_beta`); members `mp2_ri`, `mp2_n_virt`, `mp2_os_scale`, `mp2_ss_scale` |
| Roadmap | `docs/roadmap/dft.md` §10 — Periodic MP2 includes **Γ UMP2@UKS** (`reg_periodic-ump2-si`) ✓ |

## Summary

**Unrestricted second-order Møller–Plesset (UMP2) correlation under periodic
boundary conditions**, evaluated on top of a converged **Γ-point UKS GPW PBE**
SCF. Spin-resolved occupied–virtual MO integrals \((ia|jb)\) use the product
**RI-GPW** assembly (Del Ben, Hutter, VandeVondele, *JCTC* **8**, 4177 (2012);
CP2K `&WF_CORRELATION` `METHOD MP2_GPW`): collocate
\(\rho_{ia}^\sigma=\psi_i^\sigma\psi_a^\sigma\), plane-wave Coulomb
intermediates with G=0 dropped (neutral pair densities), then same-spin
αα/ββ and opposite-spin αβ doubles in the molecular / PySCF UMP2 convention.

This page is the **equal-spin closed-shell UKS foothold** on artificial
sc-Si₂ (\(n_\alpha=n_\beta=4\)): when \(C^\alpha\approx C^\beta\),
**UMP2 ≡ RMP2**, dual-gated against the closed-shell first-green band from
`periodic-mp2-si`. Absolute CP2K UMP2 digits are **not invented** when the
oracle binary is absent — the offline twin input is ready for re-pin.

Public path: construct `cpu::gpw::GpwDft`, set UKS occupations and SCF/MP2
members (no `*Config` bag, no factory), `build` → `run` → `mp2`. Energies on
`Mp2Result` are **Hartree** (internal post-HF convention); Frame public
energies remain eV when routed through evaluators. Product gate:
`reg_periodic-ump2-si`. Fair wall vs CP2K is **perf_exempt** (no CP2K binary
on the pin host).

Related: closed-shell Γ MP2 — [periodic-mp2-si](periodic-mp2-si.md);
molecular UMP2 (PySCF loop) — [ump2-o2-sto3g](ump2-o2-sto3g.md); narrative
GPW — [Ch. 9](../ch09-gpw.md).

## Method

**GPW (Quickstep family).** Contracted Gaussian AOs; density / Hartree / XC
on a uniform real-space grid; GTH pseudopotentials. Grid resolution from the
plane-wave density cutoff [Ry]. Citation: Lippert, Hutter, Parrinello,
*Mol. Phys.* **92**, 477 (1997).

**UKS reference (Γ).** Members `n_alpha` / `n_beta` select the unrestricted
path (`is_uks()` when their sum is positive); closed-shell `n_occ` is
ignored. Separate \(C^\alpha\), \(C^\beta\) and \(\varepsilon^\alpha\),
\(\varepsilon^\beta\). This pin uses **equal-spin** occupations
\(n_\alpha=n_\beta=4\) (8 valence e⁻ from two Si GTH-PBE-q4 sites) and
**PBE** GGA (`XcKind::PBE`) — the same chemical deck as
`periodic-mp2-si`, re-hosted as UKS so the UMP2 kernel is exercised with a
dual-gate identity to RMP2.

**RI-GPW \((ia|jb)\).** Product default `mp2_ri = true` (periodic):

\[
\rho_{ia}^\sigma(\mathbf r)=\psi_i^\sigma(\mathbf r)\,\psi_a^\sigma(\mathbf r),
\qquad
\int\rho_{ia}^\sigma= S_{ia}^\sigma=0
\quad\Rightarrow\quad
\text{G=0-dropped Coulomb is finite at Γ}.
\]

Plane-wave resolution of the pair-density identity yields the MO ERI tensor
without an auxiliary Gaussian fit table (cFIT scale-up is a separate
product). `mp2_ri = false` retains the Poisson real-space foothold used by
the early DIRECT_CANONICAL-class pin.

**UMP2 energy (spin-orbital / molecular convention).** With
\(v_{ia,jb}^{\sigma\sigma'}=(i_\sigma a_\sigma|j_{\sigma'} b_{\sigma'})\):

\[
E_{\mathrm{SS}}^\sigma
  =\frac14\sum_{ijab}
   \frac{\big[v_{ia,jb}^{\sigma\sigma}-v_{ib,ja}^{\sigma\sigma}\big]^2}
        {\varepsilon_a^\sigma+\varepsilon_b^\sigma
         -\varepsilon_i^\sigma-\varepsilon_j^\sigma},
\qquad
E_{\mathrm{OS}}
  =\sum_{ijab}
   \frac{\big(v_{ia,jb}^{\alpha\beta}\big)^2}
        {\varepsilon_a^\alpha+\varepsilon_b^\beta
         -\varepsilon_i^\alpha-\varepsilon_j^\beta},
\]

\[
E_{\mathrm{corr}}
  = c_{\mathrm{OS}}\,E_{\mathrm{OS}}
  + c_{\mathrm{SS}}\,(E_{\mathrm{SS}}^\alpha+E_{\mathrm{SS}}^\beta),
\qquad
E_{\mathrm{tot}} = E_{\mathrm{SCF}} + E_{\mathrm{corr}}.
\]

Defaults \(c_{\mathrm{OS}}=c_{\mathrm{SS}}=1\) recover plain UMP2
(`mp2_os_scale` / `mp2_ss_scale`). Reported `e_corr_os` / `e_corr_ss` are
**unscaled**; `n_occ` / `n_virt` on `Mp2Result` are spin-orbital totals
\(n_\alpha+n_\beta\) and \(n_{v\alpha}+n_{v\beta}\). Virtual space per spin
is capped by `mp2_n_virt` (0 = all kept after linear dependence).

**Reference choice.** Correlation is evaluated on **KS (PBE) orbitals** —
the standard periodic MP2@KS foothold (double-hybrid / RPA ingredient
family). A true MP2@HF reference awaits periodic exact exchange (blocked
row `periodic-hybrid-gpw`).

**Key exclusions (product scope).**

- **No multi-k UMP2** — `GpwDft::mp2` throws on UKS with \(N_k>0\); multi-k
  is closed-shell RI-GPW only (`periodic-mp2-kpoints`).
- **No OT reference** — OT stores occupied orbitals only; canonical
  virtuals require diagonalization (`use_ot = false`).
- **No UMP2 gradients / stress / forces**; energy only.
- **No UGW / URPA** (hold behind this UMP2 product).
- **No absolute CP2K digit pin yet** — host had no CP2K module at pin time;
  dual-gate band vs RMP2 first-green until offline re-pin.
- Artificial **sc-Si₂** deck (not diamond Si₈ at production cutoffs).
- Molecular all-electron UMP2 is the PySCF-loop path (`ump2-o2-sto3g`), not
  this page.

## Implementation

Public surface only (`GpwDft` members + lookup basis/PP constructors; **no**
`*Config` bag, **no** `make_*` / `build_*` factories):

```cpp
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/cpu/gpw/mp2.h"
#include "atomiverse/molrs.hpp"

using namespace ATV::cpu::gpw;
constexpr double kA2B = 1.8897259885789;
constexpr double a_bohr = 5.43 * kA2B;   // cubic cell [Bohr]

const double pos0[3] = {0.0, 0.0, 0.0};
const double pos1[3] = {a_bohr / 4.0, a_bohr / 4.0, a_bohr / 4.0};
const GaussianBasis bas(molrs::Element::Si, "DZVP-MOLOPT-GTH");
const GthPotential pot(molrs::Element::Si, "GTH-PBE-q4");
std::vector<GpwAtom> atoms;
atoms.emplace_back(pos0, bas, pot, 14);
atoms.emplace_back(pos1, bas, pot, 14);

GpwDft m;
m.cell[0] = m.cell[1] = m.cell[2] = a_bohr;
m.cutoff_ry = 30.0;
m.n_alpha = 4;                 // UKS; equal-spin closed shell
m.n_beta  = 4;
m.max_iter = 80;
m.tol = 1e-6;
m.mix = 0.3;
m.level_shift = 0.3;           // small-gap artificial sc-Si₂
m.xc = XcKind::PBE;
m.mp2_ri = true;               // product RI-GPW
m.build(atoms);

const GpwResult& scf = m.run(atoms);   // must converge
m.mp2_n_virt = 0;                      // 0 = all kept virtuals per spin
const Mp2Result& r = m.mp2(atoms);
// r.e_corr (< 0), r.e_corr_os / r.e_corr_ss, r.e_total = r.e_scf + r.e_corr [Ha]
// r.n_occ == 8 (nα+nβ); r.ri == true; r.n_kpts == 0
```

Guards (fail hard, no silent fallback): `std::logic_error` before
`build()` / converged `run()`, or missing β orbitals on a UKS result;
`std::invalid_argument` for multi-k UKS, for `use_ot`, for vanishing
per-spin HOMO–LUMO gap (< 1e-6 Ha), or for empty virtual space.

Build (GPW needs HF + PW + FFTW3) and run the regression:

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
  cmake --build build -j$(nproc) --target atomiverse reg_periodic-ump2-si
  ./build/regressions/reg_periodic-ump2-si
  # or: cmake --build build --target run_reg_periodic-ump2-si
'
```

| Piece | Path |
| ----- | ---- |
| Public `Mp2Result` | `include/atomiverse/cpu/gpw/mp2.h` |
| `GpwDft::{n_alpha,n_beta,mp2_ri,mp2_n_virt,mp2_os_scale,mp2_ss_scale,mp2}` | `include/atomiverse/cpu/gpw/gpw_solver.h` |
| UMP2 dispatch (Γ UKS) | `src/cpu/gpw/mp2.cpp` |
| Kernels (private) | `include/atomiverse/cpu/gpw/internal/mp2_kernels.h` |
| Unit tests (not e2e) | `tests/cpu/gpw/test_cpu_gpw_mp2.cpp` |
| Regression | `regressions/gpw/periodic-ump2-si/reg_periodic-ump2-si.cpp` |
| Offline pin contract | `regressions/gpw/periodic-ump2-si/REFERENCE.md` |
| CP2K twin input | `tools/cp2k_oracle/inputs/periodic-ump2-si.inp` |

## Validation (vs CP2K / dual-gate)

Source:
[`regressions/gpw/periodic-ump2-si/REFERENCE.md`](../../../regressions/gpw/periodic-ump2-si/REFERENCE.md).
Oracle: offline **CP2K** twin (`tools/cp2k_oracle/inputs/periodic-ump2-si.inp`:
`UKS`, `MULTIPLICITY 1`, same cell/basis/PP/cutoff, `&WF_CORRELATION` MP2);
**never** live in ctest. At pin time **no CP2K module** on the host —
absolute `MP2 Energy` digits are **not invented**. Product gate uses the
**RMP2 dual-gate** (UMP2 ≡ RMP2 for \(n_\alpha=n_\beta\) and matched
orbitals) against the closed-shell first-green band.

### System (hard-coded)

| Item | Value |
| ---- | ----- |
| Crystal | artificial sc-Si₂ (same as `periodic-mp2-si`) |
| Cell | cubic **a = 5.43 Å**, fully periodic, **Γ only** |
| Positions | Si (0,0,0), Si (a/4, a/4, a/4) |
| Basis | **DZVP-MOLOPT-GTH** |
| Pseudopotential | **GTH-PBE-q4** → 8 valence e⁻ |
| XC (SCF) | **PBE**, UKS \(n_\alpha=n_\beta=4\) |
| Density cutoff | **30 Ry** |
| SCF | tol 1e-6 Ha, mix 0.3, **level_shift 0.3 Ha**, max_iter 80 |
| MP2 | RI-GPW product (`mp2_ri = true`); all kept virtuals per spin |

### Accuracy table

| Quantity | Pin / band [Ha] | Tol | Notes |
| -------- | --------------: | --- | ----- |
| \(E_{\mathrm{corr}}\) (RMP2 first-green twin) | **−0.2764036524** | — | from `periodic-mp2-si` |
| \(E_{\mathrm{corr}}\) (UMP2@UKS product gate) | same target | **2×10⁻³** | UKS numerical drift band |
| \(E_{\mathrm{corr}}\) sign | \(E_{\mathrm{corr}} < 0\) | hard | regression |
| OS + SS ≡ \(E_{\mathrm{corr}}\) | identity | **1×10⁻¹²** | unscaled channels, scales = 1 |
| \(E_{\mathrm{tot}} \equiv E_{\mathrm{SCF}}+E_{\mathrm{corr}}\) | identity | **1×10⁻¹²** | |
| Virtual-space monotonicity | \(E_{\mathrm{corr}}(\mathrm{all}) \le E_{\mathrm{corr}}(n_v{=}8) < 0\) | hard | |
| \(n_{\mathrm{occ}}\) | **8** (\(n_\alpha+n_\beta\)) | exact | |
| CP2K absolute \(E_{\mathrm{corr}}\) / \(E_{\mathrm{tot}}\) | *pending re-pin* | — | replace dual-gate when CP2K available |

Unit tier (`tests/cpu/gpw/test_cpu_gpw_mp2.cpp`): UMP2 ≡ explicit spin-orbital
sum; UMP2 ≡ RMP2 on identical α=β tensors; cross-spin RI ≡ Poisson.

## Performance

Source: the case `REFERENCE.md` (harness wave 5, 2026-07-26);
`REFERENCE.md` §Perf.

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r = T_{\mathrm{ATV}}/T_{\mathrm{CP2K}}\) | ≤10× | Caveat |
| ---- | ------------------------: | ----------------------: | ----------------------------------------: | ---- | ------ |
| `periodic-ump2-si` | — | (reg ≤ 60 s budget) | **perf_exempt** | n/a | No CP2K binary on pin host for fair wall |

Target wall ≤ 60 s single process (SCF ≈ RKS deck cost + UMP2 three-tensor
build: αα, ββ, αβ). Do **not** invent a numeric ratio without a fair twin.

## Limitations

- **Dual-gate only** until offline CP2K re-pin of
  `tools/cp2k_oracle/inputs/periodic-ump2-si.inp` lands absolute digits +
  release citation.
- **Γ UKS only** — multi-k UMP2, UGW, URPA are follow-ons.
- **MP2@KS (PBE)** — not MP2@HF; periodic hybrid / exact exchange blocked.
- **Energy-only** — no nuclear gradients, cell stress, or forces at UMP2.
- **Artificial sc-Si₂** at 30 Ry — not a production diamond-Si band-structure
  pin; level shift 0.3 Ha is a convergence aid for the small artificial gap.
- **OT incompatible** with canonical virtual denominators.
- CUDA / Kokkos ports out of scope for this CP2K capability loop page.
- Molecular post-HF UMP2 remains on the **PySCF** oracle loop, not CP2K.

## References

- Offline pin: [`regressions/gpw/periodic-ump2-si/REFERENCE.md`](../../../regressions/gpw/periodic-ump2-si/REFERENCE.md)
- Regression: `regressions/gpw/periodic-ump2-si/reg_periodic-ump2-si.cpp`
- CP2K twin: `tools/cp2k_oracle/inputs/periodic-ump2-si.inp`
- Oracle rules: `tools/cp2k_oracle/README.md` (offline pins only; never live ctest)
- Roadmap: [`docs/roadmap/dft.md`](../../roadmap/dft.md) §10 Periodic MP2
- Closed-shell sibling: [periodic-mp2-si](periodic-mp2-si.md)
- Molecular UMP2: [ump2-o2-sto3g](ump2-o2-sto3g.md)
- Del Ben, Hutter, VandeVondele, *J. Chem. Theory Comput.* **8**, 4177 (2012) — RI-GPW MP2
- Lippert, Hutter, Parrinello, *Mol. Phys.* **92**, 477 (1997) — GPW
