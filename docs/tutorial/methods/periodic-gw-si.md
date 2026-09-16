# Periodic Γ-point spectral G0W0 — sc-Si₂ / DZVP-MOLOPT-GTH (PBE reference)

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `periodic-gw-si` |
| Regression | `regressions/gpw/periodic-gw-si/` |
| Public API | `cpu::gpw::GpwDft::g0w0(atoms)` → `cpu::gpw::G0W0Result` (`cpu/gpw/g0w0.h`); screening virtuals via `mp2_n_virt`, product RI via `mp2_ri` |
| Roadmap | roadmap `dft.md` §10 **Periodic GW** ✓ (Γ product; multi-k residual); handoff ratio **perf_exempt** |

## Summary

Closed-shell **spectral \(G_0W_0\)** quasiparticle (QP) energies on a
converged **Γ-point periodic GPW PBE** reference for artificial sc-Si₂
(\(a=5.43\) Å, DZVP-MOLOPT-GTH + GTH-PBE-q4, density cutoff **30 Ry**).
Screening is **direct RPA** (dRPA) on the same product **RI-GPW** \((ia|jb)\)
path as `periodic-rpa-si` / `periodic-ri-mp2-si`; the correlation self-energy
is the frequency-exact spectral sum over RPA poles (same form as molecular
`RHF::g0w0` / PySCF `GWExact`). Exact-exchange diagonals use **truncated (MT)
Coulomb** so the Γ-only foothold is G=0-safe without a periodic-HF product path.
Primary product observable is the **HOMO–LUMO QP gap [eV]**. Public path:
construct `GpwDft`, set members, `build` → `run` → `g0w0` — no public `*Config`,
no factory. Offline CP2K twin input ships at
`tools/cp2k_oracle/inputs/periodic-gw-si.inp` (frequency-integrated RI-G0W0;
**not** algorithmically identical — see Validation). Fair wall vs CP2K is
**`perf_exempt`**.

## Method

**Level.** One-shot \(G_0W_0@\mathrm{KS}\): Green’s function and screened
interaction built from the closed-shell GPW RKS (PBE) solution; no
self-consistency in \(G\) or \(W\).

**Spin / k.** Closed-shell spatial MOs only (`GpwDft` RKS, `is_uks() == false`).
**Γ only** — multi-k SCF references throw (`result_.n_kpts > 1`); multi-k G0W0
is a roadmap residual.

**Screening.** Direct RPA (Coulomb-only Casida / Furche reduction) on the
occupied–virtual MO-ERI tensor. Product path: RI-GPW (`mp2_ri = true`,
periodic); Poisson foothold available when `mp2_ri = false`. Virtual space for
screening capped by `mp2_n_virt` (0 = all kept virtuals). QP levels are reported
for **every** MO of the converged spectrum.

**Self-energy.** Spectral (frequency-exact) diagonal correlation self-energy
\(\Sigma_c(\omega)\) as a sum over RPA poles and transition densities; spectral
broadening \(\eta = 10^{-8}\) Ha (impl constant in `src/cpu/gpw/g0w0.cpp`).

**Exchange and local XC.** Exact-exchange diagonals \(\Sigma_{x,p}\) use
**Martyna–Tuckerman truncated Coulomb** (G=0-safe at Γ without periodic HF).
Local \(v_{\mathrm{xc},p}\) is the SCF functional (PBE on this pin) on the
closed-shell density.

**Quasiparticle equation.** For each MO \(p\), Newton solve (tol \(10^{-8}\) Ha):

\[
\omega = \varepsilon_p
  + \mathrm{Re}\,\Sigma_{c,p}(\omega)
  + \Sigma_{x,p}
  - v_{\mathrm{xc},p}.
\]

Not the linearized \(Z\)-factor form. Requires a finite KS HOMO–LUMO gap
(\(\ge 10^{-6}\) Ha) so dRPA \((A-B)\) is non-singular.

**Observables** (`G0W0Result`):

| Symbol | Field | Unit |
| ------ | ----- | ---- |
| \(\varepsilon_p^{\mathrm{QP}}\) | `qp_energy_ha` | Ha |
| \(\varepsilon_p^{\mathrm{KS}}\) | `mf_energy_ha` | Ha |
| \(\Sigma_{x,p}\) | `sigma_x_ha` | Ha |
| \(v_{\mathrm{xc},p}\) | `vxc_ha` | Ha |
| QP gap | `gap_ev` | eV |
| KS gap | `mf_gap_ev` | eV |
| IP \(-\varepsilon_{\mathrm{HOMO}}^{\mathrm{QP}}\) | `ip_ev` | eV |
| EA \(\varepsilon_{\mathrm{LUMO}}^{\mathrm{QP}}\) | `ea_ev` | eV |

**What this pin includes**

| Item | Setting |
| ---- | ------- |
| Crystal | artificial sc-Si₂ (same deck as `periodic-rpa-si`) |
| Cell | cubic \(a = 5.43\) Å, periodic XYZ, Γ only |
| Positions | Si \((0,0,0)\), Si \((a/4,a/4,a/4)\) |
| Basis / PP | DZVP-MOLOPT-GTH / GTH-PBE-q4 |
| XC | PBE (`XcKind::PBE`) |
| Cutoff | **30 Ry** |
| SCF | tol \(10^{-6}\), mix 0.3, `level_shift` 0.3 Ha, `max_iter` 80 |
| G0W0 | all kept virtuals for dRPA; `mp2_ri = true` |
| Valence | 8 e⁻ → \(n_{\mathrm{occ}} = 4\); \(n_{\mathrm{virt}} = 22\) on this basis |

**Key exclusions (non-goals of this slug)**

- Multi-k G0W0 / BZ-sampled screening (residual after Γ product).
- UKS / open-shell GW; OT orbitals (no canonical virtuals).
- Analytic-continuation GW; evGW / qsGW / scGW; BSE on this path.
- Periodic exact exchange as the \(\Sigma_x\) operator (MT substitute only).
- Absolute digit match to CP2K **frequency-integrated RI-G0W0** (different
  algorithm; twin input is diagnostic / re-pin scaffolding).
- Live CP2K (or any third-party) inside ctest.
- Public `*Config` bags / `make_*` / `build_*` factories.
- CUDA GW (roadmap ○).

## Implementation

Public surface only (`GpwDft` members + lookup constructors; matches
`reg_periodic-gw-si.cpp`). Headers: `include/atomiverse/cpu/gpw/gpw_solver.h`,
`g0w0.h`; impl `src/cpu/gpw/g0w0.cpp`. Shared dRPA / RI MO-ERI internals:
`cpu/gpw/internal/mp2_kernels.h`.

```cpp
#include "atomiverse/cpu/gpw/g0w0.h"
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/molrs.hpp"

using namespace ATV::cpu::gpw;

constexpr double kA2B = 1.8897259885789;
constexpr double a_bohr = 5.43 * kA2B;

const GaussianBasis bas(molrs::Element::Si, "DZVP-MOLOPT-GTH");
const GthPotential pot(molrs::Element::Si, "GTH-PBE-q4");
std::vector<GpwAtom> atoms;
const double pos0[3] = {0.0, 0.0, 0.0};
const double pos1[3] = {a_bohr / 4.0, a_bohr / 4.0, a_bohr / 4.0};
atoms.emplace_back(pos0, bas, pot, 14);
atoms.emplace_back(pos1, bas, pot, 14);

GpwDft m;                              // concrete method — members as params
m.cell[0] = m.cell[1] = m.cell[2] = a_bohr;
m.cutoff_ry = 30.0;
m.n_occ = 4;
m.max_iter = 80;
m.tol = 1e-6;
m.mix = 0.3;
m.level_shift = 0.3;                   // small-gap sc-Si2 convergence aid
m.xc = XcKind::PBE;
m.mp2_ri = true;
m.build(atoms);
m.run(atoms);                          // must converge

m.mp2_n_virt = 0;                      // 0 = all kept virtuals in dRPA
const G0W0Result& gw = m.g0w0(atoms);
// gw.gap_ev [eV], gw.qp_energy_ha [Ha], gw.converged, gw.ri
```

Guards: `std::logic_error` before `build()` / converged `run()`;
`std::invalid_argument` for UKS, `use_ot`, multi-k SCF, missing virtuals, or
vanishing HOMO–LUMO gap; `std::runtime_error` if dRPA is ill-conditioned.

```bash
# configure (login shell + Lmod — see AGENTS.md)
bash -lc '
  module load buildenv-gcc/2026.03-mpich
  export OPENBLAS_ROOT=$OPENBLASDIR
  export RUSTUP_TOOLCHAIN=stable-x86_64-unknown-linux-gnu
  export LD_LIBRARY_PATH=$LIBRARY_PATH:$LD_LIBRARY_PATH
  cmake -B build -DCMAKE_BUILD_TYPE=Release \
        -DATV_BUILD_TESTS=ON -DATV_BUILD_REGRESSIONS=ON \
        -DATV_BUILD_HF=ON -DATV_BUILD_PW=ON -DATV_BUILD_GPW=ON \
        -DATV_BUILD_CUDA=OFF -DBLA_VENDOR=OpenBLAS
  cmake --build build -j$(nproc) --target atomiverse reg_periodic-gw-si
  ./build/regressions/reg_periodic-gw-si
  # or: cmake --build build --target run_reg_periodic-gw-si
'
```

Source consumer: `regressions/gpw/periodic-gw-si/reg_periodic-gw-si.cpp`.

## Validation (vs offline CP2K pins / first-green)

Source:
[`regressions/gpw/periodic-gw-si/REFERENCE.md`](../../../regressions/gpw/periodic-gw-si/REFERENCE.md)
(offline pin only; CP2K never launched from CMake/ctest).

| Setting | Value |
| ------- | ----- |
| Geometry | sc-Si₂, \(a=5.43\) Å; Si at \((0,0,0)\), \((a/4,a/4,a/4)\) |
| Basis / PP | DZVP-MOLOPT-GTH / GTH-PBE-q4 |
| XC | PBE |
| Cutoff | 30 Ry |
| SCF | tol \(10^{-6}\), mix 0.3, level_shift 0.3 Ha, max_iter 80 |
| G0W0 | spectral + dRPA + MT \(\Sigma_x\); RI-GPW; all virtuals |
| Oracle contract | CP2K offline twin input; **spectral ≠ RI-G0W0** path |
| Pin date | 2026-07-25 |

### Accuracy (hard-coded product gates)

| Quantity | Pin / band | Tol | Notes |
| -------- | ---------: | --- | ----- |
| SCF converged | required | — | finite \(E_{\mathrm{total}}\) |
| Newton QP converged | `gw.converged` | — | all MOs |
| `gw.ri` | `true` | — | product RI path |
| \(n_{\mathrm{occ}}\) | 4 | exact | closed-shell valence |
| \(\varepsilon_{\mathrm{HOMO}}^{\mathrm{QP}} < \varepsilon_{\mathrm{LUMO}}^{\mathrm{QP}}\) | required | — | ordering |
| Finite HOMO/LUMO QP shifts vs KS | required | — | setup integrity |
| \(E_{\mathrm{SCF}}\) (PBE, Γ) | **−7.5955445070** Ha | info | first-green 2026-07-25 |
| KS gap | **0.410147** eV | info | diagnostic |
| **QP gap** | **5.719545** eV | **0.05 eV** | **primary band** |
| \(\varepsilon_{\mathrm{HOMO}}^{\mathrm{QP}}\) | −0.13897686 Ha | info | Δ vs KS +0.071804 Ha |
| \(\varepsilon_{\mathrm{LUMO}}^{\mathrm{QP}}\) | +0.07121256 Ha | info | Δ vs KS +0.123312 Ha |
| IP / EA (QP) | 3.7818 / 1.9378 eV | info | product fields |

**Unit tier:** `tests/cpu/gpw/test_cpu_gpw_g0w0.cpp` pins dRPA amplitudes vs the
energy path (not e2e).

### Offline CP2K status

Input twin: `tools/cp2k_oracle/inputs/periodic-gw-si.inp` — RI-G0W0 + minimax
quadrature + Gaussian `RI_AUX` under `&WF_CORRELATION`. CP2K’s default path is
**frequency-integrated RI-GW**, not spectral plasmon G0W0 with MT exchange.
Absolute CP2K QP digits are **not invented** as a hard twin of the spectral
value.

Pin strategy (binding, mirrors `periodic-rpa-si`):

1. Hard internal gates (SCF / Newton / finite positive QP gap / HOMO below LUMO).
2. Frozen first-green band \(|\mathrm{gap}-5.719545|\le 0.05\) eV.
3. Unit-tier spectral machinery tests.
4. Offline re-pin: when a fair spectral-equivalent CP2K path exists, replace the
   band with cited CP2K QP gap digits.

## Performance

Source: REFERENCE + the case `REFERENCE.md`.
Gate: \(r = T_{\mathrm{ATV}}/T_{\mathrm{CP2K}} \le 10\) unless exempted.

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | -----------------------: | ---------------------: | ----: | ---- | ------ |
| spectral G0W0@PBE (Γ sc-Si₂) | — | ~24 (SCF≈23 + G0W0≈0.6; 4 OMP, EPYC login) | **perf_exempt** | exempt | CP2K RI-G0W0 ≠ spectral algorithm; MT \(\Sigma_x\) ≠ periodic HF; no CP2K binary on pin host |

Regression wall ≈ **24 s** single process, well under the ≤ 60 s e2e cap.
Never launch CP2K from ctest.

## Limitations

- **Γ closed-shell only.** Multi-k G0W0 residual; no UKS/UGW; no OT reference.
- **Spectral product path only.** No analytic-continuation GW public API.
- **No self-consistency.** evGW / qsGW / scGW remain roadmap-open
  ([excited-states.md](../../roadmap/excited-states.md) §5 companion molecular
  surface; periodic §10 in [dft.md](../../roadmap/dft.md)).
- **MT exchange substitute.** True periodic HF \(\Sigma_x\) blocked until
  periodic hybrid / HF rows close.
- **Deck is artificial sc-Si₂ at 30 Ry.** Not a production diamond-Si₈ /
  production-cutoff QP claim; setup-integrity band 0.05 eV until spectral CP2K
  twin digits exist.
- **CPU only.** CUDA periodic GW is ○.
- **Speed.** Fair wall **perf_exempt** (algorithm mismatch + no CP2K binary);
  ATV wall alone is budget-safe.

## Follow-ons

Multi-k spectral G0W0; spectral-equivalent CP2K re-pin of the QP gap; periodic
HF \(\Sigma_x\); physical diamond supercell at production cutoffs; BSE on
periodic G0W0; UGW.

## References

- Offline pin: `regressions/gpw/periodic-gw-si/REFERENCE.md`
- Oracle twin: `tools/cp2k_oracle/inputs/periodic-gw-si.inp`
- Oracle contract: `tools/cp2k_oracle/README.md`
- Roadmap: `docs/roadmap/dft.md` §10 Periodic GW ✓ (Γ; multi-k residual)
- Public headers: `include/atomiverse/cpu/gpw/g0w0.h`, `gpw_solver.h`
- Impl: `src/cpu/gpw/g0w0.cpp`
- Molecular spectral twin (PySCF): `docs/tutorial/methods/g0w0-h2-sto3g.md`
- Hybertsen & Louie, *Phys. Rev. B* **34**, 5390 (1986)
- Furche, *J. Chem. Phys.* **114**, 5982 (2001) (dRPA / plasmon reduction)
- Del Ben, Hutter, VandeVondele, *J. Chem. Theory Comput.* **8**, 4177 (2012)
  (GPW RI \((ia|jb)\) layout shared with MP2/RPA)
