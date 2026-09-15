# Periodic Γ-point GPW direct RPA — sc-Si₂ / DZVP-MOLOPT-GTH (dRPA@PBE)

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `periodic-rpa-si` |
| Regression | `regressions/gpw/periodic-rpa-si/` |
| Public API | `cpu::gpw::GpwDft::direct_rpa(atoms)` → `cpu::gpw::RpaResult` (`cpu/gpw/rpa.h`); members `mp2_ri`, `mp2_n_virt` |
| Roadmap | roadmap §10 Γ dRPA@KS product; handoff **perf_exempt** |

## Summary

**Direct RPA (dRPA) correlation energy under periodic boundary conditions** on
top of a converged **Γ-point closed-shell GPW PBE** SCF. Occupied–virtual MO
repulsion integrals \((ia|jb)\) are the same product **RI-GPW** plane-wave
assembly used by `GpwDft::mp2` / `periodic-ri-mp2-si` (Del Ben, Hutter,
VandeVondele, *J. Chem. Theory Comput.* **8**, 4177 (2012)); the Poisson-per-pair
foothold remains available with `mp2_ri = false`. Correlation is the **Furche
plasmon formula** on the full Coulomb-only Casida spectrum — the GPW analogue
of molecular `RHF::direct_rpa()` (`direct-rpa-h2-sto3g`). Total energy is the
**dRPA@KS** sum \(E_{\mathrm{SCF}}+E_c\) (no EXX replacement; periodic HF
remains blocked). Product deck: artificial sc-Si₂, \(a=5.43\) Å, DZVP-MOLOPT-GTH
+ GTH-PBE-q4, 30 Ry. Offline CP2K twin input ships at
`tools/cp2k_oracle/inputs/periodic-rpa-si.inp` (frequency-integrated RI-RPA +
Gaussian `RI_AUX` — **not** algorithmically identical to the plasmon path).
Fair wall is **`perf_exempt`**.

## Method

**Reference.** Converged Γ-point GPW RKS (PBE). Real lattice-summed AOs
\(\chi_\mu(\mathbf r)=\sum_{\mathbf T}\varphi_\mu(\mathbf r-\mathbf T)\),
canonical eigenvalues \(\varepsilon_p\). Virtuals from diagonalization (not OT).

**\((ia|jb)\) — product RI-GPW path (`mp2_ri = true`).** Pair density
\(\rho_{ia}=\psi_i\psi_a\) on the SCF density grid; plane-wave Coulomb
intermediate; contract to the ov|ov tensor. Occupied–virtual pairs are neutral
(\(\int\rho_{ia}=S_{ia}=0\)), so the periodic Coulomb operator with the G=0
component dropped is divergence-free at Γ. The Poisson foothold
(`mp2_ri = false`) solves \(\hat\nabla^2 V_{ia}=-4\pi\rho_{ia}\) once per pair
and contracts by grid quadrature — exact within the grid resolution for a tiny
cell; the product path and foothold must agree to \(10^{-10}\) Ha on this deck.

**Kernel (singlet, Coulomb-only dRPA).** \(\alpha_{\mathrm{hf}}=0\):

\[
A_{ia,jb} = (\varepsilon_a-\varepsilon_i)\,\delta_{ij}\delta_{ab}
  + 2\,(ia|jb),\qquad
B_{ia,jb} = 2\,(ia|jb).
\]

**Correlation — Furche plasmon formula** (exact on the spectrum; no frequency
quadrature):

\[
E_{\mathrm{c}}^{\mathrm{dRPA}}
  = \tfrac12\Bigl(\sum_s \Omega_s - \mathrm{Tr}\,A\Bigr),
\]

with \(\Omega_s\) the positive eigenvalues of the Hermitian reduction
\((A-B)^{1/2}(A+B)(A-B)^{1/2}\). Equivalent (non-DF limit) to the
imaginary-frequency integral
\(\frac{1}{2\pi}\int_0^\infty\mathrm{d}\omega\,
\mathrm{Tr}[\ln(1-\chi_0 v)+\chi_0 v]\).

**Total energy.** \(E_{\mathrm{total}}=E_{\mathrm{SCF}}+E_c\) (**dRPA@PBE**).
There is no HF exchange replacement of the KS energy — periodic exact exchange
is a separate blocked capability.

**Spectrum.** Full dRPA spectrum:
\(n_{\mathrm{states}}=n_{\mathrm{occ}}\cdot n_{\mathrm{virt}}\) (product pin:
4 × 22 = 88).

**Observables**

| Symbol | Definition | Unit |
| ------ | ---------- | ---- |
| \(E_{\mathrm{SCF}}\) | converged GPW PBE total | Ha |
| \(E_{\mathrm{c}}^{\mathrm{dRPA}}\) | plasmon correlation energy | Ha |
| \(E_{\mathrm{total}}\) | \(E_{\mathrm{SCF}}+E_c\) | Ha |
| \(\mathrm{Tr}\,A\) | diagnostic for plasmon identity | Ha |
| \(\Omega_s\) | dRPA roots (ascending, full spectrum) | Ha |

**What this pin includes**

| Item | Setting |
| ---- | ------- |
| Spin | Closed-shell RKS (Γ only) |
| XC / reference | **PBE** GGA (`XcKind::PBE`) |
| Crystal | artificial sc-Si₂ (same cell as `periodic-ri-mp2-si`) |
| Cell | cubic **a = 5.43 Å**, periodic XYZ, Γ only |
| Positions [Å] | Si (0,0,0); Si (1.3575, 1.3575, 1.3575) |
| Basis / PP | **DZVP-MOLOPT-GTH** / **GTH-PBE-q4** (zion 4) |
| Cutoff | **30 Ry** |
| SCF | tol \(10^{-6}\), mix 0.3, **level_shift 0.3 Ha**, max_iter 80 |
| Valence | 8 e⁻ → \(n_{\mathrm{occ}}=4\) |
| RPA | all kept virtuals (`mp2_n_virt = 0`); product `mp2_ri = true` |

**Key exclusions (non-goals of this slug)**

- CP2K default **frequency-integrated RI-RPA** with Gaussian `RI_AUX` +
  minimax quadrature as a hard digit twin of \(E_c\) (different algorithm and
  total-energy assembly with EXX; digits not invented).
- Multi-k RPA / residual RI-RPA scale-up; **periodic GW** (`periodic-gw-si`).
- Open-shell / UKS dRPA; OT-SCF references (no canonical virtuals).
- RPAX / SOSEX / ACFDT product paths; molecular post-HF RPA is
  `direct-rpa-h2-sto3g` (PySCF loop).
- Live CP2K (or any third-party) inside ctest.
- Public `*Config` bags / `make_*` / `build_*` factories.
- CUDA GPW RPA.

## Implementation

Public surface only: concrete `GpwDft` members + lookup constructors for
basis/PP; post-SCF `direct_rpa(atoms)`. Headers: `cpu/gpw/rpa.h`,
`cpu/gpw/gpw_solver.h`; impl `src/cpu/gpw/rpa.cpp` (plasmon kernel private in
`cpu/gpw/internal/`). No public `*Config`.

```cpp
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/cpu/gpw/rpa.h"
#include "atomiverse/molrs.hpp"

using namespace ATV::cpu::gpw;

constexpr double kA2B = 1.8897259885789;
constexpr double a_bohr = 5.43 * kA2B;

const double pos0[3] = {0.0, 0.0, 0.0};
const double pos1[3] = {a_bohr / 4.0, a_bohr / 4.0, a_bohr / 4.0};
const GaussianBasis bas(molrs::Element::Si, "DZVP-MOLOPT-GTH");
const GthPotential pot(molrs::Element::Si, "GTH-PBE-q4");

std::vector<GpwAtom> atoms;
atoms.emplace_back(pos0, bas, pot, 14);
atoms.emplace_back(pos1, bas, pot, 14);

GpwDft m;
m.cell[0] = m.cell[1] = m.cell[2] = a_bohr;  // Bohr
m.cutoff_ry = 30.0;
m.n_occ = 4;
m.max_iter = 80;
m.tol = 1e-6;
m.mix = 0.3;
m.level_shift = 0.3;          // small-gap sc-Si2 convergence aid
m.xc = XcKind::PBE;
m.mp2_ri = true;              // product RI-GPW (ia|jb)
m.build(atoms);
const GpwResult& scf = m.run(atoms);  // must converge

m.mp2_n_virt = 0;             // 0 = all kept virtuals
const RpaResult& r = m.direct_rpa(atoms);
// r.e_corr, r.e_total [Ha]; r.tr_A; r.omega_ha; r.ri
```

`RpaResult` fields: `e_corr`, `e_scf`, `e_total`, `tr_A`, `n_occ`, `n_virt`,
`ri`, `omega_ha`. Guards (throw): `build()`/`run()` not done; UKS; OT
(`use_ot`); multi-k SCF reference; vanishing HOMO–LUMO gap (< 1e-6 Ha); empty
virtual space. Regression also runs the Poisson foothold (`mp2_ri = false`) and
requires \(|E_c^{\mathrm{RI}}-E_c^{\mathrm{Poisson}}|\le 10^{-10}\) Ha.

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
  cmake --build build -j$(nproc) --target atomiverse reg_periodic-rpa-si
  ./build/regressions/reg_periodic-rpa-si
'
```

| Path | Role |
| ---- | ---- |
| `include/atomiverse/cpu/gpw/rpa.h` | public `RpaResult` |
| `include/atomiverse/cpu/gpw/gpw_solver.h` | `GpwDft::direct_rpa`, `mp2_ri` / `mp2_n_virt` |
| `include/atomiverse/cpu/gpw/mp2.h` | shared MO-ERI / `Mp2Result` infrastructure |
| `src/cpu/gpw/rpa.cpp` | `GpwDft::direct_rpa` orchestration |
| `src/cpu/gpw/internal/mp2_kernels.cpp` | `direct_rpa_rhf_energy` plasmon kernel |
| `regressions/gpw/periodic-rpa-si/reg_periodic-rpa-si.cpp` | product gate |
| `tests/cpu/gpw/test_cpu_gpw_rpa.cpp` | unit: dense A/B vs plasmon kernel |
| `tools/cp2k_oracle/inputs/periodic-rpa-si.inp` | offline CP2K twin (never ctest) |

## Validation (vs CP2K offline contract)

Source:
[`regressions/gpw/periodic-rpa-si/REFERENCE.md`](../../../regressions/gpw/periodic-rpa-si/REFERENCE.md)
(offline pin only; CP2K never imported from CMake/ctest).

Oracle twin input matches the product deck geometry / basis / PP / cutoff /
level_shift but uses CP2K `&WF_CORRELATION` **RI_RPA** (minimax, 8 quadrature
points, Gaussian `RI_AUX`) with EXX total-energy assembly. Absolute CP2K
\(E_c\) digits are therefore **not invented** as a hard twin of the plasmon
value. Pin strategy mirrors the `periodic-mp2-si` foothold:

1. **Hard internal gates:** SCF converged; \(E_c\) finite and strictly
   negative; all \(\Omega_s>0\); \(E_{\mathrm{total}}\equiv E_{\mathrm{SCF}}+E_c\);
   \(|E_c^{\mathrm{RI}}-E_c^{\mathrm{Poisson}}|\le 10^{-10}\) Ha.
2. **Frozen first-green band** \(|E_c-(-0.2658539776)|\le 10^{-3}\) Ha.
3. Unit tier pins the plasmon kernel vs an independent dense A/B path
   (`tests/cpu/gpw/test_cpu_gpw_rpa.cpp`).
4. Offline re-pin: when a fair plasmon-equivalent CP2K path exists, replace the
   band with cited CP2K correlation digits (record CP2K version).

**Setup:**

| Item | Value |
| ---- | ----- |
| System | artificial sc-Si₂, charge 0, closed-shell RKS |
| Lattice | cubic \(a = 5.43\) Å, periodic XYZ |
| Positions [Å] | Si (0, 0, 0); Si (1.3575, 1.3575, 1.3575) |
| Basis | DZVP-MOLOPT-GTH |
| Pseudopotential | GTH-PBE-q4 (zion 4) |
| XC | PBE |
| Density cutoff | 30 Ry |
| SCF | mix 0.3, level_shift 0.3 Ha, max_iter 80, tol \(10^{-6}\) |
| RPA | full spectrum, \(n_{\mathrm{occ}}=4\), \(n_{\mathrm{virt}}=22\) |
| Pin date | 2026-07-25 (first-green Atomiverse) |

**Accuracy gate (product):**

| Quantity | Atomiverse / pin | Oracle / status | Tol | Notes |
| -------- | ---------------: | --------------- | --- | ----- |
| \(E_{\mathrm{SCF}}\) (PBE, Γ) | **−7.5955445070** Ha | first-green ATV | setup | REFERENCE table |
| **\(E_{\mathrm{c}}^{\mathrm{dRPA}}\)** | **−0.2658539776** Ha | first-green band (no CP2K plasmon twin) | **\(1\times 10^{-3}\) Ha** | **primary soft gate** |
| \(E_{\mathrm{total}}\) | −7.8613984846 Ha | \(E_{\mathrm{SCF}}+E_c\) | identity \(10^{-12}\) | hard |
| Tr \(A\) | 70.122806 Ha | diagnostic | — | REFERENCE |
| \(n_{\mathrm{occ}}/n_{\mathrm{virt}}\) | 4 / 22 (\(n_{\mathrm{ov}}=88\)) | — | exact | hard |
| \(\Omega_{\min}/\Omega_{\max}\) | 0.029896 / 3.347032 Ha | all \(\Omega_s>0\) | hard | full spectrum |
| \(\lvert E_c^{\mathrm{RI}}-E_c^{\mathrm{Poisson}}\rvert\) | ≤ \(10^{-10}\) Ha | internal dual path | **\(10^{-10}\) Ha** | hard |
| SCF | converged, finite | — | `converged == true` | hard |

Offline re-pin (human only; never ctest):

```bash
# when a CP2K binary is available — diagnostic near-twin only
cp2k.psmp -i tools/cp2k_oracle/inputs/periodic-rpa-si.inp \
          -o /tmp/cp2k_oracle_periodic-rpa-si.out
# record RI-RPA correlation [Ha] into REFERENCE.md only if the path is a fair
# plasmon-equivalent twin; otherwise keep first-green band + note algorithm gap
```

## Performance

Source: [`REFERENCE.md`](../../../regressions/gpw/periodic-rpa-si/REFERENCE.md);
queue/handoff **`perf_exempt`**.

| Item | Value |
| ---- | ----- |
| `perf_exempt` | **true** |
| Reason | CP2K RI-RPA (freq + Gaussian RI_AUX + EXX total) ≠ ATV plasmon on PW-RI \((ia|jb)\); no fair wall twin. No CP2K binary required on pin host. |
| \(T_{\mathrm{ATV}}\) (OMP=4, Release, EPYC login) | ~22 s total (SCF ≈ 20 s, RI-RPA ≈ 1 s, Poisson cross-check ≈ 0.3 s) |
| e2e budget | ≤ 60 s (met) |
| \(r = T_{\mathrm{ATV}}/T_{\mathrm{CP2K}}\) | **n/a (exempt)** |
| `within_10x` | n/a (exempt) |

Campaign records the same: ratio **perf_exempt**. Gate when a
fair twin exists: \(r = T_{\mathrm{ATV}}/T_{\mathrm{CP2K}} \le 10\) on the same
host/core count (single-rank).

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | -----------------------: | ----------------------: | ----: | ---- | ------ |
| sc-Si₂ Γ dRPA@PBE (30 Ry) | — | ~22 | **exempt** | n/a | Algorithm mismatch; remeasure only on plasmon-equivalent twin |

## Limitations

- **First-green soft absolute band**, not a published CP2K plasmon \(E_c\) digit
  match. CP2K's stock RI-RPA path is frequency-integrated with Gaussian
  `RI_AUX` and EXX total-energy assembly; do **not** hard-pin against those
  correlation digits without a documented algorithm match.
- **Γ only.** Multi-k RPA and residual RI-RPA scale-up remain ○
  ([dft.md](../../roadmap/dft.md) §10).
- **dRPA@KS only.** No EXX replacement of the SCF energy; periodic HF / hybrid
  remains blocked.
- **Closed-shell RKS.** No UKS/URPA; no OT reference (canonical virtuals
  required).
- **Correlation energy product**, not a spectroscopic excitation driver. Poles
  are diagnostics for the plasmon formula; TDHF/TDDFT roots are separate
  molecular entry points (`direct-rpa-h2-sto3g` / response slugs on the PySCF
  loop).
- **Artificial sc-Si₂ foothold** at 30 Ry — not production diamond Si₈ at high
  cutoffs.
- **`perf_exempt`**: no fair \(r\) until an algorithmically matched offline
  twin is timed on the same host; do not invent a speed ratio.
- **CPU only** — CUDA GPW remains out of the CP2K loop OUT set.
- Basis/PP tables are static in-tree headers; unsupported element/name
  **throws** (no silent fallback).

## References

- Offline pin: `regressions/gpw/periodic-rpa-si/REFERENCE.md`
- Regression consumer: `regressions/gpw/periodic-rpa-si/reg_periodic-rpa-si.cpp`
- Oracle twin: `tools/cp2k_oracle/inputs/periodic-rpa-si.inp`
- Oracle notes: `tools/cp2k_oracle/README.md`
- Roadmap: `docs/roadmap/dft.md` §10 (Periodic RPA Γ dRPA@KS ✓ product)
- Related method pages:
  [periodic-mp2-si](periodic-mp2-si.md) (shared MO-ERI foothold),
  [direct-rpa-h2-sto3g](direct-rpa-h2-sto3g.md) (molecular plasmon dRPA),
  [gpw-pbesol-si](gpw-pbesol-si.md) (GPW Si technical-report shape)
- Public API: `include/atomiverse/cpu/gpw/rpa.h`,
  `include/atomiverse/cpu/gpw/gpw_solver.h`
- Impl: `src/cpu/gpw/rpa.cpp`, `src/cpu/gpw/internal/mp2_kernels.cpp`
- Unit gate: `tests/cpu/gpw/test_cpu_gpw_rpa.cpp`
- F. Furche, *J. Chem. Phys.* **114**, 5982 (2001), DOI 10.1063/1.1353585
- X. Ren et al., *New J. Phys.* **14**, 053020 (2012)
- M. Del Ben, J. Hutter, J. VandeVondele, *J. Chem. Theory Comput.* **8**, 4177
  (2012) — GPW / RI-GPW \((ia|jb)\) route
- G. Lippert, J. Hutter, M. Parrinello, *Mol. Phys.* **92**, 477 (1997);
  *Theor. Chem. Acc.* **103**, 124 (1999) — GPW / Quickstep
