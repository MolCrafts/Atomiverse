# Periodic Γ-point UGW@UKS — sc-Si₂ / DZVP-MOLOPT-GTH (spectral G0W0, PBE)

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `periodic-ugw-si` |
| Regression | `regressions/gpw/periodic-ugw-si/` |
| Public API | `cpu::gpw::GpwDft::g0w0(atoms)` → `cpu::gpw::G0W0Result` (`cpu/gpw/g0w0.h`); UKS via members `n_alpha` / `n_beta` |
| Roadmap | `docs/roadmap/dft.md` §10 — Γ UGW@UKS product (`reg_periodic-ugw-si`) |

## Summary

**Unrestricted spectral G0W0 (UGW)** quasiparticle (QP) energies on a
converged **Γ-point UKS GPW PBE** reference with closed-shell occupations
\(n_\alpha = n_\beta = 4\). Screening is **spin-conserving direct RPA**
(Coulomb-only Casida over α→α ⊕ β→β; spins explicit — no closed-shell factor
2). The correlation self-energy is the frequency-exact spectral sum over RPA
poles with same-spin \(G_\sigma W\). Exact-exchange diagonals use **truncated
(MT) Coulomb** per spin; local \(v_{\mathrm{xc}}^\sigma\) is spin-resolved PBE
on the UKS density. Primary report is the **α-channel** spectrum / HOMO→LUMO
QP gap [eV].

When \(C^\alpha \approx C^\beta\), **UGW ≡ RGW** — the product dual-gates the
QP gap against the closed-shell first-green pin of `periodic-gw-si`
(\(5.719545\) eV). Public path: `GpwDft` members → `build` → `run` →
`g0w0(atoms)` — **no** public `*Config`, **no** `make_*` / `build_*` factory.
Regression binary: `reg_periodic-ugw-si`. Fair wall vs CP2K is
**`perf_exempt`** (algorithm mismatch — see Performance).

Related closed-shell spectral twin: `regressions/gpw/periodic-gw-si`.
Molecular RHF G0W0 is a separate PySCF-loop concern (`g0w0-h2-sto3g`).

## Method

**Level.** One-shot \(G_0W_0\) on UKS orbitals: Green’s function and screened
interaction from the mean-field solution; no self-consistency in \(G\) or \(W\).
Quasiparticle equation (Newton, orbit-by-orbit, α channel for the report):

\[
\omega = \varepsilon_p + \mathrm{Re}\,\Sigma_{c,p}(\omega)
         + \Sigma_{x,p} - v_{\mathrm{xc},p}.
\]

**Spin / UKS.** Separate α/β KS eigenvalues and MO coefficients from the
converged UKS SCF. Occupations are explicit integer members `n_alpha` /
`n_beta` (not a multiplicity bag). For this pin both equal 4 (valence sc-Si₂
with GTH-PBE-q4 → 4 e⁻ per Si × 2). Spin contamination \(\langle S^2\rangle\)
is near zero when orbitals match the RKS solution.

**Screening (dRPA, spin-conserving).** Coulomb-only Casida over the direct sum
of same-spin particle–hole spaces α→α ⊕ β→β. No exchange in the RPA kernel.
`(ia|jb)` from the product **RI-GPW** path when `mp2_ri = true` and the cell is
periodic (same ovov tensors as Γ UMP2 / closed-shell G0W0).

**Self-energy.** Spectral (frequency-exact) sum over RPA poles for the
diagonal correlation self-energy \(\Sigma_c^\sigma(\omega)\), contracted with
same-spin \(G_\sigma W\).

**Exchange and \(v_{\mathrm{xc}}\).** \(\Sigma_x^\sigma\) diagonals use truncated
(Martyna–Tuckerman-style) Coulomb so the Γ-only foothold is G=0-safe without a
periodic exact-exchange product path. \(v_{\mathrm{xc}}^\sigma\) is the
spin-resolved PBE potential diagonal on the UKS density (must match the SCF
functional).

**Observables (product report = α).**

| Symbol | Definition | Unit |
| ------ | ---------- | ---- |
| \(\varepsilon_p^{\mathrm{QP}}\) | QP orbital energies (α spectrum) | Ha |
| gap | \(\varepsilon_{\mathrm{LUMO}}^{\mathrm{QP}} - \varepsilon_{\mathrm{HOMO}}^{\mathrm{QP}}\) (α) | eV |
| IP | \(-\varepsilon_{\mathrm{HOMO}}^{\mathrm{QP}}\) | eV |
| EA | \(\varepsilon_{\mathrm{LUMO}}^{\mathrm{QP}}\) | eV |
| KS gap | α HOMO→LUMO mean-field gap (diagnostic) | eV |

**Dual-gate identity.** For \(n_\alpha = n_\beta\) and \(C^\alpha \approx
C^\beta\), UGW recovers closed-shell spectral G0W0 to numerical noise
(\(\Delta_{\mathrm{gap}} \sim 2\times 10^{-5}\) eV on this deck). That is the
setup-integrity gate used by the regression.

**Key exclusions (product scope).**

- **Γ only** — multi-k UGW throws (`multi-k UGW is a follow-on`).
- No analytic-continuation GW; no evGW / qsGW / scGW.
- No BSE; no RPA total energy from this entry (use `direct_rpa` for \(E_c\)).
- No OT reference (`use_ot` — no canonical virtuals).
- No CUDA UGW.
- CP2K offline input is frequency-integrated **RI-G0W0**, not spectral MT UGW
  — absolute CP2K QP digits are **not** hard-coded as the product twin.

## Implementation

Public surface only (`GpwDft` members; Frame is not required for this low-level
GPW path — the regression uses `GpwAtom` + `GpwDft` like other periodic GPW
post-SCF footholds). **No** `*Config` bag, **no** factory:

```cpp
#include "atomiverse/cpu/gpw/g0w0.h"
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/molrs.hpp"

using namespace ATV::cpu::gpw;
constexpr double kA2B = 1.8897259885789;
const double a_bohr = 5.43 * kA2B;

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
m.n_alpha = 4;
m.n_beta = 4;              // UKS closed-shell occupations
m.max_iter = 80;
m.tol = 1e-6;
m.mix = 0.3;
m.level_shift = 0.3;       // Ha; small-gap artificial sc-Si₂
m.xc = XcKind::PBE;
m.mp2_ri = true;
m.build(atoms);

const GpwResult& scf = m.run(atoms);   // converged Γ UKS-PBE
m.mp2_n_virt = 0;                      // 0 = all kept virtuals per spin
const G0W0Result& gw = m.g0w0(atoms);
// gw.gap_ev [eV]  — α HOMO→LUMO QP gap (primary gate)
// gw.qp_energy_ha — α spectrum [Ha]; gw.ip_ev / gw.ea_ev [eV]
// gw.converged, gw.ri, gw.n_occ (= n_α), gw.n_virt (= n_vα + n_vβ)
```

`G0W0Result` fields of note: `qp_energy_ha`, `mf_energy_ha`, `sigma_x_ha`,
`vxc_ha`, `gap_ev`, `mf_gap_ev`, `ip_ev`, `ea_ev`, `n_occ`, `n_virt`, `ri`,
`n_kpts` (0 on real Γ), `converged`. Requires prior successful `run()`; throws
if called without SCF, on multi-k UKS, when β orbitals are missing, with
`use_ot`, vanishing HOMO–LUMO gap (< 1e-6 Ha), or empty virtual space.

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
  cmake --build build -j$(nproc) --target atomiverse reg_periodic-ugw-si
  ./build/regressions/reg_periodic-ugw-si
'
```

Relevant sources:

| Piece | Path |
| ----- | ---- |
| Result type | `include/atomiverse/cpu/gpw/g0w0.h` |
| Solver entry | `GpwDft::g0w0` in `include/atomiverse/cpu/gpw/gpw_solver.h` |
| Impl (UGW branch) | `src/cpu/gpw/g0w0.cpp` |
| RI/MO-ERI kernels | `include/atomiverse/cpu/gpw/internal/mp2_kernels.h` |
| Unit tests (not e2e) | `tests/cpu/gpw/test_cpu_gpw_g0w0.cpp` |
| Regression | `regressions/gpw/periodic-ugw-si/reg_periodic-ugw-si.cpp` |
| Offline pin | `regressions/gpw/periodic-ugw-si/REFERENCE.md` |
| CP2K twin input | `tools/cp2k_oracle/inputs/periodic-ugw-si.inp` |

## Validation (CP2K / dual-gate)

Source: [`regressions/gpw/periodic-ugw-si/REFERENCE.md`](../../../regressions/gpw/periodic-ugw-si/REFERENCE.md)
(offline pin only; **never** live CP2K in ctest). Oracle contract:
[`tools/cp2k_oracle/README.md`](../../../tools/cp2k_oracle/README.md).

**Product gates (hard-coded in `reg_periodic-ugw-si`).**

| Quantity | Pin / gate | Tol | Notes |
| -------- | ---------: | --- | ----- |
| SCF | converged, finite \(E_{\mathrm{tot}}\) | — | UKS-PBE, `tol=1e-6`, max 80 iters |
| \(E_{\mathrm{SCF}}\) (info) | **−7.5955445033** Ha | ~4e-9 vs RKS | matches `periodic-gw-si` RKS energy |
| KS gap α (info) | **0.410157** eV | — | diagnostic |
| **QP gap α** | **5.719545** eV dual-gate | **0.05 eV** | vs RKS first-green `periodic-gw-si` |
| UGW first-green (info) | **5.719564** eV | \(\Delta\sim 2\times 10^{-5}\) eV | 2026-07-26 product |
| Newton QP | `gw.converged` | — | all MO Newton solves |
| Gap sign / order | gap > 0; \(\varepsilon_H^{\mathrm{QP}} < \varepsilon_L^{\mathrm{QP}}\) | — | hard |
| Product path | `gw.ri == true` | — | RI-GPW ovov |
| \(n_{\mathrm{occ}}\) report | 4 | — | \(n_\alpha\) primary |

**Geometry / protocol (must match twin deck).**

| Item | Value |
| ---- | ----- |
| Crystal | artificial sc-Si₂ (same as `periodic-gw-si` / `periodic-ump2-si`) |
| Cell | cubic **a = 5.43 Å**, periodic, **Γ only** |
| Positions | Si (0,0,0), Si (a/4,a/4,a/4) |
| Basis / PP | **DZVP-MOLOPT-GTH** / **GTH-PBE-q4** |
| XC | **PBE**, UKS \(n_\alpha=n_\beta=4\) |
| Cutoff | **30 Ry** |
| SCF | tol 1e-6, mix 0.3, level_shift 0.3 Ha, max_iter 80 |
| G0W0 | all kept virtuals per spin for dRPA; `mp2_ri = true` |

**Offline CP2K status.** Twin input
`tools/cp2k_oracle/inputs/periodic-ugw-si.inp` (UKS `MULTIPLICITY 1` +
`&RI_G0W0`). CP2K’s default path is **frequency-integrated RI-GW** with
Gaussian RI_AUX and periodic HF in the GW step — **not** spectral plasmon UGW
with MT exchange. Absolute CP2K QP digits are therefore **not invented** as a
hard twin of the spectral value.

Pin strategy (binding, mirrors `periodic-gw-si` / `periodic-ump2-si`):

1. **Hard internal gates** (convergence, gap sign, HOMO/LUMO order, finite shifts).
2. **Dual-gate band** vs RKS spectral first-green \(\lvert\mathrm{gap}-5.719545\rvert\le 0.05\) eV.
3. Unit tier: `direct_rpa_uhf` dual-gates `direct_rpa_rhf` on identical α=β
   tensors (`tests/cpu/gpw/test_cpu_gpw_g0w0.cpp`); UKS unconverged guard.
4. Offline re-pin: when a fair spectral-equivalent CP2K UGW path exists,
   replace the band with cited CP2K QP gap digits.

Operator-only re-run of the CP2K input (not ctest):

```bash
module load buildtool-easybuild/5.2.1-hpca3ef7d197 GCC/14.3.0 MPICH/4.3.2 CP2K/2026.1
export CP2K_DATA_DIR=$EBROOTCP2K/share/cp2k/data
# copy tools/cp2k_oracle/inputs/periodic-ugw-si.inp; link BASIS_MOLOPT, BASIS_RI_MOLOPT, GTH_POTENTIALS
mpirun -np 1 cp2k.psmp -i periodic-ugw-si.inp -o periodic-ugw-si.out
```

## Performance

Source: `REFERENCE.md` + handoff
the case `REFERENCE.md`
(wave 6, 2026-07-26). Gate: \(r = T_{\mathrm{ATV}}/T_{\mathrm{CP2K}} \le 10\)
or **`perf_exempt`**.

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | ------------------------: | ----------------------: | ----: | ---- | ------ |
| Γ UGW@UKS-PBE sc-Si₂ / 30 Ry | — | ~14 (SCF ≈ 13 + UGW ≈ 1; 8 OMP) | **perf_exempt** | n/a | CP2K RI-G0W0 ≠ spectral MT UGW; no fair algorithm twin |

**Why exempt.** Fair wall requires the same method class. CP2K RI-G0W0
(minimax / frequency integration + periodic HF exchange) is not a wall-time
twin of spectral plasmon G0W0 with MT \(\Sigma_x\). Regression wall stays
well under the ≤ 60 s e2e cap. Never launch CP2K from ctest.

## Limitations

- **Γ UKS only.** Multi-k UGW is explicitly a follow-on (throws today).
- **Spectral product path only.** No analytic-continuation GW in the public
  GPW API; CP2K RI-G0W0 numbers are diagnostic, not the product gate.
- **No self-consistency.** evGW / qsGW / scGW remain roadmap-open.
- **MT exchange, not periodic HF.** \(\Sigma_x\) uses truncated Coulomb so
  the foothold is G=0-safe without a periodic hybrid path
  (`periodic-hybrid-gpw` blocked).
- **Primary report is α.** β QP spectrum is computed internally for screening
  consistency but not dual-gated as a separate product observable on this pin.
- **Artificial sc-Si₂ / 30 Ry.** Textbook setup-integrity deck, not production
  diamond-Si₈ cutoffs or a physical GW band gap claim.
- **CPU only.** CUDA UGW is ○.
- **Speed.** `perf_exempt` until a spectral-equivalent CP2K (or literature)
  fair twin exists; no multi-system UGW perf suite beyond this regression wall.

## References

- Offline pin: [`regressions/gpw/periodic-ugw-si/REFERENCE.md`](../../../regressions/gpw/periodic-ugw-si/REFERENCE.md)
- Regression consumer: [`reg_periodic-ugw-si.cpp`](../../../regressions/gpw/periodic-ugw-si/reg_periodic-ugw-si.cpp)
- Closed-shell dual-gate twin: [`regressions/gpw/periodic-gw-si/REFERENCE.md`](../../../regressions/gpw/periodic-gw-si/REFERENCE.md)
- CP2K twin input: [`tools/cp2k_oracle/inputs/periodic-ugw-si.inp`](../../../tools/cp2k_oracle/inputs/periodic-ugw-si.inp)
- Oracle contract: [`tools/cp2k_oracle/README.md`](../../../tools/cp2k_oracle/README.md)
- Roadmap: [`docs/roadmap/dft.md`](../../roadmap/dft.md) (Periodic GW + UGW)
- Public headers: `include/atomiverse/cpu/gpw/g0w0.h`, `gpw_solver.h`
- Impl: `src/cpu/gpw/g0w0.cpp`
- Molecular spectral twin (PySCF loop): [`g0w0-h2-sto3g.md`](g0w0-h2-sto3g.md)
- M. S. Hybertsen, S. G. Louie, *Phys. Rev. B* **34**, 5390 (1986), DOI
  10.1103/PhysRevB.34.5390
- F. Furche, *J. Chem. Phys.* **114**, 5982 (2001) — dRPA / Casida reduction
- J. Wilhelm, M. Del Ben, J. Hutter, *J. Chem. Theory Comput.* **12**, 3623
  (2016) — CP2K RI-GW family (offline twin context; algorithm differs)
