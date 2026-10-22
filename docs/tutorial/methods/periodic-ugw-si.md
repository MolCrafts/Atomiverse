# Periodic Γ-point UGW@UKS — sc-Si₂ / DZVP-MOLOPT-GTH (spectral G0W0, PBE)

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Regression | None standing (end-to-end case removed) |
| Public API | `cpu::gpw::GpwDft::g0w0()` → `cpu::gpw::G0W0Result` (`cpu/gpw/g0w0.h`); UKS via members `n_alpha` / `n_beta` |
| Roadmap | `docs/roadmap/dft.md` §10 — Periodic G0W0 / UGW ◐ (no external pin; no standing case) |

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
QP gap against the closed-shell first-green pin of [periodic-gw-si](periodic-gw-si.md)
(\(5.719545\) eV). Public path: `GpwDft` members → `build` → `run` →
`g0w0()` — **no** public `*Config`, **no** `make_*` / `build_*` factory.
No standing end-to-end regression (gates below are recorded). Fair wall vs CP2K is
**`perf_exempt`** (algorithm mismatch — see Performance).

Related closed-shell spectral twin: [periodic-gw-si](periodic-gw-si.md).
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
setup-integrity gate used by the recorded check.

**Key exclusions (product scope).**

- **Γ only** — multi-k UGW throws (`multi-k UGW is a follow-on`).
- No analytic-continuation GW; no evGW / qsGW / scGW.
- No BSE; no RPA total energy from this entry (use `direct_rpa` for \(E_c\)).
- No OT reference (`use_ot` — no canonical virtuals).
- No CUDA UGW.
- CP2K offline input is frequency-integrated **RI-G0W0**, not spectral MT UGW
  — absolute CP2K QP digits are **not** hard-coded as the product twin.

## Implementation

Public surface only (`GpwDft` members; geometry from a `cpu::Frame` —
positions [Å], Z in `kAtom_type`, orthorhombic `box` — like every GPW run).
**No** `*Config` bag, **no** factory:

```cpp
#include "atomiverse/cpu/gpw/g0w0.h"
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/molrs.hpp"

using namespace ATV;
using namespace ATV::cpu::gpw;

constexpr double a = 5.43;                    // cubic cell [Å]

cpu::Frame frame(2);                          // sc-Si₂, positions [Å]
for (int i = 0; i < 2; ++i) {
    frame.span(kAtom_type)[i] = 14;
    frame.span(kAtom_x)[i] = frame.span(kAtom_y)[i] =
        frame.span(kAtom_z)[i] = i * a / 4.0;
}
frame.box = cpu::Box::cubic(a);

GpwDft m{{Kind{molrs::Element::Si,
               GaussianBasis(molrs::Element::Si, "DZVP-MOLOPT-GTH"),
               GthPotential(molrs::Element::Si, "GTH-PBE-q4")}}, 30.0, XcKind::PBE};
m.n_alpha = 4;
m.n_beta = 4;              // UKS closed-shell occupations
m.max_iter = 80;
m.tol = 1e-6;
m.mix = 0.3;
m.level_shift = 0.3;       // Ha; small-gap artificial sc-Si₂
m.mp2_ri = true;
m.build(frame);

const GpwResult& scf = m.run(frame);   // converged Γ UKS-PBE
m.mp2_n_virt = 0;                      // 0 = all kept virtuals per spin
const G0W0Result& gw = m.g0w0();
// gw.gap_ev [eV]  — α HOMO→LUMO QP gap (primary gate)
// gw.qp_energy_ha — α spectrum [Ha]; gw.ip_ev / gw.ea_ev [eV]
// gw.converged, gw.ri, gw.n_occ (= n_α), gw.n_virt (= n_vα + n_vβ)
```

`G0W0Result` fields of note: `qp_energy_ha`, `mf_energy_ha`, `sigma_x_ha`,
`vxc_ha`, `gap_ev`, `mf_gap_ev`, `ip_ev`, `ea_ev`, `n_occ`, `n_virt`, `ri`,
`n_kpts` (0 on real Γ), `converged`. Requires prior successful `run()`; throws
if called without SCF, on multi-k UKS, when β orbitals are missing, with
`use_ot`, vanishing HOMO–LUMO gap (< 1e-6 Ha), or empty virtual space.

Build (GPW needs HF + PW + FFTW3):

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
'
```

Relevant sources:

| Piece | Path |
| ----- | ---- |
| Result type | `g0w0.h` |
| Solver entry | `GpwDft::g0w0` in `gpw_solver.h` |
| Impl (UGW branch) | `src/cpu/gpw/g0w0.cpp` |
| RI/MO-ERI kernels | `mp2_kernels.h` |
| Unit tests (not e2e) | `tests/cpu/gpw/test_cpu_gpw_g0w0.cpp` |
| Offline pin | recorded when this page was written (no standing regression) |
| CP2K twin input | CP2K oracle |

## Validation (CP2K / dual-gate)

Source: values recorded when this page was written; no standing regression.
(offline pin only; **never** live CP2K in ctest). Oracle contract:
CP2K oracle.

**Recorded gates (no standing regression).**

| Quantity | Pin / gate | Tol | Notes |
| -------- | ---------: | --- | ----- |
| SCF | converged, finite \(E_{\mathrm{tot}}\) | — | UKS-PBE, `tol=1e-6`, max 80 iters |
| \(E_{\mathrm{SCF}}\) (info) | **−7.5955445033** Ha | ~4e-9 vs RKS | matches [periodic-gw-si](periodic-gw-si.md) RKS energy |
| KS gap α (info) | **0.410157** eV | — | diagnostic |
| **QP gap α** | **5.719545** eV dual-gate | **0.05 eV** | vs RKS first-green [periodic-gw-si](periodic-gw-si.md) |
| UGW first-green (info) | **5.719564** eV | \(\Delta\sim 2\times 10^{-5}\) eV | 2026-07-26 product |
| Newton QP | `gw.converged` | — | all MO Newton solves |
| Gap sign / order | gap > 0; \(\varepsilon_H^{\mathrm{QP}} < \varepsilon_L^{\mathrm{QP}}\) | — | hard |
| Product path | `gw.ri == true` | — | RI-GPW ovov |
| \(n_{\mathrm{occ}}\) report | 4 | — | \(n_\alpha\) primary |

**Geometry / protocol (must match twin deck).**

| Item | Value |
| ---- | ----- |
| Crystal | artificial sc-Si₂ (same as [periodic-gw-si](periodic-gw-si.md) / [periodic-ump2-si](periodic-ump2-si.md)) |
| Cell | cubic **a = 5.43 Å**, periodic, **Γ only** |
| Positions | Si (0,0,0), Si (a/4,a/4,a/4) |
| Basis / PP | **DZVP-MOLOPT-GTH** / **GTH-PBE-q4** |
| XC | **PBE**, UKS \(n_\alpha=n_\beta=4\) |
| Cutoff | **30 Ry** |
| SCF | tol 1e-6, mix 0.3, level_shift 0.3 Ha, max_iter 80 |
| G0W0 | all kept virtuals per spin for dRPA; `mp2_ri = true` |

**Offline CP2K status.** Twin input
CP2K oracle (UKS `MULTIPLICITY 1` +
`&RI_G0W0`). CP2K’s default path is **frequency-integrated RI-GW** with
Gaussian RI_AUX and periodic HF in the GW step — **not** spectral plasmon UGW
with MT exchange. Absolute CP2K QP digits are therefore **not invented** as a
hard twin of the spectral value.

Pin strategy (binding, mirrors [periodic-gw-si](periodic-gw-si.md) / [periodic-ump2-si](periodic-ump2-si.md)):

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
# copy <CP2K oracle input for this case>
mpirun -np 1 cp2k.psmp -i periodic-ugw-si.inp -o periodic-ugw-si.out
```

## Performance

Source: pin notes + handoff
the case pin notes
(wave 6, 2026-07-26). Gate: \(r = T_{\mathrm{ATV}}/T_{\mathrm{CP2K}} \le 10\)
or **`perf_exempt`**.

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | ------------------------: | ----------------------: | ----: | ---- | ------ |
| Γ UGW@UKS-PBE sc-Si₂ / 30 Ry | — | ~14 (SCF ≈ 13 + UGW ≈ 1; 8 OMP) | **perf_exempt** | n/a | CP2K RI-G0W0 ≠ spectral MT UGW; no fair algorithm twin |

**Why exempt.** Fair wall requires the same method class. CP2K RI-G0W0
(minimax / frequency integration + periodic HF exchange) is not a wall-time
twin of spectral plasmon G0W0 with MT \(\Sigma_x\). Recorded wall stays
well under the ≤ 60 s e2e cap. Never launch CP2K from ctest.

## Limitations

- **Γ UKS only.** Multi-k UGW is explicitly a follow-on (throws today).
- **Spectral product path only.** No analytic-continuation GW in the public
  GPW API; CP2K RI-G0W0 numbers are diagnostic, not the recorded gate.
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
  fair twin exists; no multi-system UGW perf suite beyond this page wall.

## References

- Offline pin: recorded when this page was written (no standing regression)
- Closed-shell dual-gate twin: [periodic-gw-si](periodic-gw-si.md) (recorded value)
- CP2K twin input: CP2K oracle
- Oracle contract: CP2K oracle
- Roadmap: [`docs/roadmap/dft.md`](../../roadmap/dft.md) (Periodic GW + UGW)
- Public headers: `g0w0.h`, `gpw_solver.h`
- Impl: `src/cpu/gpw/g0w0.cpp`
- Molecular spectral twin (PySCF loop): [`g0w0-h2-sto3g.md`](g0w0-h2-sto3g.md)
- M. S. Hybertsen, S. G. Louie, *Phys. Rev. B* **34**, 5390 (1986), DOI
  10.1103/PhysRevB.34.5390
- F. Furche, *J. Chem. Phys.* **114**, 5982 (2001) — dRPA / Casida reduction
- J. Wilhelm, M. Del Ben, J. Hutter, *J. Chem. Theory Comput.* **12**, 3623
  (2016) — CP2K RI-GW family (offline twin context; algorithm differs)
