# Multi-k spectral G0W0 — diamond Si (bct 4-atom) / DZVP-MOLOPT-GTH (MP 2×1×1)

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Regression | None standing (end-to-end case removed) |
| Public API | `cpu::gpw::GpwDft::g0w0()` → `cpu::gpw::G0W0Result` (`cpu/gpw/g0w0.h`); members `k_mesh`, `k_time_reversal`, `mp2_ri`, `mp2_n_virt` |
| Roadmap | `docs/roadmap/dft.md` §10 **Periodic GW** ◐ (no standing regression); handoff ratio **perf_exempt** |

## Summary

Closed-shell **k-diagonal spectral \(G_0W_0\)** quasiparticle (QP) energies on a
converged multi-\(k\) **GPW PBE** SCF for diamond silicon (body-centred
tetragonal 4-atom cell, Monkhorst–Pack **2×1×1**). Screening is **direct RPA**
(dRPA) built from **same-\(k\)** (vertical, \(q=0\)) occupied–virtual pair
densities via the product **RI-GPW** \((ia|jb)\) path; the correlation
self-energy is the frequency-exact spectral sum over those poles; exact-exchange
diagonals use **truncated (Martyna–Tuckerman) Coulomb** so the product is G=0-safe
without a full periodic-HF path. The primary product observable is the
**minimum direct QP gap [eV]** over the sampled mesh. Public path:
`GpwDft` members → `build(frame)` → `run(frame)` → `g0w0()` — no `*Config` bag, no factory.
Offline twin:
CP2K oracle (CP2K never in ctest).
Closes the multi-\(k\) residual after the Γ foothold [periodic-gw-si](periodic-gw-si.md).

## Method

**Level.** One-shot \(G_0W_0\) on KS (PBE) orbitals: Green’s function and screened
interaction from the converged multi-\(k\) RKS solution; no self-consistency in
\(G\) or \(W\).

**Spin / BC.** Closed-shell spatial MOs only (RKS). Periodic XYZ. UKS / OT / open
shell are out of scope for this entry point.

**k sampling.** Monkhorst–Pack mesh on `GpwDft::k_mesh` with
`k_time_reversal = false` (equal weights, no BZ reduction) so the e2e deck has
**\(N_k = 2\)** points at \(w = 1/2\). Bloch MOs are complex; the multi-\(k\)
product stores per-\(k\) coefficients in `GpwResult::C_k`.

**Screening (k-diagonal).** For each sampled \(k\), dRPA poles are built only
from **same-\(k\)** occupied–virtual pairs (vertical \(q=0\)):

\[
\rho_{ia}^{(k)}(\mathbf r) = \psi_i^{(k)}(\mathbf r)\,\psi_a^{(k)*}(\mathbf r),
\]

with RI-GPW Poisson on complex pair densities (linear real Poisson on Re/Im
when needed). Full inter-\(k\) / finite-\(q\) \(W(\mathbf q)\) is a follow-on
(see finite-\(q\) dRPA product [periodic-rpa-finite-q](periodic-rpa-finite-q.md)).

**Self-energy.** Spectral (frequency-exact) diagonal correlation
\(\Sigma_{c,pp}(\omega)\) as a sum over those same-\(k\) RPA poles (molecular
`G0W0<RHF>::run` / PySCF `GWExact` form, adapted to Bloch). Exact exchange
\(\Sigma_{x,p}\) uses **MT-truncated** Coulomb (Γ-safe foothold). Local
\(v_{\mathrm{xc},p}\) is PBE on the BZ-weighted density.

**Quasiparticle equation.** Per MO \(p\) and each \(k\), Newton solve

\[
\omega = \varepsilon_p^{(k)} + \mathrm{Re}\,\Sigma_{c,p}^{(k)}(\omega)
         + \Sigma_{x,p}^{(k)} - v_{\mathrm{xc},p}^{(k)}.
\]

**Not** the linearized \(Z\)-factor form. Virtual space for screening is capped
by `mp2_n_virt` (e2e uses **2** for ≤ 60 s); QP levels are reported for the full
converged MO spectrum at the \(k\) that realises the **minimum direct** QP gap.

**Observables**

| Symbol | Definition | Unit |
| ------ | ---------- | ---- |
| \(\varepsilon_p^{\mathrm{QP}}\) | QP orbital energies (report \(k\)) | Ha |
| `gap_ev` | min direct \(\varepsilon_{\mathrm{LUMO}}^{\mathrm{QP}}-\varepsilon_{\mathrm{HOMO}}^{\mathrm{QP}}\) over mesh | eV |
| `mf_gap_ev` | min direct KS HOMO–LUMO gap over mesh | eV |
| IP / EA | \(-\varepsilon_{\mathrm{HOMO}}^{\mathrm{QP}}\), \(\varepsilon_{\mathrm{LUMO}}^{\mathrm{QP}}\) on report \(k\) | eV |

**Excluded (non-goals of this slug)**

- Full inter-\(k\) / finite-\(q\) screened \(W\); analytic-continuation GW.
- evGW / qsGW / scGW; BSE; RPA total energy as this product’s gate
  (`direct_rpa` is a separate entry point).
- Open-shell / UKS G0W0; OT (no canonical virtuals).
- Absolute digit twin to CP2K’s default **frequency-integrated RI-G0W0** path
  (algorithmically distinct from k-diagonal spectral + MT \(x\); digits not
  invented when no fair twin exists).
- Live CP2K (or any third-party) inside ctest; CUDA.
- Public `*Config` bags / `make_*` / `build_*` factories.

## Implementation

Public surface only. Headers:
`gpw_solver.h`, `g0w0.h`, `gaussian_basis.h`,
`gth_potential.h`.

```cpp
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/gpw/g0w0.h"
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/molrs.hpp"

using namespace ATV::cpu::gpw;

const double a = 5.43;                  // Å
const double at = a / std::sqrt(2.0);   // bct a_t = a/√2

const double frac[4][3] = {
    {0.0, 0.0, 0.0}, {0.5, 0.5, 0.5}, {0.5, 0.0, 0.25}, {0.0, 0.5, 0.75}};
ATV::cpu::Frame frame(4);
for (int i = 0; i < 4; ++i) {
    frame.span(ATV::kAtom_type)[i] = 14;
    frame.span(ATV::kAtom_x)[i] = frac[i][0] * at;
    frame.span(ATV::kAtom_y)[i] = frac[i][1] * at;
    frame.span(ATV::kAtom_z)[i] = frac[i][2] * a;
}
frame.box = ATV::cpu::Box::orthorhombic(at, at, a);

GpwDft m{{Kind{molrs::Element::Si,
               GaussianBasis(molrs::Element::Si, "DZVP-MOLOPT-GTH"),
               GthPotential(molrs::Element::Si, "GTH-PBE-q4")}}};
m.cutoff_ry = 80.0;
m.n_occ = 8;                 // 4 Si × zion 4 → 16 e⁻
m.max_iter = 60;
m.tol = 1e-7;
m.mix = 0.3;
m.level_shift = 0.0;
m.eps_default = 1e-7;
m.xc = XcKind::PBE;
m.mp2_ri = true;             // multi-k G0W0 requires RI-GPW
m.mp2_n_virt = 2;            // e2e screening cap (≤ 60 s)
m.k_mesh[0] = 2;
m.k_mesh[1] = 1;
m.k_mesh[2] = 1;
m.k_time_reversal = false;   // FULL_GRID / no symmetry twin
m.diis_depth = 6;
m.build(frame);

const GpwResult& scf = m.run(frame);           // scf.n_kpts == 2
const G0W0Result& gw = m.g0w0();               // multi-k, no throw
// gw.gap_ev [eV]  — min direct QP gap (primary product)
// gw.mf_gap_ev, gw.ip_ev, gw.ea_ev, gw.qp_energy_ha, gw.converged, gw.n_kpts
```

Guards: `std::logic_error` before `build()` / converged `run()`;
`std::invalid_argument` for UKS, `use_ot`, missing multi-\(k\) `C_k`,
multi-\(k\) with `!mp2_ri`, empty virtual space, or vanishing HOMO–LUMO gap
(\(< 10^{-6}\) Ha). No public intermediate bag types.

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
'
```

| Path | Role |
| ---- | ---- |
| `g0w0.h` | `G0W0Result` public value type |
| `gpw_solver.h` | `GpwDft::g0w0`, `k_mesh`, `mp2_ri`, `mp2_n_virt` |
| `gaussian_basis.h` | `GaussianBasis(Element, name)` lookup |
| `gth_potential.h` | `GthPotential(Element, name)` lookup |
| `src/cpu/gpw/internal/` (G0W0 / RI-GPW kernels) | private spectral + RI machinery |
| `tests/cpu/gpw/test_cpu_gpw_g0w0.cpp` | unit: multi-\(k\) kernel at \(N_k=1\) ≡ real spectral QP |
| CP2K oracle | offline CP2K twin (never ctest) |

Related product path: Γ-only spectral G0W0 on sc-Si₂ —
[periodic-gw-si](periodic-gw-si.md).

## Validation (vs CP2K)

Source: values recorded when this page was written (no standing regression).
Oracle status at pin time (**2026-07-26**): **no CP2K binary / module** on the
worker host. Absolute CP2K QP digits are **not invented**. CP2K’s default
`&RI_G0W0` path is **frequency-integrated** RI-GW with Gaussian `RI_AUX`, not
k-diagonal spectral G0W0 with MT exchange — so even with a binary, absolute
gaps are a **near-twin diagnostic**, not a hard digit gate, until a fair
spectral-equivalent twin is defined.

### System / settings

| Setting | Value |
| ------- | ----- |
| Crystal | diamond Si, **bct 4-atom** (same deck family as [periodic-rpa-kpoints](periodic-rpa-kpoints.md)) |
| Cell | \(a_t = a/\sqrt{2}\), \(c = a\); \(a = \mathbf{5.43}\) Å |
| Positions (frac) | \((0,0,0)\), \((\tfrac12,\tfrac12,\tfrac12)\), \((\tfrac12,0,\tfrac14)\), \((0,\tfrac12,\tfrac34)\) |
| Basis / PP | **DZVP-MOLOPT-GTH** / **GTH-PBE-q4** |
| XC | **PBE**, cutoff **80 Ry**, `eps_default` \(10^{-7}\) |
| Valence | 16 e⁻ → \(n_{\mathrm{occ}} = 8\) |
| k | **MP 2×1×1**, `k_time_reversal=false` → 2 k, \(w=\tfrac12\) |
| SCF | DIIS 6, tol \(10^{-7}\), mix 0.3, **no** level_shift |
| G0W0 | k-diagonal spectral, **`mp2_ri=true`**, **`mp2_n_virt=2`** |
| Pin date | 2026-07-26 |

### Binding gates

1. **Hard multi-\(k\):** SCF converged; `C_k.size()==n_kpts`; `g0w0` no throw;
   `n_kpts==2`; `ri`; Newton converged; finite positive min-direct QP/KS gaps;
   HOMO QP < LUMO QP on the reported spectrum.
2. **Kernel Γ recovery (unit):** multi-\(k\) kernel at \(N_k=1\) ≡ real spectral
   QP on the same orbitals (`test_cpu_gpw_g0w0`).
3. **First-green absolute band** (setup integrity) — freeze ATV digits below;
   replace with a fair CP2K spectral twin when available.

### Accuracy table (Atomiverse first-green pins)

| Quantity | Pin | Tol | Notes |
| -------- | --: | --- | ----- |
| \(E_{\mathrm{SCF}}\) (2k, PBE, 80 Ry) | **−15.5751782413** Ha | — | 7 iters; same deck family as multi-\(k\) RPA |
| KS min-direct gap | **1.561168** eV | — | diagnostic |
| **min-direct QP gap** | **8.009033** eV | **0.5 eV** | **primary regression gate** |
| \(\varepsilon_{\mathrm{HOMO}}^{\mathrm{QP}}\) (report \(k\)) | **+0.167405** Ha | — | MT \(x\) + tiny virt space may shift absolute levels |
| \(\varepsilon_{\mathrm{LUMO}}^{\mathrm{QP}}\) (report \(k\)) | **+0.461731** Ha | — | gap is the hard product observable |
| \(n_{\mathrm{occ}}\) / \(n_{\mathrm{virt}}\) / \(n_k\) | 8 / 2 / 2 | exact | product flags |

Wall (SCF + G0W0, OMP=4, EPYC login at pin): **~34 s** (≤ 60 s e2e budget).

Twin input (for offline re-pin / diagnostic only):

```text
<CP2K oracle input for this case>
```

CP2K deck mirrors geometry, PBE, 80 Ry, MONKHORST-PACK 2 1 1 with
`FULL_GRID .TRUE.` / `SYMMETRY .FALSE.`, `ADDED_MOS 2`, RI-G0W0 with
`CORR_MOS_OCC 8` / `CORR_MOS_VIRT 2`. Do **not** treat printed CP2K QP levels
as a hard twin of the spectral + MT product without documenting algorithm
parity.

## Performance

Source: the case pin notes / REFERENCE **perf_exempt**.

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | ------------------------: | ----------------------: | ----: | ---- | ------ |
| multi-k spectral G0W0 | — | ~34 (OMP=4, pin host) | **perf_exempt** | n/a | CP2K RI-G0W0 (freq-integrated) ≠ k-diagonal spectral + MT \(x\); no fair wall twin at pin time |

Gate policy: fair wall \(r = T_{\mathrm{ATV}}/T_{\mathrm{oracle}} \le 10\) when a
method-class-matched twin exists; otherwise **`perf_exempt`** (this slug). Never
launch CP2K from ctest.

## Limitations

- **k-diagonal \(W\) only.** Finite-\(q\) inter-\(k\) screening is not this
  product; dRPA finite-\(q\) lives under [periodic-rpa-finite-q](periodic-rpa-finite-q.md).
- **Closed-shell RKS only.** No UKS/OT G0W0; no hybrid / exact-exchange KS
  reference (periodic HF still ○).
- **Spectral product path only.** No public analytic-continuation GW; CP2K’s
  default RI-G0W0 is frequency-integrated and is not a digit twin.
- **MT exchange foothold.** Absolute QP levels can sit above zero on tiny
  virtual spaces; the hard observable is the **gap**, not absolute HOMO/LUMO
  vs vacuum.
- **e2e virt cap.** `mp2_n_virt = 2` is for the ≤ 60 s budget — not a
  production diamond QP claim.
- **CPU only.** CUDA multi-\(k\) G0W0 is ○.
- **Pin maturity.** First-green absolute band (0.5 eV) until a fair offline
  spectral CP2K twin replaces it; do not invent CP2K digits.
- **Speed.** `perf_exempt`; no standing multi-system fair-wall suite for this
  twin class.

## References

- Offline pin: values recorded when this page was written (no standing regression)
- Offline CP2K twin: CP2K oracle
- Oracle rules: CP2K oracle (offline pins only)
- Roadmap: `docs/roadmap/dft.md` §10 Periodic GW
- Γ foothold sibling: [periodic-gw-si](periodic-gw-si.md)
- Molecular spectral twin (PySCF campaign, separate loop):
  `docs/tutorial/methods/g0w0-h2-sto3g.md`
- Hybertsen & Louie, *Phys. Rev. B* **34**, 5390 (1986)
- Furche, *J. Chem. Phys.* **114**, 5982 (2001) (dRPA / plasmon reduction)
- Del Ben, Hutter, VandeVondele, *J. Chem. Theory Comput.* **8**, 4177 (2012)
  (periodic RI-GPW correlation layout)
- Wilhelm, Hutter, et al. — CP2K RI-GW / GW documentation (freq-integrated
  near-twin class; not algorithm-identical to this spectral product)
