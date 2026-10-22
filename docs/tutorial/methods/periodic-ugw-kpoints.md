# Multi-k unrestricted spectral G0W0 (UGW) — diamond Si₄ bct / DZVP-MOLOPT-GTH

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Regression | None standing (end-to-end case removed) |
| Public API | `cpu::gpw::GpwDft::g0w0()` → `cpu::gpw::G0W0Result` (`cpu/gpw/g0w0.h`); UKS via `n_alpha`/`n_beta`, multi-k via `k_mesh` |
| Roadmap | `docs/roadmap/dft.md` §10 — Periodic G0W0 / UGW ◐ (multi-k UGW; no external pin, no standing case) |

## Summary

**Unrestricted spectral \(G_0W_0\) (UGW)** quasiparticle (QP) energies on top of
a multi-k **GPW UKS PBE** SCF with closed-shell occupations
\(n_\alpha = n_\beta\). Screening is **spin-conserving direct RPA** over
\(\alpha\!\to\!\alpha \oplus \beta\!\to\!\beta\) transitions with **finite-\(q\)
inter-\(k\)** momentum conservation (same dielectric path as multi-k RKS
G0W0 / [periodic-gw-kpoints](periodic-gw-kpoints.md) / [periodic-gw-finite-q](periodic-gw-finite-q.md)). Correlation self-energy
is the frequency-exact spectral sum over RPA poles with same-spin
\(G_\sigma W\); exact-exchange diagonals use **truncated (MT) Coulomb** per
spin; local \(v_{\mathrm{xc}}^\sigma\) is spin-resolved PBE on the BZ-weighted
UKS density.

Primary public path: set UKS + k-mesh members on `cpu::gpw::GpwDft`,
`build` → `run` → `g0w0()`. Report is the **α-channel** min-direct QP gap
[eV] over the sampled mesh. No standing end-to-end regression. Offline twin:
CP2K oracle (CP2K never in ctest).

When \(C^\alpha \approx C^\beta\), **multi-k UGW ≡ multi-k RGW** — dual-gated
against the closed-shell multi-k spectral G0W0 first-green min-direct QP gap
from [periodic-gw-kpoints](periodic-gw-kpoints.md). Absolute CP2K RI-G0W0 digits are **not** a hard
twin (algorithm class differs; see Validation).

## Method

**Level.** One-shot \(G_0W_0\): Green’s function and screened interaction from
the multi-k UKS mean-field; no self-consistency in \(G\) or \(W\).

**Reference SCF.** GPW UKS PBE on diamond Si, **bct 4-atom** cell (same deck as
[periodic-gw-kpoints](periodic-gw-kpoints.md)): Monkhorst–Pack **2×1×1**, `k_time_reversal=false` →
\(N_k=2\) with equal weights \(w=1/2\). Occupations \(n_\alpha=n_\beta=8\)
(16 valence e⁻ with GTH-PBE-q4).

**Screening.** Spin-conserving dRPA (Coulomb kernel; no exchange in the RPA
channel) over α→α ⊕ β→β. Multi-k \(W\) is built with **finite-\(q\)**
inter-\(k\) momentum conservation — product multi-k dielectric, not a pure
k-diagonal approximation. Virtual space for screening capped by
`mp2_n_virt` (**2 per spin** on this pin → `n_virt` report = 4).

**Self-energy.** Spectral (frequency-exact) sum over RPA poles for the
diagonal correlation self-energy \(\Sigma_c^\sigma(\omega)\). Exact exchange
\(\Sigma_x^\sigma\) uses **Martyna–Tuckerman truncated Coulomb** so the product
is G=0-safe without a periodic-HF path. Local XC diagonals are
\(\langle p_\sigma|v_{\mathrm{xc}}^\sigma|p_\sigma\rangle\).

**Quasiparticle equation** (Newton, orbit-by-orbit, per \(k\)):

\[
\omega = \varepsilon_p + \mathrm{Re}\,\Sigma_{c,p}(\omega)
         + \Sigma_{x,p} - v_{\mathrm{xc},p}.
\]

**Observables (public `G0W0Result`).**

| Symbol | Definition | Unit |
| ------ | ---------- | ---- |
| `qp_energy_ha` | \(\varepsilon_p^{\mathrm{QP}}\) (α channel; \(k\) of min-direct QP gap) | Ha |
| `mf_energy_ha` | KS \(\varepsilon_p\) at the same \(k\) / spin | Ha |
| `gap_ev` | min-direct α HOMO→LUMO QP gap over mesh | eV |
| `mf_gap_ev` | min-direct α KS gap | eV |
| `ip_ev` / `ea_ev` | \(-\varepsilon_{\mathrm{HOMO}}^{\mathrm{QP}}\) / \(\varepsilon_{\mathrm{LUMO}}^{\mathrm{QP}}\) | eV |
| `n_occ` / `n_virt` | \(n_\alpha\) / \(n_{\mathrm{virt}}^\alpha+n_{\mathrm{virt}}^\beta\) after cap | — |
| `n_kpts` / `ri` / `converged` | mesh size; RI-GPW path; Newton success | — |

**Excluded.** Frequency-integrated RI-GW as in CP2K default `&RI_G0W0`;
analytic-continuation GW; evGW / qsGW / scGW; BSE; RPA total energy from this
entry point (use `direct_rpa`); multi-k URPA total \(E_c\); CUDA.

## Implementation

Public surface only (`GpwDft` members; **no** `*Config` bag, **no**
`make_*` / `build_*` factories). Headers: `cpu/gpw/gpw_solver.h`,
`cpu/gpw/g0w0.h`; kernels under `cpu/gpw/internal/`.

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

// bct diamond Si4: a = 5.43 Å, a_t = a/√2, c = a; fractional coords
// (0,0,0), (½,½,½), (½,0,¼), (0,½,¾); Z = 14 in kAtom_type.
constexpr double a = 5.43;                   // [Å]
const double at = a / std::sqrt(2.0);
const double frac[4][3] = {{0, 0, 0}, {0.5, 0.5, 0.5}, {0.5, 0, 0.25}, {0, 0.5, 0.75}};
cpu::Frame frame(4);
for (int i = 0; i < 4; ++i) {
    frame.span(kAtom_type)[i] = 14;
    frame.span(kAtom_x)[i] = frac[i][0] * at;
    frame.span(kAtom_y)[i] = frac[i][1] * at;
    frame.span(kAtom_z)[i] = frac[i][2] * a;
}
frame.box = cpu::Box::orthorhombic(at, at, a);

GpwDft m{{Kind{molrs::Element::Si,
               GaussianBasis(molrs::Element::Si, "DZVP-MOLOPT-GTH"),
               GthPotential(molrs::Element::Si, "GTH-PBE-q4")}}, 80.0, XcKind::PBE};
m.n_alpha = 8;
m.n_beta = 8;
m.max_iter = 60;
m.tol = 1e-7;
m.mix = 0.3;
m.level_shift = 0.0;
m.eps_default = 1e-7;
m.mp2_ri = true;          // required for multi-k G0W0 product path
m.mp2_n_virt = 2;         // per spin for UKS screening cap
m.k_mesh[0] = 2;
m.k_mesh[1] = 1;
m.k_mesh[2] = 1;
m.k_time_reversal = false; // FULL_GRID / no symmetry ≈ equal weights
m.diis_depth = 6;
m.build(frame);
const GpwResult& scf = m.run(frame);   // 2k UKS-PBE
const G0W0Result& gw = m.g0w0();       // multi-k UGW — no throw
// gw.n_kpts == 2; gw.ri; gw.converged
// gw.gap_ev = min_k direct α QP gap [eV]
```

Guards: `std::logic_error` before `build()` / converged `run()` or missing β
orbitals; `std::invalid_argument` for `use_ot`, missing multi-k `C_k` /
`C_k_beta`, multi-k without `mp2_ri`, vanishing HOMO–LUMO gap, empty virtual
space.

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
'
```

## Validation (vs offline CP2K / dual-gate)

Source: values recorded when this page was written; no standing
regression.
(offline pin only; **never** live CP2K in ctest).

| Setting | Value |
| ------- | ----- |
| Crystal | diamond Si, **bct 4-atom** |
| Cell | \(a_t = a/\sqrt{2}\), \(c = a\); \(a =\) **5.43 Å** |
| Basis / PP | **DZVP-MOLOPT-GTH** / **GTH-PBE-q4** |
| XC / spin | **PBE UKS**, \(n_\alpha=n_\beta=8\) |
| Cutoff | **80 Ry** |
| k-mesh | MP **2×1×1**, `k_time_reversal=false` → 2 k, \(w=\tfrac12\) |
| SCF | DIIS 6, tol \(10^{-7}\), mix 0.3, **no level_shift** |
| G0W0 | finite-\(q\) inter-\(k\) spectral UGW, `mp2_n_virt=2` / spin |
| Oracle twin input | CP2K oracle |
| Pin date | 2026-07-26 |

**Oracle class note.** CP2K default path is **frequency-integrated RI-GW** with
Gaussian `RI_AUX` — **not** multi-k spectral UGW with MT exchange. Absolute
CP2K QP gaps are therefore **not invented** as hard digits; the recorded gates
are structural + dual-gate vs Atomiverse multi-k RKS G0W0.

### Accuracy table (first green, OMP=8, 2026-07-26)

| Quantity | Atomiverse / pin | Tol / band | Notes |
| -------- | ---------------: | ---------- | ----- |
| \(E_{\mathrm{SCF}}\) (2k UKS-PBE) | **−15.5751782413** Ha | info | 7 iters; ≡ RKS multi-k twin |
| KS min-direct gap α | **1.561168** eV | finite / >0 | ≡ RKS multi-k twin |
| **min-direct QP gap α** | **7.698404** eV | dual-gate | Δ vs RKS multi-k G0W0 ≈ 0.17 eV |
| \(\varepsilon_{\mathrm{HOMO}}^{\mathrm{QP}}\) / \(\varepsilon_{\mathrm{LUMO}}^{\mathrm{QP}}\) | +0.158037 / +0.440948 Ha | HOMO < LUMO | report \(k\) |
| Dual-gate ref (RKS multi-k G0W0) | **7.529834** eV | — | [periodic-gw-kpoints](periodic-gw-kpoints.md) first-green |
| Recorded band | \|gap − 7.529834\| ≤ **0.5 eV** | hard gate | setup band for UGW≡RGW |
| \(n_{\mathrm{occ}}\) / \(n_{\mathrm{virt}}\) | 8 / 4 | exact | \(n_\alpha\); virt = 2+2 |
| `n_kpts` / `ri` / Newton | 2 / true / converged | exact | multi-k product |

### Binding gates (recorded)

1. **Hard multi-k UGW:** SCF converged; `C_k.size()==C_k_beta.size()==n_kpts`;
   `g0w0` no throw; `n_kpts==2`; `ri`; Newton converged; finite positive
   min-direct α QP/KS gaps; HOMO_QP < LUMO_QP; `n_occ==8`, `n_virt==4`.
2. **Kernel unit:** multi-k UHF at \(N_k=1\) with \(C^\alpha=C^\beta\) ≡
   multi-k RHF spectral QP (`tests/cpu/gpw/test_cpu_gpw_g0w0.cpp`).
3. **Dual-gate band** vs RKS multi-k first-green min-direct QP gap
   ([periodic-gw-kpoints](periodic-gw-kpoints.md) **7.529834 eV**): \|gap − ref\| ≤ **0.5 eV**.

## Performance

Source: the case pin notes (wave 7, 2026-07-26) +
REFERENCE.

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | ------------------------: | ----------------------: | ----: | ---- | ------ |
| multi-k spectral UGW | — | **~16** (SCF+UGW, OMP=8) | **perf_exempt** | n/a | CP2K RI-G0W0 ≠ spectral + MT; no fair wall |

**Wall budget:** regression ≤ **60 s**. Never launch CP2K from ctest. When a
fair spectral-equivalent CP2K UGW path exists on a pin host, re-pin absolute
gaps and optionally record a fair_bench ratio.

## Limitations

- **Algorithm twin is dual-gate, not absolute CP2K QP digits.** Stock CP2K
  `&RI_G0W0` is frequency-integrated RI-GW; Atomiverse product is spectral
  UGW + MT \(\Sigma_x\).
- **Tiny virtual space on the pin** (`mp2_n_virt=2` / spin) — foothold for
  multi-k UGW machinery, not a production band-structure QP claim.
- **Primary report is α-channel.** β spectrum is used in screening but not
  re-exported as a second `G0W0Result` block.
- **Requires `mp2_ri = true` on multi-k.** Poisson-only multi-k G0W0 is out of
  product scope.
- **No OT reference** (`use_ot` throws — no canonical virtuals).
- **CPU only.** CUDA multi-k UGW is ○ / out of this loop.
- **Related closed product:** Γ UGW [periodic-ugw-si](periodic-ugw-si.md); multi-k RKS G0W0
  [periodic-gw-kpoints](periodic-gw-kpoints.md) / [periodic-gw-finite-q](periodic-gw-finite-q.md). Multi-k URPA total energy
  remains a separate residual if still open on the queue.

## References

- Offline pin: recorded when this page was written (no standing regression)
- Oracle input twin: CP2K oracle
- Γ UGW twin: [periodic-ugw-si](periodic-ugw-si.md) (recorded value)
- Multi-k RKS G0W0 dual-gate: [periodic-gw-kpoints](periodic-gw-kpoints.md)
- Roadmap: `docs/roadmap/dft.md` §10
- Public headers: `g0w0.h`, `gpw_solver.h`
- Unit: `tests/cpu/gpw/test_cpu_gpw_g0w0.cpp`
- Hybertsen & Louie, *Phys. Rev. B* **34**, 5390 (1986)
- Furche, *J. Chem. Phys.* **114**, 5982 (2001) (dRPA poles)
- Molecular spectral twin: `docs/tutorial/methods/g0w0-h2-sto3g.md`
  / PySCF `gw.gw_exact.GWExact`
