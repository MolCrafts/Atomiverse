# Multi-k unrestricted GPW MP2 (UMP2) — diamond Si₄ bct / DZVP-MOLOPT-GTH

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Regression | None standing (end-to-end case removed) |
| Public API | `cpu::gpw::GpwDft::mp2()` → `cpu::gpw::Mp2Result` (`cpu/gpw/mp2.h`); UKS `n_alpha`/`n_beta`, `k_mesh[3]`, `k_time_reversal`, `mp2_ri`, `mp2_n_virt` |
| Roadmap | `docs/roadmap/dft.md` §10 — multi-k UMP2 ◐ (no standing regression) |
| Dual gate | E_corr vs multi-k RMP2 first-green; speed **`perf_exempt`** |

## Summary

**Momentum-conserving unrestricted second-order Møller–Plesset (UMP2)** on top
of a multi-k **GPW UKS PBE** SCF. Same-spin αα/ββ and opposite-spin αβ doubles
follow the molecular / PySCF UMP2 convention on **complex Bloch** MOs; Coulomb
\((ia|jb)\) is assembled by **RI-GPW** (plane-wave resolution of the identity on
occupied–virtual pair densities; Del Ben, Hutter, VandeVondele, *JCTC* **8**,
4177 (2012); CP2K `&WF_CORRELATION` `METHOD MP2_GPW` + `&KPOINTS` + `UKS`).

This page is the **product multi-k UMP2 residual** after the Γ footholds
([periodic-mp2-si](periodic-mp2-si.md), [periodic-ump2-si](periodic-ump2-si.md)) and the closed-shell
multi-k residual ([periodic-mp2-kpoints](periodic-mp2-kpoints.md)). With closed-shell occupations
\(n_\alpha = n_\beta\) and \(C^\alpha \approx C^\beta\), **UMP2 ≡ RMP2** — the
e2e dual-gate.

Public path: construct `cpu::gpw::GpwDft`, set cell / cutoff / UKS occupations /
Monkhorst–Pack mesh, `build` → `run` → `mp2`. Correlation and SCF energies on
`Mp2Result` are **Hartree** (internal); public Frame energy remains eV when the
evaluator path is used. Fair wall
vs CP2K is **`perf_exempt`** (no CP2K binary on the pin host). Offline twin:
CP2K oracle — never launched by ctest.

## Method

**GPW reference.** Wavefunction in contracted Gaussians (DZVP-MOLOPT-GTH);
density / Hartree / XC on a uniform grid at density cutoff [Ry]; GTH-PBE-q4
pseudopotentials. Multi-k SCF stores complex Bloch coefficients
`GpwResult::C_k` / `C_k_beta` and eigenvalues `eps_k` / `eps_k_beta`.

**UKS occupations (this pin).** Diamond Si bct 4-atom cell, 16 valence e⁻,
closed-shell occupations \(n_\alpha = n_\beta = 8\), multiplicity 1. The
unrestricted machinery is exercised with equal spin channels so the dual-gate
against multi-k RMP2 is meaningful; open-shell multi-k UMP2 (unequal \(n_\alpha,
n_\beta\)) reuses the same kernel.

**k-mesh.** Monkhorst–Pack \(2\times 1\times 1\) with `k_time_reversal = false`
→ \(N_k = 2\), equal weights \(w = 1/2\) (CP2K twin: `FULL_GRID .TRUE.`,
`SYMMETRY .FALSE.`). Nonzero `level_shift` is rejected on the k-path.

**RI-GPW \((ia|jb)\).** Occupied–virtual pair densities are **neutral**
(\(\int\rho_{ia}=0\)), so the G=0-dropped periodic Coulomb is divergence-free.
Complex pair densities use linear real Poisson on Re/Im (product
`mp2_ri = true`; multi-k Poisson foothold is not offered).

**Momentum-conserving UMP2 sum.** With equal-weight mesh

\[
E^{(2)} = N_k^{-1}\sum_{k_i k_j k_a}\sum_{ijab}
  \frac{\cdots}{\Delta_{ij}^{ab}(k)}\,,
  \qquad
  k_b \equiv k_i + k_j - k_a
\]

(fractional wrap into the sampled set). Spin channels (molecular UMP2 /
PySCF convention; \(v\) complex):

\[
E_{\mathrm{SS}}^\sigma
  = \tfrac14\sum_{ijab}
    \frac{\lvert v_{iajb}^\sigma - v_{ibja}^\sigma\rvert^2}
         {\Delta}\,,
  \qquad
E_{\mathrm{OS}}
  = \sum_{ijab}
    \frac{\lvert v_{i_\alpha a_\alpha j_\beta b_\beta}\rvert^2}
         {\Delta}
\]

(no exchange in OS). Then \(E_{\mathrm{corr}} = E_{\mathrm{OS}} + E_{\mathrm{SS}}\)
(before optional Grimme `mp2_os_scale` / `mp2_ss_scale`; defaults 1.0). When
\(C^\alpha \equiv C^\beta\) and occupations match, OS/SS recover multi-k RMP2.

**Virtual cap.** `mp2_n_virt` is **per spin** for UMP2. This e2e pin uses
`mp2_n_virt = 2` → reported `n_virt = 4` (≤ 60 s budget). `ADDED_MOS 2` on the
CP2K twin matches that cap.

**Key exclusions**

- Absolute CP2K digits invented without a binary (offline re-pin when available).
- Poisson multi-k path (`mp2_ri = false` + \(N_k>1\)).
- OT SCF (`use_ot` — no canonical virtuals).
- Periodic hybrid / exact-exchange reference (blocked; correlation is @KS-PBE).
- UMP2 analytic forces / stress; URPA@UKS (queue residual [periodic-urpa-si](periodic-urpa-si.md)).
- Production diamond at full virtual space / production cutoffs.

## Implementation

Public surface only — **no** `*Config` bag, **no** `make_*` / `build_*`
factories. Geometry is a `cpu::Frame` (positions [Å], Z in `kAtom_type`, orthorhombic
`box`) for `GpwDft`; basis/PP via
lookup constructors `GaussianBasis` / `GthPotential`.

```cpp
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/cpu/gpw/mp2.h"
#include "atomiverse/molrs.hpp"

using namespace ATV;

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

cpu::gpw::GpwDft m{{cpu::gpw::Kind{molrs::Element::Si,
                                   cpu::gpw::GaussianBasis(molrs::Element::Si, "DZVP-MOLOPT-GTH"),
                                   cpu::gpw::GthPotential(molrs::Element::Si, "GTH-PBE-q4")}}, 80.0, cpu::gpw::XcKind::PBE};
m.n_alpha = 8;
m.n_beta = 8;
m.max_iter = 60;
m.tol = 1e-7;
m.mix = 0.3;
m.level_shift = 0.0;           // k-path rejects nonzero
m.eps_default = 1e-7;
m.diis_depth = 6;
m.k_mesh[0] = 2;
m.k_mesh[1] = 1;
m.k_mesh[2] = 1;
m.k_time_reversal = false;     // equal weights (CP2K FULL_GRID / no symmetry)
m.mp2_ri = true;
m.mp2_n_virt = 2;              // per spin → n_virt = 4 total

m.build(frame);
const cpu::gpw::GpwResult& scf = m.run(frame);  // 2k UKS, C_k + C_k_beta
const cpu::gpw::Mp2Result& r = m.mp2();         // multi-k UMP2, no throw
// r.e_corr, r.e_corr_os, r.e_corr_ss, r.e_total [Ha]; r.n_kpts==2; r.ri
```

```bash
# Scoped build/test (never bare cmake --build build)
```

Relevant headers: `gpw_solver.h` (`GpwDft::mp2`),
`mp2.h` (`Mp2Result`), kernels private under `cpu/gpw/internal/mp2_kernels.h`
(`mp2_uhf_energy_kpoints`).

## Validation (vs CP2K offline / dual-gate)

Source: values recorded when this page was written (no standing regression) (offline pin;
never live in ctest). Twin input:
CP2K oracle.

### System (binding)

| Item | Value |
| ---- | ----- |
| Crystal | diamond Si, **bct 4-atom** (same as [periodic-mp2-kpoints](periodic-mp2-kpoints.md)) |
| Cell | a_t = a/√2, c = a; a = **5.43 Å** |
| Positions (frac) | (0,0,0), (½,½,½), (½,0,¼), (0,½,¾) |
| Basis / PP | **DZVP-MOLOPT-GTH** / **GTH-PBE-q4** |
| XC | **PBE UKS**, \(n_\alpha=n_\beta=8\), cutoff **80 Ry** |
| k | **MP 2×1×1**, `k_time_reversal=false` → 2 k, \(w=\tfrac12\) |
| SCF | DIIS 6, tol 1e-7, mix 0.3, **no level_shift** |
| MP2 | RI-GPW, **n_virt = 2 per spin** |

### Binding gates

1. **Hard multi-k UMP2:** SCF converged; `C_k.size()==C_k_beta.size()==n_kpts`;
   `e_corr` finite and **strictly negative**; OS+SS ≡ e_corr;
   e_total ≡ e_scf+e_corr; `n_occ=16`, `n_virt=4`, `ri`, `n_kpts=2`.
2. **Dual-gate vs RKS multi-k first-green**
   ([periodic-mp2-kpoints](periodic-mp2-kpoints.md) E_corr = **−0.0657637511** Ha): band
   \(\lvert E_{\mathrm{corr}}(\mathrm{UMP2}) - \mathrm{ref}\rvert \le\)
   **2e-3** Ha (UKS numerical drift).
3. **Kernel unit** (`tests/cpu/gpw/test_cpu_gpw_mp2.cpp`): multi-k UMP2 α=β
   recovers multi-k RMP2 OS/SS (no full SCF e2e in unit tier).

### Accuracy table (first green, 2026-07-26, OMP=4)

| Quantity | Atomiverse / pin [Ha] | Oracle | Tol | Notes |
| -------- | --------------------: | ------ | --- | ----- |
| \(E_{\mathrm{SCF}}\) (2k UKS PBE, 80 Ry) | **−15.5751782413** | dual-gate RKS multi-k | setup | 7 iters; matches RKS multi-k |
| \(E_{\mathrm{corr}}\) (2k UMP2, n_virt=2/spin) | **−0.0657637511** | dual-gate RMP2 multi-k | **2e-3** | exact match on first green |
| \(E_{\mathrm{corr}}^{\mathrm{OS}}\) | −0.0598553603 | internal | identity | OS+SS ≡ e_corr @ 1e-12 |
| \(E_{\mathrm{corr}}^{\mathrm{SS}}\) | −0.0059083908 | internal | identity | |
| \(E_{\mathrm{total}}\) | −15.6409419924 | internal | identity | e_scf + e_corr |
| Dual-gate ref (RMP2 multi-k) | −0.0657637511 | [periodic-mp2-kpoints](periodic-mp2-kpoints.md) | — | first-green twin |

**Offline CP2K status:** no CP2K binary/module on the pin host at first green.
Absolute CP2K `MP2 Energy` digits are **not invented**. When CP2K is available,
re-pin from the twin input and cite the release; replace the dual-gate band if
needed.

## Performance

Source: the case pin notes (wave 6) and REFERENCE.md.

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | ------------------------: | ----------------------: | ----: | ---- | ------ |
| multi-k UMP2 Si₄-bct | — | **~3.7** (SCF+UMP2, OMP=4) | **perf_exempt** | n/a | no CP2K binary on pin host |

Gate: \(r = T_{\mathrm{ATV}}/T_{\mathrm{CP2K}} \le 10\) when a fair wall exists.
This slug: **`perf_exempt`**. Recorded wall stayed within 60 s.

## Limitations

- Dual-gate vs closed-shell multi-k RMP2, not absolute CP2K digits, until an
  offline re-pin lands.
- Virtual space heavily capped (`mp2_n_virt = 2`) for e2e ≤ 60 s — not a
  production correlation energy for solid Si.
- Correlation on **KS-PBE** orbitals (standard periodic foothold); true MP2@HF
  needs periodic exact exchange (blocked hybrid row).
- Requires `mp2_ri = true` and a fully periodic cell for multi-k.
- No UMP2 forces/stress; URPA@UKS remains open ([periodic-urpa-si](periodic-urpa-si.md)).
- Header comment in `mp2.h` may still say multi-k UMP2 is a “follow-on” —
  product path is multi-k UKS as of this slug; trust `GpwDft::mp2` docs + this
  regression.

## References

- Offline pin: values recorded when this page was written (no standing regression)
- Oracle twin: CP2K oracle
- Closed-shell multi-k twin: [periodic-mp2-kpoints](periodic-mp2-kpoints.md)
- Γ UMP2 foothold: [periodic-ump2-si](periodic-ump2-si.md)
- Roadmap: `docs/roadmap/dft.md` §10
- Del Ben, Hutter, VandeVondele, *J. Chem. Theory Comput.* **8**, 4177 (2012)
  (RI-GPW / CP2K `MP2_GPW`)
- Related tutorials: [periodic-mp2-si](periodic-mp2-si.md) (Γ RMP2),
  [ump2-o2-sto3g](ump2-o2-sto3g.md) (molecular UMP2 / PySCF)
