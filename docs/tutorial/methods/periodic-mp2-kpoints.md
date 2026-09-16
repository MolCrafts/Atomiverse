# Multi-k RI-GPW MP2 — diamond Si (bct 4-atom) / DZVP-MOLOPT-GTH (MP 2×1×1)

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `periodic-mp2-kpoints` |
| Regression | `regressions/gpw/periodic-mp2-kpoints/` |
| Public API | `cpu::gpw::GpwDft::{build,run,mp2}` → `cpu::gpw::Mp2Result` (`cpu/gpw/mp2.h`); members `k_mesh`, `k_time_reversal`, `mp2_ri`, `mp2_n_virt`, `mp2_os_scale`, `mp2_ss_scale` |
| Roadmap | handoff **perf_exempt**; roadmap `dft.md` §10 multi-k RI-GPW MP2 closed |

## Summary

**Momentum-conserving second-order Møller–Plesset correlation on a Monkhorst–Pack
k-mesh**, evaluated on top of a converged closed-shell multi-k **GPW PBE** SCF.
This is the multi-k residual that closes the Periodic MP2 product after the
Γ-only Poisson foothold (`periodic-mp2-si`) and the Γ RI product
(`periodic-ri-mp2-si`). Occupied–virtual pair densities are **complex Bloch
products**; the Coulomb contraction uses the **RI-GPW** plane-wave resolution of
the pair-density identity (Del Ben, Hutter, VandeVondele, *JCTC* **8**, 4177
(2012); CP2K `&WF_CORRELATION` `METHOD MP2_GPW` + `&KPOINTS`). The e2e pin is
diamond Si in the body-centred-tetragonal 4-atom cell (same geometry as
`gpw-kpoints-si`), MP **2×1×1** with equal weights, RI on, and a tight
virtual cap (`mp2_n_virt = 2`) for the ≤ 60 s regression budget. Fair wall vs
CP2K is **`perf_exempt`** (no CP2K binary on the pin host); absolute CP2K digits
are not invented — the offline twin is
`tools/cp2k_oracle/inputs/periodic-mp2-kpoints.inp`.

## Method

### Multi-k reference

The SCF is the multi-k GPW RKS path already productized by `gpw-kpoints-si`:
complex Bloch orbitals \(C_k\), phased lattice sums / KB, per-k DIIS, and
k-summed density. Sampled k come from a Monkhorst–Pack mesh

\[
\kappa_d = \frac{2r - N_d - 1}{2N_d},\qquad r = 1,\ldots,N_d
\]

(CP2K `SCHEME MONKHORST-PACK` convention; even \(N\) is Γ-avoiding). For this
pin: \(N = (2,1,1)\), `k_time_reversal = false` → **two** k-points with equal
weights \(w_k = 1/2\) (matches CP2K `FULL_GRID .TRUE.` / `SYMMETRY .FALSE.`).

### Momentum-conserving doubles

Closed-shell spatial-orbital MP2 on complex Bloch MOs. Pair densities
\(\rho_{ia}^{k_i k_a}(\mathbf r) = \psi_i^{k_i}(\mathbf r)\,
[\psi_a^{k_a}(\mathbf r)]^*\) are **neutral** (\(S_{ia}=0\)), so the periodic
Coulomb with the \(G=0\) component dropped remains divergence-free. RI-GPW
assembles \((ia|jb)\) via the plane-wave identity on the density grid (same
product path as Γ RI-MP2; multi-k requires `mp2_ri = true` — the real-space
Poisson multi-k foothold is **not** offered).

Canonical equal-weight sum over the sampled mesh (OS/SS decomposition,
identical channel convention to molecular `cpu::scf::MP2Result` and Γ
`cpu::gpw::Mp2Result`):

\[
E^{(2)}
=
\frac{1}{N_k}\sum_{k_i k_j k_a}
\sum_{ijab}
\frac{
  (ia|jb)_{k}\,
  \bigl[2(ia|jb)_{k}-(ib|ja)_{k}\bigr]
}{
  \varepsilon_a^{k_a}+\varepsilon_b^{k_b}
  -\varepsilon_i^{k_i}-\varepsilon_j^{k_j}
}
=
E_{\mathrm{OS}}+E_{\mathrm{SS}},
\]

with \(k_b\) fixed by crystal momentum conservation from
\((k_i,k_j,k_a)\). Reported `e_corr` applies optional Grimme scales
`mp2_os_scale` / `mp2_ss_scale` (defaults 1.0 recover plain MP2); the unscaled
pieces stay on `e_corr_os` / `e_corr_ss`.

### Reference orbitals

Correlation is evaluated on **KS (PBE) orbitals** — the standard periodic
MP2@KS foothold (RPA / double-hybrid ingredient). A true MP2@HF reference
awaits periodic exact exchange (`periodic-hybrid-gpw`, blocked). Virtual
space after linear dependence is capped by `mp2_n_virt` (0 = all kept
virtuals). A vanishing min HOMO–LUMO gap on the mesh (< 1e-6 Ha) throws —
canonical denominators are singular.

### What this pin includes

| Item | Setting |
| ---- | ------- |
| Crystal | diamond Si, **bct 4-atom** (same as `gpw-kpoints-si`) |
| Cell | \(a_t = a/\sqrt{2}\), \(c = a\); \(a = 5.43\) Å |
| Positions (frac) | \((0,0,0)\), \((\tfrac12,\tfrac12,\tfrac12)\), \((\tfrac12,0,\tfrac14)\), \((0,\tfrac12,\tfrac34)\) |
| Basis / PP | **DZVP-MOLOPT-GTH** / **GTH-PBE-q4** |
| XC | **PBE**, cutoff **80 Ry**, `eps_default` 1e-7 |
| Valence | 16 e⁻ → \(n_{\mathrm{occ}} = 8\) |
| k | **MP 2×1×1**, `k_time_reversal=false` → \(n_{\mathrm{kpts}}=2\), \(w=1/2\) |
| SCF | DIIS 6, tol 1e-7, mix 0.3, **no level_shift** (k-path rejects it) |
| MP2 | RI-GPW (`mp2_ri=true`), **n_virt = 2** (e2e ≤ 60 s) |

**Note:** the artificial sc-Si₂ deck used for Γ RI-MP2 needs `level_shift` at Γ
and does not converge on the k-path without it — product multi-k MP2 uses the
physical diamond bct cell instead.

### Key exclusions (non-goals of this slug)

- Absolute total-energy digit match to a **live** CP2K process at pin time (no
  binary on host; do **not** invent CP2K digits).
- Poisson multi-k foothold (`mp2_ri = false` throws on multi-k).
- UMP2 / open-shell multi-k MP2; OT reference (no canonical virtuals).
- Multi-k RPA / RI-RPA / periodic GW (separate residual / ○ rows).
- Production cutoffs, full virtual space, or dense MP meshes under the 60 s
  e2e budget.
- Live CP2K (or any third-party) inside ctest.
- Public `*Config` bags / `make_*` / `build_*` factories.

## Implementation

Public surface only (`GpwDft` members + lookup constructors; matches
`reg_periodic-mp2-kpoints.cpp`):

```cpp
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/cpu/gpw/mp2.h"
#include "atomiverse/molrs.hpp"

using namespace ATV::cpu::gpw;

// bct diamond Si4: a = 5.43 Å → Bohr; positions via fractional coords
// (0,0,0), (½,½,½), (½,0,¼), (0,½,¾) with DZVP-MOLOPT-GTH + GTH-PBE-q4.

GpwDft m;
m.cell[0] = m.cell[1] = a_bohr / std::sqrt(2.0);
m.cell[2] = a_bohr;
m.cutoff_ry = 80.0;
m.n_occ = 8;
m.max_iter = 60;
m.tol = 1e-7;
m.mix = 0.3;
m.level_shift = 0.0;         // k-path rejects nonzero
m.eps_default = 1e-7;
m.xc = XcKind::PBE;
m.mp2_ri = true;             // required for multi-k
m.mp2_n_virt = 2;            // e2e budget
m.k_mesh[0] = 2;
m.k_mesh[1] = 1;
m.k_mesh[2] = 1;
m.k_time_reversal = false;   // equal weights
m.diis_depth = 6;
m.build(atoms);

const GpwResult& scf = m.run(atoms);   // stores C_k / eps_k / k_frac / weights
const Mp2Result& r = m.mp2(atoms);     // multi-k RI-GPW; r.ri && r.n_kpts == 2
// r.e_corr (< 0), r.e_corr_os / r.e_corr_ss, r.e_total = r.e_scf + r.e_corr
```

```bash
cmake --build build -j$(nproc) --target atomiverse reg_periodic-mp2-kpoints
./build/regressions/reg_periodic-mp2-kpoints
# or: cmake --build build --target run_reg_periodic-mp2-kpoints
```

**Guards:** `std::logic_error` before `build()` / converged `run()`;
`std::invalid_argument` for UKS, `use_ot`, missing multi-k `C_k`,
`mp2_ri = false` on multi-k, vanishing min gap, or no virtuals. No public
`*Config`, no factory — kernels are private
(`cpu/gpw/internal/mp2_kernels.h`).

Relevant paths:

| Layer | Path |
| ----- | ---- |
| Public result type | `include/atomiverse/cpu/gpw/mp2.h` |
| Method + members | `include/atomiverse/cpu/gpw/gpw_solver.h` |
| Implementation | `src/cpu/gpw/mp2.cpp` |
| Kernels (private) | `cpu/gpw/internal/mp2_kernels.{h,cpp}` |
| Unit (Γ recovery) | `tests/cpu/gpw/test_cpu_gpw_mp2.cpp` |
| E2E | `regressions/gpw/periodic-mp2-kpoints/` |
| Oracle twin | `tools/cp2k_oracle/inputs/periodic-mp2-kpoints.inp` |

## Validation (vs CP2K offline pin)

Source: `regressions/gpw/periodic-mp2-kpoints/REFERENCE.md` (offline pin;
**never** live CP2K in ctest). Oracle twin:
`tools/cp2k_oracle/inputs/periodic-mp2-kpoints.inp`.

### Binding gates

1. **Hard multi-k:** SCF converged; `C_k.size() == n_kpts`; `e_corr` finite and
   **strictly negative**; OS+SS ≡ `e_corr` (1e-12); `e_total` ≡ `e_scf + e_corr`;
   `n_occ`/`n_virt` match; `r.ri == true` and `r.n_kpts == 2`.
2. **Kernel Γ recovery (unit):** `mo_on_grid_k` at \(\kappa=0\) ≡ real
   `mo_on_grid`; complex OS/SS formula ≡ real `mp2_rhf_energy` (no full SCF in
   unit tier).
3. **First-green absolute band** (setup integrity) — freeze ATV digits below;
   replace with CP2K when an offline binary re-pin is available.

### Atomiverse numbers (first green, 2026-07-25, OMP=4)

| Quantity | Atomiverse / pin | CP2K | Tol | Notes |
| -------- | ---------------: | ---: | --- | ----- |
| \(n_{\mathrm{kpts}}\) (SCF / MP2) | 2 | — | exact | `C_k` stored |
| \(n_{\mathrm{occ}}\) / \(n_{\mathrm{virt}}\) | 8 / 2 | ADDED_MOS 2 | exact | virt cap |
| \(E_{\mathrm{SCF}}\) [Ha] | **−15.5751782413** | *pending re-pin* | — | 7 iters; setup |
| \(E_{\mathrm{corr}}\) [Ha] | **−0.0657637511** | *pending re-pin* | **1e-4** | absolute band |
| \(E_{\mathrm{corr}}^{\mathrm{OS}}\) [Ha] | −0.0598553603 | — | identity | unscaled |
| \(E_{\mathrm{corr}}^{\mathrm{SS}}\) [Ha] | −0.0059083908 | — | identity | unscaled |
| \(E_{\mathrm{total}}\) [Ha] | −15.6409419924 | — | 1e-12 | \(E_{\mathrm{SCF}}+E_c\) |
| Wall (SCF + MP2) [s] | **3.6** | — | ≤ 60 | e2e budget |

Geometry, basis, `FULL_GRID` / no symmetry, `ADDED_MOS 2`, and cutoffs match the
oracle twin. Absolute CP2K correlation digits are **not** filled until the twin
is run offline; the regression gate is the ATV first-green band at **1e-4 Ha**.

## Performance

Source: the case `REFERENCE.md` + REFERENCE.

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | ------------------------: | ----------------------: | ----: | ---- | ------ |
| Si4-bct MP 2×1×1 RI-GPW MP2 (n_virt=2, 80 Ry) | n/a | ~3.6 | **perf_exempt** | n/a | no CP2K binary on pin host |

Gate when a binary exists: \(r = T_{\mathrm{ATV}}/T_{\mathrm{CP2K}} \le 10\).
Never launch CP2K from ctest.

## Limitations

- Virtual space and cutoff are e2e-budgeted (`n_virt=2`, 80 Ry); not a
  production bulk-Si correlation number.
- MP2@PBE reference only — no periodic exact exchange, no multi-k hybrid
  double hybrids.
- Multi-k path requires product RI (`mp2_ri`); Poisson multi-k not implemented.
- Closed-shell RKS only; no UMP2 / ROMP2 on the k-mesh.
- Absolute oracle digits pending offline CP2K re-pin of
  `tools/cp2k_oracle/inputs/periodic-mp2-kpoints.inp`.
- Multi-k RPA / periodic GW remain open residuals (`dft.md` §10).

## References

- Offline pin: `regressions/gpw/periodic-mp2-kpoints/REFERENCE.md`
- Oracle twin: `tools/cp2k_oracle/inputs/periodic-mp2-kpoints.inp`
- Oracle rules: `tools/cp2k_oracle/README.md` (offline only; never in ctest)
- Roadmap: `docs/roadmap/dft.md` §10 Periodic MP2
- Prior Γ products: [periodic-mp2-si](periodic-mp2-si.md) (Poisson foothold);
  `regressions/gpw/periodic-ri-mp2-si` (Γ RI)
- Literature: Del Ben, Hutter, VandeVondele, *J. Chem. Theory Comput.* **8**,
  4177 (2012) (RI-GPW MP2 / CP2K `MP2_GPW`)
