# Γ-point UGW@UKS-RPBE — sc-Si₂ / DZVP-MOLOPT-GTH

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Regression | None standing (end-to-end case removed) |
| Public API | `cpu::gpw::GpwDft` (`n_alpha`/`n_beta`, `XcKind::RPBE`, `g0w0()` → `G0W0Result`) |
| Roadmap | `docs/roadmap/dft.md` — Periodic G0W0 / UGW ◐ and RPBE ◐ (no standing case) |

## Summary

**Unrestricted spectral G0W0 (UGW)** on a converged **Γ-point UKS GPW**
reference with **RPBE** spin-GGA exchange–correlation: spin-conserving
direct RPA screening (α→α ⊕ β→β), same-spin \(G_\sigma W\) correlation
self-energy, truncated (MT) exact-exchange diagonals per spin, and
spin-resolved RPBE \(v_{\mathrm{xc}}^\sigma\) on the UKS density. This page
is the recorded gate that RPBE is active on the UGW path (not only on the
UKS SCF), dual-gated against closed-shell spectral G0W0@RPBE when
\(n_\alpha=n_\beta\).

Public path: construct `cpu::gpw::GpwDft` with one `Kind` per element, pass `cutoff_ry` and `XcKind::RPBE` to the constructor, set members (`n_alpha`/`n_beta`, …), then
`build(frame)`, `run(frame)` (positions [Å], Z in `kAtom_type`, box [Å]),
then `g0w0()`. Quasiparticle gap on `G0W0Result::gap_ev` is **eV**
(α channel primary report). No standing end-to-end regression. Fair wall vs
CP2K: **`perf_exempt`** (CP2K RI-G0W0 ≠ spectral MT algorithm).

Related: UGW@UKS-PBE twin [periodic-ugw-si](periodic-ugw-si.md); Γ-UKS RPBE
SCF without GW [gpw-uks-rpbe](gpw-uks-rpbe.md); closed-shell G0W0
[periodic-gw-si](periodic-gw-si.md). Narrative GPW: [Ch. 9](../ch09-gpw.md). Molecular
all-electron G0W0 is a separate PySCF-loop concern (`g0w0-h2-sto3g`) —
not this page.

## Method

**GPW (Quickstep family).** Contracted Gaussian AOs; density collocation
and Hartree on a uniform FFT grid; GTH pseudopotentials. Cutoff in Ry.
Citation: Lippert, Hutter, Parrinello, *Mol. Phys.* **92**, 477 (1997).

**UKS with closed-shell occupations.** \(N_\alpha = N_\beta = 4\) on
artificial sc-Si₂ (8 valence e⁻ from 2× GTH-PBE-q4). Separate α/β
densities and KS matrices; when \(C^\alpha \approx C^\beta\), the UKS
solution dual-gates RKS energy and the UGW spectrum dual-gates RGW.

**RPBE spin GGA (not revPBE).** Hammer–Hansen–Nørskov exchange enhancement

\[
F_x(s) = 1 + \kappa\bigl(1 - e^{-\mu s^2/\kappa}\bigr)
\]

with PBE \(\kappa,\mu\) and **PBE correlation** (libxc `GGA_X_RPBE` +
`GGA_C_PBE`). Grid kernel `xc_grid_rpbe_spin` is shared by UKS SCF and the
UGW \(v_{\mathrm{xc}}^\sigma\) diagonals. **Not** Zhang–Yang revPBE
(κ-revision). CP2K 2026.1 requires nested `&GGA_X_RPBE` / `&GGA_C_PBE`
blocks — short name `XC_FUNCTIONAL RPBE` is not a valid enumeration
(same as [gpw-uks-rpbe](gpw-uks-rpbe.md)). Citation: Hammer, Hansen, Nørskov,
*Phys. Rev. B* **59**, 7413 (1999).

**Spectral G0W0 / UGW.** Screening is **spin-conserving direct RPA**
(Coulomb-only Casida over α→α ⊕ β→β; spins explicit — no closed-shell
factor 2). Correlation self-energy is the frequency-exact spectral sum
over RPA poles with same-spin \(G_\sigma W\). Exact-exchange diagonals use
**truncated (MT) Coulomb** per spin so the product is G=0-safe without a
periodic-HF path. Quasiparticle equation (Newton, orbit-by-orbit):

\[
\omega = \varepsilon_p + \mathrm{Re}\,\Sigma_{c,p}(\omega)
         + \Sigma_{x,p} - v_{\mathrm{xc},p}.
\]

Primary report: **α-channel** spectrum (`qp_energy_ha`, `gap_ev` =
HOMO→LUMO QP gap [eV]). Citations: Hybertsen & Louie, *Phys. Rev. B*
**34**, 5390 (1986); Furche, *J. Chem. Phys.* **114**, 5982 (2001);
molecular twin `G0W0<RHF>::run` / PySCF `gw.gw_exact.GWExact` form.

**Deck choice.** Artificial sc-Si₂ (a = 5.43 Å, 30 Ry, DZVP-MOLOPT-GTH)
matches [periodic-ugw-si](periodic-ugw-si.md) / [periodic-gw-si](periodic-gw-si.md) so PBE vs RPBE functional
integrity is a direct digit compare. Level shift 0.3 Ha aids the
small-gap artificial cell; eigenvalues used in denominators are unshifted.

**Key exclusions (scope-out).**

- No absolute CP2K QP-gap hard twin (CP2K path is frequency-integrated
  RI-G0W0, not spectral plasmon UGW + MT exchange).
- No multi-k UGW on this page ([periodic-ugw-kpoints](periodic-ugw-kpoints.md) is separate).
- No open-shell \(N_\alpha \neq N_\beta\) pin here (occupations closed-shell
  for dual-gate UGW≡RGW).
- No analytic GW forces / stress; no BSE@UGW product on this slug.
- Not all-electron molecular G0W0 (Gaussian SCF / PySCF path).

## Implementation

Public surface only (`cpu::Frame` + `GpwDft` members; **no** `*Config` bag,
**no** `make_*` / `build_*` factories):

```cpp
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/gpw/g0w0.h"
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/molrs.hpp"

using namespace ATV;
using namespace ATV::cpu::gpw;

constexpr double a = 5.43; // Å

// sc-Si₂: Si at (0,0,0) and (a/4,a/4,a/4) [Å]
cpu::Frame frame(2);
for (int i = 0; i < 2; ++i) {
    frame.span(kAtom_type)[i] = 14;
    frame.span(kAtom_x)[i] = frame.span(kAtom_y)[i] =
        frame.span(kAtom_z)[i] = i * a / 4.0;
}
frame.box = cpu::Box::cubic(a);

GpwDft m{{Kind{molrs::Element::Si,
               GaussianBasis(molrs::Element::Si, "DZVP-MOLOPT-GTH"),
               GthPotential(molrs::Element::Si, "GTH-PBE-q4")}}, 30.0, XcKind::RPBE};  // Hammer RPBE + PBE c (not revPBE)
m.max_iter = 80;
m.tol = 1e-6;
m.mix = 0.3;
m.level_shift = 0.3;          // Ha; SCF aid
m.n_alpha = 4;
m.n_beta = 4;                 // UKS with closed-shell occupations
m.mp2_ri = true;              // product RI-GPW tensors for dRPA / G0W0
m.build(frame);
const GpwResult& scf = m.run(frame);

m.mp2_n_virt = 0;             // 0 = all kept virtuals per spin
const G0W0Result& gw = m.g0w0();
// gw.gap_ev [eV]     α HOMO→LUMO QP gap
// gw.mf_gap_ev [eV]  KS gap (diagnostic)
// gw.qp_energy_ha    α QP spectrum [Ha]
// gw.converged       Newton QP solve
```

Dual-gate RKS twin (same deck, `n_occ = 4` instead of `n_alpha`/`n_beta`)
is exercised in the recorded check; callers that only need UGW skip it.

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
| Method (`GpwDft::g0w0`) | `gpw_solver.h`, `src/cpu/gpw/` |
| RPBE spin XC | `xc_grid_gga.h` (`xc_grid_rpbe_spin`) |
| Unit tests (not e2e) | `tests/cpu/gpw/test_cpu_gpw_g0w0.cpp` |
| CP2K twin input | CP2K oracle |

## Validation (CP2K)

Source: values recorded when this page was written; no standing regression. (offline pin; **never**
live CP2K in ctest). Oracle contract: CP2K oracle.
CP2K twin input ships for operator re-pin only; Atomiverse gates are
**internal + dual-gate**, not absolute CP2K QP digits (algorithm mismatch:
frequency-integrated RI-G0W0 vs spectral UGW + MT exchange).

| Quantity | Pin / gate | Tol | Notes |
| -------- | ---------- | --- | ----- |
| \|gap(UGW@RPBE) − gap(RKS G0W0@RPBE)\| | dual-gate in-binary | **≤ 0.05 eV** | UGW ≡ RGW when \(C^\alpha\approx C^\beta\) |
| \|gap(UGW@RPBE) − 5.719545 eV\| | vs PBE UGW ([periodic-ugw-si](periodic-ugw-si.md)) | **≥ 1e-3 eV** | functional integrity — RPBE must not equal PBE |
| SCF + Newton | both converged | hard | `max_iter=80`, `tol=1e-6` |
| QP gap | \(> 0\), finite | hard | α HOMO→LUMO |
| \(\varepsilon_{\mathrm{HOMO}}^{\mathrm{QP}} < \varepsilon_{\mathrm{LUMO}}^{\mathrm{QP}}\) | order | hard | spectrum not re-sorted |
| QP shifts | finite vs KS | hard | \(\Sigma_c + \Sigma_x - v_{\mathrm{xc}}\) |
| `G0W0Result.ri` | true | hard | product RI-GPW path |

**First-green digits** (Atomiverse Release, 2026-07-26, EPYC login 8 OMP;
not a CP2K absolute twin):

| Quantity | Value | Notes |
| -------- | ----: | ----- |
| QP gap UGW@RPBE | **5.674272 eV** | α channel |
| QP gap RKS G0W0@RPBE | **5.674251 eV** | closed-shell twin; \(\lvert\Delta\rvert \approx 2\times 10^{-5}\) eV |
| \(E_{\mathrm{SCF}}\) (UKS-RPBE) | **−7.6173910139 Ha** | matches RKS to ~7e-9 Ha |
| KS gap α | **0.384145 eV** | diagnostic |
| PBE UGW first-green (compare) | **5.719545 eV** | [periodic-ugw-si](periodic-ugw-si.md); \(\lvert\Delta\rvert \gg 1\) meV |

**Geometry / protocol (must match twin).**

| Item | Value |
| ---- | ----- |
| Crystal | artificial sc-Si₂ (same as [periodic-ugw-si](periodic-ugw-si.md)) |
| Cell | cubic **a = 5.43 Å**, `PERIODIC XYZ` |
| Positions | Si (0,0,0), Si (a/4,a/4,a/4) |
| Basis / PP | DZVP-MOLOPT-GTH / GTH-PBE-q4 |
| XC | **RPBE** (nested GGA_X_RPBE + GGA_C_PBE) |
| Cutoff | **30 Ry** |
| Spin | UKS MULT 1 → \(N_\alpha=N_\beta=4\); dual-gate RKS \(n_{\mathrm{occ}}=4\) |
| G0W0 | all kept virtuals; `mp2_ri = true` |
| Level shift | 0.3 Ha (SCF aid) |

Offline re-run (operator only; not ctest):

```bash
module load buildtool-easybuild/5.2.1-hpca3ef7d197 GCC/14.3.0 MPICH/4.3.2 CP2K/2026.1
export CP2K_DATA_DIR=$EBROOTCP2K/share/cp2k/data
cd /tmp && ln -sf $CP2K_DATA_DIR/{BASIS_MOLOPT,BASIS_RI_MOLOPT,GTH_POTENTIALS} .
# copy <CP2K oracle input for this case>
mpirun -np 1 cp2k.psmp -i ugw-uks-rpbe.inp -o ugw-uks-rpbe.out
```

Printed CP2K QP levels are a **near-twin diagnostic** only; do not replace
the dual-gate band with uncited RI-GW digits.

## Performance

Source: same pin notes / handoff (the case pin notes).
**Not** a ctest gate.

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r = T_{\mathrm{ATV}}/T_{\mathrm{CP2K}}\) | ≤10× | Caveat |
| ---- | ------------------------: | ----------------------: | ---------------------------------------: | ---- | ------ |
| sc-Si₂ UGW@UKS-RPBE + RKS G0W0@RPBE dual-gate, Γ, 30 Ry | — | **≈ 22** (UKS SCF + UGW + RKS SCF + G0W0) | **perf_exempt** | exempt | CP2K RI-G0W0 ≠ spectral MT algorithm; no fair wall twin required on pin host |

Wall budget: dual-gate deck ≤ 60 s (regression constraint). Never launch
CP2K from ctest.

## Limitations

- **Algorithm mismatch with CP2K default GW:** Atomiverse implements
  **spectral** plasmon G0W0 with **MT exchange**; CP2K `RI_G0W0` is
  frequency-integrated RI-GW. Absolute QP parity is deferred until a
  fair spectral-equivalent twin exists.
- **Closed-shell occupations only** on this pin — proves RPBE on the UGW
  path via UGW≡RGW; true open-shell \(N_\alpha\neq N_\beta\) UGW@RPBE is
  covered by the general UKS g0w0 path but not by the recorded band.
- **Γ only** — multi-k UGW is [periodic-ugw-kpoints](periodic-ugw-kpoints.md).
- **Tiny cell / low cutoff** — artificial sc-Si₂ at 30 Ry is a foothold,
  not production diamond Si₈ / plane-wave complete basis.
- **No GW forces / BSE@UGW** productization on this slug.
- **`perf_exempt`** — do not invent a CP2K wall ratio from RI-G0W0 logs.

## References

- Offline pin: recorded when this page was written (no standing regression)
- CP2K twin: CP2K oracle
- Oracle contract: CP2K oracle
- PBE UGW twin: [periodic-ugw-si](periodic-ugw-si.md) (recorded value)
- Narrative GPW: [`docs/tutorial/ch09-gpw.md`](../ch09-gpw.md)
- Roadmap: [`docs/roadmap/dft.md`](../../roadmap/dft.md)
- Public headers: `g0w0.h`, `gpw_solver.h`
- G. Lippert, J. Hutter, M. Parrinello, *Mol. Phys.* **92**, 477 (1997) — GPW
- B. Hammer, L. B. Hansen, J. K. Nørskov, *Phys. Rev. B* **59**, 7413 (1999) — RPBE
- M. S. Hybertsen, S. G. Louie, *Phys. Rev. B* **34**, 5390 (1986) — G0W0
- F. Furche, *J. Chem. Phys.* **114**, 5982 (2001) — dRPA / spectral poles
