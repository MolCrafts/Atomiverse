# Maximally localized Wannier functions — Foster–Boys foothold

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `wannier-centers` |
| Regression | `regressions/scf/wannier-centers/` |
| Public API | `cpu::scf::WannierCenters` + `WannierResult` (`cpu/scf/wannier.h`) |
| Roadmap / queue | [dft.md](../../roadmap/dft.md) §11 MLWF; queue row `wannier-centers` |
| Dual gate | Geometric + unit-algebra pins (REFERENCE.md); speed **`perf_exempt`** |
| Wave | CP2K capability loop — wave 3 / staffed 2026-07-25 |

## Summary

**Maximally localized Wannier functions (MLWF)** turn delocalized canonical
occupied MOs into compact orbitals whose centroids are the **Wannier
centers**. This foothold ships **Foster–Boys** localization (maximize
\(\sum_i|\langle\mathbf r\rangle_i|^2\)) for **closed-shell molecular**
RHF/RKS via Jacobi pair rotations on the occupied MO block. AO multipole
matrices reuse `scf::MultipoleIntegrals` about the Cartesian origin.

**Not in this foothold:** periodic multi-k Marzari–Vanderbilt MLWF, GPW-grid
or plane-wave Wannier, Pipek–Mezey / ER criteria. Those remain roadmap ○
beyond the molecular product foothold.

## Method

Occupied spatial MOs \(\{\varphi_i\}\) from a restricted determinant. Define

\[
\mathbf R_{ij} = \langle\varphi_i|\mathbf r|\varphi_j\rangle,
\qquad
\Omega_i = \langle\varphi_i|r^2|\varphi_i\rangle - |\mathbf R_{ii}|^2.
\]

Boys maximizes \(\sum_i|\mathbf R_{ii}|^2\) (equivalently minimizes
\(\sum_i\Omega_i\)) under unitary mixing of occupied orbitals. Each pair
\((i,j)\) is rotated by

\[
\gamma = \tfrac14\operatorname{atan2}(B,A),
\]

with \(A^a=R_{ii}^a-R_{jj}^a\), \(B^a=2R_{ij}^a\),
\(A=\sum_a((A^a)^2-(B^a)^2)\), \(B=\sum_a 2A^a B^a\).

Outputs: centers \(\mathbf R_{ii}\) **[Å]**, spreads \(\Omega_i\) **[Å²]**.

## Usage

```cpp
#include "atomiverse/cpu/scf/methods.h"
#include "atomiverse/cpu/scf/wannier.h"

cpu::scf::RHF rhf;
rhf.basis = cpu::scf::BasisSet::STO3G;
rhf.tol = 1e-10;
rhf.run(frame);   // closed-shell water, H2, …

cpu::scf::WannierCenters wb;   // members: max_sweeps, tol
wb.tol = 1e-12;
auto r = wb.compute(rhf.result_handle(), frame);

// r.centers[i]  — Wannier center of occupied orbital i [Å]
// r.spreads[i]  — Boys spread Ω_i [Å²]
// r.total_spread, r.total_spread_initial, r.converged, r.sweeps
```

Throws `std::invalid_argument` for open-shell / GHF / not-ready results
(foothold refuses silently degraded paths).

## Pin

- **Unit:** `test_cpu_scf_wannier` — `boys_pair_angle` quadrants; one-AO
  centroid; two-AO spread non-increase; not-ready / open-shell throws.
- **E2E:** `reg_wannier-centers`
  - H₂ / STO-3G: single center at bond midpoint (±1e-4 Å).
  - H₂O / STO-3G: five centers; \(\Omega_\mathrm{tot}\) decreases; O-core +
    two OH-bond + two opposite-\(x\) lone pairs (rabbit ears); C₂ᵥ sum of
    \(x,y\) components ≈ 0.

See `regressions/scf/wannier-centers/REFERENCE.md`. CP2K `LOCALIZE` numerical
oracle is offline-only when a binary is available.
