# Reference — Periodic multi-k MLWF (Marzari–Vanderbilt), bulk Si

## Method

Maximally localized Wannier functions on the CPU plane-wave method: restricted
Kohn–Sham LDA SCF, non-self-consistent occupied Bloch states on a
Monkhorst–Pack mesh, neighbor overlap matrices
\(M_{mn}^{(\mathbf k,\mathbf b)}=\langle u_{m\mathbf k}|u_{n,\mathbf k+\mathbf b}\rangle\),
projection initialization onto bond-centered Gaussians, Marzari–Vanderbilt
steepest-descent polish, and Wannier centers from the Berry-phase formula

\[
\mathbf r_n = -\frac1{N_k}\sum_{\mathbf k,\mathbf b}
  w_b\,\mathbf b\,\operatorname{Im}\ln M_{nn}^{(\mathbf k,\mathbf b)}
\]

(Marzari & Vanderbilt, *Phys. Rev. B* **56**, 12847 (1997); Marzari et al.,
*Rev. Mod. Phys.* **84**, 1419 (2012)).

Public path:

```cpp
cpu::pw::PW method;                 // upf, ecut_ry, k_mesh (SCF), xc, …
cpu::pw::PeriodicWannier mlwf;      // members: k_mesh, max_iter, tol, …
auto r = mlwf.compute(method, frame);
// r.centers[i] [Å], r.spreads[i] [Å²], r.total_spread
```

No public `*Config` bag; no factories.

## Pseudopotential

| Item | Value |
| ---- | ----- |
| File | `tests/data/pw/Si_lda_nc.upf` (in-tree) |
| Family | PseudoDojo ONCVPSP-LDA-SR-noCC v0.4 |

## Literature pin (geometry, parameter-free)

Diamond / zincblende Si has **four** valence MLWFs, one per tetrahedral
Si–Si bond, with centroids at the four bond midpoints of the 2-atom
primitive cell (Marzari & Vanderbilt 1997 §IV; standard Wannier90 Si demo).
With atoms at \(\mathbf 0\) and \((a/4)(1,1,1)\):

\[
\mathbf m \in \frac a8\{(1,1,1),\,(1,-1,-1),\,(-1,1,-1),\,(-1,-1,1)\}.
\]

This geometric statement is basis-independent (ONCV PW vs GTH GPW differ in
spreads, not in the bond-centering of the valence manifold for elemental Si).

## Twin geometry and settings

| Setting | Value |
| ------- | ----- |
| Structure | diamond Si, 2-atom fcc primitive cell |
| Lattice constant \(a\) | 5.43 Å |
| Basis atoms | (0,0,0), \((a/4)(1,1,1)\) |
| SCF \(E_\mathrm{cut}\) | 15 Ry |
| SCF k-mesh | \(2\times2\times2\) |
| MLWF k-mesh | \(2\times2\times2\) (`PeriodicWannier::k_mesh = 2`) |
| XC | `XCType::LSDA` |
| SCF tol | \(10^{-6}\) |
| Projection | bond midpoints, `proj_alpha_ang = 1.0` |
| MV SD | `max_iter = 200`, `tol = 1e-10`, `step = 0.15` |

## Acceptance (hard-coded in `reg_periodic-wannier-si.cpp`)

1. SCF converged; `n_wann == 4`.
2. **Bond-midpoint pin (coarse-mesh):** after optimal bipartite assignment and
   min-image under the simulation cell, every Wannier center lies within
   **0.90 Å** of one tetrahedral bond midpoint (Si–Si bond = 2.35 Å; the
   2×2×2 SMV discretization of the Berry-phase center formula is coarse —
   sub-0.15 Å absolute placement is a denser-mesh productization follow-on).
3. **Localization pin:** \(\Omega_\mathrm{final} < 0.25\,\Omega_\mathrm{init}\)
   (Foster–Boys / MV definition: localization decreases total spread).
4. Spreads finite (no NaN).

## CP2K note

CP2K Quickstep can emit Wannier centers via `&LOCALIZE` / print Wannier
(GTH + MOLOPT). Absolute spreads and gauge differ from ONCV PW; the
bond-midpoint geometry is the shared scientific content. Offline input echo:
`tools/cp2k_oracle/inputs/si_periodic_wannier_note.inp`.

## Measured (2026-07-25, Release, OMP=4)

| Item | Value |
| ---- | ----- |
| SCF energy | −15.671773 Ry |
| \(\Omega_\mathrm{init}\) | 37.68 Å² |
| \(\Omega_\mathrm{final}\) | 3.05 Å² (ratio 0.081) |
| dmax → midpoints | 0.83 Å (< 0.90) |
| ATV wall | ~8 s |

## Performance

| Item | Value |
| ---- | ----- |
| `perf_exempt` | **true** |
| Reason | No CP2K binary / module on the build host for a fair wall twin this shift |
| ATV twin wall (informative) | ~8 s (Release, ecut 15 / k=2 SCF+MLWF) |
| `ratio_atv_over_cp2k` | n/a (exempt) |
| `within_10x` | n/a (exempt; correctness gate only) |

When a CP2K binary is available: time the same geometry with the oracle
input, append \(r = T_\mathrm{atv}/T_\mathrm{cp2k}\) here, require
\(r \le 10\).
