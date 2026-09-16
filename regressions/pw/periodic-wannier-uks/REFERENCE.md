# Reference — Collinear spin-polarized multi-k MLWF (Marzari–Vanderbilt), bulk Si

## Method

Maximally localized Wannier functions on the CPU plane-wave method with
**collinear UKS** fixed occupations: spin densities \(\rho_\uparrow,\rho_\downarrow\),
spin XC \(V_{\mathrm{xc},\sigma}\), frozen \(V_{\mathrm{eff},\sigma}\), NSCF
Bloch states on a Monkhorst–Pack mesh **per spin**, neighbor overlaps
\(M_{mn}^{(\mathbf k,\mathbf b)}\), projection initialization onto
bond-centered Gaussians, Marzari–Vanderbilt steepest-descent polish, and
Wannier centers from the Berry-phase formula (independent α / β manifolds).

(Marzari & Vanderbilt, *Phys. Rev. B* **56**, 12847 (1997); Marzari et al.,
*Rev. Mod. Phys.* **84**, 1419 (2012)).

Public path:

```cpp
cpu::pw::PW method;
method.spin_polarized = true;
method.multiplicity = 1;          // M = N↑ − N↓ = 0
// upf, ecut_ry, k_mesh (SCF), xc, …
cpu::pw::PeriodicWannier mlwf;    // members: k_mesh, max_iter, tol, …
auto r = mlwf.compute(method, frame);
// r.n_wann_up / r.n_wann_dn; r.centers [Å] = α then β; r.spin[i] ∈ {0,1}
```

No public `*Config` bag; no factories.

## Pseudopotential

| Item | Value |
| ---- | ----- |
| File | `tests/data/pw/Si_lda_nc.upf` (in-tree) |
| Family | PseudoDojo ONCVPSP-LDA-SR-noCC v0.4 |

## Literature pin (geometry, parameter-free)

Diamond Si valence MLWFs sit on the four tetrahedral Si–Si bond midpoints
(Marzari & Vanderbilt 1997 §IV). With closed-shell collinear occupations
\(N_\uparrow=N_\downarrow=4\) (multiplicity 1, 8 valence electrons from 2×Si
\(Z_\mathrm{val}=4\)), **each spin channel** has four bond-centered MLWFs.
With atoms at \(\mathbf 0\) and \((a/4)(1,1,1)\):

\[
\mathbf m \in \frac a8\{(1,1,1),\,(1,-1,-1),\,(-1,1,-1),\,(-1,-1,1)\}.
\]

This geometric statement is basis-independent for elemental Si valence
manifolds (ONCV PW vs GTH GPW differ in spreads, not bond-centering).

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
| Spin | `spin_polarized = true`, `multiplicity = 1` (\(M=0\)) |
| SCF tol | \(10^{-6}\) |
| Projection | bond midpoints, `proj_alpha_ang = 1.0` |
| MV SD | `max_iter = 200`, `tol = 1e-10`, `step = 0.15` |

## Acceptance (hard-coded in `reg_periodic-wannier-uks.cpp`)

1. SCF converged; `n_wann_up == n_wann_dn == 4`, `n_wann == 8`.
2. **Bond-midpoint pin (per spin, coarse mesh):** after optimal bipartite
   assignment and min-image, every Wannier center of each spin lies within
   **0.90 Å** of one tetrahedral bond midpoint (same SMV discretization
   window as `reg_periodic-wannier-si`).
3. **Localization pin:** \(\Omega_\mathrm{final} < 0.25\,\Omega_\mathrm{init}\)
   (sum over both spins).
4. Spreads finite (no NaN); `spin` labels size 8 with values in \(\{0,1\}\).

## CP2K note

CP2K Quickstep: collinear `LSD` / UKS + `&LOCALIZE` / print Wannier
(GTH + MOLOPT). Absolute spreads and gauge differ from ONCV PW; the
bond-midpoint geometry per spin channel at \(M=0\) is the shared scientific
content. Offline input echo:
`tools/cp2k_oracle/inputs/si_periodic_wannier_uks.note`.

## Performance

| Item | Value |
| ---- | ----- |
| `perf_exempt` | **true** |
| Reason | No fair CP2K PW/NC-UPF wall twin on the build host this shift (Quickstep GTH path is not the same Hamiltonian) |
| ATV twin wall (informative) | ~2× restricted Si MLWF (two spin channels) |
| `ratio_atv_over_cp2k` | n/a (exempt) |
| `within_10x` | n/a (exempt; correctness gate only) |

When a CP2K binary is available: time the same geometry with the oracle
input, append \(r = T_\mathrm{atv}/T_\mathrm{cp2k}\) here, require
\(r \le 10\).
