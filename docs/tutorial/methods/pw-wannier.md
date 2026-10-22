# Plane-wave periodic maximally localized Wannier functions

Maximally localized Wannier functions (MLWF) on the CPU plane-wave method: a
`cpu::pw::PW` Kohn–Sham SCF, non-self-consistent occupied Bloch states on a
Monkhorst–Pack mesh, neighbor overlap matrices
\(M_{mn}^{(\mathbf k,\mathbf b)}\), projection initialization onto
bond-centered Gaussians, and Marzari–Vanderbilt (MV) steepest-descent polish.
`cpu::pw::PeriodicWannier::compute(method, frame)` owns the whole
SCF → NSCF → MV sequence and returns an owned `PeriodicWannierResult`
(centers, spreads, totals, SCF energy in Ry); parameters are members of
`PeriodicWannier`, and `diamond_bond_midpoints(a_ang)` is public pure geometry
for pins and tests. The sections cover restricted and collinear spin-polarized
(\(M=0\)) bulk silicon on the CPU backend. The molecular closed-shell
Foster–Boys path is a separate method,
[wannier-centers](scf-properties.md#wannier-centers).

Case: `ctest -R '^pw/wannier$'` — sources and oracle record in [`regressions/pw/wannier/`](../../../regressions/pw/wannier/). One section of the case per `##` heading below: `atv_regressions --case pw/wannier --section <name>`.

Roadmap: [dft.md](../../roadmap/dft.md) §11 MLWF. Requires `ATV_BUILD_PW=ON`
(CPU build environment `buildenv-gcc/2026.03-mpich`).

## Method

**Level.** Periodic plane-wave Kohn–Sham DFT from the same PW stack as
[pw-lda-si](pw-rks.md#pw-lda-si) (Hartree in \(G\)-space, LDA XC on the FFT
grid, Davidson diagonalization, NC UPF). After SCF, MLWF is
**non-self-consistent**: \(V_\mathrm{eff}\) is frozen; occupied Bloch states
\(\{u_{n\mathbf k}\}\) are recomputed on the MLWF k-mesh
(`PeriodicWannier::k_mesh`, independent of `PW::k_mesh`); unitary gauges
\(U(\mathbf k)\) mix the occupied manifold only.

**Neighbor overlaps.** For each \(\mathbf k\) and finite-difference neighbor
\(\mathbf b\) on the cubic six-neighbor shell,

\[
M_{mn}^{(\mathbf k,\mathbf b)}
  = \langle u_{m\mathbf k}\,|\,u_{n,\mathbf k+\mathbf b}\rangle .
\]

**Centers (Berry phase).** Diagonal phases of \(M\) give the Wannier centroids

\[
\mathbf r_n
  = -\frac1{N_k}\sum_{\mathbf k,\mathbf b}
    w_b\,\mathbf b\,\operatorname{Im}\ln M_{nn}^{(\mathbf k,\mathbf b)},
\]

with Souza–Marzari–Vanderbilt (SMV) weights \(w_b = 3/(N_b b^2)\)
(`mv_pair_weight_cubic`). Units: \(\mathbf r_n\) **[Å]**.

**Spreads.** Per-orbital SMV spreads
\(\Omega_n=\langle r^2\rangle_n-|\mathbf r_n|^2\) **[Å²]**; totals
\(\Omega=\sum_n\Omega_n\) before and after localization
(`total_spread_initial`, `total_spread`).

**Initialization.** Default projection onto \(n_\mathrm{wann}\) real-space
Gaussians of width `proj_alpha_ang` [Å] at the diamond Si–Si bond midpoints
(`diamond_bond_midpoints`, automatic for 2-atom Si-like cells with 4 WFs per
spin; optional `trial_centers`). Identity gauge when `use_projection = false`.

**MV polish.** Steepest descent on the occupied gauge:
\(U\leftarrow U\exp(-\alpha G)\) with step `step`, stop when
\(\|G\|_\infty < \texttt{tol}\) or `max_iter` is reached. The MV loop may leave
`converged == false` under the tutorial settings while still satisfying the
midpoint + spread gates; the regression warns and accepts.

**Literature geometry pin (parameter-free).** Diamond Si, 2-atom primitive
cell: \(n_e=8\) valence electrons → four occupied spatial Wannier functions
(per spin), one per tetrahedral bond. With atoms at \(\mathbf 0\) and
\((a/4)(1,1,1)\), bond midpoints are

\[
\mathbf m \in \frac a8\{(1,1,1),\,(1,-1,-1),\,(-1,1,-1),\,(-1,-1,1)\}
\]

(Marzari & Vanderbilt 1997 §IV; standard Wannier90 Si demo). The Si–Si bond
length is \(\sqrt3\,a/4 \approx 2.35\) Å at \(a=5.43\) Å. This geometric
statement is **basis-independent**: ONCV PW and GTH+Gaussian differ in spreads,
not in the bond-centering of the valence manifold of elemental Si.

**Observables.**

| Symbol | Definition | Unit |
| ------ | ---------- | ---- |
| \(\mathbf r_n\) | `PeriodicWannierResult::centers` | Å |
| \(\Omega_n\) | `PeriodicWannierResult::spreads` | Å² |
| \(\Omega\), \(\Omega_\mathrm{init}\) | totals after / before localization | Å² |
| \(d_\mathrm{max}\) | max min-image distance after optimal bipartite assignment to midpoints | Å |
| \(E_\mathrm{SCF}\) | `energy_ry` | Ry |
| \(n_\mathrm{wann}\) | occupied bands (= 4 per spin for Si) | — |

## Common setup

| Setting | Value |
| ------- | ----- |
| Structure | diamond Si, 2-atom fcc primitive |
| \(a\) | 5.43 Å |
| Basis atoms | \((0,0,0)\), \((a/4,a/4,a/4)\) |
| Cell | \((a/2)(0,1,1)\), \((a/2)(1,0,1)\), \((a/2)(1,1,0)\) |
| PP | `tests/data/pw/Si_lda_nc.upf` (in-tree), PseudoDojo ONCVPSP-LDA-SR-noCC v0.4; ONCVPSP 3.3.0 (D. R. Hamann), \(Z_\mathrm{val}=4\), no NLCC — same family as `pw-lda-si`, `bands-dos-si` |
| SCF \(E_\mathrm{cut}\) | 15 Ry (tutorial; ONCV production ~60 Ry) |
| SCF k-mesh | \(2\times 2\times 2\) (`PW::k_mesh = 2`) |
| MLWF k-mesh | \(2\times 2\times 2\) (`PeriodicWannier::k_mesh = 2`) |
| XC | `XCType::LSDA`, fixed occupations (`Smearing::none`) |
| SCF tol / `max_iter` | \(10^{-6}\) (density change) / 50 |
| Projection | bond midpoints, `proj_alpha_ang = 1.0` |
| MV SD | `max_iter = 200`, `tol = 1e-10`, `step = 0.15` |

```cpp
cpu::Frame frame = make_si_diamond(5.43);   // type 14, mass 28.0855; atom 1 at a/4.0 (1,1,1);
                                            // cell h = a/2.0: (0,h,h), (h,0,h), (h,h,0)

PW method{UpfPseudopotential{ATV_REG_DATA_DIR "/pw/Si_lda_nc.upf"}, /*ecut Ry*/ 15.0,
          /*k_mesh 2×2×2*/ 2, cpu::scf::XCType::LSDA};
method.tol = 1e-6;
method.max_iter = 50;

PeriodicWannier mlwf;
mlwf.k_mesh = 2;                   // MLWF MP mesh (independent of SCF)
mlwf.max_iter = 200;
mlwf.tol = 1e-10;
mlwf.step = 0.15;
mlwf.use_projection = true;
mlwf.proj_alpha_ang = 1.0;

const PeriodicWannierResult r = mlwf.compute(method, frame);
// r.centers[i] [Å], r.spreads[i] [Å²], r.total_spread, r.total_spread_initial,
// r.n_wann, r.energy_ry [Ry], r.scf_converged, r.converged (MV)
const auto mids = diamond_bond_midpoints(5.43);  // four tetrahedral midpoints [Å]
```

Headers: `cpu/planewave/periodic_wannier.h`, `cpu/planewave/pw_solver.h`;
implementation `src/cpu/planewave/periodic_wannier.cpp` (SCF → freeze
\(V_\mathrm{eff}\) → NSCF occupied manifold → \(M^{(\mathbf k,\mathbf b)}\) →
projection + MV SD → centers/spreads). No unit test in the tree covers the
diamond midpoints, cubic SMV weight, `center_from_diagonal_phases`, the spin
gate or the invalid-argument throws on `compute`.

**Coarse-mesh rationale.** Ecut and MLWF k are deliberately light (same class
as `pw-lda-si`). The **0.90 Å** midpoint window is a large fraction of the
2.35 Å Si–Si bond but still rejects random / delocalized centers; sub-0.15 Å
absolute placement needs a denser k-mesh / higher ecut. The **spread-ratio**
gate ensures localization actually runs (\(\Omega\) drops), independent of the
absolute \(\Omega\), which depends on PP/basis.

## Oracle

The gates are the **literature bond-midpoint geometry** plus spread reduction,
hard-coded in the section sources; nothing runs CP2K or Wannier90 at ctest time.
CP2K Quickstep `&LOCALIZE` (`METHOD JACOBI`, `MAX_ITER 200`) / print Wannier
centers runs on **GTH-PADE + DZVP-MOLOPT**, not ONCV plane waves, so absolute
spreads and gauge are **not** comparable; bond-midpoint geometry of the four
valence MLWFs is the shared scientific content. The CP2K input (same geometry)
is an offline wall / character skeleton only.

**Performance (both sections): `perf_exempt`** — no CP2K binary / module on the
build host for a fair wall twin, and the Quickstep GTH path is a different
Hamiltonian from ONCV PW, so \(T_\mathrm{CP2K}\), the ratio and the ≤10× check
are n/a. When a CP2K binary is available offline, time the same geometry,
record \(r = T_\mathrm{ATV}/T_\mathrm{CP2K}\) in REFERENCE.md and require
\(r\le 10\) (same-geometry wall only, not absolute-spread parity).

## Limitations (both sections)

- Tutorial ecut / k only — not production-converged MLWF spreads or dense-mesh
  center placement; the coarse \(2\times2\times2\) SMV discretization limits
  absolute center accuracy to \(\lesssim 0.9\) Å (bond association, not
  crystallography-grade centroids).
- Occupied-only insulator; no Wannier disentanglement (Marzari–Souza–Vanderbilt
  outer window), no metallic smearing / entangled metals.
- No GPW-grid / Gaussian MLWF; no Pipek–Mezey / Edmiston–Ruedenberg criteria;
  no hybrid / meta-GGA XC; no noncollinear / SOC.
- Absolute \(\Omega\) and total energy are not comparable to CP2K Quickstep GTH
  decks.
- Empty `upf_path` selects a toy \(V_\mathrm{loc}\) (unphysical); the sections
  always use the in-tree ONCV Si UPF.
- The CUDA PW surface is not covered.

## References

- N. Marzari and D. Vanderbilt, *Phys. Rev. B* **56**, 12847 (1997)
- N. Marzari et al., *Rev. Mod. Phys.* **84**, 1419 (2012)
- D. R. Hamann, *Phys. Rev. B* **88**, 085117 (2013)
- M. J. van Setten et al., *Comput. Phys. Commun.* **226**, 39 (2018)
- Related: [wannier-centers](scf-properties.md#wannier-centers) (molecular
  Boys), [pw-lda-si](pw-rks.md#pw-lda-si),
  [bands-dos-si](pw-bands-dos.md#bands-dos-si),
  [pw-uks-kmesh-fe](pw-collinear-spin.md#pw-uks-kmesh-fe) (multi-k collinear
  UKS energy), [Chapter 8 — Plane-Wave DFT](../ch08-plane-wave.md)

## periodic-wannier-si — Restricted multi-k MLWF of bulk Si {#periodic-wannier-si}

Restricted LDA SCF (closed-shell, fixed occupations) followed by MV
localization of the single occupied manifold: four valence MLWFs.

### Accuracy (Release, OMP=4)

| Quantity | Atomiverse / pin | Reference | Tol / band | Notes |
| -------- | ---------------: | --------: | ---------- | ----- |
| \(n_\mathrm{wann}\) | **4** | 4 valence MLWFs | exact | closed-shell Si |
| SCF | `scf_converged == true` | — | — | preceding PW SCF |
| \(d_\mathrm{max}\) → midpoints | **&lt; 0.90 Å** | bond midpoints | hard gate | bipartite min-image assignment |
| Measured \(d_\mathrm{max}\) (informative) | **≈ 0.83 Å** | 0 | &lt; 0.90 | |
| \(\Omega/\Omega_\mathrm{init}\) | **&lt; 0.25** | localization decreases \(\Omega\) | hard gate | MV / Boys definition |
| Measured ratio (informative) | **≈ 0.081** | — | &lt; 0.25 | \(\Omega_\mathrm{init}\approx 37.7\), \(\Omega\approx 3.05\) Å² |
| Spreads | finite | no NaN | strict | every orbital |
| SCF energy (informative) | ≈ −15.672 Ry | same class as `pw-lda-si` | not a gate | incomplete ecut/k |

### Performance

ATV wall (informative): ~8 s (Release, OMP=4, ecut 15 / k=2 SCF+MLWF), inside
the ≤ 60 s single-rank CPU budget. `perf_exempt` (see [Oracle](#oracle)).

### Limitations

Restricted closed shell only — collinear spin is the
[periodic-wannier-uks](#periodic-wannier-uks) section.

## periodic-wannier-uks — Collinear spin-polarized multi-k MLWF of bulk Si {#periodic-wannier-uks}

The spin version of [periodic-wannier-si](#periodic-wannier-si): `PW` with
`spin_polarized = true` and `multiplicity = 1` (closed-shell collinear
\(M = N_\uparrow - N_\downarrow = 0\)) feeds `PeriodicWannier::compute`; the
occupied α and β manifolds are localized **independently** on frozen
\(V_{\mathrm{eff},\sigma}\) after a multi-k LSDA SCF.

### Method

1. **SCF.** Collinear UKS fixed occupations (same spin channel as
   [pw-spin-polarized-o2](pw-collinear-spin.md#pw-spin-polarized-o2)):
   densities \(\rho_\uparrow\), \(\rho_\downarrow\); Hartree from
   \(\rho = \rho_\uparrow + \rho_\downarrow\); spin XC
   \(V_{\mathrm{xc},\sigma}(\rho_\uparrow,\rho_\downarrow)\); dual Davidson per
   \(\mathbf{k}\) and spin.
2. **NSCF / overlaps / MV.** Per spin: occupied
   \(u_{n\mathbf{k}\sigma}\) on the MLWF mesh,
   \(M_{mn}^{(\mathbf{k},\mathbf{b},\sigma)}
     = \langle u_{m\mathbf{k}\sigma}\mid u_{n,\mathbf{k}+\mathbf{b},\sigma}\rangle\),
   projection, MV steepest descent, Berry-phase centers and SMV spreads as in
   [Method](#method).

**Spin / occupations.**

\[
N_\uparrow = N_\downarrow = 4,\qquad
M = N_\uparrow - N_\downarrow = 0
\]

from multiplicity \(2S+1 = 1\) and 8 valence electrons (2×Si ONCV
\(Z_{\mathrm{val}} = 4\)). Each spin carries the same four-bond valence manifold
as the restricted pin; **bond-centering per spin at \(M=0\)** is the shared
content with the CP2K Quickstep `LSD` / `MULTIPLICITY 1` + `&LOCALIZE` (JACOBI)
input (GTH-PADE-q4 + DZVP-MOLOPT-SR-GTH).

Additional result fields: `n_wann_up` / `n_wann_dn`, `spin[i]` (0 = ↑, 1 = ↓);
`centers` / `spreads` list α first, then β.

```cpp
method.spin_polarized = true;
method.multiplicity = 1;           // M = 0 closed-shell collinear
const PeriodicWannierResult r = mlwf.compute(method, frame);
// r.n_wann_up == r.n_wann_dn == 4; r.n_wann == 8; r.spin[i] ∈ {0,1}
```

### Accuracy

| Quantity | Atomiverse / pin | Reference | Tol | Notes |
| -------- | ---------------: | --------: | --- | ----- |
| \(n_\uparrow\) / \(n_\downarrow\) | **4** / **4** | 4 valence per spin at \(M=0\) | exact | `n_wann_up` / `n_wann_dn` |
| \(n_{\mathrm{wann}}\) | **8** | \(n_\uparrow+n_\downarrow\) | exact | `n_wann` |
| `spin[i]` | \(\{0,1\}\) size 8 | α then β layout | exact | labels only |
| SCF | must converge | — | — | fails if `!scf_converged` |
| \(d_{\max}\rightarrow\mathbf{m}\) spin ↑ [Å] | \(< 0.90\) | bond midpoints \(\mathbf{m}\) | 0.90 Å | min-image + optimal bipartite assignment |
| \(d_{\max}\rightarrow\mathbf{m}\) spin ↓ [Å] | \(< 0.90\) | same \(\mathbf{m}\) | 0.90 Å | independent β manifold |
| \(\Omega_{\mathrm{final}}/\Omega_{\mathrm{init}}\) | \(< 0.25\) | localization decreases total spread | 0.25 | sum over **both** spins |
| \(\Omega_{\mathrm{init}}\) | \(> 0\) | — | — | identity-gauge baseline |
| Spreads | finite (no NaN) | — | — | both channels |

### Performance

ATV wall (informative): ~2× the restricted Si MLWF (two spin channels; ~order
10 s against the restricted ~8 s at ecut 15 / k=2). `perf_exempt` (see
[Oracle](#oracle)).

### Limitations

- **Closed-shell collinear only (\(M=0\)).** Open-shell spin-polarized MLWF
  (\(N_\uparrow\neq N_\downarrow\)) is not pinned.
- Berry polarization is a separate method
  ([berry-spin-si](pw-berry-phase.md#berry-spin-si)); DFPT phonons, CUDA PW and
  hybrids are out of scope.
