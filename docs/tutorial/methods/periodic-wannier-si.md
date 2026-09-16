# Periodic multi-k MLWF — bulk silicon (Marzari–Vanderbilt)

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `periodic-wannier-si` |
| Regression | `regressions/pw/periodic-wannier-si/` |
| Public API | `cpu::pw::PW` + `cpu::pw::PeriodicWannier` / `PeriodicWannierResult` (`cpu/planewave/periodic_wannier.h`) |
| Roadmap | [dft.md](../../roadmap/dft.md) §11 MLWF |
| Dual gate | Bond-midpoint geometry + spread reduction (REFERENCE.md); speed **`perf_exempt`** |
| Wave | CP2K capability loop — harness wave 2026-07-25 |

## Summary

**Maximally localized Wannier functions (MLWF)** on the CPU **plane-wave**
method for bulk silicon: restricted LDA SCF, non-self-consistent occupied
Bloch states on a Monkhorst–Pack mesh, neighbor overlap matrices
\(M_{mn}^{(\mathbf k,\mathbf b)}\), projection initialization onto
bond-centered Gaussians, and Marzari–Vanderbilt (MV) steepest-descent polish.
Centers come from the Berry-phase formula; spreads use the Souza–Marzari–Vanderbilt
(SMV) discretization.

This is the **periodic multi-k** product path (`cpu::pw::PeriodicWannier`).
The molecular closed-shell Foster–Boys foothold is a separate slug
([wannier-centers](wannier-centers.md)). Product regression:
`regressions/pw/periodic-wannier-si` (`reg_periodic-wannier-si`). Scientific
gate is **literature bond-midpoint geometry** for diamond Si valence MLWFs
(Marzari & Vanderbilt 1997 §IV), not absolute-spread parity with CP2K
Quickstep GTH+Gaussian. Performance: **`perf_exempt`** (no fair CP2K wall twin
on the pin host).

## Method

**Level.** Periodic plane-wave Kohn–Sham DFT. Self-consistent density and
effective potential from the same PW stack as `pw-lda-si` (Hartree in
\(G\)-space, LDA XC on the FFT grid, Davidson diagonalization, NC UPF). After
SCF, MLWF is **non-self-consistent**: \(V_\mathrm{eff}\) is frozen; occupied
Bloch states \(\{u_{n\mathbf k}\}\) are recomputed on the MLWF k-mesh; unitary
gauges \(U(\mathbf k)\) mix the occupied manifold only.

**Neighbor overlaps.** For each \(\mathbf k\) and finite-difference neighbor
\(\mathbf b\) (cubic six-neighbor shell in this foothold),

\[
M_{mn}^{(\mathbf k,\mathbf b)}
  = \langle u_{m\mathbf k}\,|\,u_{n,\mathbf k+\mathbf b}\rangle.
\]

**Centers (Berry phase).** Diagonal phases of \(M\) give Wannier centroids
(Marzari–Vanderbilt):

\[
\mathbf r_n
  = -\frac1{N_k}\sum_{\mathbf k,\mathbf b}
    w_b\,\mathbf b\,\operatorname{Im}\ln M_{nn}^{(\mathbf k,\mathbf b)},
\]

with SMV weights \(w_b\) (`mv_pair_weight_cubic` for the cubic shell). Units:
\(\mathbf r_n\) **[Å]**.

**Spreads.** Per-orbital SMV spreads \(\Omega_n=\langle r^2\rangle_n-|\mathbf r_n|^2\)
**[Å²]**; totals \(\Omega=\sum_n\Omega_n\) before and after localization
(`total_spread_initial`, `total_spread`).

**Initialization.** Default projection onto \(n_\mathrm{wann}\) real-space
Gaussians of width `proj_alpha_ang` at diamond Si–Si bond midpoints
(`diamond_bond_midpoints`; optional `trial_centers`). Optional identity
gauge when `use_projection = false`.

**MV polish.** Steepest descent on the occupied gauge:
\(U\leftarrow U\exp(-\alpha G)\) with step `step`, stop when
\(\|G\|_\infty < \texttt{tol}\) or `max_iter` reached.

**XC / spin / occupations.** `XCType::LSDA` on the PW grid; restricted fixed
occupations (`Smearing::none`). Closed-shell insulator only — no
disentanglement, no metallic / spin-polarized path on this slug.

**Si valence manifold.** Diamond Si, 2-atom primitive cell: \(n_e=8\) valence
electrons → \(n_\mathrm{wann}=4\) occupied spatial Wannier functions, one per
tetrahedral bond.

**Literature geometry pin (parameter-free).** With atoms at \(\mathbf 0\) and
\((a/4)(1,1,1)\), bond midpoints are

\[
\mathbf m \in \frac a8\{(1,1,1),\,(1,-1,-1),\,(-1,1,-1),\,(-1,-1,1)\}
\]

(Marzari & Vanderbilt, *Phys. Rev. B* **56**, 12847 (1997) §IV; standard
Wannier90 Si demo). This geometric statement is **basis-independent**: ONCV PW
vs GTH+Gaussian differ in spreads, not in bond-centering of the valence
manifold for elemental Si.

**Observables.**

| Symbol | Definition | Unit |
| ------ | ---------- | ---- |
| \(\mathbf r_n\) | `PeriodicWannierResult::centers` | Å |
| \(\Omega_n\) | `PeriodicWannierResult::spreads` | Å² |
| \(\Omega\), \(\Omega_\mathrm{init}\) | totals after / before localization | Å² |
| \(d_\mathrm{max}\) | max min-image distance after optimal bipartite assignment to midpoints | Å |
| \(E_\mathrm{SCF}\) | `energy_ry` | Ry |
| \(n_\mathrm{wann}\) | occupied bands (= 4 for Si) | — |

**Key exclusions (this slug).**

- Molecular Foster–Boys → [wannier-centers](wannier-centers.md).
- GPW-grid / Gaussian MLWF (unqueued residual).
- Pipek–Mezey / Edmiston–Ruedenberg criteria; disentanglement; entangled
  metals.
- Sub-0.15 Å absolute center placement (needs denser k / higher ecut —
  productization follow-on; twin uses coarse \(2\times2\times2\) SMV).
- Absolute \(\Omega\) parity with CP2K Quickstep GTH+MOLOPT.
- Live CP2K / Wannier90 at ctest time (offline pins only).
- Public `*Config` bags; `make_*` / `build_*` factories.
- CUDA PW surface as a CPU compatibility API for this pin.

## Implementation

Public surface only (`cpu::Frame` + concrete method / post-processor members;
no `*Config`, no factories). Headers:
`include/atomiverse/cpu/planewave/periodic_wannier.h`,
`pw_solver.h`. Implementation:
`src/cpu/planewave/periodic_wannier.cpp` (SCF → freeze \(V_\mathrm{eff}\) →
NSCF occupied manifold → \(M^{(\mathbf k,\mathbf b)}\) → projection + MV SD →
centers/spreads). Unit pure-math tests (no full SCF e2e):
`tests/cpu/planewave/test_periodic_wannier.cpp`.

Minimal sequence (matches `reg_periodic-wannier-si.cpp`):

```cpp
#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/planewave/periodic_wannier.h"
#include "atomiverse/cpu/planewave/pw_solver.h"
#include "atomiverse/cpu/scf/xc_functional.h"

using namespace ATV;
using namespace ATV::cpu::pw;

constexpr double kA = 5.43;       // Å
constexpr double kEcutRy = 15.0;  // tutorial incomplete basis
constexpr int kMeshScf = 2;       // 2×2×2 MP SCF
constexpr int kMeshMlwf = 2;      // 2×2×2 MLWF mesh

cpu::Frame frame(2);
frame.span(kAtom_type)[0] = 14;
frame.span(kAtom_type)[1] = 14;
frame.span(kAtom_x)[0] = 0.0;
frame.span(kAtom_y)[0] = 0.0;
frame.span(kAtom_z)[0] = 0.0;
frame.span(kAtom_x)[1] = kA / 4.0;
frame.span(kAtom_y)[1] = kA / 4.0;
frame.span(kAtom_z)[1] = kA / 4.0;

const double h = kA / 2.0;
Mat3 cell;
// fcc primitive: (0,h,h), (h,0,h), (h,h,0)
cell(0, 0) = 0; cell(0, 1) = h; cell(0, 2) = h;
cell(1, 0) = h; cell(1, 1) = 0; cell(1, 2) = h;
cell(2, 0) = h; cell(2, 1) = h; cell(2, 2) = 0;
frame.box.set_cell(cell);

PW method;
method.upf_path = "…/tests/data/pw/Si_lda_nc.upf";
method.ecut_ry = kEcutRy;
method.k_mesh = kMeshScf;
method.tol = 1e-6;
method.max_iter = 50;
method.xc = cpu::scf::XCType::LSDA;

PeriodicWannier mlwf;
mlwf.k_mesh = kMeshMlwf;
mlwf.max_iter = 200;
mlwf.tol = 1e-10;
mlwf.step = 0.15;
mlwf.use_projection = true;
mlwf.proj_alpha_ang = 1.0;

const PeriodicWannierResult r = mlwf.compute(method, frame);
// r.centers[i] [Å], r.spreads[i] [Å²]
// r.total_spread, r.total_spread_initial, r.n_wann, r.scf_converged
// diamond_bond_midpoints(kA) — four tetrahedral midpoints [Å]
```

Build and run the regression (`ATV_BUILD_PW=ON`, `ATV_BUILD_REGRESSIONS=ON`):

```bash
bash -lc '
  module load buildenv-gcc/2026.03-mpich
  export OPENBLAS_ROOT="$OPENBLASDIR"
  export RUSTUP_TOOLCHAIN=stable-x86_64-unknown-linux-gnu
  export LD_LIBRARY_PATH="$LIBRARY_PATH:$LD_LIBRARY_PATH"
  cmake -B build -DCMAKE_BUILD_TYPE=Release \
        -DATV_BUILD_TESTS=ON -DATV_BUILD_HF=ON -DATV_BUILD_PW=ON \
        -DATV_BUILD_REGRESSIONS=ON -DATV_BUILD_CUDA=OFF \
        -DBLA_VENDOR=OpenBLAS
  cmake --build build -j$(nproc) --target atomiverse test_cpu_pw_periodic_wannier \
        reg_periodic-wannier-si
  ctest --test-dir build -R "^test_cpu_pw_periodic_wannier$" --output-on-failure
  cmake --build build --target run_reg_periodic-wannier-si
'
```

Unit (no full SCF e2e): `test_cpu_pw_periodic_wannier` — diamond midpoints,
cubic SMV weight, `center_from_diagonal_phases`, invalid-arg throws on
`compute`.

## Validation (vs literature / CP2K note)

Source: `regressions/pw/periodic-wannier-si/REFERENCE.md` (offline pin; never
live third party in ctest). Queue / handoff: done, **`perf_exempt`**.

### Pseudopotential

| Item | Value |
| ---- | ----- |
| File | `tests/data/pw/Si_lda_nc.upf` (in-tree) |
| Family | PseudoDojo ONCVPSP-LDA-SR-noCC v0.4 |
| Generator | ONCVPSP 3.3.0 (D. R. Hamann), \(Z_\mathrm{val}=4\), no NLCC |
| Cite | Hamann, *Phys. Rev. B* **88**, 085117 (2013); van Setten et al., *Comput. Phys. Commun.* **226**, 39 (2018) |
| Same family as | `regressions/pw/pw-lda-si`, `bands-dos-si` |

### Literature pin (geometry)

Bulk diamond Si has **four** valence MLWFs with centroids on the four
Si–Si bond midpoints of the 2-atom primitive cell.

| Quantity | Value | Source |
| -------- | ----- | ------ |
| Valence MLWF count | 4 | closed-shell Si, \(n_e=8\) |
| Center locations | tetrahedral bond midpoints \(\mathbf m\) | Marzari & Vanderbilt, PRB **56**, 12847 (1997) §IV |
| Si–Si bond length | \(\sqrt3\,a/4 \approx 2.35\) Å at \(a=5.43\) Å | diamond geometry |

### Twin geometry and settings (regression)

| Setting | Value |
| ------- | ----- |
| Structure | diamond Si, 2-atom fcc primitive |
| \(a\) | 5.43 Å |
| Basis atoms | \((0,0,0)\), \((a/4,a/4,a/4)\) |
| Cell | \((a/2)(0,1,1)\), \((a/2)(1,0,1)\), \((a/2)(1,1,0)\) |
| SCF \(E_\mathrm{cut}\) | 15 Ry (tutorial; ONCV production ~60 Ry) |
| SCF k-mesh | \(2\times 2\times 2\) (`PW::k_mesh = 2`) |
| MLWF k-mesh | \(2\times 2\times 2\) (`PeriodicWannier::k_mesh = 2`) |
| XC | `XCType::LSDA` |
| SCF tol | \(10^{-6}\) (density change) |
| `max_iter` (SCF) | 50 |
| Projection | bond midpoints, `proj_alpha_ang = 1.0` |
| MV SD | `max_iter = 200`, `tol = 1e-10`, `step = 0.15` |

### Accuracy table

| Quantity | Atomiverse / pin | External | Tol / band | Notes |
| -------- | ---------------: | -------: | ---------- | ----- |
| \(n_\mathrm{wann}\) | **4** | 4 valence MLWFs | exact | closed-shell Si |
| SCF | `scf_converged == true` | — | — | preceding PW SCF |
| \(d_\mathrm{max}\) → midpoints | **&lt; 0.90 Å** | bond midpoints | hard gate | bipartite min-image assignment; coarse \(2\times2\times2\) SMV |
| Measured \(d_\mathrm{max}\) (informative) | **≈ 0.83 Å** | 0 | &lt; 0.90 | REFERENCE host 2026-07-25, Release OMP=4 |
| \(\Omega/\Omega_\mathrm{init}\) | **&lt; 0.25** | localization decreases \(\Omega\) | hard gate | MV / Boys definition |
| Measured ratio (informative) | **≈ 0.081** | — | &lt; 0.25 | \(\Omega_\mathrm{init}\approx 37.7\), \(\Omega\approx 3.05\) Å² |
| Spreads | finite | no NaN | strict | every orbital |
| SCF energy (informative) | ≈ −15.672 Ry | same twin class as `pw-lda-si` | not a gate | incomplete ecut/k |

**Coarse-mesh rationale.** Twin ecut and MLWF k are deliberately light (same
class as `pw-lda-si`). The **0.90 Å** midpoint window is a large fraction of
the 2.35 Å Si–Si bond but still rejects random / delocalized centers; sub-0.15
Å absolute placement is a denser-mesh productization follow-on, not this
foothold. The **spread-ratio** gate ensures localization actually runs
(\(\Omega\) drops), independent of absolute \(\Omega\) which depends on
PP/basis.

**CP2K.** Quickstep `&LOCALIZE` / print Wannier centers use **GTH-PADE +
DZVP-MOLOPT**, not ONCV plane waves. Absolute spreads and gauge are **not**
comparable. Bond-midpoint geometry of the four valence MLWFs is the shared
scientific content. Optional offline wall / character skeleton:
`tools/cp2k_oracle/inputs/si_periodic_wannier_note.inp` (same geometry;
`METHOD JACOBI`, `MAX_ITER 200`).

## Performance

Source: `regressions/pw/periodic-wannier-si/REFERENCE.md`. Queue / handoff row:
**`perf_exempt`** (literature midpoint + spread gates; no fair GTH twin wall on
pin host).

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | ------------------------: | ----------------------: | ----: | ---- | ------ |
| Si MLWF ecut=15 / scf k=2 + MLWF k=2 | n/a | ≈ 8 (informative) | n/a | n/a (**exempt**) | No CP2K module on pin host; absolute \(\Omega\) not a CP2K twin |

| Item | Value |
| ---- | ----- |
| `perf_exempt` | **true** |
| Reason | No CP2K binary / module on the build host for a fair wall twin this shift |
| ATV twin wall (informative) | ~8 s (Release, OMP=4, ecut 15 / k=2 SCF+MLWF) |
| `ratio_atv_over_cp2k` | n/a (exempt) |
| `within_10x` | n/a (exempt; correctness gate only) |
| Wall budget | e2e ≤ 60 s single-rank CPU (PASS) |

When a CP2K binary is available offline: time the same geometry with
`tools/cp2k_oracle/inputs/si_periodic_wannier_note.inp`, append
\(r = T_\mathrm{ATV}/T_\mathrm{CP2K}\) to REFERENCE.md, require \(r\le 10\).
Wall comparison is same-geometry only, not absolute-spread parity
(GTH+Gaussian vs NC-UPF PW).

## Limitations

- Tutorial ecut / k only — not production-converged MLWF spreads or dense-mesh
  center placement.
- Coarse \(2\times2\times2\) SMV discretization limits absolute center
  accuracy to \(\lesssim 0.9\) Å (bond association, not crystallography-grade
  centroids).
- Occupied-only, closed-shell insulator; no disentanglement, no metallic
  smearing, no collinear spin.
- No GPW-grid MLWF on this slug (unqueued residual after molecular Boys +
  periodic PW MV).
- Absolute \(\Omega\) not comparable to CP2K Quickstep GTH decks.
- Empty `upf_path` selects a toy \(V_\mathrm{loc}\) (unphysical); this
  regression always uses the in-tree ONCV Si UPF.
- CUDA PW surface is not a CPU compatibility API for this pin.
- MV loop may leave `converged == false` under tutorial settings while still
  satisfying midpoint + spread gates (regression warns, accepts foothold).

## References

- Offline pin: [`regressions/pw/periodic-wannier-si/REFERENCE.md`](../../../regressions/pw/periodic-wannier-si/REFERENCE.md)
- Regression: [`regressions/pw/periodic-wannier-si/reg_periodic-wannier-si.cpp`](../../../regressions/pw/periodic-wannier-si/reg_periodic-wannier-si.cpp)
- Public API: [`include/atomiverse/cpu/planewave/periodic_wannier.h`](../../../include/atomiverse/cpu/planewave/periodic_wannier.h)
- Optional CP2K wall / character skeleton: `tools/cp2k_oracle/inputs/si_periodic_wannier_note.inp`
- Molecular Boys foothold: [wannier-centers](wannier-centers.md)
- Related energy / bands footholds: [pw-lda-si](pw-lda-si.md),
  [bands-dos-si](bands-dos-si.md)
- Narrative style ref: [Chapter 8 — Plane-Wave DFT](../ch08-plane-wave.md)
- N. Marzari and D. Vanderbilt, *Phys. Rev. B* **56**, 12847 (1997)
- N. Marzari et al., *Rev. Mod. Phys.* **84**, 1419 (2012)
- D. R. Hamann, *Phys. Rev. B* **88**, 085117 (2013)
- M. J. van Setten et al., *Comput. Phys. Commun.* **226**, 39 (2018)
