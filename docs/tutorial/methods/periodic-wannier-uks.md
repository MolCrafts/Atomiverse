# Collinear spin-polarized multi-k MLWF — bulk Si (UKS / Marzari–Vanderbilt)

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `periodic-wannier-uks` |
| Regression | `regressions/pw/periodic-wannier-uks/` |
| Public API | `cpu::pw::PW` (`spin_polarized`, `multiplicity`) + `cpu::pw::PeriodicWannier` / `PeriodicWannierResult` (`cpu/planewave/periodic_wannier.h`) |
| Track | CP2K capability (); response / PW |
| Roadmap | [dft.md](../../roadmap/dft.md) §11 MLWF; queue `periodic-wannier-uks` (done) |
| Dual gate | Bond midpoints per spin + total-spread reduction; speed **`perf_exempt`** |
| Wave | CP2K harness wave 10 / staffed 2026-07-26 |

## Summary

**Collinear spin-polarized multi-k maximally localized Wannier functions
(MLWF)** on the CPU plane-wave stack — the spin productization of
`periodic-wannier-si` (restricted Marzari–Vanderbilt; `regressions/pw/periodic-wannier-si`).
Concrete method `cpu::pw::PW` with `spin_polarized = true` and
`multiplicity = 1` (closed-shell collinear \(M = N_\uparrow - N_\downarrow = 0\))
feeds `cpu::pw::PeriodicWannier::compute`. Occupied α and β manifolds are
localized **independently** on frozen \(V_{\mathrm{eff},\sigma}\) after a multi-k
LSDA SCF. Target system is **diamond Si** (2-atom fcc primitive cell). Primary
gates: four valence MLWFs **per spin** at the tetrahedral Si–Si bond midpoints
(Marzari & Vanderbilt 1997 §IV) and total spread reduction
\(\Omega_{\mathrm{final}} < 0.25\,\Omega_{\mathrm{init}}\). Regression binary:
`reg_periodic-wannier-uks`. **No live CP2K in ctest** — offline oracle pins
only (`tools/cp2k_oracle/`).

## Method

**Level.** Periodic plane-wave Kohn–Sham with **collinear UKS** fixed
occupations (same spin channel as `pw-spin-polarized-o2` / multi-k UKS),
followed by Marzari–Vanderbilt (MV) localization on each spin manifold.

1. **SCF.** Collinear densities \(\rho_\uparrow\), \(\rho_\downarrow\); Hartree
   from \(\rho = \rho_\uparrow + \rho_\downarrow\); spin XC
   \(V_{\mathrm{xc},\sigma}(\rho_\uparrow,\rho_\downarrow)\); dual Davidson per
   \(\mathbf{k}\) and spin on a Monkhorst–Pack mesh (`PW::k_mesh`).
2. **NSCF Bloch states.** Freeze \(V_{\mathrm{eff},\sigma}\); recompute occupied
   \(u_{n\mathbf{k}\sigma}\) on the MLWF mesh (`PeriodicWannier::k_mesh`,
   independent of SCF mesh).
3. **Overlaps.** Neighbor matrices
   \[
   M_{mn}^{(\mathbf{k},\mathbf{b},\sigma)}
     = \langle u_{m\mathbf{k}\sigma}
       \mid u_{n,\mathbf{k}+\mathbf{b},\sigma}\rangle
   \]
   on the SMV six-neighbor cubic shell (weights
   \(w_b = 3/(N_b b^2)\)).
4. **Initialization.** Projection onto bond-centered Gaussians
   (`use_projection = true`, `proj_alpha_ang` [Å]; auto diamond midpoints for
   2-atom Si-like cells with 4 WFs/spin).
5. **MV polish.** Steepest descent on the unitary gauge \(U(\mathbf{k})\) per
   spin: \(U\leftarrow U\exp(-\alpha G)\) until
   \(\|G\|_\infty < \texttt{tol}\) or `max_iter`.
6. **Centers.** Berry-phase formula (per spin channel \(n\)):
   \[
   \mathbf{r}_n = -\frac1{N_k}\sum_{\mathbf{k},\mathbf{b}}
     w_b\,\mathbf{b}\,
     \operatorname{Im}\ln M_{nn}^{(\mathbf{k},\mathbf{b})}.
   \]
   Spreads \(\Omega_n = \langle r^2\rangle_n - |\langle\mathbf{r}\rangle_n|^2\)
   [Å²] (SMV discretization).

**Spin / occupations.** Closed-shell collinear product pin:

\[
N_\uparrow = N_\downarrow = 4,\qquad
M = N_\uparrow - N_\downarrow = 0
\]

from multiplicity \(2S+1 = 1\) and 8 valence electrons (2×Si ONCV
\(Z_{\mathrm{val}} = 4\)). Each spin therefore carries the same four-bond
valence manifold as the restricted Si pin. Open-shell \(M\neq 0\) MLWF and
metallic disentanglement are **out of scope** for this slug.

**Literature geometry (parameter-free).** With atoms at \(\mathbf{0}\) and
\((a/4)(1,1,1)\), tetrahedral bond midpoints are

\[
\mathbf{m}\in\frac{a}{8}
  \bigl\{(1,1,1),\,(1,-1,-1),\,(-1,1,-1),\,(-1,-1,1)\bigr\}.
\]

ONCV PW vs GTH GPW differ in absolute spreads; **bond-centering of the valence
manifold per spin at \(M=0\)** is the shared scientific content.

**Observables**

| Symbol | Definition | Unit |
| ------ | ---------- | ---- |
| \(\mathbf{r}_{n\sigma}\) | Wannier center (`PeriodicWannierResult::centers`, α then β) | Å |
| \(\Omega_{n\sigma}\) | Orbital spread (`spreads`) | Å² |
| \(\Omega_{\mathrm{tot}}\) | \(\sum_{n\sigma}\Omega_{n\sigma}\) after localization | Å² |
| \(\Omega_{\mathrm{init}}\) | Identity-gauge total spread | Å² |
| \(n_\uparrow,n_\downarrow\) | `n_wann_up` / `n_wann_dn` | — |
| `spin[i]` | 0 = ↑, 1 = ↓ | — |
| \(E\) | SCF total energy (`energy_ry`) | Ry |

**Excluded (this slug).** Restricted-only path remains `periodic-wannier-si`.
Molecular Foster–Boys is `wannier-centers` (Gaussian RHF/RKS). GPW-grid MLWF,
Pipek–Mezey / ER criteria, metallic smearing / disentanglement, noncollinear /
SOC, open-shell \(M\neq 0\) product pin, CUDA MLWF, and hybrid / meta-GGA XC
are not accepted here.

## Implementation

Public surface only: construct `cpu::pw::PW` and `cpu::pw::PeriodicWannier`, set
**members** (no public `*Config` bag, no `make_*` / `build_*` factory), call
`mlwf.compute(method, frame)`. Header:
`include/atomiverse/cpu/planewave/periodic_wannier.h`. Internals stay under
`cpu/planewave/internal/` (or implementation TU). Helper
`diamond_bond_midpoints(a_ang)` is public pure geometry for pins/tests.

```cpp
#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/planewave/periodic_wannier.h"
#include "atomiverse/cpu/planewave/pw_solver.h"
#include "atomiverse/cpu/scf/xc_functional.h"

using namespace ATV;
using namespace ATV::cpu::pw;

// Diamond Si 2-atom fcc primitive cell, a = 5.43 Å
cpu::Frame frame(2);
for (int i = 0; i < 2; ++i) {
    frame.span(kAtom_type)[i] = 14;
    frame.span(kAtom_mass)[i] = 28.0855;
}
const double a = 5.43;
const double h = a / 2.0;
Mat3 cell;
cell(0, 0) = 0; cell(0, 1) = h; cell(0, 2) = h;
cell(1, 0) = h; cell(1, 1) = 0; cell(1, 2) = h;
cell(2, 0) = h; cell(2, 1) = h; cell(2, 2) = 0;
frame.box.set_cell(cell);
frame.span(kAtom_x)[0] = 0.0;
frame.span(kAtom_y)[0] = 0.0;
frame.span(kAtom_z)[0] = 0.0;
frame.span(kAtom_x)[1] = a / 4.0;
frame.span(kAtom_y)[1] = a / 4.0;
frame.span(kAtom_z)[1] = a / 4.0;

PW method;
method.upf_path = "tests/data/pw/Si_lda_nc.upf";
method.ecut_ry = 15.0;
method.k_mesh = 2;                 // SCF MP mesh
method.tol = 1e-6;
method.max_iter = 50;
method.xc = cpu::scf::XCType::LSDA;
method.spin_polarized = true;
method.multiplicity = 1;           // M = 0 closed-shell collinear

PeriodicWannier mlwf;
mlwf.k_mesh = 2;                   // MLWF MP mesh (independent of SCF)
mlwf.max_iter = 200;
mlwf.tol = 1e-10;
mlwf.step = 0.15;
mlwf.use_projection = true;
mlwf.proj_alpha_ang = 1.0;

const PeriodicWannierResult r = mlwf.compute(method, frame);
// r.n_wann_up == r.n_wann_dn == 4; r.n_wann == 8
// r.centers[i] [Å]  (α first, then β); r.spin[i] ∈ {0,1}
// r.spreads[i] [Å²]; r.total_spread, r.total_spread_initial
// r.energy_ry [Ry]; r.scf_converged; r.converged (MV)
```

`PeriodicWannier::compute` owns the SCF + NSCF + MV sequence for this path;
there is no separate user-facing “Wannier Config”.

```bash
bash -lc '
  module load buildenv-gcc/2026.03-mpich
  export OPENBLAS_ROOT=$OPENBLASDIR
  export RUSTUP_TOOLCHAIN=stable-x86_64-unknown-linux-gnu
  export LD_LIBRARY_PATH=$LIBRARY_PATH:$LD_LIBRARY_PATH
  cmake -B build -DCMAKE_BUILD_TYPE=Release \
        -DATV_BUILD_TESTS=ON -DATV_BUILD_REGRESSIONS=ON \
        -DATV_BUILD_HF=ON -DATV_BUILD_PW=ON -DATV_BUILD_GPW=ON \
        -DATV_BUILD_CUDA=OFF -DBLA_VENDOR=OpenBLAS
  cmake --build build -j$(nproc) --target atomiverse \
        test_cpu_pw_periodic_wannier reg_periodic-wannier-uks
  ctest --test-dir build -R "^(test_cpu_pw_periodic_wannier)$" \
        --output-on-failure
  cmake --build build --target run_reg_periodic-wannier-uks
'
```

Regression binary: `./build/regressions/reg_periodic-wannier-uks`.

**File pointers**

| Role | Path |
| ---- | ---- |
| Public API | `include/atomiverse/cpu/planewave/periodic_wannier.h` |
| PW method | `include/atomiverse/cpu/planewave/pw_solver.h` |
| Unit (midpoints, SMV math, spin gate) | `tests/cpu/planewave/test_periodic_wannier.cpp` |
| Regression | `regressions/pw/periodic-wannier-uks/reg_periodic-wannier-uks.cpp` |
| Offline pin | `regressions/pw/periodic-wannier-uks/REFERENCE.md` |
| Oracle note | `tools/cp2k_oracle/inputs/si_periodic_wannier_uks.note` |
| Restricted twin | `regressions/pw/periodic-wannier-si/` |

## Validation (CP2K)

Source:
[`regressions/pw/periodic-wannier-uks/REFERENCE.md`](../../../regressions/pw/periodic-wannier-uks/REFERENCE.md)
(offline only; **no live CP2K in ctest**). Closed acceptance: collinear
spin-polarized multi-k MLWF, bond midpoints **per spin** + spread reduction,
`perf_exempt`.

| Setting | Value |
| ------- | ----- |
| Structure | diamond Si, 2-atom fcc primitive cell |
| Lattice constant \(a\) | 5.43 Å |
| Basis atoms | \((0,0,0)\), \((a/4)(1,1,1)\) |
| PP | `tests/data/pw/Si_lda_nc.upf` (PseudoDojo ONCVPSP-LDA-SR-noCC v0.4) |
| SCF \(E_{\mathrm{cut}}\) | 15 Ry |
| SCF k-mesh | \(2\times2\times2\) |
| MLWF k-mesh | \(2\times2\times2\) (`PeriodicWannier::k_mesh = 2`) |
| XC | `XCType::LSDA` |
| Spin | `spin_polarized = true`, `multiplicity = 1` (\(M=0\)) |
| SCF tol | \(10^{-6}\) |
| Projection | bond midpoints, `proj_alpha_ang = 1.0` |
| MV SD | `max_iter = 200`, `tol = 1e-10`, `step = 0.15` |
| Twin note | CP2K Quickstep is **GPW/GTH + MOLOPT** + `LSD` + `&LOCALIZE`, not NC-UPF PW; absolute spreads/gauge differ |

### Counts and spin labels

| Quantity | Atomiverse / pin | Reference | Tol | Notes |
| -------- | ---------------: | --------: | --- | ----- |
| \(n_\uparrow\) | **4** | 4 (valence / spin at \(M=0\)) | exact | `n_wann_up` |
| \(n_\downarrow\) | **4** | 4 | exact | `n_wann_dn` |
| \(n_{\mathrm{wann}}\) | **8** | \(n_\uparrow+n_\downarrow\) | exact | `n_wann` |
| `spin[i]` | \(\{0,1\}\) size 8 | α then β layout | exact | labels only |
| SCF | must converge | — | — | regression fails if `!scf_converged` |

### Bond-midpoint geometry (literature + hard-coded)

Diamond / zincblende Si valence MLWFs sit on the four tetrahedral Si–Si bond
midpoints **per spin channel** at \(M=0\) (Marzari & Vanderbilt, *Phys. Rev. B*
**56**, 12847 (1997) §IV). Same coarse \(2\times2\times2\) SMV midpoint window
as `reg_periodic-wannier-si`.

| Quantity | Atomiverse / pin | Reference | Tol | Notes |
| -------- | ---------------: | --------: | --- | ----- |
| \(d_{\max}\rightarrow\mathbf{m}\) spin ↑ [Å] | \(< 0.90\) | bond midpoints \(\mathbf{m}\) (literature geometry) | 0.90 Å | min-image + optimal bipartite assignment |
| \(d_{\max}\rightarrow\mathbf{m}\) spin ↓ [Å] | \(< 0.90\) | same \(\mathbf{m}\) | 0.90 Å | independent β manifold |
| Spreads | finite (no NaN) | — | — | both channels |

Si–Si bond length \(\approx 2.35\) Å at \(a=5.43\) Å; sub-0.15 Å absolute
placement is a denser-mesh productization follow-on, not this pin.

### Localization (MV)

| Quantity | Atomiverse / pin | Reference | Tol | Notes |
| -------- | ---------------: | --------: | --- | ----- |
| \(\Omega_{\mathrm{final}}/\Omega_{\mathrm{init}}\) | \(< 0.25\) | MV / Boys: localization decreases total spread | 0.25 | sum over **both** spins |
| \(\Omega_{\mathrm{init}}\) | \(> 0\) | — | — | identity-gauge baseline |

MV loop flag `converged` may warn without failing if centers still pin (same
product policy as restricted twin).

### CP2K oracle (algorithm twin, not energy-compared)

Offline input echo:
`tools/cp2k_oracle/inputs/si_periodic_wannier_uks.note` — Quickstep
`LSD` / `MULTIPLICITY 1` + `&LOCALIZE` (JACOBI) + print Wannier centers on
diamond Si with GTH-PADE-q4 + DZVP-MOLOPT-SR-GTH. Shared content: **bond-midpoint
geometry of the four valence MLWFs per spin channel at \(M=0\)**. Absolute
spreads and gauge are **not** compared to ONCV PW.

## Performance

Source: `REFERENCE.md` § Performance; queue/handoff mark this slug
**`perf_exempt`**.

| Field | Value |
| ----- | ----- |
| `perf_exempt` | **true** |
| Reason | No fair CP2K PW/NC-UPF wall twin on the build host (Quickstep GTH path ≠ ONCV Hamiltonian) |
| \(T_{\mathrm{ATV}}\) (informative) | ~2× restricted Si MLWF (two spin channels); restricted twin ~8 s Release ecut 15 / k=2 |
| \(r = T_{\mathrm{ATV}}/T_{\mathrm{CP2K}}\) | **n/a** |

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | -----------------------: | ----------------------: | ----: | ---- | ------ |
| Si PW collinear multi-k MLWF (NC-UPF, \(M=0\)) | n/a | ~2× restricted (~order 10 s) | **n/a** | exempt | GTH-GPW LOCALIZE ≠ NC-UPF PW |

Dual gate: (1) SCF + per-spin bond midpoints + spread ratio; (2) perf exempt
with reason above. Queue row: ratio **perf_exempt** (2026-07-26). When a CP2K
binary is available: time the oracle note geometry, append
\(r = T_{\mathrm{ATV}}/T_{\mathrm{CP2K}}\), require \(r\le 10\).

## Limitations

- **Closed-shell collinear only (\(M=0\)).** Open-shell spin-polarized MLWF
  (\(N_\uparrow\neq N_\downarrow\)) and noncollinear / SOC manifolds are not
  this product pin.
- **Insulator / fixed occupations.** No metallic smearing, no Wannier
  disentanglement (Marzari–Souza–Vanderbilt outer window).
- **Coarse \(2\times2\times2\) SMV mesh.** Midpoint tol 0.90 Å is a
  discretization window, not chemical Å-level placement accuracy.
- **PP / basis twin debt.** CP2K twin is GTH+MOLOPT Quickstep LOCALIZE; ATV is
  ONCV PW — spreads and total energy are not cross-engine absolute pins.
- **GPW-grid MLWF** remains unqueued residual (roadmap §11 prose).
- **Out of scope for this page:** molecular Boys (`wannier-centers`),
  restricted periodic only (`periodic-wannier-si`), Berry polarization
  (`berry-polarization` / `berry-spin-si`), DFPT phonons, CUDA PW, hybrids.

## References

- Offline pin: [`regressions/pw/periodic-wannier-uks/REFERENCE.md`](../../../regressions/pw/periodic-wannier-uks/REFERENCE.md)
- Regression consumer: `regressions/pw/periodic-wannier-uks/reg_periodic-wannier-uks.cpp`
- Oracle note: `tools/cp2k_oracle/inputs/si_periodic_wannier_uks.note`
- Restricted twin pin: `regressions/pw/periodic-wannier-si/REFERENCE.md`
- Restricted twin oracle: `tools/cp2k_oracle/inputs/si_periodic_wannier_note.inp`
- Oracle rules: `tools/cp2k_oracle/README.md` (never live in ctest)
- Roadmap: [dft.md](../../roadmap/dft.md) §11 MLWF
- Related molecular Boys: [`wannier-centers`](wannier-centers.md)
- Related restricted periodic MLWF: `regressions/pw/periodic-wannier-si`
- Related multi-k collinear UKS energy: [`pw-uks-kmesh-fe`](pw-uks-kmesh-fe.md)
- Related bands/DOS: [`bands-dos-si`](bands-dos-si.md)
- Marzari N., Vanderbilt D., *Phys. Rev. B* **56**, 12847 (1997)
- Marzari N. et al., *Rev. Mod. Phys.* **84**, 1419 (2012)
- Narrative PW chapter: [ch08-plane-wave.md](../ch08-plane-wave.md)
