# GPW PBE-MBD@rsSCS + Hirshfeld volumes — water dimer

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `gpw-mbd-hirshfeld` |
| Regression | `regressions/gpw/gpw-mbd-hirshfeld/` |
| Public API | `cpu::gpw::GpwDft::mbd` + `::hirshfeld` / `GPWEvaluator` same; `GpwResult::e_disp` [Ha], `GpwResult::volume_ratios` |
| Roadmap | `docs/roadmap/dft.md` §8 Dispersion |

## Summary

**Hirshfeld volume ratios** \(V/V_{\mathrm{free}}\) are computed from the final
Γ-point **GPW PBE** valence density and used to **density-scale** free-atom
**MBD@rsSCS** parameters (Tkatchenko–Scheffler / Ambrosetti many-body
dispersion). This product pin uses the same **periodic water-dimer cell** as
`gpw-mbd-ice` / `gpw-ts-ice` / `gpw-pbe-d3bj-ice` (12 Å cubic, ice-relevant
H-bonded pair — not bulk ice Ih) and gates **E_disp** against the fixed
product setting documented in `REFERENCE.md`.

**Why not stock CP2K QS?** CP2K 2026.1 `PAIR_POTENTIAL TYPE` is only
`DFTD2|DFTD3|DFTD3(BJ)|DFTD4` — no stock TS/MBD. Hosts may link **libMBD**
externally. Free-atom MBD is already pinned vs **libMBD Fortran**
(`reg_gpw-mbd-ice`, git `88069c6`). This slug pins the **volume-ratio layer**
(dual pure-Python Hirshfeld model + volume-scaled `mbd_rsscs_pbe`). Twin notes
live under `tools/cp2k_oracle/inputs/gpw-mbd-hirshfeld/`.

Public surface: bool members `mbd` and `hirshfeld` on `cpu::gpw::GpwDft` and
`cpu::gpw::GPWEvaluator` (no `*Config` bag). Dispersion on
`GpwResult::e_disp` [Hartree]; ratios on `GpwResult::volume_ratios`. Free-atom
path (`hirshfeld=false`) remains geometry-only (`reg_gpw-mbd-ice`). Product
gate: `regressions/gpw/gpw-mbd-hirshfeld` (`reg_gpw-mbd-hirshfeld`). Wall ratio
is **perf_exempt** (E_disp + volume-ratio pin; SCF dominates wall — see
Performance).

## Method

### Hirshfeld volumes

Per-atom effective volumes from the molecular valence density \(\rho\) on the
GPW uniform orthorhombic grid and free-atom densities
\(\rho_A^{\mathrm{free}}\):

\[
w_A(\mathbf r)=\frac{\rho_A^{\mathrm{free}}(|\mathbf r-\mathbf R_A|)}
{\sum_B\rho_B^{\mathrm{free}}(|\mathbf r-\mathbf R_B|)},
\quad
V_A=\int w_A\,\rho\,|\mathbf r-\mathbf R_A|^3\,d^3r,
\quad
v_A=\frac{V_A}{V_A^{\mathrm{free}}}.
\]

Minimum-image distances when the SCF cell is periodic. Free-atom model
(in-tree exponential, `basis_data/hirshfeld_tables.h`):

\[
\rho_Z^{\mathrm{free}}(r)=A\exp(-\alpha_Z r),\quad
A=\frac{N_{\mathrm{val}}\,\alpha_Z^3}{8\pi},\quad
V_Z^{\mathrm{free}}=\frac{60\,N_{\mathrm{val}}}{\alpha_Z^3}.
\]

| Z | \(N_{\mathrm{val}}\) | \(\alpha\) [Bohr⁻¹] |
| - | -------------------: | ------------------: |
| H | 1 | 2.0 |
| O | 6 | 2.2 |

### Density-scaled MBD@rsSCS

Free-atom \(\alpha_0\), \(C_6\), \(R_{\mathrm{vdW}}\) from libMBD
`ts_vdw_params` (in-tree `basis_data/ts_tables.h`) are scaled as

\[
\alpha_A \propto v_A,\qquad C_6^{A}\propto v_A^2,\qquad R_{\mathrm{vdW}}^{A}\propto v_A^{1/3}.
\]

MBD@rsSCS (Ambrosetti et al., *J. Chem. Phys.* **140**, 18A508 (2014);
libMBD Hermann et al., *J. Chem. Phys.* **159**, 174802 (2023)):

1. Range-separated self-consistent screening of dynamical polarizabilities
   (Fermi × Gaussian dipole tensor, \(\beta\) damping).
2. Long-range MBD Hamiltonian diagonalization → zero-point energy difference.

**PBE defaults:** \(\beta = 0.83\), Fermi steepness \(a = 6\). Product MBD is
**isolated** on the nuclear coordinates (periodic Ewald MBD is a follow-on).
SCF uses a periodic 12 Å cell for the density that feeds Hirshfeld.

Free-atom tables (same as `gpw-mbd-ice`):

| Z | \(\alpha_0\) [Bohr³] | \(C_6\) [Ha·Bohr⁶] | \(R_{\mathrm{vdW}}\) [Bohr] |
| - | -------------------- | ------------------ | -------------------------- |
| H | 4.5 | 6.5 | 3.1 |
| O | 5.4 | 15.6 | 3.19 |

**Electronic host.** Hirshfeld is **density-dependent** (post-SCF); the
regression runs Γ-RKS GPW (PBE, `DZVP-MOLOPT-SR-GTH` + `GTH-PBE-q*`, CUTOFF
200 Ry) so volumes and `e_disp` materialize on a real `GpwResult` and
`e_disp` is folded into `e_total`. Absolute QS \(E_{\mathrm{tot}}\) micro-Ha
parity is **out of scope**.

**Key exclusions.**

- Periodic Ewald / k-mesh MBD
- Analytic MBD forces / stress product gate (energy + volumes only)
- Full ice Ih supercell / phonons
- Stock CP2K QS TS/MBD pin (not in PAIR_POTENTIAL enum)
- k-points / UKS with `hirshfeld` (Γ-RKS only; k-points / UKS throw)
- Mutual exclusivity: `mbd` vs `d3` / `d3bj` / `d4` / `ts` / `vdw_df`
- CUDA GPW
- Absolute \(e_{\mathrm{tot}}\) pin vs CP2K QS

## Implementation

Public surface only (`GpwDft` / `GPWEvaluator` members; no `*Config`, no
`make_*` / `build_*` factories). Regression uses the concrete solver:

```cpp
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/molrs.hpp"

using ATV::cpu::gpw::GaussianBasis;
using ATV::cpu::gpw::GpwAtom;
using ATV::cpu::gpw::GpwDft;
using ATV::cpu::gpw::GthPotential;
using ATV::cpu::gpw::XcKind;

// Geometry [Å] → Bohr — water dimer (REFERENCE.md / same as gpw-mbd-ice).
constexpr double kA2B = 1.0 / 0.52917720859;
// … fill 6 GpwAtom rows (2×O + 4×H) with DZVP-MOLOPT-SR-GTH + GTH-PBE-q* …

GpwDft gpw;
const double L = 12.0 * kA2B;
gpw.cell[0] = gpw.cell[1] = gpw.cell[2] = L;
gpw.cutoff_ry = 200.0; // density-dependent E_disp; pin is this product setting
gpw.n_occ = 8;         // 16 valence e⁻ → 8 doubly occupied
gpw.max_iter = 16;
gpw.tol = 1e-6;
gpw.mix = 0.4;
gpw.xc = XcKind::PBE;
gpw.nonperiodic = false;
gpw.mbd = true;
gpw.hirshfeld = true;  // post-SCF volumes → mbd_rsscs_pbe(…, vol_ratio)

gpw.build(atoms);
const auto& res = gpw.run(atoms);
// res.e_disp [Ha] within 1e-8 of offline pin
// res.volume_ratios size 6, physically in (0.3, 1.5)
// res.e_total includes e_disp
```

Driver / Frame path (component; same members):

```cpp
// kinds = {O, H} Kind{Element, GaussianBasis, GthPotential}
ATV::cpu::gpw::GPWEvaluator eval(kinds, /*cutoff_ry=*/300.0, /*max_iter=*/200,
                                 /*tol=*/1e-7, /*mix=*/0.3,
                                 ATV::cpu::gpw::XcKind::PBE,
                                 /*nonperiodic=*/false);
eval.mbd = true;
eval.hirshfeld = true;
// single_point(frame, eval) or bind on Evaluate / MD / Minimize force slot
// eval.last_result().e_disp  [Hartree]
// eval.last_result().volume_ratios
```

Build and run the regression:

```bash
bash -lc '
  module load buildenv-gcc/2026.03-mpich
  export OPENBLAS_ROOT=$OPENBLASDIR
  export RUSTUP_TOOLCHAIN=stable-x86_64-unknown-linux-gnu
  export LD_LIBRARY_PATH=$LIBRARY_PATH:$LD_LIBRARY_PATH
  cmake -B build -DCMAKE_BUILD_TYPE=Release \
        -DATV_BUILD_TESTS=ON -DATV_BUILD_HF=ON \
        -DATV_BUILD_PW=ON -DATV_BUILD_GPW=ON \
        -DATV_BUILD_REGRESSIONS=ON -DATV_BUILD_CUDA=OFF \
        -DBLA_VENDOR=OpenBLAS
  cmake --build build --target reg_gpw-mbd-hirshfeld -j$(nproc)
  ./build/regressions/reg_gpw-mbd-hirshfeld
'
```

| Path | Role |
| ---- | ---- |
| `include/atomiverse/cpu/gpw/gpw_solver.h` | `GpwDft`, `GpwResult::e_disp`, `volume_ratios`, `mbd`, `hirshfeld` |
| `include/atomiverse/cpu/gpw/gpw_evaluator.h` | `GPWEvaluator::mbd`, `::hirshfeld` |
| `include/atomiverse/cpu/gpw/internal/mbd_dispersion.h` | `mbd_rsscs_pbe` (private; optional `vol_ratio`) |
| `include/atomiverse/cpu/gpw/internal/hirshfeld_volumes.h` | `hirshfeld_volume_ratios` (private) |
| `include/atomiverse/cpu/gpw/basis_data/hirshfeld_tables.h` | Free-atom exponential model |
| `include/atomiverse/cpu/gpw/basis_data/ts_tables.h` | Free-atom \(\alpha_0\) / \(C_6\) / \(R_{\mathrm{vdW}}\) |
| `tests/cpu/gpw/test_cpu_gpw_hirshfeld.cpp` | Unit gates (free-atom \(v\approx 1\), scaling dual, product path) |
| `regressions/gpw/gpw-mbd-hirshfeld/reg_gpw-mbd-hirshfeld.cpp` | Product gate |
| `regressions/gpw/gpw-mbd-hirshfeld/REFERENCE.md` | Offline pin + perf_exempt |
| `tools/cp2k_oracle/inputs/gpw-mbd-hirshfeld/` | Oracle notes (no stock CP2K QS twin) |

## Validation (libMBD / dual Hirshfeld; CP2K loop)

Source:
[`regressions/gpw/gpw-mbd-hirshfeld/REFERENCE.md`](../../../regressions/gpw/gpw-mbd-hirshfeld/REFERENCE.md).
Oracle: offline **libMBD** free-atom MBD (slug `gpw-mbd-ice`) + **dual
pure-Python Hirshfeld** on the same exponential free-atom model; product pin
is ATV `GpwDft` with `mbd`+`hirshfeld` at CUTOFF 200 Ry. External / dual
pins — not a golden of Atomiverse’s own output alone; **never** launch live
libMBD, CP2K, or Python duals from ctest. Loop domain remains **CP2K
capability** (periodic GPW dispersion roadmap §8).

**Setup:**

| Item | Value |
| ---- | ----- |
| System | Water dimer (ice-relevant H-bond pair) |
| Geometry [Å] | O (0,0,0); H (0.758602, 0, ±0.504284); O (3,0,0); H (3.758602, 0, ±0.504284) |
| Cell (SCF) | ABC **12.0 12.0 12.0 Å**, PERIODIC XYZ |
| MBD BC | **Isolated** on nuclear coordinates with Hirshfeld \(v_A\) |
| Basis / PP | `DZVP-MOLOPT-SR-GTH` + `GTH-PBE-q6` (O) / `GTH-PBE-q1` (H) |
| XC + disp | PBE host + MBD@rsSCS (Hirshfeld-scaled) |
| Occupations | RKS, 16 valence e⁻ → `n_occ = 8` |
| CUTOFF / SCF | 200 Ry, max_iter 16, tol 1e-6, mix 0.4 |
| Pin E_disp | −1.1479852469005181×10⁻³ Ha |

Representative volume ratios at the pin (order O, H, H, O, H, H):

`0.7403, 0.6529, 0.6529, 0.7153, 0.5784, 0.5784`

| Quantity | Atomiverse / pin | Offline cross-check | Tol | Notes |
| -------- | ---------------: | ------------------: | --- | ----- |
| \(E_{\mathrm{disp}}\) Hirshfeld-MBD [Ha] | −1.1479852469005181×10⁻³ | product dual gate (REFERENCE) | \(1\times 10^{-8}\) Ha | Hard gate |
| \(E_{\mathrm{disp}}\) free-atom MBD [Ha] | −1.4928804171407961×10⁻³ | libMBD Fortran `88069c6` | \(1\times 10^{-8}\) / unit 1e-12 | Context (`gpw-mbd-ice`) |
| Volume ratios \(v_A\) | six values \(\in(0.3,1.5)\) | dual pure-Python free-atom \(v\approx 1\) | range + unit 3e-2 on free-atom O | Regression bounds |
| \(E_{\mathrm{tot}}\) QS [Ha] | finite; includes \(e_{\mathrm{disp}}\) | (no stock CP2K MBD) | — | Not absolute-parity gated |

**Tolerance rationale.** Matching free-atom tables, Hirshfeld model, and
isolated MBD@rsSCS track the dual stack to nano-Hartree on E_disp at this
fixed CUTOFF/basis product setting. Density dependence means E_disp is
basis/cutoff sensitive at mHa level if settings change — the hard gate is
**this** product configuration. Hirshfeld scaling reduces |E_disp| vs
free-atom (~23% less attractive here), as expected for compressed molecular
volumes (\(v_A\in[0.58,0.74]\)).

Offline re-pin notes (human only; never ctest):
`tools/cp2k_oracle/inputs/gpw-mbd-hirshfeld/water_dimer_hirshfeld.note`.

## Performance

Source: `REFERENCE.md` (queue / handoff: **perf_exempt**).

Hirshfeld grid integral is \(O(N_{\mathrm{grid}} N_{\mathrm{atoms}})\); MBD is
an \(O(N^3)\) eigensolve of a \(3N\times 3N\) matrix on a 6-atom system
(microseconds–milliseconds). Full GPW SCF wall dominates. A fair
**dispersion-only** microbenchmark vs libMBD is not the product gate; stock
CP2K QS has no TS/MBD twin wall.

| Twin | \(T_{\mathrm{oracle}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | --------------------------: | ----------------------: | ----: | ---- | ------ |
| Full QS (context) | n/a (no stock CP2K MBD) | — (not fair-twinned) | **perf_exempt** | — | Twin notes only |
| E_disp + volumes product gate | n/a | short SCF + Hirshfeld + MBD (reg) | **perf_exempt** | — | Density-dependent deliverable |

**perf_exempt = true.** Reason (from REFERENCE / queue): primary deliverable
is E_disp + volume-ratio parity; same policy as `gpw-mbd-ice` / `gpw-ts-ice` /
`gpw-pbe-d3bj-ice` / `gpw-d4-water`. Re-measure a strict full-SCF twin later
if product policy requires \(r = T_{\mathrm{ATV}}/T_{\mathrm{CP2K}} \le 10\) on a
host with libMBD-linked CP2K.

## Limitations

- **Cell is a water dimer**, not bulk ice Ih — sufficient for Hirshfeld +
  isolated MBD parity; not a structural ice benchmark.
- **E_disp + volume-ratio hard gate only** — absolute GPW \(e_{\mathrm{tot}}\)
  vs CP2K QS is not asserted (and stock CP2K has no TS/MBD PAIR_POTENTIAL).
- **Isolated MBD** — periodic Ewald / k-mesh MBD is a follow-on.
- **Energy only** — MBD analytic forces / stress out of scope for this slug.
- **Γ-RKS only** for `hirshfeld` — multi-k / UKS throw when Hirshfeld is on.
- **Density-dependent** — changing CUTOFF / basis / tol changes E_disp; pin is
  the fixed product settings in REFERENCE.
- **`mbd` exclusive** vs `d3` / `d3bj` / `d4` / `ts` / `vdw_df`.
- Free-atom path remains `hirshfeld=false` (`reg_gpw-mbd-ice`); do not mix
  free-atom and Hirshfeld pins when comparing.
- Wall **perf_exempt** for this wave; not a speed optimization target.

## References

- Offline pin: `regressions/gpw/gpw-mbd-hirshfeld/REFERENCE.md`
- Free-atom MBD pin: `regressions/gpw/gpw-mbd-ice/REFERENCE.md`
- Oracle notes: `tools/cp2k_oracle/inputs/gpw-mbd-hirshfeld/`
- Oracle rules: `tools/cp2k_oracle/README.md`
- Roadmap: `docs/roadmap/dft.md` §8 Dispersion
- Related free-atom MBD regression: `regressions/gpw/gpw-mbd-ice/`
- Related TS free-atom: `regressions/gpw/gpw-ts-ice/`
- Related D3(BJ) / D4 GPW pins: [gpw-pbe-d3bj-ice](gpw-pbe-d3bj-ice.md),
  [gpw-d4-water](gpw-d4-water.md)
- GPW narrative chapter: `docs/tutorial/ch09-gpw.md`
- Closed spec (git history): `.claude/specs/gpw-mbd-hirshfeld/` (deleted at done)
- A. Tkatchenko, M. Scheffler, *Phys. Rev. Lett.* **102**, 073005 (2009)
  (Hirshfeld / TS volume scaling)
- A. Ambrosetti et al., *J. Chem. Phys.* **140**, 18A508 (2014) (MBD@rsSCS)
- J. Hermann et al., *J. Chem. Phys.* **159**, 174802 (2023) (libMBD)
