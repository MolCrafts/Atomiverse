# GPW PBE-MBD@rsSCS — periodic Ewald + k-mesh (water dimer)

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `gpw-mbd-periodic` |
| Regression | `regressions/gpw/gpw-mbd-periodic/` |
| Public API | `cpu::gpw::GpwDft::mbd` + `mbd_k_mesh[3]`; `cpu::gpw::GPWEvaluator::mbd` + `mbd_k_mesh[3]`; `GpwResult::e_disp` [Ha] |
| Roadmap | `docs/roadmap/dft.md` §8 Dispersion / §10 |

## Summary

**Many-body dispersion MBD@rsSCS** (range-separated self-consistent screening)
is wired into the **Γ-point GPW** path as an additive London correction on top
of PBE. This product pin uses a **periodic water-dimer cell** (same ice-relevant
H-bonded pair as `gpw-mbd-ice` / `gpw-ts-ice` / `gpw-pbe-d3bj-ice`, 12 Å cubic)
and gates the **dispersion energy** for the **periodic Ewald + Brillouin-zone
k-mesh** path against offline **pure-Python pymbd** (`libmbd` algorithm family).

Default `mbd_k_mesh = {0,0,0}` keeps free-atom **isolated** MBD
(`reg_gpw-mbd-ice`). Non-zero mesh (here **2×2×2**, libMBD shift-½) enables
periodic Ewald dipole tensors and k-mesh ZPE averaging; free-atom volume
ratios \(V/V_{\mathrm{free}}=1\) (no Hirshfeld on this pin — see
`gpw-mbd-hirshfeld`).

Public surface: bool `mbd` and `int mbd_k_mesh[3]` on `cpu::gpw::GpwDft` and
`cpu::gpw::GPWEvaluator` (no `*Config` bag). Dispersion is reported on
`GpwResult::e_disp` [Hartree] and folded into `e_total`. Mutually exclusive
with `d3` / `d3bj` / `d4` / `ts` / `vdw_df` / `vdw_df2` / `rvv10`. Product
gate: `regressions/gpw/gpw-mbd-periodic` (`reg_gpw-mbd-periodic`). Wall ratio
is **perf_exempt** (E_disp pin; full QS twin wall unfair — see Performance).

## Method

**MBD@rsSCS** (Ambrosetti et al., *J. Chem. Phys.* **140**, 18A508 (2014);
libMBD: Hermann et al., *J. Chem. Phys.* **159**, 174802 (2023)):

1. **Range-separated SCS** of dynamical polarizabilities over lattice images
   (Fermi × Gaussian dipole tensor, range-separation \(\beta\); **no Ewald**
   on this step).
2. **Long-range MBD Hamiltonian** with **Ewald** dipole tensor; eigenvalues
   (zero-point energy difference) averaged over a shifted k-mesh (libMBD
   `get_kpts`, shift \(\tfrac12\)).

\[
E_{\mathrm{MBD}} = \frac{1}{2N_k}\sum_{\mathbf{k}}\sum_{p}
  \sqrt{\lambda_p(\mathbf{k})}
  - \frac{3}{2}\sum_A\omega_A
\]

(schematic ZPE form; implementation matches libMBD / pymbd `mbd_energy`).

**PBE defaults** (this pin):

| Parameter | Value | Unit / notes |
| --------- | ----: | ------------ |
| \(\beta\) | 0.83 | range separation |
| Fermi steepness \(a\) | 6 | dimensionless |
| Free-atom \(\alpha_0\), \(C_6\), \(R_{\mathrm{vdW}}\) | libMBD `ts_vdw_params` | in-tree `basis_data/ts_tables.h` |
| Volume ratios | \(V/V_{\mathrm{free}}=1\) | free-atom; no Hirshfeld |
| k-mesh | \(2\times 2\times 2\) | libMBD shift-½ |

**Why periodic differs from isolated.** On the same nuclei, isolated free-atom
MBD is \(-1.4928804171407961\times 10^{-3}\) Ha (`reg_gpw-mbd-ice`). Periodic
k-mesh averages image coupling; at 12 Å the 2×2×2 vs isolated difference is
only \(\sim 3.7\times 10^{-6}\) Ha (weak coupling), but the code path is the
full Ewald + k-mesh product surface.

**k-mesh convergence (context, offline):**

| k-mesh | \(E_{\mathrm{disp}}\) [Ha] |
| ------ | -------------------------: |
| \(1\times 1\times 1\) | −1.4474832766229184×10⁻³ |
| \(2\times 2\times 2\) | **−1.4965720704989138×10⁻³** (hard pin) |
| \(3\times 3\times 3\) | −1.5008181948612531×10⁻³ |

**Electronic host.** Dispersion is geometry-only (free-atom volumes); the
regression still runs a short Γ-RKS GPW (PBE, `DZVP-MOLOPT-SR-GTH` +
`GTH-PBE-q*`) so `e_disp` is materialized on a real `GpwResult` and included
in `e_total`. Absolute QS \(E_{\mathrm{tot}}\) micro-Ha parity is **out of
scope** (OT/grid/DIIS differences).

**Oracle split (CP2K campaign).** Stock **CP2K 2026.1**
`PAIR_POTENTIAL TYPE` is only `DFTD2|DFTD3|DFTD3(BJ)|DFTD4` — MBD is external
libMBD when linked. Hard pin is **pure-Python pymbd** (same algorithm as
libMBD Fortran), same policy as `gpw-mbd-ice`. Twin notes under
`tools/cp2k_oracle/inputs/gpw-mbd-periodic/`.

**Key exclusions (from closed product scope).**

- Hirshfeld volume scaling (productized separately: `gpw-mbd-hirshfeld`)
- MBD analytic forces / stress product gate (energy-only v1)
- Full ice Ih supercell / phonons
- Absolute \(e_{\mathrm{tot}}\) pin vs CP2K QS
- k-points / UKS on the GPW electronic host as part of this slug
- CUDA GPW
- Mutual use with pairwise D3/D4/TS or nonlocal vdW-DF/rVV10

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

// Geometry [Å] → Bohr — water dimer (REFERENCE.md).
constexpr double kA2B = 1.0 / 0.52917720859;
// … fill 6 GpwAtom rows (2×O + 4×H) with DZVP-MOLOPT-SR-GTH + GTH-PBE-q* …

GpwDft gpw;
const double L = 12.0 * kA2B;
gpw.cell[0] = gpw.cell[1] = gpw.cell[2] = L;
gpw.cutoff_ry = 200.0; // e_disp is geometry-only; SCF only needs to run
gpw.n_occ = 8;         // 16 valence e⁻ → 8 doubly occupied
gpw.max_iter = 8;
gpw.tol = 1e-6;
gpw.mix = 0.4;
gpw.xc = XcKind::PBE;
gpw.nonperiodic = false;
gpw.mbd = true;
gpw.mbd_k_mesh[0] = gpw.mbd_k_mesh[1] = gpw.mbd_k_mesh[2] = 2; // periodic

gpw.build(atoms);
const auto& res = gpw.run(atoms);
// res.e_disp [Ha] within 1e-8 of offline pymbd pin
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
eval.mbd_k_mesh[0] = eval.mbd_k_mesh[1] = eval.mbd_k_mesh[2] = 2;
// single_point(frame, eval) or bind on Evaluate / MD / Minimize force slot
// eval.last_result().e_disp  [Hartree]
```

Isolated free-atom MBD (contrast; not this pin): leave `mbd_k_mesh` at
`{0,0,0}` with `mbd = true` — product gate `reg_gpw-mbd-ice`.

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
  cmake --build build --target reg_gpw-mbd-periodic -j$(nproc)
  ./build/regressions/reg_gpw-mbd-periodic
'
```

| Path | Role |
| ---- | ---- |
| `include/atomiverse/cpu/gpw/gpw_solver.h` | `GpwDft`, `GpwResult::e_disp`, `mbd`, `mbd_k_mesh` |
| `include/atomiverse/cpu/gpw/gpw_evaluator.h` | `GPWEvaluator::mbd`, `mbd_k_mesh` |
| `include/atomiverse/cpu/gpw/internal/mbd_dispersion.h` | `mbd_rsscs_pbe` (private; periodic flag + k-mesh) |
| `include/atomiverse/cpu/gpw/basis_data/ts_tables.h` | Free-atom \(\alpha_0\), \(C_6\), \(R_{\mathrm{vdW}}\) |
| `regressions/gpw/gpw-mbd-periodic/reg_gpw-mbd-periodic.cpp` | Product gate |
| `regressions/gpw/gpw-mbd-periodic/REFERENCE.md` | Offline pin + perf_exempt |
| `tools/cp2k_oracle/inputs/gpw-mbd-periodic/` | Oracle notes (pymbd reproduce) |

## Validation (pymbd / libMBD family)

Source:
[`regressions/gpw/gpw-mbd-periodic/REFERENCE.md`](../../../regressions/gpw/gpw-mbd-periodic/REFERENCE.md).
Oracle: offline **pure-Python pymbd** (`libmbd/src/pymbd/pymbd.py` `mbd_energy`),
ATV Å→Bohr factor `1/0.52917720859`. External engine pin — not a golden of
Atomiverse’s own output; **never** launch live CP2K / libMBD / pymbd from
ctest. Twin notes:
`tools/cp2k_oracle/inputs/gpw-mbd-periodic/water_dimer_mbd_periodic.note`.

**Setup:**

| Item | Value |
| ---- | ----- |
| System | Water dimer (ice-relevant H-bond pair) |
| Geometry [Å] | O (0,0,0); H (0.758602, 0, ±0.504284); O (3,0,0); H (3.758602, 0, ±0.504284) |
| Cell | ABC **12.0 12.0 12.0 Å**, PERIODIC XYZ |
| Basis / PP | `DZVP-MOLOPT-SR-GTH` + `GTH-PBE-q6` (O) / `GTH-PBE-q1` (H) (E_disp is basis-independent) |
| XC + disp | PBE host + MBD@rsSCS free-atom, **k = 2×2×2** |
| Occupations | RKS, 16 valence e⁻ → `n_occ = 8` |
| Pin line | \(E_{\mathrm{disp}} = -1.4965720704989138\times 10^{-3}\) Ha |

| Quantity | Atomiverse / pin | pymbd (offline) | Tol | Notes |
| -------- | ---------------: | --------------: | --- | ----- |
| \(E_{\mathrm{disp}}\) [Ha] | −1.4965720704989138×10⁻³ | −1.4965720704989138×10⁻³ | \(1\times 10^{-8}\) Ha | Hard gate (k=2×2×2) |
| \(E_{\mathrm{disp}}\) isolated [Ha] | (sibling pin) | −1.4928804171407961×10⁻³ | — | `reg_gpw-mbd-ice` / libMBD Fortran |
| \(E_{\mathrm{disp}}\) k=1×1×1 [Ha] | (context) | −1.4474832766229184×10⁻³ | — | Offline cross-check only |
| \(E_{\mathrm{tot}}\) QS [Ha] | finite; includes \(e_{\mathrm{disp}}\) | (CP2K stock MBD N/A) | — | Not absolute-parity gated |

**Tolerance rationale.** Matching geometry, free-atom TS tables, PBE
\(\beta=0.83\), \(a=6\), and libMBD Ewald + shift-½ k-mesh are expected to
track pymbd to nano-Hartree on this cell. Unit tests use **1e-10 Ha** on the
standalone `mbd_rsscs_pbe` periodic kernel.

Offline re-pin (human only; never ctest) — after cloning libmbd:

```python
# coords Å→Bohr with ATV factor; lattice diag(12*kA2B); free-atom alpha_0, C6, R_vdw
# k_grid=(2,2,2); beta=0.83
mbd_energy(coords, alpha_0, C6, R_vdw, beta=0.83, lattice=lattice, k_grid=(2,2,2))
# → -1.4965720704989138e-03 Ha
```

## Performance

Source: `REFERENCE.md` (queue / handoff: **perf_exempt**).

MBD evaluation is \(O(N_k \cdot (3N)^3)\) on a 6-atom system (milliseconds).
Full GPW SCF wall is dominated by the electronic structure, not MBD. A fair
**dispersion-only** microbenchmark vs pymbd is not the product gate; a full
twin SCF ratio vs CP2K is unavailable for stock MBD (external libMBD only when
linked) and would be unfair for a geometry-only hard gate.

| Twin | \(T_{\mathrm{oracle}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | --------------------------: | ----------------------: | ----: | ---- | ------ |
| Full QS (context) | n/a (stock CP2K lacks MBD PAIR_POTENTIAL) | — (not fair-twinned) | **perf_exempt** | — | Twin notes retained for libMBD-linked hosts |
| E_disp product gate | n/a | short SCF + MBD (reg) | **perf_exempt** | — | Geometry-only deliverable |

**perf_exempt = true.** Reason (from REFERENCE / handoff): primary deliverable
is E_disp pin parity; same policy as `gpw-mbd-ice` / `gpw-ts-ice` /
`gpw-pbe-d3bj-ice`. Re-measure a strict full-SCF twin later if product policy
requires \(r = T_{\mathrm{ATV}}/T_{\mathrm{oracle}} \le 10\) on a host with
libMBD-linked CP2K.

## Limitations

- **Cell is a water dimer**, not bulk ice Ih — sufficient for periodic Ewald /
  k-mesh MBD parity; not a structural ice benchmark.
- **E_disp hard gate only** — absolute GPW \(e_{\mathrm{tot}}\) vs CP2K QS is
  not asserted (and stock CP2K has no MBD PAIR type).
- **Energy only** — MBD analytic forces / stress are out of scope (v1).
- **Free-atom volumes** — density-scaled Hirshfeld is a separate product
  (`gpw-mbd-hirshfeld`); do not mix pin values.
- **`mbd` exclusive** vs `d3` / `d3bj` / `d4` / `ts` / `vdw_df` / `vdw_df2` /
  `rvv10` — any second flag throws at run.
- **Non-zero `mbd_k_mesh` requires periodic cell** (`!nonperiodic`); all-zero
  mesh is isolated free-atom MBD.
- **Oracle is pymbd / libMBD**, not live CP2K QS — CP2K campaign still owns the
  GPW host and dispersion product surface; pin backend is the libMBD family.
- Wall **perf_exempt** for this wave; not a speed optimization target.

## References

- Offline pin: `regressions/gpw/gpw-mbd-periodic/REFERENCE.md`
- Oracle notes: `tools/cp2k_oracle/inputs/gpw-mbd-periodic/`
- Oracle rules: `tools/cp2k_oracle/README.md`
- Roadmap: `docs/roadmap/dft.md` §8 Dispersion
- Related isolated MBD: `regressions/gpw/gpw-mbd-ice/` (no methods page yet)
- Related Hirshfeld MBD: `regressions/gpw/gpw-mbd-hirshfeld/`
- Related pairwise D3(BJ): [gpw-pbe-d3bj-ice](gpw-pbe-d3bj-ice.md)
- Related D4: [gpw-d4-water](gpw-d4-water.md)
- GPW narrative chapter: `docs/tutorial/ch09-gpw.md`
- Closed product history: git log for `reg_gpw-mbd-periodic` / queue entry
  2026-07-26
- A. Ambrosetti, A. M. Reilly, R. A. DiStasio Jr., A. Tkatchenko,
  *J. Chem. Phys.* **140**, 18A508 (2014) (MBD@rsSCS)
- J. Hermann, M. Stöhr, S. Góger, S. Chaudhuri, B. Aradi, R. J. Maurer,
  A. Tkatchenko, *J. Chem. Phys.* **159**, 174802 (2023) (libMBD)
