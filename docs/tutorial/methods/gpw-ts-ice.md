# GPW PBE-TS — periodic water dimer (Tkatchenko–Scheffler free-atom)

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `gpw-ts-ice` |
| Regression | `regressions/gpw/gpw-ts-ice/` |
| Public API | `cpu::gpw::GpwDft::ts` / `cpu::gpw::GPWEvaluator::ts`; `GpwResult::e_disp` [Ha] |
| Roadmap | `docs/roadmap/dft.md` §8 Dispersion |

## Summary

**Tkatchenko–Scheffler (TS)** pairwise dispersion (free-atom volume ratios
\(V/V_{\mathrm{free}}=1\)) is wired into the **Γ-point GPW** path as an additive
London correction on top of PBE. This product pin uses the same **periodic
water-dimer cell** as `gpw-pbe-d3bj-ice` / `gpw-d4-water` (12 Å cubic,
ice-relevant H-bonded pair — not bulk ice Ih) and gates the **dispersion
energy** against an offline **libMBD-class pure real-space** pin.

**Why libMBD rather than live CP2K QS?** CP2K 2026.1 stock
`PAIR_POTENTIAL TYPE` is only `DFTD2|DFTD3|DFTD3(BJ)|DFTD4` — there is **no**
`TYPE TS`. libMBD is the canonical offline TS/MBD oracle (also the backend
FHI-aims / Quantum ESPRESSO / DFTB+ use). Twin notes are retained under
`tools/cp2k_oracle/inputs/gpw-ts-ice/` for hosts that may later expose TS via
an external library. The hard pin is **libMBD free-atom TS + pure-RS lattice
sum** — never launched from ctest.

Public surface: bool member `ts` on `cpu::gpw::GpwDft` and
`cpu::gpw::GPWEvaluator` (no `*Config` bag). Dispersion is reported on
`GpwResult::e_disp` [Hartree] and folded into `e_total`. Mutually exclusive
with zero-damping `d3`, BJ `d3bj`, and `d4` (any two true throws at run).
Product gate: `regressions/gpw/gpw-ts-ice` (`reg_gpw-ts-ice`). Wall ratio is
**perf_exempt** (E_disp pin; full QS twin wall unfair / unavailable — see
Performance).

## Method

**Tkatchenko–Scheffler free-atom pairwise dispersion** (Tkatchenko & Scheffler,
*Phys. Rev. Lett.* **102**, 073005 (2009)), matching libMBD’s
`get_ts_energy` / `cmbd_ts_energy` real-space path with **periodic** lattice
sums over image vectors \(\mathbf{T}\):

\[
E_{\mathrm{disp}} = -\sum_{i\le j}\sum_{\mathbf{T}}{}^{\prime}
  C_{6,ij}\,f_{\mathrm{damp}}(R)\,R^{-6},
\]

\[
C_{6,ij}=\frac{2\,C_{6,i}C_{6,j}}
  {(\alpha_j/\alpha_i)\,C_{6,i}+(\alpha_i/\alpha_j)\,C_{6,j}},
\quad
R_{0,ij}=R_i+R_j,
\]

\[
f_{\mathrm{damp}}=\bigl(1+\exp[-d\,(R/(s_R R_{0,ij})-1)]\bigr)^{-1}.
\]

The prime excludes the singular \(R=0\) self term. **PBE damping:**
\(s_R=0.94\), \(d=20\). Free-atom \(\alpha_0\), \(C_6\), \(R_{\mathrm{vdW}}\)
from libMBD `ts_vdw_params` (in-tree `basis_data/ts_tables.h`):

| Z | \(\alpha_0\) [Bohr³] | \(C_6\) [Ha·Bohr⁶] | \(R_{\mathrm{vdW}}\) [Bohr] |
| - | -------------------: | -----------------: | -------------------------: |
| H | 4.5 | 6.5 | 3.1 |
| O | 5.4 | 15.6 | 3.19 |

**Pair cutoff.** Pure real-space orthorhombic lattice sum with
**rcut = 50 Å ≈ 94.561 Bohr** (kernel default 94.2 Bohr; no Ewald). libMBD’s
Ewald path on the same cell differs by \({\sim}7\times 10^{-8}\) Ha (long-range
\(1/R^6\) tail) — documented; the regression hard gate is the pure-RS value.

**Volume ratios.** Optional per-atom \(v = V/V_{\mathrm{free}}\) scale
\(\alpha\leftarrow\alpha_{\mathrm{free}}v\),
\(C_6\leftarrow C_{6,\mathrm{free}}v^2\),
\(R\leftarrow R_{\mathrm{free}}v^{1/3}\). Empty ratios ⇒ free-atom limit
(all \(v=1\)) — the product pin. Hirshfeld density volumes from the GPW density
are **out of scope** (follow-on with MBD).

**Electronic host.** Dispersion is geometry-only; the regression still runs a
short Γ-RKS GPW (PBE, `DZVP-MOLOPT-SR-GTH` + `GTH-PBE-q*`) so `e_disp` is
materialized on a real `GpwResult` and included in `e_total`. Absolute QS
\(E_{\mathrm{tot}}\) micro-Ha parity is **out of scope** (OT/grid/DIIS
differences; CP2K has no stock TS).

**Isolated two-body on the dimer nuclei (context / unit test):**
**−5.9764872295816794×10⁻⁴ Ha** (libMBD `cmbd_ts_energy`, git `88069c6`).

**Key exclusions (from closed acceptance / REFERENCE).**

- Hirshfeld density volume scaling (free-atom \(v=1\) only)
- Many-body dispersion (MBD) — queue `gpw-mbd-ice`
- vdW-DF nonlocal functionals
- Full ice Ih supercell / phonons
- Absolute \(e_{\mathrm{tot}}\) pin vs CP2K QS (no stock `TYPE TS`)
- k-points, UKS, cell stress as part of this slug
- CUDA GPW

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

// Geometry [Å] → Bohr — water dimer (REFERENCE.md / same as D3(BJ) / D4).
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
gpw.ts = true;         // exclusive vs gpw.d3 / d3bj / d4

gpw.build(atoms);
const auto& res = gpw.run(atoms);
// res.e_disp [Ha] within 1e-8 of offline pure-RS / libMBD-class pin
// res.e_total includes e_disp
```

Driver / Frame path (component; same member):

```cpp
// kinds = {O, H} Kind{Element, GaussianBasis, GthPotential}
ATV::cpu::gpw::GPWEvaluator eval(kinds, /*cutoff_ry=*/300.0, /*max_iter=*/200,
                                 /*tol=*/1e-7, /*mix=*/0.3,
                                 ATV::cpu::gpw::XcKind::PBE,
                                 /*nonperiodic=*/false);
eval.ts = true;
// single_point(frame, eval) or bind on Evaluate / MD / Minimize force slot
// eval.last_result().e_disp  [Hartree]
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
  cmake --build build --target reg_gpw-ts-ice -j$(nproc)
  ./build/regressions/reg_gpw-ts-ice
'
```

| Path | Role |
| ---- | ---- |
| `include/atomiverse/cpu/gpw/gpw_solver.h` | `GpwDft`, `GpwResult::e_disp`, `ts` |
| `include/atomiverse/cpu/gpw/gpw_evaluator.h` | `GPWEvaluator::ts` |
| `include/atomiverse/cpu/gpw/internal/ts_dispersion.h` | `ts_pbe` kernel (private) |
| `include/atomiverse/cpu/gpw/basis_data/ts_tables.h` | Free-atom libMBD `ts_vdw_params` |
| `tests/cpu/gpw/test_cpu_gpw_ts.cpp` | Unit gate (isolated libMBD + periodic pure-RS + FD forces) |
| `regressions/gpw/gpw-ts-ice/reg_gpw-ts-ice.cpp` | Product gate |
| `regressions/gpw/gpw-ts-ice/REFERENCE.md` | Offline pin + perf_exempt |
| `tools/cp2k_oracle/inputs/gpw-ts-ice/` | Twin notes (no stock CP2K `TYPE TS`) |

## Validation (libMBD / pure-RS)

Source:
[`regressions/gpw/gpw-ts-ice/REFERENCE.md`](../../../regressions/gpw/gpw-ts-ice/REFERENCE.md).
Oracle: offline **libMBD** (git `88069c6`, built 2026-07-25) for the isolated
dimer, plus dual pure-Python lattice sum / Atomiverse `ts_pbe` for the periodic
hard gate. External engine pin — not a golden of Atomiverse’s own output;
**never** launch live libMBD or CP2K from ctest.

**Setup:**

| Item | Value |
| ---- | ----- |
| System | Water dimer (ice-relevant H-bond pair) |
| Geometry [Å] | O (0,0,0); H (0.758602, 0, ±0.504284); O (3,0,0); H (3.758602, 0, ±0.504284) |
| Cell | ABC **12.0 12.0 12.0 Å**, PERIODIC XYZ |
| Basis / PP | `DZVP-MOLOPT-SR-GTH` + `GTH-PBE-q6` (O) / `GTH-PBE-q1` (H) (E_disp is basis-independent) |
| XC + disp | PBE host + free-atom TS (\(s_R=0.94\), \(d=20\)) |
| Pair cutoff | 50 Å ≈ 94.561 Bohr (pure real-space; no Ewald) |
| Occupations | RKS, 16 valence e⁻ → `n_occ = 8` |
| Hard pin | \(E_{\mathrm{disp}} = -6.099630773386013\times 10^{-4}\) Ha |

| Quantity | Atomiverse / pin | Oracle (offline) | Tol | Notes |
| -------- | ---------------: | ----------------: | --- | ----- |
| \(E_{\mathrm{disp}}\) periodic pure-RS [Ha] | −6.099630773386013×10⁻⁴ | −6.099630773386013×10⁻⁴ | \(1\times 10^{-8}\) Ha | Hard gate (`reg_gpw-ts-ice`) |
| \(E_{\mathrm{disp}}\) isolated [Ha] | (unit test `ts_pbe`) | −5.9764872295816794×10⁻⁴ (libMBD) | \({\sim}10^{-19}\) Ha match | Unit hard pin |
| \(E_{\mathrm{disp}}\) libMBD Ewald [Ha] | (context) | −6.100334450881942×10⁻⁴ | — | Δ ≈ \(7\times 10^{-8}\) vs pure-RS; not the gate |
| \(E_{\mathrm{tot}}\) QS [Ha] | finite; includes \(e_{\mathrm{disp}}\) | (CP2K stock TS N/A) | — | Not absolute-parity gated |

**Tolerance rationale.** Matching geometry, free-atom tables, PBE Fermi damping,
and pure-RS rcut 50 Å track the dual lattice-sum pin to nano-Hartree. Isolated
libMBD and Atomiverse `ts_pbe` agree to machine epsilon. Unit tests also check
analytic TS forces vs central finite difference and
\(e_{\mathrm{total}}(\mathrm{ts}) - e_{\mathrm{total}}(\mathrm{off}) = e_{\mathrm{disp}}\).

Offline re-pin (human only; never ctest):

```bash
# Build libmbd (once); pin isolated free-atom TS on the water dimer
# (see tools/cp2k_oracle/inputs/gpw-ts-ice/water_dimer_ts.note).
# Isolated: Dispersion energy = -5.9764872295816794e-04 Eh
# Periodic pure-RS hard gate: -6.099630773386013e-04 Ha (REFERENCE.md).
```

## Performance

Source: `REFERENCE.md` (queue / handoff: **perf_exempt**).

TS evaluation itself is \(O(N^2 \times N_{\mathrm{images}})\) over a 6-atom
cell (microseconds). Full GPW SCF wall is dominated by the electronic
structure, not TS. A fair **dispersion-only** microbenchmark vs libMBD CLI is
not the product gate; a full twin SCF ratio vs CP2K is unavailable (no stock
`TYPE TS`) and would be unfair for a geometry-only hard gate.

| Twin | \(T_{\mathrm{oracle}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | --------------------------: | ----------------------: | ----: | ---- | ------ |
| Full QS (context) | n/a (CP2K lacks stock TS) | — (not fair-twinned) | **perf_exempt** | — | Twin notes retained under `tools/cp2k_oracle/inputs/gpw-ts-ice/` |
| E_disp product gate | n/a | short SCF + TS (reg) | **perf_exempt** | — | Geometry-only deliverable |

**perf_exempt = true.** Reason (from REFERENCE / handoff): primary deliverable
is E_disp pin parity; CP2K QS wall is unavailable (no stock TS) and a full twin
SCF ratio is unfair for a geometry-only hard gate (same policy as
`gpw-pbe-d3bj-ice` / `gpw-d4-water`). Re-measure a strict full-SCF twin later if
product policy requires \(r = T_{\mathrm{ATV}}/T_{\mathrm{oracle}} \le 10\) on a
host with an external TS-capable engine.

## Limitations

- **Cell is a water dimer**, not bulk ice Ih — sufficient for periodic TS
  lattice-sum parity; not a structural ice benchmark.
- **E_disp hard gate only** — absolute GPW \(e_{\mathrm{tot}}\) vs CP2K QS is
  not asserted (and CP2K stock TS is missing).
- **Free-atom volumes only** — no Hirshfeld \(V/V_{\mathrm{free}}\) from density.
- **Two-body only** — no MBD / many-body screening (queue `gpw-mbd-ice`).
- **Pure real-space lattice sum** — not Ewald; documented nano-Ha offset vs
  libMBD Ewald on this cell.
- **PBE damping parameters only** on this pin (\(s_R=0.94\)); other functionals’
  \(s_R\) are not e2e-gated here.
- **`d3`, `d3bj`, `d4`, and `ts` mutually exclusive** — any two true throws at run.
- **Oracle split:** GPW pin is libMBD / pure-RS; D3(BJ) is CP2K; D4 is dftd4 —
  do not mix oracles when comparing.
- Wall **perf_exempt** for this wave; not a speed optimization target.

## References

- Offline pin: `regressions/gpw/gpw-ts-ice/REFERENCE.md`
- Oracle notes: `tools/cp2k_oracle/inputs/gpw-ts-ice/water_dimer_ts.note`
- Oracle rules: `tools/cp2k_oracle/README.md`
- Roadmap: `docs/roadmap/dft.md` §8 Dispersion
- Related D3(BJ) GPW pin: [gpw-pbe-d3bj-ice](gpw-pbe-d3bj-ice.md)
- Related D4 GPW pin: [gpw-d4-water](gpw-d4-water.md)
- GPW narrative chapter: `docs/tutorial/ch09-gpw.md`
- Closed spec (git history): `.claude/specs/gpw-ts-ice/` (deleted at done;
  harness wave 2026-07-25)
- A. Tkatchenko, M. Scheffler, *Phys. Rev. Lett.* **102**, 073005 (2009)
  (TS dispersion)
- libMBD (canonical TS/MBD library; free-atom `ts_vdw_params` tables)
