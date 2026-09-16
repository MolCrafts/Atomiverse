# GPW PBE-MBD@rsSCS — free-atom water dimer (ice-relevant H-bond)

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `gpw-mbd-ice` |
| Regression | `regressions/gpw/gpw-mbd-ice/` |
| Public API | `cpu::gpw::GpwDft::mbd` / `cpu::gpw::GPWEvaluator::mbd`; `GpwResult::e_disp` [Ha] |
| Roadmap | `docs/roadmap/dft.md` §8 Dispersion |

## Summary

**Many-body dispersion MBD@rsSCS** (range-separated self-consistent screening)
is wired into the **Γ-point GPW** path as an additive London correction on top
of PBE, with **free-atom** volume ratios \(V/V_{\mathrm{free}}=1\). This product
pin uses a **water-dimer cell** (ice-relevant H-bonded pair in a 12 Å cubic box —
not a full ice Ih supercell) and gates the **dispersion energy** against offline
**libMBD** (Fortran `cmbd_mbd_scs_energy`), the same external backend CP2K uses
when hosts link libMBD for MBD (stock CP2K 2026.1 `PAIR_POTENTIAL TYPE` has no
MBD — only DFTD2/D3/D3BJ/D4).

Public surface: bool members `mbd` on `cpu::gpw::GpwDft` and
`cpu::gpw::GPWEvaluator` (no `*Config` bag). Dispersion is reported on
`GpwResult::e_disp` [Hartree] and folded into `e_total`. Mutually exclusive with
`d3` / `d3bj` / `d4` / `ts`. Product path is **isolated** free-atom MBD@rsSCS
(geometry-only; ignores the SCF lattice). Product gate:
`regressions/gpw/gpw-mbd-ice` (`reg_gpw-mbd-ice`). Wall ratio is
**perf_exempt** (E_disp pin; full QS twin wall unfair — see Performance).

## Method

**MBD@rsSCS** (Ambrosetti et al., *J. Chem. Phys.* **140**, 18A508 (2014);
libMBD: Hermann et al., *J. Chem. Phys.* **159**, 174802 (2023)):

1. **Range-separated self-consistent screening** of dynamical polarizabilities
   on a frequency grid (Fermi × Gaussian-damped dipole tensor, \(\beta\) damping).
2. **Long-range MBD Hamiltonian** diagonalization → zero-point energy difference
   relative to the non-interacting atomic oscillators.

**PBE defaults** (product path):

| Parameter | Value | Unit / note |
| --------- | ----: | ----------- |
| \(\beta\) | 0.83 | range separation |
| Fermi \(a\) | 6 | damping steepness |
| \(V/V_{\mathrm{free}}\) | 1 | free-atom (no Hirshfeld volumes) |
| Frequency grid | \(n=15\), \(L=0.6\) | libMBD `freq_grid` |

**Free-atom parameters** (libMBD `ts_vdw_params`, in-tree
`basis_data/ts_tables.h`):

| Z | \(\alpha_0\) [Bohr³] | \(C_6\) [Ha·Bohr⁶] | \(R_{\mathrm{vdW}}\) [Bohr] |
| - | -------------------: | -----------------: | --------------------------: |
| H | 4.5 | 6.5 | 3.1 |
| O | 5.4 | 15.6 | 3.19 |

Characteristic frequencies \(\omega = \tfrac{4}{3} C_6 / \alpha_0^2\). After
rsSCS, \(C_6^{\mathrm{rsscs}} = \tfrac{3}{\pi}\sum_u w(u)\,\alpha(u)^2\) and
\(R_{\mathrm{vdW}}\) is rescaled by \((\alpha(0)/\alpha_0)^{1/3}\).

**Product path = isolated.** The hard gate evaluates MBD with
`periodic=false` on the nuclear coordinates, even when the host GPW SCF is
periodic (`nonperiodic = false`, cell ABC 12 Å). Periodic Ewald + k-mesh MBD
is a follow-on.

**Electronic host.** Dispersion is geometry-only; the regression still runs a
short Γ-RKS GPW (PBE, `DZVP-MOLOPT-SR-GTH` + `GTH-PBE-q*`) so `e_disp` is
materialized on a real `GpwResult` and included in `e_total`. Absolute QS
\(E_{\mathrm{tot}}\) micro-Ha parity is **out of scope** (OT/grid/DIIS
differences). E_disp itself is **basis-independent**.

**Key exclusions.**

- Full ice Ih supercell / phonons
- Hirshfeld density volumes (volume-ratio scaling exists on the internal API
  but product pin is free-atom)
- Periodic Ewald MBD / k-mesh MBD
- MBD forces (energy-only v1)
- DFT-D3 / D3(BJ) / D4 / pairwise TS on this pin (mutually exclusive flags)
- CUDA GPW
- Absolute \(e_{\mathrm{tot}}\) pin vs CP2K QS
- Nonlocal vdW-DF (separate slug `gpw-vdw-df-ice`)

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

// Geometry [Å] → Bohr — CP2K / libMBD water dimer (REFERENCE.md).
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
gpw.mbd = true;        // mutually exclusive with d3 / d3bj / d4 / ts

gpw.build(atoms);
const auto& res = gpw.run(atoms);
// res.e_disp [Ha] within 1e-8 of offline libMBD pin
// res.e_total includes e_disp
```

Driver / Frame path (component; same member):

```cpp
// kinds = {O, H} Kind{Element, GaussianBasis, GthPotential}
ATV::cpu::gpw::GPWEvaluator eval(kinds, /*cutoff_ry=*/300.0, /*max_iter=*/200,
                                 /*tol=*/1e-7, /*mix=*/0.3,
                                 ATV::cpu::gpw::XcKind::PBE,
                                 /*nonperiodic=*/false);
eval.mbd = true;
// single_point(frame, eval) or bind on Evaluate / MD / Minimize force slot
// eval.last_result().e_disp  [Hartree]
// Public Frame energy remains eV via meta kEnergy; internal Ha on GpwResult.
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
  cmake --build build --target reg_gpw-mbd-ice -j$(nproc)
  ./build/regressions/reg_gpw-mbd-ice
'
```

| Path | Role |
| ---- | ---- |
| `include/atomiverse/cpu/gpw/gpw_solver.h` | `GpwDft`, `GpwResult::e_disp`, `mbd` |
| `include/atomiverse/cpu/gpw/gpw_evaluator.h` | `GPWEvaluator::mbd` |
| `include/atomiverse/cpu/gpw/internal/mbd_dispersion.h` | `mbd_rsscs_pbe` (private engine) |
| `include/atomiverse/cpu/gpw/basis_data/ts_tables.h` | Free-atom \(\alpha_0\), \(C_6\), \(R_{\mathrm{vdW}}\) |
| `src/cpu/gpw/internal/mbd_dispersion.cpp` | rsSCS + MBD Hamiltonian |
| `regressions/gpw/gpw-mbd-ice/reg_gpw-mbd-ice.cpp` | Product gate |
| `regressions/gpw/gpw-mbd-ice/REFERENCE.md` | Offline pin + perf_exempt |
| `tools/cp2k_oracle/inputs/gpw-mbd-ice/` | Oracle notes (libMBD twin) |

## Validation (libMBD / CP2K MBD backend)

Source:
[`regressions/gpw/gpw-mbd-ice/REFERENCE.md`](../../../regressions/gpw/gpw-mbd-ice/REFERENCE.md).
Hard pin: offline **libMBD** Fortran `cmbd_mbd_scs_energy` (git `88069c6`),
geometry converted with ATV Å→Bohr \(1/0.52917720859\). Same policy as
`gpw-ts-ice`: CP2K 2026.1 stock pair types lack MBD; hosts that enable VDW via
libMBD share this engine. Twin notes:
`tools/cp2k_oracle/inputs/gpw-mbd-ice/water_dimer_mbd.note`. External engine
pin — not a golden of Atomiverse’s own output; **never** launch live CP2K or
libMBD from ctest.

**Setup:**

| Item | Value |
| ---- | ----- |
| System | Water dimer (ice-relevant H-bond pair) |
| Geometry [Å] | O (0,0,0); H (0.758602, 0, ±0.504284); O (3,0,0); H (3.758602, 0, ±0.504284) |
| Cell (SCF only) | ABC **12.0 12.0 12.0 Å**, PERIODIC XYZ |
| MBD geometry | **Isolated** free-atom on the same nuclear coordinates |
| Basis / PP | `DZVP-MOLOPT-SR-GTH` + `GTH-PBE-q6` (O) / `GTH-PBE-q1` (H) |
| XC + disp | PBE + MBD@rsSCS (\(\beta=0.83\), \(a=6\), free-atom) |
| Occupations | RKS, 16 valence e⁻ → `n_occ = 8` |
| Pin line | `E_disp = -1.4928804171407961e-03` Ha |

| Quantity | Atomiverse / pin | libMBD (offline) | Tol | Notes |
| -------- | ---------------: | ---------------: | --- | ----- |
| \(E_{\mathrm{disp}}\) [Ha] | −1.4928804171407961×10⁻³ | −1.4928804171407961×10⁻³ | \(1\times 10^{-8}\) Ha | Hard gate |
| pure-Python `pymbd` (same ATV Å→Bohr) | — | −1.4928804171425725×10⁻³ | — | Cross-check \(\sim 2\times 10^{-15}\) Ha |
| pure-Python (libmbd 0.529177249) | — | −1.4928806007885598×10⁻³ | — | Conversion-sensitivity note |
| \(E_{\mathrm{tot}}\) QS [Ha] | (not gated) | n/a (no stock CP2K QS MBD twin) | — | Context only |

**Tolerance rationale.** Matching free-atom tables, \(\beta\)/\(a\), frequency
grid, and isolated dipole tensors track libMBD Fortran to \(\sim 10^{-15}\) Ha;
regression hard gate is \(10^{-8}\) Ha. Unit tests assert the standalone
`mbd_rsscs_pbe` kernel at \(10^{-12}\) Ha.

## Performance

Source: `REFERENCE.md` (queue / handoff: **perf_exempt**).

MBD evaluation is \(O(N^3)\) eigensolve of a \(3N\times 3N\) matrix on a 6-atom
system (microseconds). Full GPW SCF wall is dominated by the electronic
structure, not MBD. A fair **dispersion-only** microbenchmark vs CP2K is not
meaningful without a linked-libMBD host twin; a full twin SCF ratio is not the
product gate.

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | ------------------------: | ----------------------: | ----: | ---- | ------ |
| Full QS (context) | n/a (stock no MBD) | — | **perf_exempt** | — | Hard pin is libMBD E_disp |
| E_disp product gate | n/a | short SCF + MBD (reg) | **perf_exempt** | — | Geometry-only deliverable |

**perf_exempt = true.** Reason (from REFERENCE / handoff wave 2 2026-07-25):
primary deliverable is E_disp pin parity vs libMBD; same policy as
`gpw-ts-ice` / `gpw-pbe-d3bj-ice` / `gpw-d4-water`. Re-measure a strict
full-SCF twin later if product policy requires
\(r = T_{\mathrm{ATV}}/T_{\mathrm{oracle}} \le 10\).

## Limitations

- **Cell is a water dimer**, not bulk ice Ih — sufficient for free-atom
  MBD@rsSCS parity; not a structural ice benchmark.
- **Isolated MBD only** — SCF may be periodic, but MBD ignores the lattice;
  periodic Ewald MBD is follow-on.
- **Free-atom volumes** — no Hirshfeld \(V/V_{\mathrm{free}}\) from density.
- **Energy-only** — no MBD analytic forces in v1.
- **E_disp hard gate only** — absolute GPW \(e_{\mathrm{tot}}\) vs any QS twin
  is not asserted.
- **`d3` / `d3bj` / `d4` / `ts` / `mbd` mutually exclusive** — more than one
  true throws at run.
- **Stock CP2K has no MBD pair type** — pin is libMBD (CP2K’s external MBD
  backend when linked), documented under `tools/cp2k_oracle/inputs/gpw-mbd-ice/`.
- Wall **perf_exempt** for this wave; not a speed optimization target.
- Nonlocal **vdW-DF** remains open (`gpw-vdw-df-ice`).

## References

- Offline pin: `regressions/gpw/gpw-mbd-ice/REFERENCE.md`
- Oracle notes: `tools/cp2k_oracle/inputs/gpw-mbd-ice/`
- Roadmap: `docs/roadmap/dft.md` §8 Dispersion
- GPW narrative chapter: `docs/tutorial/ch09-gpw.md`
- Sibling pins: `gpw-ts-ice`, `gpw-pbe-d3bj-ice`, `gpw-d4-water`
- Closed spec (git history): `.claude/specs/gpw-mbd-ice/` (deleted at done)
- A. Ambrosetti et al., *J. Chem. Phys.* **140**, 18A508 (2014) (MBD@rsSCS)
- J. Hermann et al., *J. Chem. Phys.* **159**, 174802 (2023) (libMBD)
