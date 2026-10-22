# GPW free-atom TS ionic forces — water dimer / PBE

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `gpw-ts-forces` |
| Public API | `cpu::gpw::GpwDft::ts` (bound by `cpu::gpw::GPWEvaluator`); `GpwResult::e_disp` [Ha]; force assembly via `GpwDft::forces(frame)` / `GPWEvaluator` `compute_forces` |
| Roadmap | `docs/roadmap/dft.md` §8 Dispersion |

## Summary

**Tkatchenko–Scheffler (TS)** pairwise London dispersion with **free-atom**
volume ratios \(V/V_{\mathrm{free}}=1\) is folded into the **Γ-point GPW**
force path on top of PBE. This product pin uses the same **periodic
water-dimer** cell as `gpw-ts-ice` / `gpw-mbd-forces` (12 Å cubic, ice-relevant
H-bonded pair — not bulk ice Ih) and gates both **dispersion energy** and
**analytic ionic forces** \(F_A = -\partial E_{\mathrm{disp}}/\partial R_A\).

**Oracle policy.** CP2K 2026.1 stock `PAIR_POTENTIAL TYPE` is only
`DFTD2|DFTD3|DFTD3(BJ)|DFTD4` — there is **no** stock `TYPE TS`. The hard pin is
offline **libMBD** free-atom TS energy (`cmbd_ts_energy`) plus a **force
dual-gate** (product analytic \(\nabla E_{\mathrm{disp}}\) vs independent
central finite differences of that energy) and a **product wiring** check
(\(F(\mathrm{ts})-F(\mathrm{plain})\) matches standalone `ts_pbe` forces). Twin
notes live under CP2K oracle. Same policy as
`gpw-ts-ice` / `gpw-mbd-forces`.

Public surface: bool member `ts` on `cpu::gpw::GpwDft`, bound by
`cpu::gpw::GPWEvaluator` (no `*Config` bag). Dispersion is reported on
`GpwResult::e_disp` [Hartree] and folded into `e_total`; force assembly adds
geometry-only TS forces to electronic Hellmann–Feynman / Pulay terms.
Mutually exclusive with `d3` / `d3bj` / `d4` / `mbd` / `xdm` / nonlocal vdW.
Product gate: `gpw-ts-forces` (`reg_gpw-ts-forces`). Wall ratio
is **perf_exempt** (E_disp + force dual-gate; no stock CP2K TS twin — see
Performance).

## Method

**Tkatchenko–Scheffler pairwise TS** (Tkatchenko & Scheffler, *Phys. Rev. Lett.*
**102**, 073005 (2009)), free-atom limit matching libMBD’s real-space
`get_ts_energy` path (Fermi damping, PBE \(s_R = 0.94\), \(d = 20\)):

\[
E_{\mathrm{disp}}
= -\sum_{i\le j}\sum_{\mathbf{T}}{}^{\prime}
  C_6^{ij}\,
  \frac{f_{\mathrm{damp}}(R_{ij,T})}{R_{ij,T}^{6}},
\]

\[
C_6^{ij}
= \frac{2\,C_6^{i}\,C_6^{j}}
       {(\alpha_j/\alpha_i)\,C_6^{i} + (\alpha_i/\alpha_j)\,C_6^{j}},
\quad
f_{\mathrm{damp}}
= \frac{1}{1+\exp\!\bigl(-d\bigl(R/(s_R R_0^{ij})-1\bigr)\bigr)},
\quad
R_0^{ij} = R_{\mathrm{vdW}}^{i}+R_{\mathrm{vdW}}^{j}.
\]

The prime excludes the singular \(R=0\) self term. Free-atom \(\alpha_0\),
\(C_6\), and \(R_{\mathrm{vdW}}\) come from in-tree
`basis_data/ts_tables.h` (libMBD `ts_vdw_params`). Optional per-atom volume
ratios scale free-atom parameters as in TS
(\(\alpha\leftarrow\alpha_{\mathrm{free}}\,v\),
\(C_6\leftarrow C_6^{\mathrm{free}}\,v^{2}\),
\(R\leftarrow R_{\mathrm{free}}\,v^{1/3}\)); **this pin uses empty ratios**
⇒ all \(v=1\) (free-atom). Hirshfeld density volumes
(`GpwDft::hirshfeld`) are a separate product path.

**Forces.** Analytic product kernel:

\[
\mathbf{F}_A = -\frac{\partial E_{\mathrm{disp}}}{\partial\mathbf{R}_A}
\quad[\mathrm{Ha/Bohr}].
\]

Dual-gated against central FD of the same geometry-only energy
(\(h=10^{-5}\) Bohr). On the product path, `GpwDft::forces(frame)` with
`GpwDft::ts = true` **adds** these forces to the
electronic HF / Pulay / grid-core terms.

**PBE-TS parameters (binding):**

| Parameter | Value | Unit / notes |
| --------- | ----: | ------------ |
| \(s_R\) | 0.94 | PBE range scaling |
| \(d\) | 20 | Fermi steepness |
| Pair cutoff (periodic) | **50 Å** | pure real-space lattice sum |
| Volume ratios | all 1 | free-atom foothold |
| XC host | PBE | geometry-only disp; short Γ-RKS SCF materializes `e_disp` |

**Electronic host.** Dispersion is geometry-only; the regression still runs a
short Γ-RKS GPW (PBE, `DZVP-MOLOPT-SR-GTH` + `GTH-PBE-q*`) so `e_disp` is
materialized on a real `GpwResult`, included in `e_total`, and force wiring
can be checked on the same density. Absolute QS \(E_{\mathrm{tot}}\) micro-Ha
parity vs CP2K is **out of scope** (no stock TS twin; OT/grid/DIIS residuals
anyway).

**Key exclusions.**

- Hirshfeld volume-scaled TS (`gpw-mbd-hirshfeld` family) as the hard pin
- MBD@rsSCS many-body forces (`gpw-mbd-forces`) — different method
- Full ice Ih supercell / phonons
- Absolute \(e_{\mathrm{tot}}\) pin vs CP2K QS
- Stock CP2K `TYPE TS` twin (does not exist in 2026.1)
- k-point force assembly as the e2e gate (unit / related rows may cover multi-k fold)
- CUDA GPW
- Live libMBD / CP2K from ctest

## Implementation

Public surface only (`GpwDft` / `GPWEvaluator` members; no `*Config`, no
`make_*` / `build_*` factories). Regression uses the concrete solver +
standalone `ts_pbe` dual-gate:

```cpp
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/cpu/gpw/internal/ts_dispersion.h"
#include "atomiverse/molrs.hpp"

using ATV::cpu::gpw::GaussianBasis;
using ATV::cpu::gpw::Kind;
using ATV::cpu::gpw::GpwDft;
using ATV::cpu::gpw::GthPotential;
using ATV::cpu::gpw::ts_pbe;
using ATV::cpu::gpw::XcKind;

// Geometry [Å] — water dimer (REFERENCE.md).
constexpr double kA2B = 1.0 / 0.52917720859;
// … fill a 6-atom cpu::Frame (2×O + 4×H; positions [Å], Z in kAtom_type)
//   and kinds = {O, H}: Kind{Element, DZVP-MOLOPT-SR-GTH, GTH-PBE-q*} …
// … Z, pos [Bohr] for ts_pbe …

// Isolated free-atom TS energy + analytic forces (libMBD pin dual-gate)
const double cell0[3] = {0.0, 0.0, 0.0};
const auto ts_iso = ts_pbe(Z, pos, cell0, /*periodic=*/false);
// |ts_iso.energy + 5.9764872295816794e-4| ≤ 1e-8 Ha
// max |F − FD(h=1e-5)| ≤ 1e-6 Ha/Bohr; |∑F| ≤ 1e-8 Ha/Bohr

// Periodic pure-RS pin (rcut 50 Å)
const double L = 12.0 * kA2B;
const double cell[3] = {L, L, L};
const auto ts_per = ts_pbe(Z, pos, cell, /*periodic=*/true, 50.0 * kA2B);
// |ts_per.energy + 6.099630773386013e-4| ≤ 1e-8 Ha

frame.box = ATV::cpu::Box::cubic(12.0);  // Å
GpwDft gpw{kinds, 200.0, XcKind::PBE};
gpw.n_occ = 8;         // 16 valence e⁻ → 8 doubly occupied
gpw.max_iter = 80;
gpw.tol = 1e-6;
gpw.mix = 0.4;
gpw.nonperiodic = false;
gpw.local_pp_on_grid = true;
gpw.ts = true;         // exclusive vs d3 / d3bj / d4 / mbd / xdm / …

gpw.build(frame);
const auto& res = gpw.forces(frame);  // SCF + analytic forces → kAtom_f* [eV/Å]
// res.e_disp [Ha] within 1e-8 of periodic pin; folded into res.e_total

// F(ts) − F(plain), converted to Ha/Bohr (÷ units::kHartreePerBohrInEvPerAngstrom),
// matches ts_per.forces within 1e-7 Ha/Bohr
```

Driver / Frame path (component; same member):

```cpp
// kinds = {O, H} Kind{Element, GaussianBasis, GthPotential}
ATV::cpu::gpw::GpwDft dft{kinds, 300.0, ATV::cpu::gpw::XcKind::PBE};
dft.max_iter = 200;  dft.tol = 1e-7;  dft.mix = 0.3;
dft.nonperiodic = false;
dft.ts = true;
ATV::cpu::gpw::GPWEvaluator eval(std::move(dft), /*compute_forces=*/true);
// single_point(frame, eval) or bind on Evaluate / MD / Minimize force slot
// eval.last_result().e_disp  [Hartree]
// frame.span(kAtom_f*) [eV/Å] includes TS contribution
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
  cmake --build build --target reg_gpw-ts-forces -j$(nproc)
  reg_gpw-ts-forces
'
```

| Path | Role |
| ---- | ---- |
| `gpw_solver.h` | `GpwDft`, `GpwResult::e_disp`, `ts` |
| `gpw_evaluator.h` | `GPWEvaluator` (binds the `GpwDft`; `compute_forces`) |
| `ts_dispersion.h` | `ts_pbe`, `TsResult` |
| `ts_tables.h` | Free-atom \(\alpha_0\), \(C_6\), \(R_{\mathrm{vdW}}\) |
| `src/cpu/gpw/internal/gpw_forces.cpp` | Product force assembly folds TS |
| `tests/cpu/gpw/test_cpu_gpw_ts.cpp` | Unit energy + force + wiring gates |
| `reg_gpw-ts-forces.cpp` | Product gate |
| case `gpw-ts-forces` pin | Offline pins + perf_exempt |
| CP2K oracle | Geometry + oracle policy note |

## Validation (libMBD / force dual-gate)

Source:
case `gpw-ts-forces` pin.
Oracle: offline **libMBD** free-atom TS energy (`cmbd_ts_energy`, git
`88069c6`, ATV Å→Bohr) + **analytic-vs-FD dual-gate** + product wiring.
External engine pin — not a golden of Atomiverse’s own output; **never**
launch live libMBD or CP2K from ctest.

**Setup:**

| Item | Value |
| ---- | ----- |
| System | Water dimer (ice-relevant H-bond pair) |
| Geometry [Å] | O (0,0,0); H (0.758602, 0, ±0.504284); O (3,0,0); H (3.758602, 0, ±0.504284) |
| Cell (periodic pin + SCF) | ABC **12.0 12.0 12.0 Å**, PERIODIC XYZ |
| Isolated dual-gate | Same nuclei, no lattice images |
| Basis / PP | `DZVP-MOLOPT-SR-GTH` + `GTH-PBE-q6` (O) / `GTH-PBE-q1` (H) (E_disp / F_TS basis-independent) |
| XC + disp | PBE host + free-atom TS (\(s_R=0.94\), \(d=20\), rcut 50 Å) |
| Occupations | RKS, 16 valence e⁻ → `n_occ = 8` |
| Å→Bohr | `1/0.52917720859` (ATV) |

| Quantity | Atomiverse / pin | Offline source | Tol | Notes |
| -------- | ---------------: | -------------- | --- | ----- |
| \(E_{\mathrm{disp}}\) isolated [Ha] | −5.9764872295816794×10⁻⁴ | libMBD `cmbd_ts_energy` | \(1\times 10^{-8}\) Ha | Hard gate |
| \(E_{\mathrm{disp}}\) periodic [Ha] | −6.099630773386013×10⁻⁴ | pure-RS / `ts_pbe` (`gpw-ts-ice`) | \(1\times 10^{-8}\) Ha | Hard gate; also `GpwDft::e_disp` |
| max\|F_product − F_fd\| [Ha/Bohr] | ≤ \(1\times 10^{-6}\) | FD \(h=10^{-5}\) Bohr of \(E_{\mathrm{disp}}\) | \(1\times 10^{-6}\) | Isolated dual-gate |
| \(\|\sum_A\mathbf{F}_A\|\) [Ha/Bohr] | ≤ \(1\times 10^{-8}\) | translation | \(1\times 10^{-8}\) | Isolated |
| max\|F(ts)−F(plain)−F_geom\| [Ha/Bohr] | ≤ \(1\times 10^{-7}\) | product wiring | \(1\times 10^{-7}\) | Same density; SCF short-host |
| \(E_{\mathrm{tot}}\) QS [Ha] | finite; includes \(e_{\mathrm{disp}}\) | (no CP2K TS twin) | — | Not absolute-parity gated |

**Tolerance rationale.** Matching free-atom tables, Fermi damping, and pure
real-space lattice sums track libMBD to nano-Hartree on this cell. Force
agreement at \(10^{-6}\) Ha/Bohr is the \(O(h^2)\) residual of central FD with
\(h=10^{-5}\) Bohr on a smooth pairwise energy — not chemical force accuracy
vs a QS twin. Wiring at \(10^{-7}\) Ha/Bohr asserts that force assembly adds
exactly the geometry-only TS contribution without double-counting or omission.

Offline re-pin notes (human only; never ctest):

```text
# <CP2K oracle input for this case>
# Isolated: libMBD cmbd_ts_energy on the six nuclei (ATV Å→Bohr)
# Periodic: pure-RS rcut 50 Å (same as gpw-ts-ice)
# Forces: product ∇E vs independent FD; optional libMBD C gradients on host
```

## Performance

Source: pin notes (queue / handoff: **perf_exempt**).

TS evaluation is \(O(N^2\times N_{\mathrm{images}})\) over a 6-atom cell
(microseconds). Full GPW SCF wall dominates when the short host SCF is run.
There is **no** stock CP2K `TYPE TS` twin for a fair
\(r=T_{\mathrm{ATV}}/T_{\mathrm{CP2K}}\) ratio.

| Twin | \(T_{\mathrm{oracle}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | --------------------------: | ----------------------: | ----: | ---- | ------ |
| Full QS ENERGY_FORCE | n/a (CP2K has no stock TS) | — | **perf_exempt** | — | Policy note under CP2K oracle |
| E_disp + force dual-gate | n/a | short SCF + TS (reg) | **perf_exempt** | — | Geometry-only deliverable + wiring |

**perf_exempt = true.** Reason (from REFERENCE / handoff wave 11): primary
deliverable is E_disp pin parity + force dual-gate + product wiring; CP2K has
no stock TS twin for a fair wall ratio (same policy as `gpw-ts-ice` /
`gpw-mbd-forces` / `gpw-xdm-forces`). Re-measure a strict full-SCF twin only if
a future CP2K build exposes a documented TS pair potential and product policy
requires \(r\le 10\).

## Limitations

- **Cell is a water dimer**, not bulk ice Ih — sufficient for free-atom TS
  energy/force parity; not a structural ice benchmark.
- **Free-atom volumes only** on this pin — Hirshfeld \(V/V_{\mathrm{free}}\)
  scaling is a separate member / queue family.
- **E_disp + force dual-gate / wiring** — absolute GPW \(e_{\mathrm{tot}}\) vs
  CP2K QS is not asserted (and stock CP2K has no TS).
- **Pairwise only** — no MBD many-body screening on this slug
  (`gpw-mbd-forces` / `gpw-mbd-ice`).
- **`ts` mutually exclusive** with `d3` / `d3bj` / `d4` / `mbd` / `xdm` /
  nonlocal vdW flags — any two true throws at run.
- **Oracle split:** energy pin is libMBD; force pin is analytic-vs-FD of ATV’s
  (libMBD-matched) energy — not a CP2K FORCE_EVAL absolute force table.
- Wall **perf_exempt** for this wave; not a speed optimization target.
- Frame forces are eV/Å on the public Evaluate path; literature / dual-gate
  comparison uses Ha/Bohr.

## References

- Offline pin: case `gpw-ts-forces` pin
- Oracle notes: CP2K oracle,
  CP2K oracle
- Roadmap: `docs/roadmap/dft.md` §8 Dispersion
- Related energy pin: `gpw-ts-ice` (E_disp only)
- Related many-body forces: `gpw-mbd-forces`
- Related XDM forces: `gpw-xdm-forces`
- Related D4 forces: `gpw-d4-forces`
- GPW analytic forces (no disp): [gpw-forces-water](gpw-forces-water.md)
- GPW D4 energy: [gpw-d4-water](gpw-d4-water.md)
- GPW narrative chapter: `docs/tutorial/ch09-gpw.md`
- Closed spec (git history): `.claude/specs/gpw-ts-forces/` (deleted at done)
- A. Tkatchenko, M. Scheffler, *Phys. Rev. Lett.* **102**, 073005 (2009)
- libMBD (TS tables / `cmbd_ts_energy` energy twin)
