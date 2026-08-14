# QEq fluctuating-charge equilibration — LAMMPS `fix qeq/point` twin

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `qeq-fluctuating-charge` |
| Regression | `regressions/md/qeq-fluctuating-charge/` |
| Public API | `cpu::md::QEq` (Component on `cpu::md::MD`, default `pre_force` order −50) |
| Roadmap / queue | `docs/developer/metalwalls-method-queue.md` → wave 5 (2026-07-27); map §2 "Fluctuating-charge / QEq" |
| Oracle | **LAMMPS** QEQ package (`fix qeq/point`), offline pin — ratio **0.45** |

## Summary

**Charge equilibration** (electronegativity equalization, Rappé & Goddard
1991) for electrolyte / molecular systems: per-atom fluctuating charges
minimize a second-order charge energy with per-type electronegativity
\(\chi\) [eV] and hardness \(\eta\) [eV/e²] under a total-charge constraint.
This closes the MetalWalls capability map's open "fluctuating-charge / QEq"
polarization slice, and is deliberately **not** the constant-potential
electrode family (`ConstPotBO` / `ConstPotCG` / `ConstPotMaZe` solve fixed-Ψ
Gaussian electrode charges); QEq equilibrates **every parameterized atom**
against its own electronegativity.

Public surface is `cpu::md::QEq` — a **Component** bound at
`md::Phase::pre_force` (order −50, the same slot as the electrode solvers),
never a Driver category and never inside `KSpace*`. Each launch it solves the
equilibration and writes the charges to `kAtom_charge` **before** the pair /
k-space force components run, so downstream Coulomb styles see equilibrated
charges every step. The regression pins per-atom charges, the QEq objective,
and the downstream `coul/cut` energy against an offline LAMMPS
`fix qeq/point` run, plus a fair 1000-solve perf twin (ratio 0.45 — ATV
faster).

## Method

**Physics.** Rappé, Goddard, *J. Phys. Chem.* **95**, 3358 (1991), DOI
[10.1021/j100161a070](https://doi.org/10.1021/j100161a070). The energy model
is the exact LAMMPS `fix qeq/point` identity (verified against LAMMPS source
`src/QEQ/fix_qeq_point.cpp`):

\[
E(q)=\sum_i\Bigl(\chi_i q_i+\tfrac12\eta_i q_i^2\Bigr)
 +\sum_{i<j,\;r_{ij}\le r_c}\frac{q_i q_j}{r_{ij}},
\qquad \sum_i q_i = q_{\mathrm{tot}}
\]

with the literal LAMMPS convention of a **bare** \(1/r\) coupling: no qqrd2e
prefactor on the off-diagonal (`compute_H` fills `0.5/r` on a full neighbor
list). The physical Coulomb energy of the resulting charges is computed
downstream by the pair style with \(k_e = 14.399645\) eV·Å/e²
(`units::coulomb`). Stationarity gives the linear system

\[
(\eta_i\delta_{ij}+J_{ij})\,q=-\chi+\lambda\mathbf 1,
\]

solved with the same two-vector scheme as LAMMPS `calculate_Q`:
\(Hs=-\chi\), \(Ht=-\mathbf 1\), then
\(q=s-((\Sigma s-q_{\mathrm{tot}})/\Sigma t)\,t\) — so \(\Sigma q =
q_{\mathrm{tot}}\) holds exactly, not to solver tolerance. Atomiverse factors
the dense matrix once per launch (Cholesky) and substitutes both right-hand
sides; distances use the frame box minimum image. The matrix is rebuilt every
launch, so charges follow the moving geometry in MD.

### System (H₂CO organic toy)

Formaldehyde-like 4-atom cluster, LAMMPS `boundary f f f`, 40 Å cube,
`units metal` (eV / Å / e):

| id | type | element | x [Å] | y [Å] | z [Å] |
| -- | ---- | ------- | ----- | ----- | ----- |
| 1 | 1 | C | 20.0 | 20.0 | 20.0 |
| 2 | 2 | O | 20.0 | 20.0 | 21.205 |
| 3 | 3 | H | 20.943 | 20.0 | 19.412 |
| 4 | 3 | H | 19.057 | 20.0 | 19.412 |

QEq parameters (`param.qeq`; χ [eV], η [eV/e²]):

| type | χ [eV] | η [eV/e²] |
| ---- | -----: | --------: |
| 1 (C) | 5.343 | 10.126 |
| 2 (O) | 8.741 | 13.364 |
| 3 (H) | 4.528 | 13.8904 |

Oracle command: `fix q all qeq/point 1 10.0 1.0e-10 500 param.qeq` with
`pair_style coul/cut 10.0`, `run 0` (setup equilibrates charges before
output). Inputs preserved at
`tools/metalwalls_oracle/inputs/lammps/qeq-fluctuating-charge/`
(`in.qeq_point`, `in.qeq_perf`, `param.qeq`); procedure in
`tools/metalwalls_oracle/pin_notes_qeq-fluctuating-charge.md`. ctest never
launches LAMMPS.

### Key exclusions (non-goals of this slug)

- Live LAMMPS inside ctest (offline pin only).
- Shielded / Slater kernels (`qeq/shielded`, `qeq/slater`, `qeq/dynamic`) —
  point kernel only; variants are follow-up queue rows if mapped.
- Ewald-summed QEq — primary-image minimum-image coupling (matches the
  non-periodic oracle toy).
- Sparse / CG solve path — dense Cholesky, toy/small-N scope.
- QEq as a Driver, solver class, or part of `KSpace*` — Component only.
- Public `*Config` bags or `make_*` / `build_*` factories.

## Implementation

Public surface only (`cpu::Frame` + `QEq` ctor / members; no Config bag).
Minimal standalone equilibration matching the regression:

```cpp
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/qeq.h"
#include "atomiverse/cpu/frame/frame.h"

using namespace ATV;

cpu::Frame frame(4);                   // H2CO toy: fill x/y/z, type, charge=0
// ... positions / types / masses / tags as in the System table ...
frame.box = cpu::Box::cubic(40.0);

// Default binding is pre_force order −50, before every pair / k-space force.
cpu::md::QEq qeq(/*cutoff [Å]=*/10.0);
qeq.set_type(/*type=*/1, /*chi [eV]=*/5.343, /*eta [eV/e²]=*/10.126); // C
qeq.set_type(2, 8.741, 13.364);                                      // O
qeq.set_type(3, 4.528, 13.8904);                                     // H
// qeq.qtot = 0.0;  // constrained total charge [e] (default 0, LAMMPS behavior)

cpu::md::MD sim;
sim.add_ref(qeq);  // or .on(md::Phase::pre_force).add(std::move(...))
sim.set_evaluator(cpu::PairCoulCut{/*cutoff [Å]=*/10.0});  // sees equilibrated q
sim.set_neighbor(cpu::md::NeighborList<>{10.0, 0.3});
sim.set_integrator(cpu::md::VelocityVerlet{0.5});
sim.build();
sim.run(frame, /*nsteps=*/1);   // charges re-equilibrated every step

const auto& q = qeq.charges();   // [e], frame-row order (also on kAtom_charge)
const double e_qeq = qeq.qeq_energy();   // [eV], bare-1/r kernel objective
const double net = qeq.net_charge();     // [e], equals qtot exactly
```

Architecture rules:

- **Driver stays `cpu::md::MD`.** QEq is a Component; never a new Driver
  category, never a charge solve inside `KSpace*`.
- Parameters are **constructor arguments and public members** (`cutoff`,
  `qtot`) plus `set_type` rows — no `QEqConfig`.
- Frame is the sole I/O currency; equilibrated charges land on
  `kAtom_charge`.
- **No fallback:** every atom type present in the frame must be registered
  via `set_type` — `build()` throws on an unparameterized type; `cutoff ≤ 0`
  and `eta ≤ 0` throw.

Build and run the regression (MD-only configure; CPU login pattern):

```bash
bash -lc '
  module load buildenv-gcc/2026.03-mpich
  export OPENBLAS_ROOT="$OPENBLASDIR"
  export RUSTUP_TOOLCHAIN=stable-x86_64-unknown-linux-gnu
  export LD_LIBRARY_PATH="$LIBRARY_PATH:$LD_LIBRARY_PATH"
  cmake -B build -DCMAKE_BUILD_TYPE=Release \
        -DATV_BUILD_MD=ON \
        -DATV_BUILD_TESTS=ON -DATV_BUILD_REGRESSIONS=ON \
        -DATV_BUILD_HF=OFF -DATV_BUILD_PW=OFF -DATV_BUILD_GPW=OFF \
        -DATV_BUILD_CUDA=OFF -DBLA_VENDOR=OpenBLAS
  cmake --build build --target reg_qeq-fluctuating-charge -j"$(nproc)"
  ./build/regressions/reg_qeq-fluctuating-charge
'
```

| Path | Role |
| ---- | ---- |
| `include/atomiverse/cpu/dynamics/qeq.h` | Public `QEq` (ctor, `set_type`, `cutoff`, `qtot`, observables) |
| `src/cpu/dynamics/qeq.cpp` | Dense \(H=\eta\delta+1/r\) build + Cholesky + two-vector constraint |
| `tests/cpu/dynamics/test_qeq.cpp` | Unit pins: 2-atom closed form, constraint, cutoff decoupling, energy, throws |
| `regressions/md/qeq-fluctuating-charge/reg_qeq-fluctuating-charge.cpp` | Product gate + fair perf twin |
| `tools/metalwalls_oracle/inputs/lammps/qeq-fluctuating-charge/` | Offline LAMMPS inputs (`in.qeq_point`, `in.qeq_perf`, `param.qeq`) |

## Validation (vs LAMMPS `fix qeq/point`)

Source:
[`regressions/md/qeq-fluctuating-charge/REFERENCE.md`](../../../regressions/md/qeq-fluctuating-charge/REFERENCE.md)
(offline pin; **never** live LAMMPS in ctest). Oracle tooling:
[`tools/metalwalls_oracle/README.md`](../../../tools/metalwalls_oracle/README.md).

**Oracle:** LAMMPS 30 Mar 2026 development
(patch_30Mar2026-1074-g7f680de296), QEQ package, run offline 2026-07-27
(`lmp -in in.qeq_point`). Kernel identity verified against LAMMPS source
`src/QEQ/fix_qeq_point.cpp`: off-diagonal bare `1/r` (no qqrd2e), diagonal
η, RHS −χ, constraint `q = s − (Σs/Σt)t`. The `coul/cut` pair energy of the
equilibrated charges *does* use qqr2e = 14.399645 eV·Å/e².

### Accuracy

| Quantity | Atomiverse gate | LAMMPS pin | Tol | Notes |
| -------- | --------------: | ---------: | --- | ----- |
| \(q(\mathrm C)\) [e] | matches pin | **+0.046279970820** | abs ≤ **1e-6** | `charges()[0]` |
| \(q(\mathrm O)\) [e] | matches pin | **−0.230795571743** | abs ≤ **1e-6** | `charges()[1]` |
| \(q(\mathrm H)\) ×2 [e] | matches pin | **+0.092257800461** | abs ≤ **1e-6** | symmetric H pair |
| \(\Sigma q\) [e] | two-vector constraint | **0** | abs ≤ **1e-12** | exact by construction |
| \(E_{\mathrm{QEq}}\) [eV] | `qeq_energy()` | **−0.467311783767** | abs ≤ **1e-6** | bare-1/r kernel objective, derived from pinned q |
| \(E_{\mathrm{coul}}\) (`coul/cut`) [eV] | `PairCoulCut` on equilibrated q | **−0.254699311955** | abs ≤ **1e-6** | LAMMPS thermo `ecoul`, qqr2e = 14.399645 |

Unit tests (`tests/cpu/dynamics/test_qeq.cpp`) additionally pin the two-atom
closed form \(q=(\chi_2-\chi_1)/(\eta_1+\eta_2-2/r)\), constraint
conservation, beyond-cutoff decoupling, the energy closed form, and the
throw contracts.

### Offline pin repro (never ctest)

```bash
cd tools/metalwalls_oracle/inputs/lammps/qeq-fluctuating-charge
lmp -in in.qeq_point   # charges + ecoul pins
lmp -in in.qeq_perf    # 1000-solve fair perf wall
```

## Performance

Source: REFERENCE.md fair perf twin; handoff
[`docs/developer/metalwalls-method-handoff.md`](../../developer/metalwalls-method-handoff.md).
Fair twin is `in.qeq_perf`: 1000 repeated `fix qeq/point` solves (dt = 0)
vs 1000 `QEq::launch` equilibrations (dense rebuild + Cholesky each launch)
on the same geometry, measured 2026-07-27 on the x86_64 epyc login node,
1 thread / 1 proc.

| Twin | \(T_{\mathrm{LAMMPS}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r = T_{\mathrm{ATV}}/T_{\mathrm{LAMMPS}}\) | ≤10× | Caveat |
| ---- | --------------------------: | -----------------------: | -------------------------------------------: | ---- | ------ |
| 1000 QEq solves, H₂CO N=4 | 0.000610 | 0.000275 | **0.45** | yes | micro-workload (both sides sub-ms); ratio reported, not asserted |

| Field | Value |
| ----- | ----- |
| `perf_exempt` | **false** |
| `ratio_atv_over_oracle` | **0.45** |
| `within_10x` | yes |

At N = 4 a dense factor + two substitutions beats LAMMPS's warm-started CG —
do **not** extrapolate the 0.45 to production N (dense is \(O(N^3)\); see
Limitations).

## Limitations

- **Dense Cholesky solve** rebuilt every launch — toy/small-N scope (≤ a few
  hundred atoms), matching the const-potential dense path; a CG +
  neighbor-list path is the natural extension for production N.
- **Point kernel only** (`qeq/point`); shielded / Slater variants
  (`qeq/shielded`, `qeq/slater`) are follow-up queue rows if mapped.
- **Primary-image minimum-image coupling** (matches the non-periodic oracle
  toy); no Ewald-summed QEq.
- **Micro perf twin.** Both walls are sub-millisecond; the 0.45 ratio is a
  fairness datapoint at N = 4, not a scaling claim.
- **Scope of this slug only:** constant-potential electrodes (`ConstPotBO` /
  `ConstPotCG` / `ConstPotMaZe`), induced dipoles (PIM), and charge-dynamics
  (MaZe) are sibling slugs.

## References

- Offline pin: [`regressions/md/qeq-fluctuating-charge/REFERENCE.md`](../../../regressions/md/qeq-fluctuating-charge/REFERENCE.md)
- Product gate: [`regressions/md/qeq-fluctuating-charge/reg_qeq-fluctuating-charge.cpp`](../../../regressions/md/qeq-fluctuating-charge/reg_qeq-fluctuating-charge.cpp)
- Public header: [`include/atomiverse/cpu/dynamics/qeq.h`](../../../include/atomiverse/cpu/dynamics/qeq.h)
- Pin procedure: [`tools/metalwalls_oracle/pin_notes_qeq-fluctuating-charge.md`](../../../tools/metalwalls_oracle/pin_notes_qeq-fluctuating-charge.md)
- Queue / handoff: [`docs/developer/metalwalls-method-queue.md`](../../developer/metalwalls-method-queue.md), [`docs/developer/metalwalls-method-handoff.md`](../../developer/metalwalls-method-handoff.md), [`docs/developer/metalwalls-method-loop.md`](../../developer/metalwalls-method-loop.md)
- Oracle notes: [`tools/metalwalls_oracle/README.md`](../../../tools/metalwalls_oracle/README.md)
- Capability map: `.claude/notes/metalwalls-capability-map.md` (§2 "Fluctuating-charge / QEq")
- Literature: Rappé, Goddard, *J. Phys. Chem.* **95**, 3358 (1991), DOI [10.1021/j100161a070](https://doi.org/10.1021/j100161a070) — QEq / electronegativity equalization
- Oracle: LAMMPS 30 Mar 2026 development (patch_30Mar2026-1074-g7f680de296), QEQ package, `src/QEQ/fix_qeq_point.cpp`
- Related electrode family: [`const-pot-bo-parallel-plate.md`](const-pot-bo-parallel-plate.md), [`maze-symplectic-charge-dynamics.md`](maze-symplectic-charge-dynamics.md)
