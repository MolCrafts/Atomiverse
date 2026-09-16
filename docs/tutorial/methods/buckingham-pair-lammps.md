# Buckingham pair — ion dimer / LAMMPS `pair_style buck`

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `buckingham-pair-lammps` |
| Regression | `regressions/forcefield/buckingham-pair-lammps/` |
| Public API | `cpu::PairBuck(A, rho, C, cutoff)` on `cpu::Evaluate` / MD force slot |
| Roadmap / queue | `docs/roadmap/force-fields.md` → Buckingham **✓** (CPU); MetalWalls map **E8** wave 1; handoff **merged → done** |
| Oracle | **LAMMPS** (not MetalWalls) — offline pin only; never in ctest |

## Summary

`cpu::PairBuck` evaluates the **Buckingham** (Born–Mayer repulsion + \(C/r^{6}\)
dispersion) pair potential in metal units (eV, Å), matching LAMMPS
`pair_style buck` with energy shift at the cutoff (`pair_modify shift yes`).
This page is the **force-field foothold** for molten-salt / electrode wall
potentials in the MetalWalls method loop (map E8): a two-atom single-type
dimer, cubic PBC cell, O(N²) all-pairs walk (no NeighborList). Public path is
constructor arguments only — no `*Config` bag, no factory. Product gate
`reg_buckingham-pair-lammps` asserts energy and forces against hard-coded
offline LAMMPS pins to \(10^{-10}\) eV (eV/Å). Fair wall is **`perf_exempt`**
(startup-dominated dimer; bulk NL twin deferred with Fumi–Tosi).

## Method

**Potential** (LAMMPS `pair_style buck`; same algebra as Atomiverse):

\[
E(r) = A\,e^{-r/\rho} - \frac{C}{r^{6}},\quad r < r_c
\]

**Energy shift** (`pair_modify shift yes` / ATV always-on shift so \(U(r_c)=0\)):

\[
E_{\mathrm{shift}}(r) = E(r) - E(r_c),\qquad
E(r_c)=A\,e^{-r_c/\rho}-\frac{C}{r_c^{6}}
\]

**Force** (shift is constant → zero contribution to \(\nabla\)):

\[
\frac{\mathrm{d}E}{\mathrm{d}r}
  = -\frac{A}{\rho}\,e^{-r/\rho} + \frac{6C}{r^{7}}
\]

Newton third-law pair forces along \(\mathbf{r}_{ij}\); min-image under PBC.

**What this pin includes**

| Item | Setting |
| ---- | ------- |
| Atoms | 2, single type |
| Separation \(r\) | 2.5 Å along \(x\) |
| Box | cubic 40 Å, PBC |
| \(A\) | 1000.0 eV |
| \(\rho\) | 0.3 Å |
| \(C\) | 50.0 eV·Å⁶ |
| \(r_c\) | 8.0 Å |
| Shift | yes (ATV default; LAMMPS twin `pair_modify shift yes`) |
| Units | metal = public ATV surface (eV, Å) |
| Pair walk | O(N²) all-pairs (`as_neighbor_consumer() == nullptr`) |

**Key exclusions (non-goals of this slug)**

- Multi-type coefficients / Lorentz–Berthelot-style mixing / Fumi–Tosi product
  form (queue: `fumi-tosi-nacl`).
- NeighborList production path (same foothold shape as `PairYukawa` /
  `PairWCA` / Morse demo styles).
- Coulomb / long-range / special_bonds scales on this pair style.
- CUDA backend.
- Live LAMMPS (or MetalWalls) inside ctest.
- Fair bulk performance gate (see Performance).

## Implementation

Public surface only (`cpu::Frame` + `PairBuck` constructor args; no Config bag):

```cpp
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/potentials/pair_buck.h"

using namespace ATV;

const double A = 1000.0;   // eV
const double rho = 0.3;    // Å
const double C = 50.0;     // eV·Å⁶
const double rc = 8.0;     // Å
const double r = 2.5;      // Å

cpu::Frame f(2);
f.span(kAtom_x)[0] = 0;
f.span(kAtom_y)[0] = 0;
f.span(kAtom_z)[0] = 0;
f.span(kAtom_x)[1] = r;
f.span(kAtom_y)[1] = 0;
f.span(kAtom_z)[1] = 0;
for (int i = 0; i < 2; ++i) {
    f.span(kAtom_mass)[i] = 1.0;
    f.span(kAtom_type)[i] = 0;
    f.span(kAtom_tag)[i] = i + 1;
}
f.box = cpu::Box::cubic(40.0);

cpu::Evaluate eval;
eval.set_evaluator(cpu::PairBuck{A, rho, C, rc});
eval.build();
eval.run(f);

const double E = f.meta.get(kEnergy);           // eV
const double fx0 = f.span(kAtom_fx)[0];         // eV/Å
const double fx1 = f.span(kAtom_fx)[1];         // eV/Å
```

On classical MD, the same component binds via
`sim.set_evaluator(cpu::PairBuck{…})` (or the force-phase plan); the regression
uses `cpu::Evaluate` for a single-point energy/force pin.

Build and run (MD-only flags for this loop; modules as in `AGENTS.md`):

```bash
bash -lc '
  module load buildenv-gcc/2026.03-mpich
  export OPENBLAS_ROOT="$OPENBLASDIR"
  export RUSTUP_TOOLCHAIN=stable-x86_64-unknown-linux-gnu
  export LD_LIBRARY_PATH="$LIBRARY_PATH:$LD_LIBRARY_PATH"
  cmake -B build -DCMAKE_BUILD_TYPE=Release \
        -DATV_BUILD_MD=ON -DATV_BUILD_KSPACE=ON \
        -DATV_BUILD_TESTS=ON -DATV_BUILD_REGRESSIONS=ON \
        -DATV_BUILD_HF=OFF -DATV_BUILD_PW=OFF -DATV_BUILD_GPW=OFF \
        -DATV_BUILD_CUDA=OFF \
        -DBLA_VENDOR=OpenBLAS
  cmake --build build --target reg_buckingham-pair-lammps -j$(nproc)
  ./build/regressions/reg_buckingham-pair-lammps
'
```

| Path | Role |
| ---- | ---- |
| `include/atomiverse/cpu/potentials/pair_buck.h` | Public `PairBuck` |
| `src/cpu/potentials/pair_buck.cpp` | Shifted energy + force kernel |
| `tests/cpu/potentials/test_pair_buck.cpp` | Unit: analytic \(E,F\); zero at \(r_c\); invalid params |
| `regressions/forcefield/buckingham-pair-lammps/reg_buckingham-pair-lammps.cpp` | Product gate |
| `tools/metalwalls_oracle/inputs/lammps/in.buckingham_dimer` | Offline LAMMPS twin (never ctest) |

## Validation (vs LAMMPS)

Source:
[`regressions/forcefield/buckingham-pair-lammps/REFERENCE.md`](../../../regressions/forcefield/buckingham-pair-lammps/REFERENCE.md).
Oracle: offline **LAMMPS** `pair_style buck` with `pair_modify shift yes`,
**metal** units. External-engine pin — **not** a golden of Atomiverse’s own
output; **never** launch LAMMPS from ctest. Twin input:
`tools/metalwalls_oracle/inputs/lammps/in.buckingham_dimer`.

```text
oracle: lammps
oracle_version: 30 Mar 2026 - Development (develop / patch_30Mar2026-1074-g7f680de296)
```

**Setup:**

| Item | Value |
| ---- | ----- |
| System | 2-atom ion dimer, one type |
| Geometry | atom 0 at origin; atom 1 at \((2.5, 0, 0)\) Å |
| Cell | cubic \(L = 40\) Å, PBC |
| Coefficients | \(A=1000\), \(\rho=0.3\), \(C=50\), \(r_c=8\) |
| Twin | `lmp -in in.buckingham_dimer` (`run 0`; dump forces %.16e) |
| Pin date | 2026-07-23 |
| Analytic cross-check | Independent algebra agrees with LAMMPS dump to machine precision |

**Accuracy gate (product):**

| Quantity | Atomiverse / pin | LAMMPS offline pin | Tol | Notes |
| -------- | ---------------: | -----------------: | --- | ----- |
| \(E\) [eV] | **0.03576020865970150342** | **0.03576020865970150342** | \(1\times 10^{-10}\) eV | Hard-coded in `reg_*` |
| \(F_{1x}\) [eV/Å] | **−0.30971158806504689** | **−0.30971158806504689** | \(1\times 10^{-10}\) eV/Å | Atom 0 (at origin) |
| \(F_{2x}\) [eV/Å] | **+0.30971158806504689** | **+0.30971158806504689** | \(1\times 10^{-10}\) eV/Å | Atom 1; Newton III |
| \(F_{iy}, F_{iz}\) | **0** | **0** | \(1\times 10^{-10}\) eV/Å | Transverse components |

Tolerance rationale (REFERENCE): dimer single-point is algebraically exact
against the shifted formula; \(10^{-10}\) catches wrong shift, units, or force
sign conventions without depending on NL rebuild noise.

Offline re-pin (human only; never ctest):

```bash
# when a LAMMPS binary is available (metal units twin)
lmp -in tools/metalwalls_oracle/inputs/lammps/in.buckingham_dimer
# record pe + forces.dump → REFERENCE.md pins
```

## Performance

Source:
[`REFERENCE.md`](../../../regressions/forcefield/buckingham-pair-lammps/REFERENCE.md)
and the case `REFERENCE.md`
(wave 1, 2026-07-23).

| Engine | Wall (single `run 0` / Evaluate) | Note |
| ------ | -------------------------------: | ---- |
| LAMMPS `run 0` | ~ms (startup-dominated) | full MD stack spin-up |
| ATV `Evaluate` dimer | ~µs–ms (startup-dominated) | O(N²) foothold |

| Item | Value |
| ---- | ----- |
| `perf_exempt` | **true** |
| Reason | Ion-dimer single-point pin; wall dominated by process startup, not the pair kernel. Fair bulk NeighborList twin deferred with Fumi–Tosi multi-type + NL path. |
| \(r = T_{\mathrm{ATV}}/T_{\mathrm{LAMMPS}}\) | **n/a (exempt)** |
| `within_10x` | exempt |
| `ratio_atv_over_oracle` | n/a |

Optional bulk microbench (not a ctest gate): when a production NL `PairBuck`
lands, re-measure vs LAMMPS `pair_style buck` on the same lattice.

## Limitations

- **Single-type O(N²) foothold only.** No multi-type table, no NeighborList
  consumer, no special_bonds. Production molten-salt / wall styles ride the
  Fumi–Tosi / Steele / Born follow-ons (queue E8).
- **CPU only.** CUDA row remains ○ on the force-field roadmap.
- **No Coulomb on this component.** Real-space / long-range electrostatics are
  separate pair / `KSpace*` styles — charge solve must not live inside
  `KSpace*` for const-pot electrodes (loop architecture rule).
- **Performance not gated.** `perf_exempt` for this wave; do not treat dimer
  wall as a speed target.
- **Oracle is LAMMPS**, not MetalWalls: the feature exists in LAMMPS
  (`pair_style buck`), so the binding offline pin + any future fair bench use
  LAMMPS (MetalWalls is reserved for MW-only physics).

## References

- Offline pin: `regressions/forcefield/buckingham-pair-lammps/REFERENCE.md`
- Twin input: `tools/metalwalls_oracle/inputs/lammps/in.buckingham_dimer`
- Oracle notes: `tools/metalwalls_oracle/README.md`
- Capability map: `.claude/notes/metalwalls-capability-map.md` (E8)
- Roadmap: `docs/roadmap/force-fields.md` (Buckingham CPU ✓)
- Public header: `include/atomiverse/cpu/potentials/pair_buck.h`
- LAMMPS docs: `pair_style buck` / `pair_modify shift` (metal units)
