# Born–Mayer–Huggins pair — ion dimer foothold (LAMMPS `pair_style born`)

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `born-repulsion-foothold` |
| Regression | `regressions/forcefield/born-repulsion-foothold/` |
| Public API | `cpu::PairBorn` (`Potential` on `cpu::Evaluate` / `cpu::md::MD`) |
| Roadmap | map slice E8 |
| Oracle | **LAMMPS** `pair_style born` + `pair_modify shift yes` (metal units) |

## Summary

**Born–Mayer–Huggins (BMH)** short-range pair energy and force: exponential
core repulsion plus attractive \(r^{-6}\) and \(r^{-8}\) tails, shifted so
\(U(r_c)=0\). Public surface is `cpu::PairBorn` — a **Potential** Component
with constructor arguments \((A,\rho,\sigma,C,D,r_c)\) (single type) or a
`std::vector<BornTypeCoeff>` multi-type table (Fumi–Tosi ion–ion short range;
no Lorentz–Berthelot mixing). This regression is the **E8 foothold**: a
two-atom ion dimer single-point against offline LAMMPS `pair_style born`
(metal units, shift on). Coulomb is **not** included here — compose with
`PairCoulCut` / `coul/long` for full ionic force fields (see sibling
`fumi-tosi-nacl`). Product gate
`reg_born-repulsion-foothold` asserts energy and \(F_x\) to \(10^{-10}\) eV
(eV/Å). Fair wall vs LAMMPS is **`perf_exempt`** (2-atom evaluate is
startup-dominated; bulk NeighborList twin deferred).

## Method

**Physics.** LAMMPS `pair_style born` binary-verified form (coeff order
`A rho sigma C D`; \(\rho\) is the softness length in Å):

\[
E(r) = A\,e^{(\sigma-r)/\rho} - \frac{C}{r^{6}} + \frac{D}{r^{8}},
\quad r < r_c
\]

Energy shift (`pair_modify shift yes` / Atomiverse default):

\[
E_{\mathrm{shift}}(r) = E(r) - E(r_c)
\]

Force follows from the unshifted derivative (shift is constant):

\[
\frac{\mathrm{d}E}{\mathrm{d}r}
  = -\frac{A}{\rho}\,e^{(\sigma-r)/\rho}
  + \frac{6C}{r^{7}}
  - \frac{8D}{r^{9}}
\]

Units on the public surface: **eV**, **Å** (LAMMPS **metal** twin).

**What this pin includes**

| Item | Setting |
| ---- | ------- |
| Atoms | 2 (single type, `kAtom_type = 0`) |
| Separation \(r\) | 2.5 Å along \(x\) |
| Box | cubic 40 Å, fully periodic |
| \(A\) | 1000.0 eV |
| \(\rho\) | 3.0 Å |
| \(\sigma\) | 1.0 Å |
| \(C\) | 50.0 eV·Å⁶ |
| \(D\) | 20.0 eV·Å⁸ |
| \(r_c\) | 8.0 Å |
| Shift | yes (hard-coded in `PairBorn`) |
| Driver | `cpu::Evaluate` + `set_evaluator(PairBorn{…})` |

**Key exclusions (non-goals of this slug)**

- Coulomb / long-range electrostatics (compose separately; multi-type Born+Coul
  product is `fumi-tosi-nacl`).
- NeighborList-backed \(O(N)\) production path — v1 is **O(N²)** all-pairs
  (`as_neighbor_consumer()` returns `nullptr`, same foothold shape as
  `PairBuck` / `PairYukawa`).
- Special-bond scaling tables beyond the base `Pair` infrastructure used by
  this foothold.
- Live LAMMPS inside ctest.
- Public `*Config` bags or `make_*` / `build_*` factories.

## Implementation

Public surface only (`cpu::Frame` + `PairBorn` ctor args; no Config bag).
Minimal single-point matching the regression:

```cpp
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/potentials/pair_born.h"

using namespace ATV;

const double A = 1000.0;   // eV
const double rho = 3.0;    // Å
const double sigma = 1.0;  // Å
const double C = 50.0;     // eV·Å⁶
const double D = 20.0;     // eV·Å⁸
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
eval.set_evaluator(cpu::PairBorn{A, rho, sigma, C, D, rc});
eval.build();
eval.run(f);

const double E = f.meta.get(kEnergy);  // [eV]
// forces on kAtom_fx / fy / fz [eV/Å]
```

Multi-type Fumi–Tosi short-range table (unlisted type pairs do not interact;
no silent mixing):

```cpp
std::vector<cpu::BornTypeCoeff> coeffs{
    {/*type_i=*/0, /*type_j=*/0, A00, rho00, sigma00, C00, D00},
    {0, 1, A01, rho01, sigma01, C01, D01},
    {1, 1, A11, rho11, sigma11, C11, D11},
};
cpu::PairBorn born(std::move(coeffs), /*cutoff [Å]=*/rc);
// compose with PairCoulCut / coul long for full ionic salt
```

MD binding (categorical potential API):

```cpp
cpu::md::MD sim;
sim.set_evaluator(cpu::PairBorn{A, rho, sigma, C, D, rc});
// NeighborList optional for this O(N²) foothold; production NL path deferred
sim.set_integrator(cpu::md::VelocityVerlet{/*dt [fs]=*/1.0});
sim.build();
sim.run(frame, /*nsteps=*//*…*/);
```

Architecture rules:

- **Potential Component** on Evaluate/MD — not a Driver category.
- Parameters are **constructor arguments** — no `PairBornConfig`.
- Frame is the sole I/O currency; energy on `kEnergy`, forces on `kAtom_f*`.

Build and run the regression (MD-only configure; CPU login pattern from
`AGENTS.md` / metalwalls loop):

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
        -DATV_BUILD_CUDA=OFF -DBLA_VENDOR=OpenBLAS
  cmake --build build --target reg_born-repulsion-foothold -j"$(nproc)"
  ./build/regressions/reg_born-repulsion-foothold
'
```

| Path | Role |
| ---- | ---- |
| `include/atomiverse/cpu/potentials/pair_born.h` | Public `PairBorn`, `BornTypeCoeff` |
| `src/cpu/potentials/pair_born.cpp` | Shifted energy/force, single- and multi-type tables |
| `regressions/forcefield/born-repulsion-foothold/reg_born-repulsion-foothold.cpp` | Product gate |
| `tools/metalwalls_oracle/inputs/lammps/in.born_dimer` | Offline LAMMPS twin input |
| `tests/cpu/potentials/test_pair_born.cpp` | Analytic unit checks (not e2e) |

## Validation (vs LAMMPS)

Source:
[`regressions/forcefield/born-repulsion-foothold/REFERENCE.md`](../../../regressions/forcefield/born-repulsion-foothold/REFERENCE.md)
(offline pin; **never** live LAMMPS in ctest). Oracle tooling notes:
[`tools/metalwalls_oracle/README.md`](../../../tools/metalwalls_oracle/README.md).
Twin input: `tools/metalwalls_oracle/inputs/lammps/in.born_dimer`.

**Oracle:** LAMMPS **30 Mar 2026 - Development**
(`develop` / `patch_30Mar2026-1074-g7f680de296`), local binary
`/home/jicli594/.local/x86_64/bin/lmp`, `units metal`,
`pair_style born 8.0`, `pair_coeff 1 1 1000.0 3.0 1.0 50.0 20.0`,
`pair_modify shift yes`, `run 0` (thermo PE + dump forces `%.16e`).

**Why LAMMPS (not MetalWalls):** BMH / `pair born` exists in LAMMPS; binding
oracle priority for this loop is **LAMMPS first** when the feature overlaps.
MetalWalls is reserved for MW-only physics.

### Accuracy

Offline pins recorded **2026-07-24**. Independent analytic evaluation of the
same algebra agrees with the LAMMPS dump and thermo PE.

| Quantity | Atomiverse / pin | LAMMPS pin | Tol | Notes |
| -------- | ---------------: | ---------: | --- | ----- |
| \(E\) [eV] | **509.3671885909988** | **509.3671885909988** | \(\lvert\Delta E\rvert\le 10^{-10}\) | shifted pair energy |
| \(F_{1x}\) [eV/Å] | **−2.0172730961087780×10²** | **−2.0172730961087780×10²** | \(\lvert\Delta F\rvert\le 10^{-10}\) | atom at \(x=0\) |
| \(F_{2x}\) [eV/Å] | **+2.0172730961087780×10²** | **+2.0172730961087780×10²** | \(\lvert\Delta F\rvert\le 10^{-10}\) | atom at \(x=2.5\) Å |
| \(F_{1y},F_{1z},F_{2y},F_{2z}\) | **0** | **0** | \(10^{-10}\) | transverse components |

Geometry, coefficients, cutoff, and tolerances: **Method** table and
REFERENCE (match `reg_born-repulsion-foothold`).

Related unit surface (not the e2e pin):
`tests/cpu/potentials/test_pair_born.cpp` (analytic shifted \(U\) and
\(\mathrm{d}U/\mathrm{d}r\) to \(10^{-12}\)).

### Offline pin repro (never ctest)

```bash
# Requires a LAMMPS metal build with pair born
lmp -in tools/metalwalls_oracle/inputs/lammps/in.born_dimer
# thermo PE; dump forces.dump id fx fy fz (%.16e)
# pins → regressions/forcefield/born-repulsion-foothold/REFERENCE.md
```

## Performance

Source: REFERENCE.md performance block; handoff
the case `REFERENCE.md`.

| Twin | \(T_{\mathrm{LAMMPS}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | --------------------------: | -----------------------: | ----: | ---- | ------ |
| 2-atom dimer `run 0` / `Evaluate` | ~ms (startup-dominated) | ~µs–ms (startup-dominated) | **perf_exempt** | n/a | wall dominated by process startup, not pair kernel |

| Field | Value |
| ----- | ----- |
| `perf_exempt` | **true** |
| `perf_exempt_reason` | ion-dimer single-point pin; wall dominated by process startup, not pair kernel. Fair bulk NL twin deferred with Fumi–Tosi multi-type + NeighborList path |
| `ratio_atv_over_oracle` | **n/a** |
| `within_10x` | exempt |

Do **not** invent wall-clock ratios. When a production NeighborList
`PairBorn` path lands, re-measure offline vs LAMMPS `pair_style born` on the
same lattice, record version + walls, and replace `perf_exempt` in REFERENCE.

## Limitations

- **Foothold only.** Two atoms, O(N²) all-pairs walk; not a bulk MD performance
  claim.
- **No NeighborList consumer** on this component yet (`as_neighbor_consumer`
  withdrawn) — same class as early `PairBuck` / `PairYukawa` footholds.
- **Coulomb is out of scope** for this slug; ionic salt production uses
  multi-type `PairBorn` + Coulomb (`fumi-tosi-nacl`).
- **No fair speed ratio** until a bulk NL twin is instrumented.
- **Scope of E8 wave:** Steele wall (`steele-wall-argon`) and Fumi–Tosi NaCl
  (`fumi-tosi-nacl`) are sibling forcefield slugs with their own pages when
  written.

## References

- Offline pin: [`regressions/forcefield/born-repulsion-foothold/REFERENCE.md`](../../../regressions/forcefield/born-repulsion-foothold/REFERENCE.md)
- Product gate: [`regressions/forcefield/born-repulsion-foothold/reg_born-repulsion-foothold.cpp`](../../../regressions/forcefield/born-repulsion-foothold/reg_born-repulsion-foothold.cpp)
- Public header: [`include/atomiverse/cpu/potentials/pair_born.h`](../../../include/atomiverse/cpu/potentials/pair_born.h)
- Oracle notes: [`tools/metalwalls_oracle/README.md`](../../../tools/metalwalls_oracle/README.md)
- Offline twin input: [`tools/metalwalls_oracle/inputs/lammps/in.born_dimer`](../../../tools/metalwalls_oracle/inputs/lammps/in.born_dimer)
- LAMMPS doc: `pair_style born` (Born–Mayer–Huggins); metal units; `pair_modify shift`
- Literature (method family): Born–Mayer; Huggins–Mayer / Fumi–Tosi molten-salt
  short-range forms (compose with Coulomb for production salts)
