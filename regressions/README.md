# regressions/ — end-to-end capability pins

The **only** home of end-to-end in this repository. One case per method proves
that the method works through the public API, against numbers that do not come
from Atomiverse. A case is a directory of sections — one per pinned system,
quantity or backend — run together.

Every section links into one executable, `atv_regressions`; every case is one
ctest test labelled `regression`:

```bash
cmake -B build -G Ninja -DATV_BUILD_REGRESSIONS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --target atv_regressions
ctest --test-dir build -L regression               # every case
ctest --test-dir build -R '^scf/coupled-cluster$'    # one case
build/regressions/atv_regressions --list           # cases and sections
build/regressions/atv_regressions --case scf/coupled-cluster --section ccsd-h2-sto3g
ctest --test-dir build -LE regression              # unit tests only
```

| | answers | runs | executable |
|---|---|---|---|
| `tests/` | does this function / method do what it says? | whole suite in seconds | `atv_tests` (+ `_cuda`, `_mpi`) |
| `regressions/` | does this method still give the right physics? | ≤ 60 s per section | `atv_regressions` |
| `benchmarks/` | how fast is it next to a production twin? | at acceptance | one per harness, never ctest |

## When a case is required

A case exists **for one method**: one row of `docs/roadmap/`, checked for
every quantity it claims (energy, forces, stress, its variants — spin,
k-points, smearing) on every backend it claims. A cell is ✓ only when a case
pins it; a capability without a case is ◐ at most.

| Needs a case | Does not get a case |
|---|---|
| a new user-visible capability: method, ensemble, integrator, bias, observable, I/O path | single-function math or logic → `tests/` |
| an existing method gaining a backend (CUDA), a quantity (forces, stress) or a variant (spin, k-points) — the **same** case gains the gate | internal refactors and staged ports ("move step X to the device") → unit test or nothing |
| a ◐ cell being promoted to ✓ | parameter variants on the same code path (another functional, basis, k-mesh, thermostat constant) |
| | speed → `benchmarks/`; API shape, layering, includes → code review |

One method × its quantities × its backends = **one case**.

## The oracle

The oracle is where the expected number comes from. Test: *if our
implementation were wrong, would this number change?* If it would, it is not
an oracle.

| Kind | Example | `REFERENCE.md` must record |
|---|---|---|
| `external` | PySCF, CP2K, LAMMPS, DFTB+, tblite, MOPAC, OpenMM printed it | program + version/sha, input file (in `tools/<code>_oracle/`), the output line |
| `literature` | Madelung constant, published reference energy | citation (DOI/book, section/table), value, units |
| `analytic` | closed form, exact identity (virial = energy for a degree −1 potential), conservation law | the formula and its derivation source |
| `cpu-twin` | a CUDA cell of a capability whose CPU cell is pinned by one of the above | the CPU case that carries the external pin |

Never an oracle: Atomiverse's own earlier output ("golden", "first green",
"frozen"), an Atomiverse-vs-Atomiverse finite difference as the only gate, a
numpy re-implementation of the algorithm, or a "twin" with a different
Hamiltonian (GTH-GPW numbers for NC-UPF plane waves). These may appear as
*extra* consistency checks next to a real oracle, never instead of one. Full
provenance rules: `.claude/notes/oracle-pin-provenance.md`.

## Layout

```
regressions/<domain>/<method>/
  case.cmake         one atv_regression(SECTION …) per section (below)
  <section>.cpp      or .cu when the section runs on CUDA; one registered run()
  REFERENCE.md       the oracle record, one block per section (template below)
  <small inputs>     optional, named <section>.<what>; shared data in tests/data/
```

- `<domain>` is the product module whose sources the case exercises
  (`scf` = HF, `gpw`, `pw`, `semiempirical`, `md`, `minimizer`, `mc`, `path`,
  `sampling`, `multiscale`, `response`, `forcefield`, `longrange`, …).
- `<method>` names the method (`rhf`, `d3`, `constant-potential`); a section
  names the system and quantity it pins (`h2o-forces`, `si-kpoint-stress`).
  No wave/slug/backend prefixes; backends are a runtime argument.
- A new quantity, variant or backend of an existing method is a new section,
  never a new case. Sections on one system share one SCF where they can.

### `<section>.cpp`

```cpp
#include "atomiverse/…"          // public headers only
#include "harness/registry.h"

namespace atv_reg_<domain>_<section> {   // every section in its own namespace
int run() {                              // or run(int argc, char** argv)
    …                                    // print got / want / tol per gate
    return failures == 0 ? 0 : 1;
}
} // namespace atv_reg_<domain>_<section>

ATV_REGRESSION("<domain>/<method>", "<section>", atv_reg_<domain>_<section>::run);
```

Sections are compiled in unity batches: nothing may leak out of the section
namespace (no file-scope `using`, no macros). A case runs its sections in name
order and fails when any fails; `--section <name>` runs one.

### `case.cmake`

```cmake
atv_regression(SECTION <section>
  REQUIRES MD POTENTIALS KSPACE     # product modules (ATV_BUILD_<X>); any OFF ⇒ section absent
  [DEVICES cpu|cuda|both]           # default cpu; cuda/both need a .cu source and ATV_BUILD_CUDA
  [THREADS <n>]                     # OMP width, default 1; the case runs at its widest section
  [MPIEXEC]                         # the section relaunches itself through mpiexec
  [GPUS <n>]                        # minimum visible GPUs
  [LIBS <targets…>])                # extra link targets (CUDA::cufft, …)
```

`regressions/CMakeLists.txt` discovers every `*/*/case.cmake`; nothing is
registered by hand. Shared inputs resolve through the compile-time root
`ATV_REG_DATA_DIR` (`tests/data`); inputs next to the section through
`atv_reg::case_dir(__FILE__)`.

## Rules

1. **Out-of-library consumer.** Public `atomiverse/…` headers only, link only
   `atomiverse`. No `internal/` headers. A capability that cannot be driven
   through the public API is the defect the case just found.
2. **A real oracle** (table above), recorded in `REFERENCE.md`.
3. **≤ 60 s per section** at the declared `THREADS` on one node, measured. A
   section that needs longer is too big: shrink the system, not the rule.
4. **No CWD-relative input.** Resolve inputs through `ATV_REG_DATA_DIR` /
   `atv_reg::case_dir(__FILE__)`. A missing input fails; never substitute a different
   model.
5. **Exit non-zero on failure**, printing `got / want / tol` per gate.
6. **One flat `run()`.** Code that runs once is not carved into one-call
   helpers. `--device=cpu|cuda|both` selects backends for `DEVICES both`.
7. **Deterministic.** Fixed seeds; tolerances justified by physics or by the
   oracle's own precision, not loosened until green.

## `REFERENCE.md` template

```markdown
# <case> — <capability in one line>

| | |
|---|---|
| Roadmap | docs/roadmap/<chapter>.md — <row> (<quantities>, <backends>) |
| Oracle | external / literature / analytic / cpu-twin |
| Source | <program version + input + output line> or <citation> |
| Quantity | <what is compared, units> |
| Tolerance | <value> — <why this tolerance> |
| Runtime | <measured s> at <THREADS> threads (<machine>) |

<System and settings: geometry, basis/cutoffs, ensemble, steps — enough to
re-run the oracle.>
```
