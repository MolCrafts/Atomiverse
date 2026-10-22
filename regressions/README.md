# regressions/ — end-to-end capability pins

The **only** home of end-to-end in this repository. One case proves that one
delivered capability works through the public API, against a number that does
not come from Atomiverse.

```bash
cmake -B build -DATV_BUILD_REGRESSIONS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --target regressions        # build and run every case
cmake --build build --target run_reg_<case>     # one case
```

| | answers | runs | registered in ctest |
|---|---|---|---|
| `tests/` | does this function / method do what it says? | whole suite in seconds | yes |
| `regressions/` | does this capability still give the right physics? | ≤ 60 s per case | **never** |
| `benchmarks/` | how fast is it next to a production twin? | at acceptance | never |

## When a case is required

A case exists **for one roadmap cell**: one capability (a row of
`docs/roadmap/`) on the backends it claims. A cell is ✓ only when a case pins
it; a capability without a case is ◐ at most.

| Needs a case | Does not get a case |
|---|---|
| a new user-visible capability: method, ensemble, integrator, bias, observable, I/O path | single-function math or logic → `tests/` |
| an existing capability gaining a backend (CUDA) — the **same** case gains `--device=cuda` | internal refactors and staged ports ("move step X to the device") → unit test or nothing |
| a ◐ cell being promoted to ✓ | parameter variants on the same code path (another functional, basis, k-mesh, thermostat constant) |
| | speed → `benchmarks/`; API shape, layering, includes → code review |

One capability × its backends = **one case**. A second case for the same
capability needs a different code path (forces vs stress vs k-points are
different paths; PBE vs PBEsol through libxc is not).

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
regressions/<domain>/<case>/
  case.cmake         registration (below)
  reg_<case>.cpp     or .cu when the case runs on CUDA; one flat main()
  REFERENCE.md       the oracle record (template below)
  <small inputs>     optional; shared data lives in tests/data/
```

- `<domain>` is the product module whose sources the case exercises
  (`scf` = HF, `gpw`, `pw`, `semiempirical`, `md`, `minimizer`, `mc`, `path`,
  `sampling`, `multiscale`, `response`, `forcefield`, `longrange`, …).
- `<case>` names the capability, not the history: system + method + quantity
  (`h2o-rhf-sto3g-energy`, `nacl-pme-npt`). No wave/slug/backend prefixes;
  backends are a runtime argument.

### `case.cmake`

```cmake
atv_regression(
  REQUIRES MD POTENTIALS KSPACE     # product modules (ATV_BUILD_<X>); any OFF ⇒ case absent
  [DEVICES cpu|cuda|both]           # default cpu; cuda/both need a .cu source and ATV_BUILD_CUDA
  [THREADS <n>]                     # OMP width, default 1; raise only on a measurement, ≤ 8
  [RANKS <n>]                       # run under mpiexec -n <n> (needs ATV_USE_MPI)
  [GPUS <n>]                        # minimum visible GPUs
  [LIBS <targets…>]                 # extra link targets (CUDA::cufft, …)
  [ARGS <args…>])                   # extra command-line arguments
```

`regressions/CMakeLists.txt` discovers every `*/*/case.cmake`; nothing is
registered by hand. The binary gets the compile-time data root
`ATV_REG_DATA_DIR` (`tests/data`) and `ATV_REG_CASE_DIR` (its own directory).

## Rules

1. **Out-of-library consumer.** Public `atomiverse/…` headers only, link only
   `atomiverse`. No `internal/` headers. A capability that cannot be driven
   through the public API is the defect the case just found.
2. **A real oracle** (table above), recorded in `REFERENCE.md`.
3. **≤ 60 s** at the declared `THREADS` on one node, measured. A case that
   needs longer is too big: shrink the system, not the rule.
4. **No CWD-relative input.** Resolve inputs through `ATV_REG_DATA_DIR` /
   `ATV_REG_CASE_DIR`. A missing input fails; never substitute a different
   model.
5. **Exit non-zero on failure**, printing `got / want / tol` per gate.
6. **One flat `main()`.** Code that runs once is not carved into one-call
   helpers. `--device=cpu|cuda|both` selects backends for `DEVICES both`.
7. **Deterministic.** Fixed seeds; tolerances justified by physics or by the
   oracle's own precision, not loosened until green.

## `REFERENCE.md` template

```markdown
# <case> — <capability in one line>

| | |
|---|---|
| Roadmap | docs/roadmap/<chapter>.md — <row> (<backends>) |
| Oracle | external / literature / analytic / cpu-twin |
| Source | <program version + input + output line> or <citation> |
| Quantity | <what is compared, units> |
| Tolerance | <value> — <why this tolerance> |
| Runtime | <measured s> at <THREADS> threads (<machine>) |

<System and settings: geometry, basis/cutoffs, ensemble, steps — enough to
re-run the oracle.>
```
