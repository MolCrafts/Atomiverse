# Archived scientific regression cases

The regression runner, registry, CMake registration and `ATV_BUILD_REGRESSIONS`
option have been removed. This directory is retained as scientific source and
reference material; it is not built or registered with CTest.

The case sources retain their former registration calls and include paths as
archival context. They are not standalone executables. Commands recorded in
case references and historical diagnostics describe the removed runner and
cannot be used with the current checkout.

Each method directory preserves its section sources, inputs and `REFERENCE.md`
with external results, units and tolerances. Retaining this evidence does not
establish that the current implementation passes those checks. Active unit tests
remain under `tests/`; performance benchmarks remain under `benchmarks/`.

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
