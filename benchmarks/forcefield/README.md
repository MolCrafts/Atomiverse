# GAFF molecular parameterization

This method compares the public Atomiverse/molpy molecular APIs with independent
AmberTools 26.1 commands. It uses the existing scientific benchmark runner,
result contract and numerical/timing eligibility rules. It is outside molpy's
unit tests and does not execute an external scientific program from pytest.

`bench_gaff.py` declares four combinations: GAFF and GAFF2, each with AM1-BCC
(`bcc`) and ABCG2 (`abcg2`). Atom typing and charge correction are independent
choices. The canonical inputs in `../systems/gaff.json` contain complete atom properties,
connectivity, stated bond orders and explicit hydrogens, with positions rounded
once to 0.001 Å. Their accompanying MOL2 text is the exact AmberTools input.
Conformer generation is excluded from every participant's measured task.

The acceptance/performance tiers cover ethanol, phenol, acetate, pyridine,
acetamide, methylammonium, DMSO, thiophene, an F/Cl/Br organic molecule,
trimethylphosphate, octane and hexadecane (7–50 atoms). The smoke tier selects
ethanol, acetate and trimethylphosphate. This is a declared representative set,
not a claim to cover arbitrary organic chemistry, radicals or metal complexes.

## Compared task

Each native invocation reconstructs a fresh `molpy.Atomistic` and computes
charges with `atomiverse.AM1BCCCharges(method=atomiverse.AM1(...),
correction=...).assign(molecule)`. It stores the returned charges on its local
molecule. Typing then runs in two steps, both with the row's `parameter_set`
(`gaff` or `gaff2`):

1. `molpy.ff.typifier.AtdTypifier(parameter_set=...).typify(molecule)` applies
   antechamber's `ATOMTYPE_*.DEF` rules and returns a clone with GAFF atom types.
2. `molpy.ff.typifier.GaffTypifier(parameter_set=...).typify(labelled)` matches
   the bonded terms against the compiled `gaff.dat`/`gaff2.dat` table (with
   parmchk2-style estimates), and `.forcefield()` returns the parameters.

From the typed frame and force field, it materializes atom-ordered
charges/types and explicit parameter records.

Each independent invocation writes a fresh MOL2 in a fresh job directory,
runs `antechamber -at ... -c ... -eq 1 -ek "qm_theory='AM1', maxcyc=0,
scfconv=1.d-10"`, runs `parmchk2 -s ...`, and assembles a prmtop with tleap.
The prmtop is parsed to materialize the same parameter records. This is a
comparison of complete parameterization routes, including the external
programs' process and file overhead. Component timings identify antechamber,
parmchk2 and tleap separately, and identify native AM1-BCC and GAFF typing.

Both routes use a fixed input geometry. They do not implement antechamber's
default geometry-optimization task. Fresh solvers, fresh typifiers and fresh
external job directories prevent a converged-density or parameter cache from
becoming an undeclared scientific intermediate. Native static parameter
libraries may initialize during the declared untimed runtime warmup.

## Numerical checks

The generic checker compares exact numeric encodings of the GAFF atom labels,
per-atom charges, explicit bond/angle/proper/improper parameter instances,
per-atom masses and Lennard-Jones self parameters, and 1–4 scale factors. Parameter records
retain atom indices; only the full reversal of an undirected proper/bond/angle
is normalized. Improper slot order is retained up to full quartet reversal at
phase zero or pi; tleap sometimes reverses a quartet to avoid storing the
unencodable negative-zero fourth pointer. The whole reversal is equivalent at
these phases, and no arbitrary permutation of the outer atoms is accepted.
Zero Fourier coefficients are omitted on both sides. Empty relation categories
have an explicit numeric sentinel, so they cannot disappear from acceptance.

The per-atom charge tolerance is declared as 0.001 e: sqm prints the AM1 base
Mulliken charges to three decimals (up to 0.0005 e rounding), and 0.0005 e is
reserved for the independent AM1 implementation. The result preserves actual
per-atom deviations and oracle net-charge rounding residuals. The native
charge sum must independently equal the molecular charge within 1e-6 e.
This bound is not tuned per molecule or widened after a failed measurement.
GAFF type codes are exact; parameter tolerances allow prmtop decimal
serialization and radians/degrees conversion, not a different parameter row.

Every raw sqm output, antechamber/parmchk2/tleap log, frcmod, prmtop and input
file is kept. The numerical results retain estimated native parameter types
and their provenance, plus the complete reference frcmod text. A parmchk2
row marked as needing revision is a scientific failure.

Parameter agreement is stronger than simply obtaining a finite energy, but it
does not constitute an independent energy/force implementation comparison.
No force/energy-evaluator, molecular dynamics, geometry optimization, RESP or
CUDA claim is made by these rows.

## Running

Use one environment with the rebuilt molpy wheel and Atomiverse native module
from the same molpy/Rust ABI build. The benchmark fingerprints both loaded
native libraries, the external binaries and reference parameter tables.
The default AmberTools prefix is the cluster's 26.1 installation; set
`GAFF_AMBERHOME` to use another installation and inspect the recorded identity.

For repeated isolated workers, an allocation's controller may capture the
sibling molpy source with the existing `source_info` helper once before the
campaign, together with hashes of both loaded native libraries, and expose that
immutable JSON through `GAFF_MOLPY_SOURCE_RECEIPT`. Each worker validates the
receipt's source-tree digest and the actual loaded binary hashes, recording the
receipt SHA-256 and sibling tree SHA-256. The controller must recompute the full
sibling source tree after the campaign and verify all observations used the
same receipt, tree and libraries. A changed sibling tree or inconsistent receipt
invalidates timing ratios explicitly. Without a receipt, the
adapter computes sibling provenance directly in every worker; it never omits
source information to save time. Source hashing and provenance checks are
outside the scientific E2E timer.

```bash
export ATV_GAFF_ARTIFACT_ROOT="$PWD/build/gaff-raw-jobs"
python -m benchmarks.forcefield.bench_gaff --tier smoke --threads 1 \
  --repeats 1 --warmup 0 --no-snapshot --out build/gaff-smoke
python -m benchmarks.forcefield.bench_gaff --tier performance --threads 1 \
  --repeats 5 --warmup 1 --no-snapshot --out build/gaff-performance
```

Each run writes the standard results JSON, matrix, raw observation artifacts,
CSV and JUnit under the `--out` directory; keep it, and `ATV_GAFF_ARTIFACT_ROOT`,
under `build/` or outside the source tree. `--no-snapshot` disables the separate
CI envelope and records that choice in `bundle.protocol.ci_snapshot`; use it
when the installed molci predates the 0.3 protocol (its `Manifest` rejects
`observation_id`). Existing observations can be re-reported with
`python -m benchmarks report --input <results.json> --out <directory>` once
the compatible protocol is available; no scientific rerun is needed for that
envelope alone.

Participants execute serially with an interleaved
schedule; every sample has its own worker process, and warm samples run an
untimed fresh-state invocation before measurement. Cold process wall time
includes imports and runtime initialization. Five accepted warm samples are
required before a speedup can be reported. A dirty source tree cannot promote
the roadmap's official ✓/⚡ marks; measured local results still remain visible.

A known tleap behavior: for DMSO, tleap's GAFF2 torsion assignment depends on
atom numbering. Those comparison cells remain failures with no speedup; the
native exact-table assignment is symmetric under the atom permutation.
