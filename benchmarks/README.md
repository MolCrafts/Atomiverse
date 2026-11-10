# Benchmarks

Benchmarks measure accuracy, speed and resources outside the unit suite. None
of them is registered with CTest. The tree has one axis per directory:

```text
benchmarks/
  README.md  __init__.py  __main__.py  framework/  roadmap.toml  requirements.txt
  systems/      hf.json dft.json gaff.json
  scf/          bench_{rhf,uhf,rohf,ghf,rks,uks,rmp2,rmp3}.py  hf.py  dft.py
  forcefield/   bench_gaff.py
  perf/         CMakeLists.txt  md/  scf/  pw/  semiempirical/{nddo_energy,nddo_forces}/
  compare/      lj-lammps/  gaff-liquid/  gaff-peo-openmm/  gaff-peo-lammps/  am1-sqm/
```

- **Framework spine** (`__main__.py`, `framework/`, `roadmap.toml`): the
  `python -m benchmarks` CLI. It discovers methods, runs isolated workers,
  compares observables, and writes reports and roadmap evidence.
- **`systems/`**: canonical inputs that the Atomiverse and reference
  participants share. Methods read these files and do not embed their own copies.
- **`scf/`, `forcefield/`**: framework methods. Each `bench_*.py` compares
  public Atomiverse calls with a live independent engine. See
  [scf/README.md](scf/README.md) and [forcefield/README.md](forcefield/README.md).
- **`perf/`**: native C++/CUDA timing harnesses, built by CMake into
  `<build>/bin`. See [perf/README.md](perf/README.md).
- **`compare/`**: scripts that drive Atomiverse and an external engine
  (LAMMPS, OpenMM, AmberTools SQM) on the same input. Each subdirectory has
  its own README.

## Rules

1. A framework method is `benchmarks/<domain>/bench_*.py`, exactly one level
   deep. `__main__.py` globs `*/bench_*.py`, and each match must export a
   `Method` named `SPEC`. A file that matches without a `SPEC` is an error.
   Shared helpers use other names (`hf.py`, `dft.py`).
2. Native harnesses live only under `perf/<domain>/`. Each one is declared in
   `perf/CMakeLists.txt` and requires `-DATV_BUILD_BENCHMARKS=ON`.
3. `compare/` holds only scripts against external engines. They may call
   binaries from `perf/`, but they don't declare framework methods.
4. Inputs shared by several participants belong in `systems/`.
5. **Results are never committed.** `.gitignore` ignores `benchmarks/**/results/`.
   Campaigns write under `build/` or outside the source tree. A document may
   quote a measured number together with its hardware, but it must not link
   to a result bundle.

## Framework CLI

Install the HF-enabled Atomiverse extension and `molcrafts-molpy==0.20.0` in
one environment. Until the molcrafts-ci 0.3 observation protocol is released,
install the sibling checkout before `requirements.txt`. The native bridge
must pass the runtime's ABI fingerprint check; do not bypass it to accept a
different Rust build. Optional GPU oracles count as available only if they
actually import and run; an installed package name is not enough.

```bash
python -m pip install -e ../molcrafts-ci
python -m pip install -r benchmarks/requirements.txt
python -m benchmarks list        # methods and their row IDs
python -m benchmarks catalog     # declared capabilities, including gaps
python -m benchmarks run --tier smoke --threads 1 --repeats 1 --warmup 0 \
  --out build/bench-smoke
```

`run` selects coverage with `--tier {smoke,acceptance,performance}`,
`--method` (repeatable) and `--system` (explicit IDs, marked partial). Its
other options are `--threads`, `--repeats`, `--warmup`, `--timeout` and
`--require-cuda`. Each method can also run as a module, for example
`python -m benchmarks.scf.bench_rhf --tier smoke --out build/rhf-smoke`.

A campaign writes:

- the authoritative `results.json`;
- every cold and warm observation and log;
- a detailed CSV, `matrix.md` and `matrix.html`;
- roadmap evidence, JUnit and `snapshots/benchmark.json`.

The matrix's last column identifies the actual hardware and environment. A
controlled speedup requires all of these:

- five samples;
- passing observables;
- consistent native binary fingerprints;
- a matched resource policy.

Without them, the report shows raw times and no ratio.

`--no-snapshot` writes the scientific results, tables and raw evidence without
the optional molci CI envelope. The result protocol records this choice. By
default, the coordinated molcrafts-ci 0.3 API is required, and an older
envelope schema is never substituted.

- `validate --input <results.json>` checks the strict result contract.
  `molci validate-snapshot <snapshot.json>` checks the storage envelope.
- `report --input <results.json> --out <dir>` regenerates the views from the
  stored observations.
- `source --out <manifest.json>` captures source hashes for moving a frozen
  source snapshot. `run --source-manifest` checks every declared source before
  and after execution. It doesn't infer that an installed library was compiled
  from that source.
- `roadmap --input <results.json> --chapter hartree-fock.md --write` updates
  only the chapter's generated evidence block. `--check` detects drift. A
  local dirty build, missing coverage or an unqualified CUDA window cannot
  promote a ✓ or ⚡ mark. Performance qualification also needs the
  capability's acceptance coverage.
- `attest-build` gives a source-qualified build. Configure CMake first; this
  command then runs the build and records the unchanged source, the compiler
  settings and the native library hash. Install that exact library and pass
  the receipt to `run`:

```bash
python -m benchmarks attest-build --build-dir build/hf-native --jobs 2 \
  --out build/hf-build-receipt.json
python -m benchmarks run --tier acceptance --build-receipt build/hf-build-receipt.json \
  --out build/hf-acceptance
```

Every Atomiverse worker checks its loaded binary against the receipt. A
receipt doesn't make a dirty tree a clean release, and it doesn't qualify
CUDA-only residency. Keep receipts and logs under `build/` or outside the tree.

The manual CI workflow includes an HF benchmark job. CUDA and HPC campaigns
run through molq on their declared hardware and emit the same result schema.
Historical regression gates, multi-shard merging and automated profiling are
not implemented. They are future work, and nothing falls back to them.
