# Scientific benchmarks

The Python suite compares public Atomiverse methods with live independent
engines, measuring accuracy, end-to-end wall time and resources. Its entry point
is `python -m benchmarks`. It does not use the removed regression runner or
register scientific calculations with CTest.

The first delivery is [molecular HF](scf/README.md): RHF, UHF, ROHF and GHF,
20 method/model/task rows, 25 acceptance workloads and an additional 16-water
performance workload. Unsupported backends and unmigrated capabilities are
explicit cells/catalog entries, not passes.

Install the public HF-enabled ATV extension and published
`molcrafts-molpy==0.20.0` in one environment. Install the coordinated
molcrafts-ci 0.3 checkout before `requirements.txt` until that protocol is
released. The native bridge must satisfy the runtime's ABI fingerprint check;
never bypass it to accept a different Rust build. Optional GPU oracles are
discovered by actual import/execution, not assumed from a package name.

```bash
python -m pip install -e ../molcrafts-ci
python -m pip install -r benchmarks/requirements.txt
python -m benchmarks list
python -m benchmarks catalog
python -m benchmarks run --tier smoke --threads 1 --repeats 1 --warmup 0 \
  --out build/hf-smoke
```

Each campaign writes the authoritative `results.json`, every cold/warm
observation and log, detailed CSV, `matrix.md`/`matrix.html`, roadmap evidence,
JUnit and `snapshots/benchmark.json`. The final table column identifies actual
hardware/environment. A controlled speedup needs five samples, passing
observables, consistent native binary fingerprints and matched resource policy;
raw times remain visible otherwise.

Local runs can explicitly add `--no-snapshot` to produce scientific results,
tables and raw evidence without the optional molci CI envelope. This choice is
recorded in the result protocol; the default still requires the coordinated
molcrafts-ci 0.3 API and never substitutes an older envelope schema.

Run `python -m benchmarks validate --input <results.json>` for the strict result
contract and `molci validate-snapshot <snapshot.json>` for its storage envelope.
`report` regenerates views from the observations. `source` captures hashes for
transporting a frozen source snapshot; `run --source-manifest` verifies every
declared source before and after execution. It does not infer that an arbitrary
installed library was compiled from that source.

`roadmap --input <results.json> --chapter hartree-fock.md --write` updates only
the chapter's generated evidence block; `--check` detects drift. Local dirty
builds, missing coverage and unqualified CUDA windows cannot promote ✓/⚡.
Performance qualification also needs the capability's acceptance coverage;
a performance-only subset cannot inherit an unmeasured basis or method branch.
Historical regression gates, multi-shard merging and automated profiling are
future work, not hidden fallback behavior.

For a source-qualified build, configure CMake first, then use `attest-build`
to perform the build and record unchanged source, compiler settings and the
native library hash. Install that exact library and pass its receipt to `run`:

```bash
python -m benchmarks attest-build --build-dir build/hf-native --jobs 2 \
  --out build/hf-build-receipt.json
python -m benchmarks run --tier acceptance --build-receipt build/hf-build-receipt.json \
  --out build/hf-acceptance
```

Every ATV worker verifies the loaded binary against that receipt. A receipt
does not make a dirty tree a clean release, and it does not qualify CUDA-only
residency. Store receipts/logs under ignored `build/` or outside the source tree.

The existing manual CI workflow contains an HF benchmark job. CUDA/HPC campaigns
run through molq on their declared hardware and emit the same result schema.
The molcrafts-ci Benchmark view reads the method/backend matrix, hardware notes,
scientific summary and per-observation provenance. Current local snapshots are
untracked; coordinated CI/package changes must land before public publishing.
