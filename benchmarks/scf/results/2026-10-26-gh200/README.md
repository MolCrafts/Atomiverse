# HF benchmark: GH200, 2026-10-26

The new Python benchmark suite ran live Atomiverse/PySCF comparisons through the
public APIs. This record replaces no old measurement numerically: it has a new
complete-call boundary, public Python client, molpy runtime and explicit inputs.

| Campaign | Full method/backend matrix | Required acceptance | Timing samples |
| --- | --- | --- | --- |
| Full molecular HF acceptance | [20-row matrix](acceptance-cpu8-gpu1/matrix.md) · [HTML](acceptance-cpu8-gpu1/matrix.html) | 60/60; no failures or incomplete required cells | One cold + one warm, no warmup; no qualifying speed ratio |
| 16 waters, CPU threads 1 | [Matrix](water16-performance-cpu1-gpu1/matrix.md) · [HTML](water16-performance-cpu1-gpu1/matrix.html) | 3/3, plus stored CPU passes | Five warm samples, one untimed fresh-state warmup before each |
| 16 waters, CPU threads 8 | [Matrix](water16-performance-cpu8-gpu1/matrix.md) · [HTML](water16-performance-cpu8-gpu1/matrix.html) | 3/3, plus stored CPU passes | Five warm samples, one untimed fresh-state warmup before each |

Full acceptance covers 25 method/system/task workloads in RHF, UHF, ROHF and
GHF. Required cells include live PySCF, the declared native CPU strategy and
implemented CUDA workloads. Eight additional optional CPU cells passed. The
43 optional unavailable cells comprise 25 missing gpu4pyscf cells and 18
explicitly unimplemented strategies/backends. They are not passes.

## Complete-call performance

RHF, STO-3G, 16 waters, 48 atoms, 112 AOs; canonical 4-Angstrom water chain.
All science starts from fresh state. Times include engine input conversion,
construction/setup, integrals, initial guess, SCF, required output and cleanup;
CUDA completion is synchronized. Values are medians in seconds.

| Requested CPU threads | ATV CPU direct | ATV CPU stored | ATV CUDA | PySCF CPU | Hardware / environment notes |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 5.703858 | 14.243243 | 0.570900 | 1.254915 | H1, all recorded thread resources match |
| 8 | 1.464194 | 3.297145 | 0.570016 | 1.854957 | H1, PySCF internal BLAS is fixed at one thread; external ratios withheld |

With CPU math threads set to 1, CUDA is **2.19814x** faster than the live PySCF
CPU reference. At 8 configured threads, the same-product CPU-direct/CUDA median
ratio is **2.56869x**. Both profiles use the same eight-core allocation; the
thread setting does not claim that a separate one-core allocation was reserved.
PySCF's requested-8 configuration does expose OpenMP 8 and other BLAS pools 8,
but its bundled OpenBLAS reports `threading_layer=disabled`, `num_threads=1`.
Under this suite's strict resource policy its external speed ratios are null;
the raw time and complete pool metadata remain visible. Do not describe that
engine as entirely single-threaded.

The maximum CUDA energy error over cold and warm samples is 4.64e-11 Ha.
CPU direct's maximum error is 1.31e-8 Ha; stored CPU is below 3.21e-11 Ha.
All satisfy the existing 1e-6 Ha chain gate. Other molecular rows retain their
1e-8 Ha energy and 1e-6 Ha/Bohr force gates; the H3 GHF noncollinearity gate also
passes. Every sample, observable and error is in the compressed records.

**H1:** one NVIDIA GH200 120GB visible GPU; CUDA reports 102005473280 bytes total
device memory, and `nvidia-smi` reports 97871 MiB. The allocation provides eight
CPU cores (affinity 72–79) and 64e9 bytes host-memory budget; the node reports
288 physical/logical cores and 921772163072 bytes RAM. The CPU model field
returned only `aarch64`; no more specific model was inferred or borrowed from
the login node. Linux 5.14/el9, ATV GCC 14.3.0, CUDA runtime/driver API 13.0,
driver 580.178.04, pthread OpenBLAS 0.3.32, Python 3.13.17, NumPy 2.5.4,
SciPy 1.18.1, PySCF 2.14.0, published molpy 0.20.0 and molci 0.3.0. The native
bridge and wheel use Rust 1.99.1 (`58cf8f9c3`); the ABI guard remains enabled.

## Evidence and reproducibility

- Successful job: **4451058**. Ten public HF binding tests and the original
  21 SCF Python tests passed before the three campaigns.
- Every runnable ATV observation uses native binary SHA256
  `1cdb05d5a7507508f079b81475bccc399d616be6ad74edd10bc90b0cc724d046`.
  The primary PySCF binary is consistently
  `f18b19f24491c3b92b856497d5ef85ace7a1ec1b18b78a96baa19edaa4233657`.
- Measurement source tree:
  `cfd36d03d4a3e6e3190d2dabead270a7bee82ef1489d1fd5adbc14941a32c559`.
  The [source manifest](source-manifest.json.gz) identifies all 3570 files.
- [Measurement source patch](measurement-source.patch.gz), applied to commit
  `54d0b9c9d91bffc680d14f4f687578cf83cf70af`, reconstructs those 3570 file
  contents. Reconstruction was checked against every recorded SHA256.
  The [build-source manifest](build-source-manifest.json.gz) separately retains
  the actual compiled source inventory. Parallel non-HF/XC edits between build
  and measurement are not silently claimed to be part of that binary.
- Each campaign's `measurements.json.gz` preserves the original immutable
  returned bundle. `results.json.gz`, `payload.json.gz`, `matrix.*`,
  `roadmap-status.*`, CSV and `snapshots/benchmark.json.gz` are current report
  projections. Snapshot provenance identifies the comparison/report code hashes.
- Each campaign's `artifacts.tar.gz` preserves its canonical requests, terminal
  observations and logs. Extract it beside the bundle to resolve the recorded
  `artifacts/...` paths. [Collected setup/build/ABI provenance](provenance.tar.xz),
  [verification](verification.json) and the [file checksum inventory](evidence-sha256.json.gz)
  are retained with the public report as well.
- Setup attempts 4450713, 4450743 and 4450945 retained their Python-selection,
  binding-header and Rust-ABI failures. No failed attempt is counted as a
  scientific pass. Original logs, job references, build configuration, wheel
  identities and all 733 collected file hashes remain in the local campaign
  `.claude/campaigns/hf-benchmark-20261026/gpu/results/`.
  Native binaries and redundant full source archives stay in that complete
  local store. The public provenance archive includes their hashes and sizes,
  setup/build logs and manifests; the source reconstruction patch is separate.

To regenerate a view without rerunning science, decompress a campaign bundle
and run the current reporter:

```bash
gzip -dc measurements.json.gz > /tmp/hf-measurements.json
python -m benchmarks report --input /tmp/hf-measurements.json --out /tmp/hf-report
molci validate-snapshot /tmp/hf-report/snapshots/benchmark.json
```

Snapshots have distinct observation IDs and `tracking=false`; they have not
been published to the global dashboard. The measured tree is dirty,
`build_verified=false`, historical regression status is `unbaselined`, and
new automatic CUDA-only trace qualification is incomplete. Numerical acceptance
does not bypass those independent roadmap requirements. The generated status
records retain those gaps instead of upgrading every HF feature to ✓/⚡.

No compatible native aarch64 CUDA13 gpu4pyscf wheel was available in this
allocation. The binary-only resolver output is preserved; no GPU-oracle result
or speedup is invented. GHF direct spinor SCF, GHF forces without a native force
oracle, periodic/spinor/response features and other unmigrated HF capabilities
remain explicit catalog entries, outside these 20 migrated rows.
