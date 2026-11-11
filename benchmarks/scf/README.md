# Molecular electronic-structure benchmarks

These declarations compare public Atomiverse SCF methods with a live independent
PySCF calculation using the same molecular input and Hamiltonian. They are manual
scientific benchmarks, not CTest regressions. Each method exports `SPEC`; the
shared framework owns processes, timing, numerical comparisons and reports.
Campaign output goes under `build/` or outside the source tree, never into git.
The native C++ HF diagnostic (`bench_hf_cpu` and `profile_hf_cpu.py`) lives in
[`../perf/scf/`](../perf/README.md#scf).

## Environment and native backends

Run from the repository root with Python 3.12 or later, NumPy, SciPy, PySCF,
threadpoolctl, Pydantic and the Atomiverse native extension. Frame exchange needs
the **published `molcrafts-molpy` 0.20.0 wheel** (`import molpy`); the distribution
name is not `molpy`. Rebuild the extension against the compatible molpy interface.
An older extension missing the public controls is reported as a binding gap.

CPU participants use `atomiverse.scf.RHF`, `UHF`, `ROHF` and `GHF` with the public
`max_memory_mb`, `init_guess`, damping, gradient convergence and DIIS controls.
CPU exact HF has one automatic J/K engine: it caches eightfold-symmetric
integrals within the memory budget and streams screened shell quartets otherwise.
The former public `integral_mode` switch and separate direct/stored participants
have been removed.
OpenMP uses the runtime default unless `OMP_NUM_THREADS` or `set_threads(n)`
sets a count. MPI is off by default; build with `-DATV_USE_MPI=ON` to enable
MPI functionality. `method.mpi = true` enables collective CPU HF J/K on an
initialized `MPI_COMM_WORLD`; every rank supplies the same full Frame. The
cache is replicated and J/K contraction is distributed. CUDA HF remains local
to its device.
The binding also exposes actual `backend_info()` and GHF
`spin_density_block("alpha", "beta")`. CUDA builds additionally expose
`CudaRHF`, `CudaUHF`, `cuda_backend_info()` and `cuda_synchronize()`.
`run(frame)` is energy-only; `forces(frame)` is a separate complete solve with
analytic forces. CUDA participants reject reported CPU SCF/force fallback.

Snapshot output requires the updated **molcrafts-ci / `molci` observation API**:
`Manifest(record="benchmark", observation_id=...)`, plus `Snapshot`, `Source`
and `Tracking`. An older installation without `observation_id` is incompatible;
a package version string alone does not certify that API. Check the installed
interfaces before starting a long campaign:

```python
import importlib.metadata as metadata
import molci
from atomiverse import scf

assert metadata.version("molcrafts-molpy") == "0.20.0"
assert "observation_id" in molci.Manifest.model_fields
print(scf.backend_info())
```

`gpu4pyscf` and a compatible CuPy/CUDA installation are optional. Missing optional
engines remain visible in the matrix. They do not become zero-error results or
CPU fallbacks. CUDA compilation and execution follow the repository's approved
GH200 workflow; a CPU preflight does not validate those binaries.

## Coverage and explicit gaps

HF has four method declarations (`bench_rhf`, `bench_uhf`, `bench_rohf`,
`bench_ghf`); `python -m benchmarks list` prints their row IDs. Energy and
nonzero analytic-force tasks have separate rows. Inputs live in
[`../systems/hf.json`](../systems/hf.json).

| Method | Current scientific scope | Native ERI strategies |
| --- | --- | --- |
| RHF | H2/water; STO-3G, Cartesian 6-31G*, spherical def2-SVP; 16-water STO-3G chain | CPU automatic exact J/K; optional CUDA energy, STO-3G forces |
| UHF | H2+ doublet and O2 triplet; distinct Hcore/MINAO starts; STO-3G and def2-SVP | CPU automatic exact J/K; optional CUDA STO-3G MINAO energy/forces |
| ROHF | H2+ and O2 triplet, STO-3G and def2-SVP energy/forces | CPU automatic exact J/K; CUDA not implemented |
| GHF | H2 collapse in STO-3G/def2-SVP; noncollinear H3/STO-3G energy and spin-mixing gate | Required native CPU automatic exact J/K; CUDA is an explicit gap |

GHF uses the same J/K engine for nonsymmetric spin-off-diagonal exchange.
H3 uses the declared localized 120-degree Neel seed, constructed during each
solve, and requires a noncollinear converged density as well as agreement with
live PySCF.

PySCF's native `GHF.nuc_grad_method()` is unimplemented. This benchmark therefore
does not invent a GHF force oracle from archived values or a custom derivative.
That force capability remains an explicit catalog migration gap. Other HF
capabilities, including periodic/spinor methods, response and integral/solver
infrastructure, also remain explicit gaps in [`../roadmap.toml`](../roadmap.toml).
Ordinary molecular rows do not silently certify those broader capabilities.

The O2/def2-SVP CPU rows are deliberately retained. Their open-shell
higher-angular-momentum coverage can be expensive, particularly at one thread;
a time budget should accommodate that cost rather than drop a slow case after
seeing its result.

## Commands and tiers

Use a fresh output directory for each campaign. Examples below do not submit
jobs or provision hardware.

`--method rmp2` compares complete RHF-to-canonical-MP2 calls on Cartesian
STO-3G water and an eight-water chain. The smoke tier covers one water. Each
engine computes its own RHF reference and MP2 correlation, and the benchmark
compares the reference, total, correlation and two spin-channel energies.

`--method rmp3` uses the same Cartesian STO-3G systems and times fresh RHF and
all-electron canonical MP3. It compares the RHF, total, correlation, second-order
and third-order energies against PySCF RHF and MO integrals with the NumPy
[spatial MP3 diagram sum](../../tools/pyscf_oracle/mp3.py).

```bash
python -m benchmarks list
python -m benchmarks catalog

# All small STO-3G smoke workloads; one cold and one warm call, no warmup.
python -m benchmarks run --tier smoke --threads 1 --repeats 1 --warmup 0 \
  --out build/hf-smoke

# Full declared acceptance inputs, including O2/def2-SVP.
python -m benchmarks run --tier acceptance --threads 1 --repeats 1 --warmup 0 \
  --timeout 300 --out build/hf-acceptance

# On an approved GPU allocation: a targeted, explicitly partial performance run.
python -m benchmarks run --method rhf --tier performance --system water_chain16 \
  --threads 1 --repeats 5 --warmup 1 --require-cuda --timeout 300 \
  --out build/hf-water16-performance

# Standalone method entry point uses the same framework and exit status.
python -m benchmarks.scf.bench_rhf --tier smoke --threads 1 --repeats 1 --warmup 0 \
  --out build/rhf-smoke
```

`smoke` contains seven small STO-3G method/task workloads. `acceptance` contains
25 workloads and excludes the large chain. `performance` includes all acceptance
systems plus the 16-water chain, so an unrestricted performance campaign is not
a small workload. `--system` makes selection explicit and marks coverage partial;
it cannot promote the entire HF roadmap to complete coverage.

The chain contains exactly 48 atoms and 112 STO-3G AOs. Water coordinates are
O `(4w, 0, 0.1173)`, H `(4w, 0.7572, -0.4692)`, H
`(4w, -0.7572, -0.4692)` Angstrom for `w=0..15`, matching the native
`perf/scf/bench_hf_cpu` water geometry. Ordinary small/high-basis energy tolerances are 1e-8 Ha,
analytic-force component gates are 1e-6 Ha/Bohr, and the chain energy gate is
1e-6 Ha. These tolerances are not loosened for an unavailable engine or timeout.

## Timing, metadata and interpretation

Every timed call creates a fresh Frame/molecule and solver, builds the initial
density, runs the full public calculation, reads the necessary outputs and
cleans up. CUDA work is synchronized before returning. Imports, runtime setup
and binary fingerprints are prepared by the adapter factory; scientific
integrals, densities and orbitals are not. Warmup also uses fresh state; no
converged molecular solution is reused in a measured sample.

The framework records complete-call time separately from process time, which
also includes imports and startup. It retains every cold/warm sample, failures,
raw logs and observables. Performance ratios require the framework's sample,
numerical, source/build and resource gates. Acceptance with one warm sample
provides no qualifying performance ratio.

These are **matched-accuracy** comparisons. Native CPU, CUDA and PySCF do not
have identical stopping criteria. Resolved metadata records actual public
controls; CUDA uses its native density/energy tolerance and orbital threshold,
while unavailable getters/final residuals are marked `not_exposed`. Requested
CPU-only controls are not presented as applied CUDA settings.

ATV observations include the loaded binary path/SHA256, actual package and molpy
versions, compiler/backend information, thread pools and resolved settings.
PySCF reports its actual thread configuration. For example, a wheel without
OpenMP may still report one PySCF thread after a request for eight;
`resource_mismatch=True` prevents a ratio. File/build identity is not inferred
from a repository commit or a clean-looking source tree.

Device free-memory checkpoints are not allocation peaks. A numerical CUDA result
or a public on-device flag alone is not complete CUDA-only residency evidence;
trace qualification must cover the relevant scientific windows. Missing oracle
results cannot pass comparisons. Reports retain unsupported strategy/backend
columns and catalog gaps.

## Molecular density-functional theory

`rks` and `uks` use the public Python KS methods and a live PySCF reference.
Inputs live in [`../systems/dft.json`](../systems/dft.json).
Run the complete declared acceptance matrix on a CPU compute allocation:

```sh
python -m benchmarks run --method rks --method uks --tier acceptance \
  --threads 1 --repeats 1 --warmup 0 --timeout 300 --no-snapshot \
  --out /path/to/fresh/dft-acceptance
```

The rows name their XC ingredients explicitly: LSDA uses VWN5; B3LYP uses
VWN-RPA; PBE0 includes 25% exact exchange. Standard CAM-B3LYP uses libxc
functional 433 on the reference side, including short-range ITYH-B88 and
VWN5. PBEsol, revPBE and RPBE use their explicit exchange/correlation pairs.
Both engines use the declared
MatchPySCF/level-3 quadrature. Native `RKS.grid` and `UKS.grid` accept `SG0`,
`SG1` and `MatchPySCF`; this benchmark explicitly selects `MatchPySCF` rather
than relying on their different defaults. PySCF forces include moving-grid
response, and every force component is compared at 1e-6 Ha/Bohr, alongside
the 1e-8 Ha energy gate. Spherical def2-SVP PBE/PBE0/CAM rows cover the Cartesian
XC-density transformation needed by analytic gradients.

The PBEsol, revPBE and RPBE rows cover only the declared STO-3G molecules.

Each observation constructs and releases a fresh solver, molecule and grid.
Gaussian CUDA KS is explicitly unavailable in this matrix; CUDA PW and GPW
have different representations and require their own validation. This command
records numerical acceptance and diagnostic timings; it does not qualify a
performance ratio or replace the remaining DFT roadmap gaps.
