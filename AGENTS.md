# AGENTS.md — Atomiverse

This file holds the project's shared instructions for coding agents. Step-by-step
procedures live in `.agents/skills/`. Codex reads `AGENTS.md` by name. Claude Code
v2.1.277 and later can load it directly according to its
[project instruction settings](https://code.claude.com/docs/en/memory#agentsmd).

## What this is

Atomiverse is an engine for multiscale, multiphysics atomistic simulation:
classical MD, Monte Carlo, geometry optimization, vibrational analysis and
electronic structure (Gaussian SCF, plane-wave DFT, GPW, NDDO, DFTB/xTB) as
one staged loop over one system object. C++20, namespace `ATV::`, with a
nanobind Python package on top and a Rust toolchain for the molpy CXX bridge.
Two backends, CPU and CUDA, sit behind one Driver/Component/Frame API; the
CPU side is far ahead, and the CUDA electronic-structure surface is narrow.

The full binding contract is `docs/developer/project-contract.md`; this file
carries its stable core; the contract holds the full mechanism detail.

| Path | What |
|---|---|
| `include/atomiverse/`, `src/` | mirrored header / implementation trees: `core/` (the CUDA-free contract both backends share), `cpu/`, `cuda/` |
| `tests/` | unit tests only (gtest `atv_tests`, plus `atv_tests_cuda` / `atv_tests_mpi`) |
| `regressions/` | archived scientific case sources and oracle records (`REFERENCE.md`); no runner, not built, not registered with CTest |
| `benchmarks/` | scientific benchmark framework (`python -m benchmarks`), native perf harnesses (`perf/`), cross-code comparisons (`compare/`); never registered with ctest; no results in git |
| `examples/` | API examples by audience: `cpp/` (numbered), `python/`, `cli/`, `extensions/`; compiled by the unit gate, never run by it |
| `docs/` + `zensical.toml` | the manual (Zensical); `docs/roadmap/` is the **only** place that states what is implemented on which backend |
| `app/`, `python/` | the `atomiverse` CLI and the nanobind wheel (`python/tests/` for Python tests) |
| `tools/` | data generators, oracle pin scripts (`tools/*_oracle/`), debug harnesses, `cpu_job.sh` / `gpu_job.sh` batch templates |
| `cmake/AtvProductModules.cmake` | the `ATV_BUILD_<DOMAIN>` flags and module DAG |

**molpy** (MolCrafts/molpy) supplies the Frame, I/O and analysis. The engine
*measures*; molpy *analyzes and stores*. Interactive builds use the local
checkout in `ATV_MOLPY_PATH` (default `../molpy`): the `molcxx/` bridge and
the `molpy/` wheel must come from that same checkout, and no revision is
pinned. molpy never imports Atomiverse. Capsule names come from the linked
bridge (`molrs::frameref_capsule_name()`), are engine-private, and are never
inferred from package names; capsule validation remains required.

**License.** GPL-2.0-only, like LAMMPS, so parallel machinery (domain
decomposition, halo/forward/reverse comm, pack/unpack, k-space parallel FFT
decomposition and remap, load balancing, neighbor-build parallel structure)
is **ported from LAMMPS, not independently derived**. Read the LAMMPS
implementation first (`../../lammps/src`, at `e8c4c278a8`; deepen or
re-clone rather than trust memory) and transplant its algorithm and
communication structure; do not invent an alternative and A/B it. Every
ported file carries a provenance header naming the LAMMPS source file and
keeps the Sandia copyright notice; stamp the SHA you read into every header
you touch, and do not wholesale rewrite headers naming the older `78a7ddf` or
`970d4ba`. Public API stays ATV-native. Physics/serial kernels (SCF, GPW, TB,
pair math) are outside the mandate.

## Design preferences

### Iron laws

Binding; enforced by this file and code review, never by grep gates
registered as tests.

1. **Tests are unit tests — nothing else** (see Testing).
2. **Frame is the only public token.** System state (positions, species,
   charges, masks/regions, topology, energies, forces, box, meta) lives on
   `Frame` = `blocks` + `meta` + `box`. Algorithm parameters live on the
   owning type; other components go in categorical slots (`set_evaluator`,
   `set_engine`, `set_integrator`). `cpu::Context` / `cuda::Context`,
   `LaunchContext`, `SolverContext` and service slices are engine-private:
   users never define, fill, subclass or pass a Context, and never hand-drive
   Component lifecycle. No parallel `*System`, public `AtomList` or
   `SinglePoint` class.
3. **No public Config bags, no factories.** Parameters are constructor
   arguments (`RHF{BasisSet::STO3G, charge, tol}`) or public members.
   Forbidden: `*Config` / nested `::Config`, `obj.param = …` after default
   construction as the parameter story, `make_*` / `build_*` / `create_*` /
   `*_config()` returning objects. Narrow exceptions: true free math with no
   owner, lookup-table constructors (`GthPotential(Element, name)`),
   subsystem-private logging/async infra, `ATV::make_binding`, the comm
   `make_*_plan` makers.
4. **Region and selection are Frame fields**, never `set_region` /
   `set_mask` / `set_links` / constructor index lists. A missing or invalid
   required field throws with a how-to. Derived membership is fine when it is
   a pure function of Frame data. Named exception: Gaussian SCF ghost indices
   (`MeanFieldMethod::ghost_atoms`).
5. **Frame columns are `span`, meta is `get`**: `frame.span(ColKey)`,
   `frame.meta.get(MetaKey)`. No `get_column`; pythonic names belong on
   nanobind.
6. **Composition over inheritance.** AIMD = `MD` + an electronic evaluator;
   QM/MM = the same + MM evaluators + a connection-layer evaluator — no
   `QmmmDriver`, no second MD. A new preset needs a genuinely new loop
   topology. Inheritance depth ≤ 2 (`Concrete → Kind → Component`); no
   category driver bases.
7. **No freestanding integral bag.** Integral state is private fields on the
   concrete method, rebuilt from Frame; engines take a non-owning
   `internal::IntegralView`. Public API is `RHF{basis, …}.run(frame)` /
   `.forces(frame)`, same shape on CUDA.
8. **Every method domain has its own build flag** `ATV_BUILD_<DOMAIN>`; OFF
   excludes its sources, headers, tests and regressions. A new domain adds
   its flag in the same change. Dependencies live in the module DAG
   (`atv_force_on`). molpy is derived: products force `ATV_BUILD_MOLPY=ON`,
   never the inverse; no capability probe gates sources.
9. **No silent debt.** On finding an anti-pattern, failing test, broken
   invariant or clear bug in the surface you touch or depend on: fix it now
   if local, otherwise stop and report `path:line`. Never "pre-existing",
   never skip-mark, never weaken an assert, never land on known rot. Name it
   in the summary (found / fixed / blocking). Outranks "stay in scope" and
   "minimal diff".
10. **CUDA-only means no device→host synchronization** inside a run loop (MD
    step, SCF iteration, force body): no stream/event/device synchronize, no
    synchronous `cudaMemcpy` / `cudaMemset`, no D2H into pageable memory, no
    allocation, no host branch on a value computed this step. Decisions are
    made on the device; host consumers get data asynchronously. Only setup,
    the end-of-run retrieve and the error path may synchronize. The one
    sanctioned wait is lossless capture backpressure on a host condition
    variable in `Component::before_launch`.

### Defaults

- **OOP by default.** Domain concepts are types with methods
  (`NeighborList.build`); free functions only for true free operations.
- **Primitive, single-responsibility public APIs**: construct → configure →
  one concern → read result. No `run_everything` / `compute_all` /
  `pipeline` façade.
- **Inline until the second real use** (or until a unit test must target it).
- **No fallback values.** Lookup tables throw on unsupported input.
- **Hot-path data carry only fundamental physical quantities**, never
  indices, caches or derived maps.
- **No backward compatibility**: no shims, alias headers, deprecated
  forwarding or `#error` tombstones.
- **No analysis math in C++** on either backend; it belongs to molpy.
- **Known debt is not extended.** The CPU/CUDA execution split, CUDA
  multi-GPU (`cuda::MultiGpu`), nested electronic methods, residual `int`
  step counters and divergent ES public shapes are listed in the contract's
  "Known debt"; do not build on them. Namespace/directory alignment is a
  deliberate follow-up, not a drive-by.

### Conventions

- **Namespaces** `ATV`, `ATV::cpu`, `ATV::cuda`, `ATV::md`, `ATV::minimizer`,
  `ATV::sampling`, `ATV::path`, `ATV::scf`, `ATV::pw`, `ATV::evaluate`; code
  under `internal/` lives in the matching `internal` namespace.
- **Extensions**: CPU `.h`/`.cpp`, CUDA `.cuh`/`.cu`, core `.h`; CUDA-free
  bridge headers under `cuda/` keep `.h`.
- **Units** follow the run's LAMMPS style (`set_units` in C++, `units=` in
  Python). Frame results use the run style and electronic solvers convert at
  their native-unit boundaries. Public results with fixed units document them.
  See [`docs/developer/unit-contract.md`](docs/developer/unit-contract.md).
- **Precision**: state it or state the integer role (`LocalInt`, `TagInt`,
  `ImageInt`, `BigInt`); `Real`/`Int` do not exist. Step counters are
  `BigInt`.
- **SIMD is portable xsimd only** (14.3.0): no arch intrinsics, per-arch
  kernels or global `XTENSOR_USE_XSIMD`. A vectorized reduction lands only
  with a stated tolerance; never behind a loop-invariant branch.
- **libxc**, when enabled: native XC matches it to 1e-12.
- **API naming**: `declare_*` = structure, `require_*` = dependency, `on_*` =
  event; never mix. Hand-written `static_cast<int>` over a Phase enum is
  banned; use `ATV::slot_of(phase)` and `ATV::make_binding(phase, order)`.
- **Directories**: each backend root holds directories only, full English
  names, mirrored across backends; a CPU-only domain is absent under `cuda/`,
  never renamed. Shared code goes in the matching pillar directory.
- **C++ style** is pinned by `.clang-format`.

## Testing

`tests/` holds only single-function / single-method unit tests mirroring the
source layout, linked into one gtest executable `atv_tests` (plus
`atv_tests_cuda` / `atv_tests_mpi`; a dedicated executable only for a test
that needs its own link line). The suite runs in seconds; presets time out a
test at 10 s.

- **No end-to-end of any kind** in `tests/`: no full SCF/MD/vib/MC solve,
  convergence loop, timing or external-engine comparison.
- **No reading outside the repository** at build or test time. In-tree
  parameter data (`basis_data/`) is first-party code with provenance.
- **No style / surface / architecture gates**: no grep scrubs, no scripts via
  `add_test`, no test reading the source tree. No opt-in switch: if it is
  registered, it runs.
- **Python** tests live in `python/tests/` (pytest); benchmark-framework
  units in `tests/benchmarks/`.
- **`regressions/` is archived evidence**, not a current acceptance gate. An
  oracle is a number another code printed (program + version/sha + input +
  output line), a literature value with citation, an analytic result, or a
  `cpu-twin` of a pinned CPU case. Never an oracle: Atomiverse's own earlier
  output, an Atomiverse-only finite difference as the sole check, a numpy
  re-implementation, or a twin with a different Hamiltonian. `tools/*_oracle/`
  only parses oracle output, never reimplements the algorithm.
- After class-layout changes, rebuild every consuming translation unit;
  mixed-layout binaries are not numerical evidence.
- Never set `OMP_PROC_BIND` / `OMP_PLACES` when timing a BLAS-bound path
  (SCF, GPW, plane-wave, DFTB SCC): it single-threads OpenBLAS.

## Build and gates

Hooks and CI run the pinned MolCrafts CI (`molci`, profile `cpp-python`,
revision in `.github/partners.env`); `.pre-commit-config.yaml` installs the
pre-commit and pre-push hooks. Commit checks cover clang-format, hygiene,
workflow policy, dependency locks and release versions. Push checks
(`molci verify`) build the full CPU product set, all serial CPU units,
examples and perf harnesses, the native wheel, Python tests, benchmark units,
strict docs and the pure-Python packages; `--full` reruns every gate.
Validation uses its own pinned molpy under `build/ci-partners/molpy`, separate
from `ATV_MOLPY_PATH`. Never kill a `git commit` / `git push` mid-hook.

```bash
CI_REF=$(sed -n 's/^CI_REF=//p' .github/partners.env)
uv tool install --force "molcrafts-ci @ git+https://github.com/MolCrafts/molcrafts-ci@$CI_REF"
molci run partners                 # prepare the pinned validation checkout
molci doctor --install-hooks
molci check                        # commit gate
molci verify                       # push gate (--full: rerun all)
molci run native                   # native build + tests
molci run cuda                     # CUDA unit gate (aarch64 GH200 only)
molci run package --full           # pure wheel + sdist into dist/
cmake --workflow --preset cpu-unit # manual classical configure+build+test
cmake --preset cpu-unit-hf && cmake --build --preset cpu-unit-hf && ctest --preset cpu-unit-hf
```

Presets (`CMakePresets.json`): `cpu-unit` (classical, optional physics OFF),
`cpu-unit-hf`, `cpu-unit-pw`, `cpu-unit-gpw`, `cpu-unit-kspace`,
`cpu-unit-all`. Work on one domain with its closure only (e.g. SE-only:
`SEMIEMPIRICAL` + `EVALUATE` ON, everything else OFF). Key options:
`ATV_BUILD_CUDA`, `ATV_BUILD_HF` (needs OpenMP), `ATV_USE_MPI`,
`ATV_BUILD_PW` / `ATV_BUILD_KSPACE` (FFTW3), `ATV_BUILD_GPW` (HF + PW),
`ATV_BUILD_TESTS` / `_EXAMPLES`, `ATV_REAL_STORAGE` / `ATV_REAL_WORK`,
`ATV_ID_SIZE`. `-march=native` is required for the SCF ERI kernels.

**Cluster.** Interactive builds run in a login shell (`bash -lc`) so Lmod is
defined: `module load buildenv-gcc/2026.03-mpich`, export
`OPENBLAS_ROOT="$OPENBLASDIR"`,
`RUSTUP_TOOLCHAIN=stable-x86_64-unknown-linux-gnu` and mirror `LIBRARY_PATH` into `LD_LIBRARY_PATH` (skipping that is the usual
cause of `libopenblas.so.0` / `GLIBCXX` errors). Never pipe `module load`.
CUDA builds and runs only on the aarch64 GH200 node: submit and inspect jobs
through the public `molq` CLI within the authorized budget, reuse a running
job, never duplicate an uncertain submission. `tools/gpu_job.sh` /
`tools/cpu_job.sh` are parameterised templates (`ATV_CMAKE_FLAGS`,
`ATV_TARGETS`, `ATV_CTEST_RE`), not one script per task.

CI (`.github/workflows/test.yml`): upstream PRs and master pushes run
independent native validation plus platform lanes (Linux full CPU, macOS
native, Windows pure-Python); `test / result` must succeed. `nightly.yml`
runs the HF scientific campaign (`molci run benchmark-hf`); `release.yml`
publishes on tags. CUDA is never scheduled on GitHub runners.

## Architecture

Pillars: **Driver** (owns the loop), **Component** (one single-step
operation), **Frame** (state). `core/` holds only what both backends name.

- **A Driver is a staged loop, not a category**: typed phase enum + iteration
  + termination policy. Presets (`MD`, `Minimize`, `MC`, `Evaluate`,
  `Hessian`, `PathDriver`) attach default criteria — defaults, not
  semantics. `EvaluateDriver` / `DynamicsDriver` / `OptimizeDriver` are never
  the answer.
- **Algorithms never judge themselves.** `LBFGS`/`CG` publish
  `ConvergenceMetrics`; a `Criterion` judges; a component may request a stop,
  never declare convergence. An `OutcomePolicy` (MC accept/reject) decides
  whether an iteration counts.
- **A method is concrete.** `RHF`, `UHF`, `RKS`, `AM1`, `PW`, `GpwDft`,
  `DFTB2` own their Hamiltonian, parameters and stages; no generic solver
  base or method category. A genuine SCF/SCC loop is a `Driver<Phase>` node;
  single-pass methods do not gain artificial Drivers. Properties (charges,
  dipoles) are component outputs written to the Frame.
- **A new algorithm is a Component** (integrator, optimizer, mover), never a
  driver subclass. Lifecycle `build()` → `launch()` × N → `destroy()`; no
  allocation in `launch()`. `on(phase)` is the only public C++ registration
  path: `add(rvalue)` transfers, `add_ref(lvalue)` borrows. Observer
  computes; Recorder writes I/O; never both. A protocol or declared
  requirement earns a place only when something scans for it.
- **Ghost data is a lifetime contract**: a component that reads a ghost
  column declares it in `declare_ghosts()` (`core/ghost_contract.h`) in the
  same change, or it reads garbage silently. The contract→wire mapping lives
  once, in `make_wire_plan` (`src/cpu/communication/atom_pack.cpp`); never
  borrow the migration wire for anything else.
- **Box** stores lattice vectors in columns (`r = H s`) on both backends.
  `Frame` alone owns `nlocal`, `nghost` and capacity. Consumers resolve atoms
  by tag where rows move.
- **Frame identity** is two columns: `kAtom_type` (force-field type index)
  and `kAtom_atomic_number` (species Z). A source lacking one leaves it out —
  never zero-fill or derive one from the other — and the consumer throws.
  `from_molpy()` / `to_molpy()` are a strict schema boundary, once per run.
  No method scratch on the Frame.
- **CUDA classical MD is an OpenMM cluster/tile engine**: no per-atom Verlet
  default path, no `NonbondedStyle` dual architecture. Generic algorithms use
  CUB, Cooperative Groups, Thrust, cuRAND Philox and cuFFT directly — no
  handwritten twin, no forwarding wrapper.
- **Electronic-structure order**: CPU architecture → CPU performance → CUDA
  by method slice (same-settings CPU is the correctness oracle, plus external
  pins) → large-system CUDA performance. Small molecules are
  correctness-only. Mid-path device↔host exchange is a defect.

Mechanism detail lives in `docs/developer/` (`architecture.md`,
`cpu-internals.md`, `md-execution-model.md`, `frame-architecture.md`,
`extending-components.md`) and the contract.

## Documentation

The manual is built with Zensical and the shared MolCrafts theme from
`zensical.toml`; the C++ reference is generated separately by Doxygen
(`doxygen Doxyfile` → `build/doxygen/html`, mapped by `docs/api/index.md`).

```bash
uv run --with zensical --with molcrafts-zensical-theme zensical serve   # live preview
cmake -B build/docs -DATV_DOCS_ONLY=ON && cmake --build build/docs --target docs
```

- **Adding a page is two edits**: the markdown under `docs/` and its `nav`
  entry in `zensical.toml`; a page absent from `nav` is not published.
- **No custom CSS or HTML** in product docs: no stylesheets,
  `extra_css`/`extra_javascript`, `custom_dir`, raw HTML or `molcrafts-*`
  class names. Missing visual treatment is fixed in the theme. The one
  exception is `[project.extra.molcrafts].accent` in `zensical.toml`.
- **`docs/roadmap/`** is the only place that states what runs on which
  backend; a capability is claimed only with evidence. Read it before
  planning work around a method.
- **Doxygen**: every public symbol has `@brief`, `@param` (with units) and
  `@return`.

## Workflow

Develop on `dev`; worktrees branch off `dev` explicitly. `master` is the
released snapshot, advanced only by merging `dev` — never commit to it. When
they diverge, `dev` describes current behaviour. Dev changes go through PRs:
push a feature branch to `MolCrafts/Atomiverse` and open the PR against
`dev` (the `Roy-Kid/Atomiverse` mirror is outside the fork network and cannot
open one).

Releases ship the pure `atomiverse` wheel + sdist to PyPI; the native
extension is built on the user's machine from the matching tag. The single
version source is `__version__` in `python/src/atomiverse/__init__.py`. To
cut one: bump `__version__` on `dev` (keep the CMake project version
consistent), run `molci verify --full`, merge `dev` into `master`, then tag
`master` (`git tag -a vX.Y.Z -m "atomiverse X.Y.Z" master`) and push the tag
upstream. `release.yml` requires the tag to match `__version__`, the commit to
be on `master` with a green master-push `test / result`, and publishes by
trusted publishing from the `pypi` environment. Manual dispatch is a dry run.
