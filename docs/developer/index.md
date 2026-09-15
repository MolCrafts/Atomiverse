# Developer Manual

This manual explains how Atomiverse is put together internally. Read it when you
want to **change** the codebase rather than just use it. Code examples and
contributor guidance assume **C++20 or later** throughout.

It still assumes the reader may be new to computational chemistry: where
chemistry terms appear, the goal is to explain why the software is organized the
way it is, not to assume domain knowledge.

## What to read

- **[Architecture](architecture.md)** — start here. One CUDA-free contract
  layer in `core/`, two execution backends, a host-side `cpu::Frame` throughout.
- **[CPU Backend Internals](cpu-internals.md)** — *why* the CPU side has one
  loop node and a handful of presets rather than a driver-category hierarchy,
  why termination is a policy object rather than something an algorithm decides
  for itself, why Role is a bitset rather than an exclusive enum, why a concrete
  SCF method is itself a Driver node with typed stages, and how the
  iteration-status channel signals convergence.
- **[MD Execution Model](md-execution-model.md)** — one timestep as a fixed
  sequence of named **phases**, and the `(phase, order)` binding model a
  component uses to attach to it. Read this before adding or moving any MD
  component.
- **[Extending Components](extending-components.md)** — the component lifecycle
  (`build` → `launch` × N → `destroy`), the registration paths, and where tests
  and examples go.
- **[Extending CPU Subsystems](extending-cpu-subsystems.md)** — three case
  studies (add an MD integrator, add a minimizer, add an SCF inner-step helper)
  at the level of "what file do I create, what do I inherit, what does the test
  look like?"
- **[The Frame](frame-architecture.md)** — what `cpu::Frame` physically is:
  named typed Blocks, exact-dtype meta, and the Box that owns the cell; the
  typed resolution boundary that keeps a column read as cheap as a raw array;
  and the schema that drives MPI packing and the molrs boundary.
- **[KSpace long-range electrostatics](kspace-long-range.md)** — CoulLong +
  Ewald/PME pairing, stress, cell refresh vs rebuild, replicated-grid MPI
  ceiling, and the accepted charged-NPT scope.
- **[Python / molpy boundary](python-molpy-boundary.md)** — the acyclic
  dependency order between molrs, molpy, the C++ library, the CLI and the
  Python package.
- **[Oracle capability loops](oracle-capability-loop.md)** — how a capability
  gets from "an external code can do this" to a landed regression with a
  citable pin.

## Codebase map

| Directory | Role |
|---|---|
| `include/atomiverse/core` | the CUDA-free contract layer both backends share: precision tiers, `Mat3`, units, device tags, Frame schema, the `Driver<PhaseEnum, DeviceTag>` template, `Plan`, `Stage`, `Role`, `Verdict` |
| `include/atomiverse/cpu`, `src/cpu` | CPU backend — frame, drivers, dynamics, montecarlo, minimize, scf, planewave, gpw, semiempirical, observe |
| `include/atomiverse/cuda`, `src/cuda` | CUDA backend — staged components, transfers, async observer bridges |
| `tests` | **unit tests only** — one function or method each, at millisecond scale |
| `regressions` | end-to-end cases, one directory per delivered feature, each checked against a cited external or analytic value |
| `benchmarks` | standing performance harnesses; never registered with ctest |
| `examples` | the numbered tutorial programs |
| `docs` | this site |

Each backend root holds **directories only** — one per pillar (`frame/`,
`driver/`, `component/`), infrastructure cluster (`communication/`,
`neighbor/`, `asynchronous/`, `device/`), or method domain (`dynamics/`,
`montecarlo/`, `minimize/`, `scf/`, `planewave/`, `gpw/`, `semiempirical/`, …).
Names are full English words, and the same feature lives in the same directory
on every backend: virial stress is always `observe/`, neighbor structures
always `neighbor/`. A CPU-only domain is simply absent under `cuda/`, never
renamed there.

## Where does my change belong?

Ask in order:

1. Is it a concept **both backends** must name? → `core`.
2. Is it CPU-only algorithm work (dense linear algebra, post-processing)? → `cpu`.
3. Is it staged GPU execution, device memory, or async host transfer? → `cuda`.
4. Is it a new **method** (a way to get energy and forces)? → a **concrete
   class** owning its own Hamiltonian, parameters and stages — `RHF`, `PW`,
   `GpwDft`, `DFTB2`. There is no generic solver base and no method *category*;
   a method that runs a loop is itself a `Driver<Phase>` node.
5. Is it a new **algorithm** (integrator, optimizer, mover)? → a **Component**,
   never a driver subclass. A new Driver preset is justified only when the loop
   *topology* is genuinely new.
6. Does it need a test? A single-function unit test in `tests/`. Anything
   end-to-end — a full solve, a convergence loop, a comparison against an
   external engine — goes to `regressions/<domain>/<case>/` as an
   out-of-library consumer, never to `tests/`.

## Maintaining these docs

The site is built with [Zensical](https://zensical.org) and the shared
[MolCrafts theme](https://github.com/MolCrafts/molcrafts-zensical-theme).
Configuration and navigation live in `zensical.toml` at the repository root.

```bash
uv run --with zensical --with molcrafts-zensical-theme zensical serve   # live preview
uv run --with zensical --with molcrafts-zensical-theme zensical build   # → site/
```

Adding a page means two edits: the markdown file under `docs/`, and its entry in
the `nav` table of `zensical.toml`. A page absent from `nav` is not published.

**Product docs carry no custom CSS or HTML.** Write plain markdown; all styling
and layout — including the home-page hero, which is rendered from the `hero:`
front-matter block in `docs/index.md` — belongs to the MolCrafts theme. Do not
add stylesheets, `extra_css`/`extra_javascript`, a `custom_dir`, raw HTML
elements, or hand-written theme class names (`molcrafts-*`) to any page. If a
page needs a visual treatment the theme lacks, the fix goes in the theme, not
here. Per-product brand colour is the one exception, and it is configuration,
not markup: `[project.extra.molcrafts].accent` in `zensical.toml`.

The generated C++ API reference is still produced by Doxygen (`doxygen Doxyfile`
→ `build/doxygen/html`) and is not yet part of this site; `docs/api/index.md` is
the hand-written map into it.
