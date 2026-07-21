# Developer Manual

This manual explains how Atomiverse is put together internally. Read it when you
want to **change** the codebase rather than just use it. Code examples and
contributor guidance assume **C++17 or later** throughout.

It still assumes the reader may be new to computational chemistry: where
chemistry terms appear, the goal is to explain why the software is organized the
way it is, not to assume domain knowledge.

## What to read

- **[Architecture](architecture.md)** — start here. One facade API, two
  execution backends, a host-side `cpu::Frame` throughout.
- **[CPU Backend Internals](cpu-internals.md)** — *why* the CPU side has one
  loop node and a handful of presets rather than a driver-category hierarchy,
  why termination is a criterion, why role and capability are orthogonal axes,
  why SCF is an inner solver rather than a driver subclass, and how the
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
- **[KSpace long-range electrostatics](kspace-long-range.md)** — CoulLong +
  Ewald/PME pairing, stress, cell refresh vs rebuild, replicated-grid MPI
  ceiling, and the accepted charged-NPT scope.

## Codebase map

| Directory | Role |
|---|---|
| `include/atomiverse/core` | shared vocabulary: role-specific precision tiers, `Mat3`, units, device tags |
| `include/atomiverse/facade` | backend-agnostic orchestration: `Driver`, `Plan`, phase enums, roles/capabilities |
| `include/atomiverse/cpu`, `src/cpu` | CPU backend — frame, drivers, MD, MC, minimizer, SCF, PW, GPW, semiempirics, observers |
| `include/atomiverse/cuda`, `src/cuda` | CUDA backend — staged components, transfers, async observer bridges |
| `tests` | **unit tests only** — one function or method each, at millisecond scale |
| ~~`benchmarks`~~ | removed for rebuild; regressions hold textbook e2e |
| `examples` | the numbered tutorial programs |
| `docs` | this site |

Each backend root holds **directories only** — one per pillar (`frame/`,
`driver/`, `component/`), infrastructure cluster (`comm/`, `neighbor/`,
`async/`), or method domain (`md/`, `mc/`, `scf/`, `pw/`, `gpw/`, …). The same
feature lives in the same directory on every backend: virial stress is always
`observer/`, neighbor structures always `neighbor/`.

## Where does my change belong?

Ask in order:

1. Is it a concept **both backends** must name? → `core` or `facade`.
2. Is it CPU-only algorithm work (xtensor linear algebra, post-processing)? → `cpu`.
3. Is it staged GPU execution, device memory, or async host transfer? → `cuda`.
4. Is it a new **method** (a way to get energy and forces)? → an inner **solver**
   plus an **Evaluator** component — never a new `Driver` subclass.
5. Is it a new **algorithm** (integrator, optimizer, mover)? → a **Component**,
   never a driver subclass.
6. Does it need a test? Unit test in `tests/`; anything end-to-end or compared
   against an external engine will live in a future benchmarks tree (currently removed). Unit tests stay in `tests/`.

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
