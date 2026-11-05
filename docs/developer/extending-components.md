# Extending Components

Most new runtime features in Atomiverse are added as components. This page is a
practical guide to doing that without fighting the architecture.

## Step 1: Pick The Right Kind Of Extension

Before writing code, decide which of these categories your change belongs to.

If the new feature is part of the staged runtime loop on the GPU, it should be a
CUDA component under `cuda` and `src/cuda`.

If the feature is host-side analysis, formatting, file output, or a reference
solver that works naturally with xtensor, it usually belongs under `cpu` as a
plain function or CPU helper type.

If the feature changes the public workflow structure itself, such as adding a
new phase family or changing how plans are compiled, it belongs in `core`.

## Step 2: Choose The Execution Phase

Every component has to answer one question clearly: at what stage of the run
should it execute?

Some components declare `default_bindings()` because their natural phase is
obvious. `VelocityVerlet`, `LBFGS`, and `CG` fall into this category.

Other components deliberately return no default bindings because they are
cross-driver tools. `NeighborList`, `Thermo`, `Recorder`, and `RDF` are examples.
Those components must be placed explicitly with `.on(phase).add(...)`.

If you cannot explain the phase choice in one sentence, you probably have not
finished designing the component yet.

## Step 3: Respect The Lifecycle Contract

The lifecycle contract is the main rule contributors must not break.

Use `build()` for one-time setup. This is where a CUDA component may allocate
device memory, create handles, or upload constant data. It is also where a CPU
component may size work buffers.

Use `launch()` for the repeated work only. In the CUDA backend, this should be
the part you would still want to run one million times. Hidden allocation,
device-to-host transfers, and synchronization in `launch()` are almost always a
design mistake; host-side output belongs in an asynchronous consumer.

Use `destroy()` for cleanup only. Free what you allocated in `build()`, and do
not assume components are destroyed in the same order they were built. The
backend tears them down in reverse plan order.

## Step 4: Decide Whether You Need `AsyncConsumer`

If your component needs state from each step on the CPU and then performs
host-side logic, do not reimplement any copy, ring or worker plumbing
yourself. Derive from `AsyncConsumer` (`cuda::AsyncObserver` /
`cuda::AsyncRecorder`, `cpu::md::AsyncObserver`): declare the columns you
read (`capture_columns()`, plus a device payload block if you need one) and
your cadence (`frequency()`), and implement `consume` / `on_snapshot`, which
runs on your own worker thread with an immutable copy of the step.

That is the right tool for observers and recorders whose real logic is CPU-side:
thermo printing, trajectory writing, RDF accumulation, Mulliken analysis, and
similar post-processing tasks. The driver's shared capture buffer captures
each step once for every consumer due on it, is lossless (a full buffer makes
the run wait), and stops the run with your error if `consume` throws.

The split should look like this:

1. The consumer subclass declares what it reads and when.
2. CPU helper owns formatting, analysis, or file writing.

This keeps GPU orchestration and host logic separate and makes the code much
easier to test.

## Step 5: Wire The New Component Into Real Usage

A component is not finished when it compiles. It is finished when there is a
clear way to use it and a regression test that protects it.

The normal checklist is:

1. Add or update the public header under `...`
2. Implement the source file under `src/...`
3. Add a runnable example if the feature is user-facing
4. Add a test under `tests`
5. Update the user or developer docs if the public workflow changed

Examples are especially important in this repository because many readers learn
the project from the examples first and the API reference second.

## Step 6: Audit The Data Path

Any change that touches simulation state must be checked along the full path.

If you add a field to `cpu::Frame`, ask whether it also needs:

- transfer into `cuda::Frame`
- retrieval back to `cpu::Frame`
- zeroing or initialization rules
- recorder or observer support
- test coverage on both CPU and CUDA paths

This audit matters because the host frame is the user-facing truth, while the
device frame is the execution-time mirror. Forgetting one side usually produces
bugs that look mysterious at runtime but are structurally simple.

## A Minimal Design Checklist

Before opening a pull request, make sure you can answer yes to these questions.

1. Does the component have a clearly justified phase and ordering?
2. Does `launch()` avoid one-time work that belongs in `build()`?
3. If the component crosses the GPU-to-CPU boundary, is it an `AsyncConsumer`?
4. Is there at least one example or test showing how the feature is meant to be used?
5. Did you update the documentation entry point that a new contributor would read first?
