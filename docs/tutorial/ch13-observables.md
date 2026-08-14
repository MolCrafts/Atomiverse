# Chapter 13. Observables

Observers measure RDF, MSD, VACF, virial stress, Mulliken, and density.
The engine measures; molrs analyzes and stores — analysis math stays out of C++.

Energy conservation is the first observable: if \(E_\mathrm{total}\) drifts,
nothing downstream (RDF, MSD) is trustworthy. Chapter 10 already shows a flat
total-energy series from a real Atomiverse NVE run.

Charges and populations are Frame outputs (`kAtom_charge`, Mulliken observers)
— colour atoms in MolVis by those columns after a SCF/AM1 evaluation
([Chapter 17](ch17-semiempirical-am1.md) water charges).

Every previous chapter produced a *trajectory* — a sequence of frames from an
integrator ([Chapter 11](ch11-ensembles.md)), an optimizer
([Chapter 12](ch12-geometry-optimization.md)), or an AIMD run
([Chapter 15](ch15-cross-method.md)). A trajectory by itself is not a scientific result.
The result is an **observable**: a number or a curve you extract from the
trajectory and compare against experiment or a reference calculation — a
diffusion coefficient, a radial distribution function, a vibrational density of
states.

This chapter is about how Atomiverse turns a running simulation into
observables, and it makes one architectural commitment explicit:

> **The engine measures; molrs analyzes.** The C++ side captures state as the
> loop runs and hands it to [molrs](https://github.com/MolCrafts/molrs) — the
> Rust infrastructure library — which owns every piece of analysis math
> (`compute::MSD`, `compute::RDF`, `compute::VACF`, `fit::EinsteinDiffusion`).
> No correlation function, histogram normalization, or curve fit is
> reimplemented in C++.

You will learn:

1. **The concept.** An observable is a functional of the trajectory, computed
   either on the fly (structure, correlation) or as a post-process at the end
   of the run.
2. **The representation.** A derived quantity is a `role() == Observer`
   **Component** (LAMMPS `compute`), fed by an asynchronous snapshot ring so it
   never stalls the integrator, and backed by molrs for the actual math.
3. **The C++ API.** `set_observer` / `set_observer_ref` attach an observer;
   `ATV::RDF`, `ATV::MSD`, and `ATV::VACF` are the three ready-made ones.
   `set_record` attaches a Recorder — `cpu::XyzRecorder` dumps the raw
   trajectory on a step stride, with `cpu::XYZ` remaining the bare writer
   beneath it for frames you persist by hand.
4. **The expected result.** g(r) with a first peak at the nearest-neighbour
   distance; a linear MSD(t) whose slope is the self-diffusion coefficient; a
   VACF that decays from ⟨v²⟩ and rings.
5. **The validation.** Each observer is checked against an analytic result or a
   physical reference, and the two transport routes (Einstein and Green-Kubo)
   must agree.
6. **The recurring pattern.** The same observer works no matter which driver or
   backend produced the frames — the snapshot ring is the universal seam.

## The Concept: An Observable Is a Functional of the Trajectory

Let a trajectory be the sequence of frames $\{r_i(t), v_i(t)\}$ for atoms $i$ at
sampled times $t = 0, \Delta t, 2\Delta t, \dots$. The observables in this
chapter are all averages over that sequence:

- **Energy conservation** is the simplest: in the microcanonical (NVE) ensemble
  the total energy $E = U + K$ should be constant, so its drift is a direct
  measure of integrator quality — an observable you get for free from
  typed `frame.meta` entries.
- **The radial distribution function** $g(r)$ is a *structural* average — the
  probability of finding a pair of atoms at separation $r$, normalised by the
  ideal-gas expectation. Its first peak is the nearest-neighbour shell.
- **The mean-squared displacement** $\text{MSD}(t) = \langle |r_i(t) -
  r_i(0)|^2 \rangle_i$ is a *transport* average. In the diffusive regime it is
  linear, and the **Einstein relation** reads off the self-diffusion
  coefficient $D$ from its slope,
  $$ \text{MSD}(t) \xrightarrow{t \to \infty} 2 d\, D\, t, $$
  where $d$ is the number of spatial dimensions.
- **The velocity autocorrelation function** $\text{VACF}(t) = \langle v_i(t)
  \cdot v_i(0) \rangle_i$ is the *dynamical* companion of the MSD. Its time
  integral gives the **Green-Kubo** diffusion coefficient (a second, independent
  route to the same $D$), and its Fourier transform is the vibrational density
  of states.

Two of these — g(r) computed from stored pair distances, and VACF/MSD from
stored coordinate or velocity histories — are trajectory functionals that could
be computed *during* the run or *after* it. Atomiverse computes them during the
run, incrementally, so a million-step trajectory never has to live in memory or
on disk all at once.

## The Representation: The Engine Measures, molrs Analyzes

A derived quantity is a **Component** with `role() == ComponentRole::Observer`
— the Atomiverse spelling of a LAMMPS `compute`. But an observer has a problem
the integrator does not: analysis (an FFT, a histogram sweep, a curve fit) is
often more expensive than a single MD step, and running it inline would throttle
the hot loop.

The solution is the **asynchronous snapshot ring**. An observer derives from
`cpu::md::AsyncObserver` and declares, through a `CaptureMask`, exactly which
fields it needs:

```cpp
cpu::CaptureMask capture_mask() const override {
    cpu::CaptureMask m;
    m.positions  = true;   // RDF, MSD
    // m.velocities = true; // VACF
    return m;
}
```

At its configured `frequency()`, the driver copies just those fields into a
lock-free ring buffer and returns to integrating immediately. A separate
analysis worker thread drains the ring and calls the observer's
`on_snapshot(const cpu::FrameSnapshot&)` — the integrator never waits. At the
end of the run the driver calls `finalize()`, where the observer produces its
result.

This design is *backend-uniform by construction*. A CUDA driver publishes into
the same CPU ring after an asynchronous device-to-host copy, so an observer is
CPU-only and works identically whether the frames came from `cpu::md::MD`,
`cuda::md::MD`, or an AIMD run driven by an `SCFEvaluator`. This is the same
"one loop, many physics" seam from [Chapter 15](ch15-cross-method.md), seen from
the analysis side.

Inside `on_snapshot`/`finalize`, the observer does **no analysis math** and holds
**no record type**. It packs each snapshot's raw numbers (positions for MSD/RDF,
velocities for VACF) into a reusable scratch and *streams* the frame into a
molrs accumulator (`cpu::compute::{RDFAccumulator, MSDAccumulator,
VACFAccumulator}`); `finalize()` just reads the finished curve back. State stays
bounded — O(bins) for g(r), O(window · n_atoms) for the correlation rings —
never O(trajectory), so million-step runs stream at constant memory. This is
the [infrastructure-in-molrs policy](https://github.com/MolCrafts/Atomiverse/blob/dev/.claude/specs/molrs-analysis-ffi.md):
molrs already ships a freud-class analysis library, validated by its own Rust
test suite (including streaming-equals-batch equivalence tests). Atomiverse
exposes it through a thin CXX FFI bridge rather than porting the numerics.

## The C++ API: Attach an Observer, Read the Result

Observers are *plural* — you can attach several — so they use `set_observer`,
which binds the component at the phase and order its own `default_bindings()`
declare. Two forms, and the distinction matters here more than anywhere else in
the book:

- `set_observer(Obs&&)` **takes ownership** of an rvalue. The observer lives
  inside the driver and you cannot read its curve afterwards.
- `set_observer_ref(Obs&)` **borrows** an observer you declared yourself. This
  is the form you want whenever you intend to call `rdf.gr()` / `msd.curve()`
  after the run. Declare the observer *before* the driver so the driver is
  destroyed first.

Passing an lvalue to `set_observer` is `= delete`d — there is no implicit
borrow. Build the MD system exactly as in [Chapter 11](ch11-ensembles.md); the
only new lines construct an observer and call `set_observer_ref`.

### The cheapest observable: energy conservation

You do not need a component for this one. Every evaluator writes into the shared
typed Frame metadata, and the integrator writes the kinetic energy, so total energy
is already available each step:

```cpp
auto frame = make_fcc_argon(4, /*a=*/5.32);        // 256 Ar atoms, Å

cpu::md::MD sim;
sim.set_evaluator(cpu::PairLJCut{/*eps=*/0.0104, /*sigma=*/3.4, /*cutoff=*/6.8});
sim.set_neighbor(cpu::md::NeighborList<>{6.8, 0.3});
sim.set_integrator(cpu::md::VelocityVerlet{/*dt=*/4.0});   // NVE — no thermostat
sim.set_thermo(1000);   // print step, T, PE, KE, E_total every 1000 steps
sim.build();
sim.run(frame, 100'000);

const double E_total = frame.meta.get(kEnergy)
                     + frame.meta.get(kKinetic);                  // (eV)
```

In NVE, `E_total` should be constant to a small drift that scales with the
timestep. This is a **physical-approximation** result: you validate it as a
bounded energy drift over the run, and it is the standard first check that an
integrator and force field are wired correctly.

### Dumping the trajectory: the `cpu::XyzRecorder` component

When you want the frames themselves — for external analysis or visualisation —
register a **Recorder**. A Recorder is the Component kind whose only job is
writing outside the process; it is Atomiverse's answer to the LAMMPS `dump`
command. The trajectory Recorder is `cpu::XyzRecorder`, and you install it with
`set_record`:

```cpp
sim.set_record(cpu::XyzRecorder{"trajectory.xyz", /*freq=*/1000});
sim.build();
sim.run(frame, 1'000'000);   // one frame every 1000 steps
```

It binds itself to `md::Phase::end_of_step` (order +10, i.e. behind the
observers that publish per-step quantities), so you never name a phase unless
you want a different one — `on(phase).add(...)` still works if you do.

Three details decide what ends up in the file:

- **The cadence is a stride on the step index.** A frame is written whenever
  `step % freq == 0`, and the driver counts steps from zero
  (`0, 1, …, n-1` for `run(frame, n)`), cumulatively across consecutive
  `run(cpu::Frame&, …)` calls. So a 10-step run at `freq=5` writes **two**
  frames — steps 0 and 5 — not three: there is no extra end-of-run write at
  step 10.
- **The first write truncates, later writes append.** One recorder therefore
  owns one trajectory file, and a freshly constructed recorder on the same path
  starts that file over instead of extending the previous run's.
- **Under MPI it is collective.** Every rank evaluates the same gate and joins
  the owner gather; only rank 0 writes, and what it writes is the complete,
  tag-ordered geometry rather than its own slice.

`set_record` takes an rvalue and *owns* the recorder. If you want to ask the
recorder afterwards how many frames it wrote, declare it first and borrow it —
the same ownership pair as the observers above:

```cpp
cpu::XyzRecorder traj{"trajectory.xyz", /*freq=*/1000};   // declared BEFORE sim
cpu::md::MD sim;
// … evaluator / neighbour list / integrator exactly as above …
sim.set_record_ref(traj);                // borrow: traj must outlive sim
sim.build();
sim.run(frame, 1'000'000);
const int n_frames = traj.n_frames_written();   // rank 0
```

The element symbol on each atom line comes from the frame's per-atom `type`
column, which the writer hands to molrs as an atomic number $Z$ — type 18 is
written as `Ar`. Frames typed with LAMMPS-style `1..N` type indices will
therefore carry the element names of those small $Z$ values, which is a
labelling artefact, not a physics one.

Underneath the Recorder sits `cpu::XYZ`, a plain writer with `write(frame)` and
`append(frame)` and no Component base. Use it directly when the frames are not
coming from a driver loop at all — dumping one final geometry, say — and use
the Recorder inside a run.

Keep dumps sparse either way: the point of the observers below is that you
rarely need the full trajectory on disk — the observable is computed as the run
proceeds.

### Structure: the radial distribution function `ATV::RDF`

`ATV::RDF` streams each snapshot's positions + cell into the molrs
`RdfAccumulator` — pair binning (per-frame LinkCell) and ideal-gas shell-volume
normalisation all live in molrs; the observer keeps only O(n_bins) state:

```cpp
// r_max (Å), n_bins, capture frequency [steps], output file
ATV::RDF rdf{/*r_max=*/6.0, /*n_bins=*/200, /*freq=*/100, "rdf.dat"};
sim.set_observer_ref(rdf);   // borrow: rdf must outlive sim
sim.build();
sim.run(frame, 1'000'000);
// rdf.dat now holds two columns: r (Å), g(r); rdf.gr() exposes the curve
```

Choose `r_max` no larger than half the shortest box length (the minimum-image
convention), and enough bins that the first peak is well resolved. The output is
a **physical** result: for the FCC argon system the first g(r) peak sits at the
nearest-neighbour distance $a/\sqrt{2}$.

> **The engine measures, molrs analyses.** Like `ATV::MSD` and `ATV::VACF`
> below, the C++ observer only *captures* state — every line of binning and
> normalisation lives in molrs `compute::RDF`, reused verbatim across the
> Python and WASM bindings. The engine holds no duplicate histogram.

### Transport, Einstein route: `ATV::MSD` and self-diffusion

`ATV::MSD` streams positions into the molrs `MsdAccumulator` (Direct mode,
first captured frame = reference) and, at `finalize()`, reads the MSD(t) curve
plus the Einstein $D$ from the windowed-MSD slope fit:

```cpp
// dt between captured frames = timestep * freq = 4 fs * 25 = 100 fs
ATV::MSD msd{/*dt=*/100.0, /*freq=*/25, "msd.dat"};
sim.set_observer_ref(msd);   // borrow: msd must outlive sim
sim.build();
sim.run(frame, 2'000'000);

const std::vector<double>& curve = msd.curve();   // MSD(t) [Å²], curve[0] == 0
const double D = msd.diffusion();                 // Einstein D [Å²/fs]
```

The `dt` you pass **must** equal `timestep * freq` — it is the physical time
between captured frames, the abscissa of MSD(t). `diffusion()` returns the
Einstein coefficient from a linear fit over the diffusive window (it is `NaN` if
there are too few frames — a degenerate input never aborts across the FFI).
The optional fourth constructor argument `window` (default 256) caps the lag
range of the windowed-MSD fit; it also bounds the observer's memory to
O(`window` · n_atoms), so arbitrarily long runs stream at constant footprint.
Convert to laboratory units with $1\ \text{Å}^2/\text{fs} = 0.1\ \text{cm}^2/\text{s}$.

### Transport, Green-Kubo route: `ATV::VACF`

`ATV::VACF` streams per-frame velocities into a molrs accumulator
(lagged-product sums over a `resolution`-deep ring buffer — numerically
identical to the FFT-based, DOF-averaged batch autocorrelation) and reads the
curve at `finalize()`:

```cpp
// dt (fs), resolution (max lag in captured frames, >= 1), freq [steps], output
ATV::VACF vacf{/*dt=*/100.0, /*resolution=*/256, /*freq=*/25, "vacf.dat"};
sim.set_observer_ref(vacf);  // borrow: vacf must outlive sim
sim.build();
sim.run(frame, 2'000'000);

const std::vector<double>& c = vacf.curve();   // VACF(t), c[0] == ⟨v²⟩
```

`resolution` bounds both the max lag of the curve and the observer's memory
(O(`resolution` · n_atoms)); runs shorter than `resolution` frames resolve
every lag they contain.

The VACF requires a constant atom count over the captured window (single rank,
no domain migration). Its running time-integral is the Green-Kubo diffusion
coefficient — a second, independent estimate of the same $D$ obtained above from
the MSD — and its power spectrum is the vibrational density of states.

## The Expected Result

For the FCC argon system near its triple point (T ≈ 94 K, the LJ parameters
above), a well-formed run gives:

- **Energy conservation:** total energy constant to a small drift; a good NVE
  run with a 4 fs step keeps the relative drift well below a percent over
  10⁵ steps.
- **g(r):** zero below the repulsive wall, a sharp first peak at the
  nearest-neighbour distance $a/\sqrt 2$, a structured second shell, and
  $g(r) \to 1$ at large $r$.
- **MSD(t):** a short ballistic ($\propto t^2$) region at very small $t$
  crossing over to a linear diffusive region; `diffusion()` reads the slope of
  the linear part.
- **VACF(t):** starts at $\langle v^2 \rangle = d\,k_B T / m$, decays, and for a
  dense liquid dips negative (backscattering from the cage) before relaxing to
  zero.

Every one of these carries its **validation class** (see [Chapter
1](ch01-foundations.md) and the table below): energy conservation and g(r) are
*physical* checks, the MSD analytic test is a *numerical* check, and the
Einstein-vs-Green-Kubo agreement is a *physical-consistency* check.

## The Validation

The analysis chain is validated at two levels, matching the "engine measures,
molrs analyzes" split.

**molrs owns the numerics.** Each `compute`/`fit` kernel has its own Rust unit
test inside molrs; those are the source of truth for the mathematics.

**The C++ side validates the FFI round-trip and the physics.** Atomiverse tests
assert that the bridge reproduces the Rust result and that the observable is
physically correct:

- **MSD** — a two-atom trajectory with known linear displacement (atom 0 moves
  $+a$ per frame in $x$, atom 1 moves $+b$ per frame in $y$) has the closed form
  $\text{MSD}(t) = t^2 (a^2 + b^2)/2$. The test asserts the FFI curve matches
  this to `1e-9` (a **numerical** check) and that a degenerate trajectory
  returns an empty curve rather than aborting.
- **Einstein D** — a valid trajectory yields a finite $D$; degenerate input
  yields `NaN`. Numerical correctness of the fit is molrs's responsibility.
- **RDF** — on an FCC argon lattice the first g(r) peak falls at the FCC
  nearest-neighbour distance (a **physical** check), and the histogram
  accumulates across snapshots at the configured frequency.
- **Energy conservation** — bounded NVE drift is exercised by the MD regression
  suite and the LJ/PE benchmark cases.

The designed physical-consistency check for this chapter is that the **Einstein
$D$ (from `ATV::MSD`) and the Green-Kubo $D$ (from the `ATV::VACF` integral)
agree** within a documented tolerance on the same trajectory — two routes, one
number.

## The Recurring Pattern: Observers Are Backend- and Method-Uniform

The through-line of this book is that you change *method* by swapping the
force-providing evaluator and change *hardware* by swapping the backend
namespace, while the loop, identity columns, and exact typed-meta keys stay fixed.
Observables are where that pattern pays off a second time.

An observer never asks *how* the trajectory was produced. It subscribes to the
snapshot ring, receives `positions` and/or `velocities`, and computes. So the
exact same `ATV::MSD` measures self-diffusion whether the forces came from
`PairLJCut` (classical MD), an `SCFEvaluator` (AIMD, [Chapter 15](ch15-cross-method.md)),
or a plane-wave evaluator — and whether the loop ran on the CPU or, after an
asynchronous device-to-host copy, on a GPU. The snapshot ring is the universal
seam between *producing* state and *analysing* it, exactly as the `Force`
capability is the universal seam between the loop and the physics.

And because the analysis itself lives in molrs, adding an observable is a matter
of bridging one more `compute`/`fit` kernel and wrapping it in a ~60-line
observer — never of writing new numerics in C++.

## Validation Reference

The three observers are exercised as fast, millisecond-scale unit tests that
assert the molrs FFI round-trip and the observer plumbing:

- MSD (analytic curve to `1e-9`, Einstein-D bridge, empty-on-degenerate):
  [`tests/cpu/observe/test_msd.cpp`](https://github.com/MolCrafts/Atomiverse/blob/dev/tests/cpu/observe/test_msd.cpp)
- VACF (FFI round-trip, buffer accumulation):
  [`tests/cpu/observe/test_vacf.cpp`](https://github.com/MolCrafts/Atomiverse/blob/dev/tests/cpu/observe/test_vacf.cpp)
- RDF (histogram accumulation, first-peak position, output format, frequency):
  [`tests/cpu/observe/test_rdf_observer.cpp`](https://github.com/MolCrafts/Atomiverse/blob/dev/tests/cpu/observe/test_rdf_observer.cpp)

Energy conservation and ensemble-level observable averages are physical
benchmark cases under [`benchmarks/md`](https://github.com/MolCrafts/Atomiverse/blob/dev/benchmarks/md) (the LJ and
polyethylene validations), not unit tests. The molrs analysis kernels
(`compute::MSD`/`RDF`/`VACF`, `fit::EinsteinDiffusion`) are validated by molrs's
own Rust test suite, which is the source of truth for the analysis mathematics;
the design and task status for the whole bridge live in
[`.claude/specs/molrs-analysis-ffi.md`](https://github.com/MolCrafts/Atomiverse/blob/dev/.claude/specs/molrs-analysis-ffi.md).
