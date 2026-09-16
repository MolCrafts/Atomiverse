# Chapter 12. Geometry Optimization

The same `cpu::minimizer::Minimize` preset and `ForceConverged` criterion work
for classical, AM1, and SCF forces — only the force-phase component changes.

> **Build flag.** The ab initio half of this chapter needs `ATV_BUILD_HF=ON`,
> which is **OFF by default** — with it off, the `cpu::scf::…` headers are not
> installed at all (a method domain OFF excludes its sources, headers, tests and
> regressions). The `Minimize` preset itself is `ATV_BUILD_MINIMIZER`, ON by
> default.
>
> ```bash
> cmake -B build -DCMAKE_BUILD_TYPE=Release \
>       -DATV_BUILD_HF=ON -DBLA_VENDOR=OpenBLAS
> cmake --build build -j$(nproc)
> ```

### Fast path: AM1 water

| | Before | After L-BFGS |
|---|---:|---:|
| \(E\) (eV) | −348.295963 | **−348.562886** |
| max\|F\| (eV/Å) | 3.3135 | **0.1513** |
| Status | — | `force_converged` |

### Ab initio path: aspirin RHF/STO-3G

21 atoms, 73 basis functions, 94 electrons. Live 25-step budget:

| | Step 1 | Step 25 |
|---|---:|---:|
| \(E\) (eV) | −16988.935735 | **−17320.369991** |
| max\|F\| (eV/Å) | 1723.87 | **0.434** |
| \(\Delta E\) | — | **−331.43 eV** |

Force thresholds (`ftol=0.05`) are not fully met in 25 steps — the tutorial
stops on the step budget with a clear descent curve. Raise `max_steps` for
`ForceConverged`.

```molplot preset="molplot" theme="auto" width="100%" aspect="16:9"
mark: {type: line, point: true, tooltip: true}
data:
  values:
    - {step: 1, series: energy, y: -16988.936}
    - {step: 5, series: energy, y: -17293.963}
    - {step: 10, series: energy, y: -17316.731}
    - {step: 15, series: energy, y: -17319.181}
    - {step: 20, series: energy, y: -17320.200}
    - {step: 25, series: energy, y: -17320.370}
encoding:
  x: {field: step, type: quantitative, title: L-BFGS step}
  y: {field: y, type: quantitative, title: "E (eV)", scale: {zero: false}}
```

```molplot preset="molplot" theme="auto" width="100%" aspect="16:9"
mark: {type: line, point: true, tooltip: true}
data:
  values:
    - {step: 1, maxF: 1723.87}
    - {step: 5, maxF: 47.59}
    - {step: 10, maxF: 7.95}
    - {step: 15, maxF: 3.36}
    - {step: 20, maxF: 0.772}
    - {step: 25, maxF: 0.434}
encoding:
  x: {field: step, type: quantitative, title: L-BFGS step}
  y: {field: maxF, type: quantitative, title: "max |F| (eV/Å)", scale: {type: log, base: 10}}
```

The `cpu::minimizer::Minimize` driver answers a different question from
the MD chapters. Instead of "how does the system evolve in time?" it
asks "where is the nearest local minimum of the potential energy
surface?" The wiring is closely related — same `cpu::Frame`, same
`SCF` at the force phase — but the outer loop is no longer a
time integrator. It is a quasi-Newton or conjugate-gradient *search*,
and it terminates when the gradient and the displacements between
successive geometries fall below tolerance.

This chapter walks through the full geometry-optimization recipe with
an aspirin (acetylsalicylic acid) molecule at RHF / STO-3G. The example
to read alongside this chapter is
the Practice block.

## What the Minimize Driver Actually Does

`cpu::minimizer::Minimize`
([`cpu/minimize/minimize.h`](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/cpu/minimize/minimize.h))
is a **preset** over the CPU loop node, `cpu::Driver<minimizer::Phase>`.
"Preset" means: it is an ordinary driver whose constructor pre-attaches
the pieces a geometry optimization wants — the zeroing components, and a
termination policy. It owns the loop:

```text
loop until a criterion says stop, or max_iter reached:
    status.reset()                       // clear last iteration's verdict
    run one pass of the compiled plan    // search_direction → line_search →
                                         // force → convergence_check → end_of_step
    ask the criteria                     // Continue / Converged / Stop
```

The driver does not know about L-BFGS, conjugate-gradient, FIRE, or any
specific algorithm — and, just as importantly, it does not know what
"converged" means either. That is the job of a **criterion**
(`cpu::Criterion`), a Component the driver polls once per iteration.
`Minimize` attaches `cpu::ForceConverged` — the five-criteria test
described below — for you. You can replace it:

```cpp
opt.add_criterion(cpu::ForceConverged{/*ftol=*/0.01, /*rms_ftol=*/0.005,
                                    /*max_dtol=*/0.002, /*rms_dtol=*/0.001,
                                    /*etol=*/1e-5});
```

The optimizer component bound at `convergence_check` does not decide
anything. It **measures** — max force, RMS force, the displacement since
the last iteration, |ΔE| — and publishes those numbers into the loop's
`ConvergenceMetrics`. The criterion reads them and returns a verdict. That split
is why you ask the *driver*, never the optimizer, whether the run
succeeded: `opt.converged()` and `opt.stop_reason()`.
(`LBFGS::converged()` used to exist; it is gone.)

Build-time validation at `Minimize::build()` checks that:

1. At least one component has `role() == ComponentRole::Optimizer`.
2. At least one component has `role() == ComponentRole::Evaluator` and
   advertises `Capability::Energy` *or* `Capability::Force`.

If either is missing, `build()` throws with an explanatory message.
Unlike MD, the optimizer side has only one composition rule — pick
*one* optimizer component. The two shipped optimizers, `LBFGS` and
`CG`, are mutually exclusive in the same way Nose-Hoover and Langevin
are.

## The Phase Order

```text
search_direction → line_search → force → convergence_check → end_of_step
```

The `Minimize` preset auto-adds three infrastructure components in its
constructor, plus its default criterion:

- `cpu::ZeroForces` at `force @ -1000`
- `cpu::ZeroEnergy` at `force @ -999`
- `cpu::Publish` (the async-analysis snapshot publisher) at `end_of_step @ +10`
- `cpu::ForceConverged{}` as the loop's termination criterion

Note that `force` is the same logical phase as in MD — same evaluator
components plug into both drivers. The `Zero*` components are
phase-agnostic and are bound explicitly by each driver, which is why the
same class serves `minimizer::Phase::force` here and `md::Phase::pre_force`
there.

The optimizer component is bound at *three* phases: `search_direction`
(it computes a step), `line_search` (it evaluates trial geometries), and
`convergence_check` (it publishes the metrics the criterion will judge).
That is why a single `LBFGS{...}` registration is enough for the entire
algorithm.

## L-BFGS, the Default Optimizer

`cpu::minimizer::LBFGS` is the limited-memory BFGS quasi-Newton
optimizer (Nocedal, *Math. Comp.* 35, 773 (1980)). At every iteration
it stores up to $m$ pairs of position and gradient differences,

$$s_k = x_{k+1} - x_k, \qquad y_k = g_{k+1} - g_k,$$

and uses Nocedal's two-loop recursion to compute a search direction
that approximates the Newton step

$$d_k = -H_k g_k$$

without ever materializing $H_k$. Step zero falls back to steepest
descent. From step one onward the L-BFGS direction is used.

The constructor takes:

```cpp
cpu::minimizer::LBFGS{
    /*memory=*/   5,         // history size m
    /*alpha0=*/   0.01,      // initial step length [Å²/eV], used by LineSearchMode::Fixed
    /*max_ls=*/   10         // maximum backtracking attempts per step
};
```

**Algorithm parameters only — no tolerances.** Those three arguments are
the whole constructor, and that is deliberate: an L-BFGS *algorithm* has
a curvature depth, an initial step and a line-search cap. It does not
have a convergence criterion. Stopping is the loop's business, and the
loop's stopping criteria are the criterion's — `cpu::ForceConverged`,
which `Minimize` attaches for you with the defaults `ftol` 0.05 eV/Å,
`rms_ftol` 0.03 eV/Å, `max_dtol` 0.002 Å, `rms_dtol` 0.001 Å, `etol`
1e-5 eV. To change a tolerance, change the criterion — never the
optimizer:

```cpp
opt.add_criterion(cpu::ForceConverged{0.05, 0.03, 0.1, 0.05, 1e-3});
```

Attaching any criterion suppresses the default one, so the five values
above are the ones that judge the run.

Convergence requires *all five* criteria — max force, RMS force, max
displacement, RMS displacement, and energy change — to be below their
tolerances simultaneously. That is stricter than a single-criterion
test (e.g. "max force < 0.05 eV/Å") and avoids the failure mode where a
flat saddle with a tiny gradient passes but the geometry is still
walking around. If you find yourself loosening tolerances to make a run
succeed, prefer to loosen them all uniformly rather than disable any.

**Line search is on by default.** `LBFGS::line_search` selects a
`LineSearchMode`, and the default is `StrongWolfe` — you opt *out* of the
line search, not into it. The three modes:

| `LineSearchMode` | What each step does |
|---|---|
| `StrongWolfe` (default) | Bracket + zoom enforcing Nocedal & Wright 3.7a+3.7b. `yᵀs > 0` is automatic, so every stored curvature pair is valid. |
| `Armijo` | Backtracking on sufficient decrease (3.6a) only — energy per trial, no gradient. |
| `Fixed` | Legacy step `x -= min(alpha0, cap)·(-p)`. No trial evaluation at all. |

In `StrongWolfe` and `Armijo`, trial geometries $x_k + \alpha p$ are evaluated
by replaying the force phase through `Context::reeval_forces`, and $\alpha$ is
a dimensionless multiplier on the quasi-Newton direction $p = -Hg$ (initial
trial 1.0). `alpha0` is only consulted in `Fixed` mode.

The first argument is the **mode**, not a tolerance:

```cpp
cpu::minimizer::LBFGS step{/*memory=*/5, /*alpha0=*/0.01, /*max_ls=*/10};
step.line_search(cpu::minimizer::LineSearchMode::Armijo,
                 /*c1=*/1e-4,        // sufficient-decrease constant, in (0,1)
                 /*c2=*/0.9,         // Wolfe curvature constant, c1 < c2 < 1
                 /*max_step=*/0.2,   // max trial displacement [Å]; caps alpha
                 /*min_alpha=*/1e-8);
opt.set_optimizer(std::move(step));
```

Two API details that bite: `line_search()` must be called **before**
`opt.build()`, and it returns `LBFGS&` — an lvalue reference — while `LBFGS`
itself is move-only (it owns the curvature history). So `auto step = LBFGS{…}
.line_search(…)` does **not** compile; declare the optimizer, configure it as
a statement, then `set_optimizer(std::move(step))` (or `set_optimizer_ref(step)`
to keep it alive and read `step.trace()` afterwards).

Every trial that fails the sufficient-decrease condition

$$E_{k+1} - E_k \le c_1 \alpha_k g_k \cdot d_k$$

rolls back: positions return to $x_k$, $\alpha_k$ is contracted, and the trial
is re-taken from $x_k$ along the same direction. The $(s, y)$ history is
updated only on accepted trials. `min_alpha` floors the contraction so the
loop cannot run forever on a pathological evaluation, and `max_ls` caps the
attempts per step.

## Conjugate Gradient as an Alternative

`cpu::minimizer::CG` is a Polak-Ribière+ conjugate-gradient optimizer.
It publishes the same metrics as `LBFGS`, and is judged by the same
criterion. The only thing that changes is the direction:

$$\beta_k = \max\!\left(0,\; \frac{g_k \cdot (g_k - g_{k-1})}{g_{k-1} \cdot g_{k-1}}\right), \qquad d_k = -g_k + \beta_k d_{k-1}.$$

The non-negativity projection on $\beta$ already restarts the algorithm
when curvature information becomes ill-conditioned, so there is no
periodic-restart parameter.

To swap CG in for L-BFGS, change one line:

```cpp
opt.set_optimizer(cpu::minimizer::CG{/*alpha0=*/0.01});
```

The rest of the driver setup is unchanged. The hot-swap is exercised in
[`tests/cpu/test_minimize_driver_swap_optimizer.cpp`](https://github.com/MolCrafts/Atomiverse/blob/dev/tests/cpu/test_minimize_driver_swap_optimizer.cpp),
which builds the same molecule twice and checks that both optimizers
converge under the same criterion — the same `ForceConverged` object, not
two copies of the same five numbers.

In practice, L-BFGS converges in fewer iterations than CG on
well-conditioned problems (rigid molecules near minimum), and the
two are roughly comparable when the surface is flat or the gradient is
noisy. CG's smaller per-step memory makes it the right default for
very large systems where storing $2m$ vectors of length $3N$ would be
expensive.

## Putting It Together: Aspirin at RHF/STO-3G

Aspirin is 21 atoms (9 C + 8 H + 4 O), 73 STO-3G basis functions, 94
electrons. It is small enough that an SCF + analytic gradient finishes
in a fraction of a second per step, but large enough that you actually
see L-BFGS doing useful work.

The setup, in full:

```cpp
auto frame = make_aspirin();              // aspirin geometry

// RHF: no XC grid, so each step is a plain SCF + analytic gradient.
// Every parameter is a constructor argument — there is no config bag, and
// the integrals are private to the method.
cpu::scf::RHF rhf{cpu::scf::BasisSet::STO3G, /*charge=*/0,
                  /*tol=*/1e-6,             // [Hartree]
                  /*max_iter=*/200};

cpu::minimizer::Minimize opt;
opt.set_optimizer(cpu::minimizer::LBFGS{5, /*alpha0=*/0.01, /*max_ls=*/10});
opt.add_criterion(cpu::ForceConverged{
    /*ftol=*/     0.05,   // (eV/Å)
    /*rms_ftol=*/ 0.03,   // (eV/Å)
    /*max_dtol=*/ 0.1,    // (Å) — slightly looser; L-BFGS takes ~0.05–0.2 Å steps
    /*rms_dtol=*/ 0.05,   // (Å)
    /*etol=*/     1e-3    // (eV)
});
opt.on(minimizer::Phase::force).add(std::move(rhf));
opt.build();

int iters = opt.run(frame, /*max_iter=*/100);
if (opt.converged())
    std::cout << "converged in " << iters << " iterations\n";
else
    std::cout << "stopped: " << opt.stop_reason() << "\n";  // empty ⇒ ran out of budget
```

Because you attached a criterion explicitly, `Minimize` does *not* also
attach its default one — the preset supplies `ForceConverged{}` only
when you have supplied no criterion of your own.

The single line that does *all* the chemistry is
`opt.on(minimizer::Phase::force).add(std::move(rhf))`.
The evaluator's `launch()` rebuilds the integrals at the current
geometry, runs the SCF cycle, computes the analytic nuclear gradient
($\partial E / \partial R_A$), and writes the negative gradient into
the `kAtom_fx/fy/fz` columns. The optimizer reads those forces at its
`search_direction` pass, decides the new geometry and writes the updated
coordinates back; at `convergence_check` it publishes the force,
displacement and energy metrics that `ForceConverged` will judge.

`Minimize::run` returns the number of iterations actually executed
(≤ `max_iter`). On a converged run that number is your iteration count.
On a non-converged run that number equals `max_iter`, and you can
inspect `LBFGS::trace()` for the per-step record.

## What Loose Tolerances Actually Cost You

The aspirin example above uses `max_dtol = 0.1 Å` — quite loose by
quantum-chemistry standards. The reason is that the L-BFGS line search
is opt-in, and a fixed-step optimizer near a minimum routinely takes
0.05 Å — 0.2 Å steps when the gradient is small but non-vanishing.
Setting `max_dtol = 0.001 Å` on a fixed-step run will keep the loop
running until `max_iter`, even when the energy and forces are clearly
at minimum.

Two clean cures, in order of effort:

1. **Match displacement tolerance to step size.** If your `alpha0` is
   $0.01\,\text{Å}^2/\text{eV}$ and the typical $|g|$ near minimum is
   $0.05\,\text{eV/Å}$, the predicted step is ~$5 \times 10^{-4}\,\text{Å}$,
   so `max_dtol = 1\text{e-}3` is appropriate. Loosen it only if you
   are far from this regime.
2. **Turn on the line search.** With Armijo backtracking the optimizer
   refuses to take steps that increase the energy, and the typical step
   length adapts to the local curvature. Tight displacement tolerances
   become physically meaningful.

The five-criterion rule means you should report all five thresholds
when you publish a geometry, not just the force tolerance.

## Recorders for the Optimization Trace

Three recorder components live under
[`include/atomiverse/cpu/minimize/`](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/cpu/minimize/)
and are designed specifically for optimization runs.

`OptLog` writes a tab-separated row per step with `step`, `E (eV)`,
`max_force (eV/Å)`, `rms_force (eV/Å)`, and `step_norm (Å)`. It
defaults to binding at `end_of_step @ 0` and works against any force
evaluator (classical or SCF). Useful as the canonical
"is-this-converging" log:

```cpp
opt.on(minimizer::Phase::end_of_step)
   .add(cpu::minimizer::OptLog{"aspirin_opt.tsv", /*also_stdout=*/true});
```

`FinalGeometry` writes a single-frame XYZ file exactly once, at the end
of the run. After the run it holds the last-visited geometry — converged
or not. Cheap to enable and ideal for "I just want the final
coordinates":

```cpp
opt.on(minimizer::Phase::end_of_step)
   .add(cpu::minimizer::FinalGeometry{"aspirin_final.xyz"});
```

`TrajectoryWriter` appends a frame per step to an XYZ trajectory.
Useful for visualizing the optimization path or re-running an
animation. Throttle long runs with the `stride` parameter:

```cpp
opt.on(minimizer::Phase::end_of_step)
   .add(cpu::minimizer::TrajectoryWriter{"aspirin_traj.xyz", /*stride=*/1});
```

All three are MPI-aware (rank-0-only writes) and respect the same
build/launch/destroy lifecycle as every other component.

## When Optimization Diverges

Geometry optimization with quantum forces fails for two qualitatively
different reasons. Distinguishing them quickly is what separates a
fifteen-minute fix from a two-day debug.

**The optimizer is the problem** when the energy is converging
monotonically and the forces are shrinking, but the loop hits
`max_iter`. Symptoms in `LBFGS::trace()`: `max_force` plateaus near
the threshold and oscillates; `step_norm` is large because the fixed
step keeps overshooting. Cure: turn on the line search, or tighten
`alpha0`.

**The SCF is the problem** when the energy jumps non-monotonically,
the forces blow up, or the atoms move by tens of Ångström in one step.
Symptoms: SCF's `converged()` is false; the
trajectory shows atoms passing through each other. Cure: this is not
an optimizer bug. Either the SCF tolerance is set too loose
(`config.tol > 1e-6` produces noisy gradients near minimum), the
initial guess is poor (try MINAO via `default_config`), or you are
running the wrong method for the chemistry (e.g. RHF on a singlet
biradical, or a small basis on an anion). The
[SCF tutorials](ch02-rhf.md) cover these cases.

If forces, energy, and SCF are all healthy but the geometry will not
move, you may be at a saddle point. None of the shipped optimizers
follow modes; they all minimize. Use a transition-state search package
or perturb the starting geometry along a guessed mode and minimize from
there.

## What's Next

The exact same recipe — `Minimize` driver, `RHF` at the force
phase, optimizer component at `search_direction` — also drives
classical relaxation. Replace the SCF with `cpu::PairLJCut` and
`cpu::BondHarmonic`, and you have a force-field minimization.
The optimizer does not know the difference, and the convergence
tolerances stay in the same units.

If you came here from [Chapter 15](ch15-cross-method.md), the symmetry is
worth pausing on. The same concrete method plugs into either driver:

```cpp
// Geometry optimization (Optimize loop)
cpu::minimizer::Minimize opt;
opt.set_optimizer(cpu::minimizer::LBFGS{5, 0.01, 10});
opt.on(minimizer::Phase::force).add(cpu::scf::RHF{basis});

// Ab initio molecular dynamics (Dynamics loop)
cpu::md::MD                sim;
sim.set_integrator(cpu::md::VelocityVerlet{dt});
sim.set_thermostat(cpu::md::NoseHoover{T, tau, dt});
sim.on(md::Phase::force).add(cpu::scf::RHF{basis}.predictor(3));
```

Two drivers, one evaluator, two ensembles. That is the architecture
paying off.
