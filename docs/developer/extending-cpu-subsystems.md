# Extending the CPU Subsystems

This page is a practical handbook. It walks through three specific
extensions — adding an MD integrator, adding a minimizer, and adding an
SCF inner-step helper — at the level of "what file do I create, what do
I inherit, what do I register, what does the test look like?" The
rationale behind the loop node, its presets, and the role/capability
split is in [CPU Internals](cpu-internals.md). The general rules of
component design are in
[Extending Components](extending-components.md). This page picks up
where they leave off.

The three case studies are deliberately chosen to span the layering:

1. **A new MD integrator** — a Component plugged into the
   `cpu::md::MD` preset.
2. **A new minimizer** — a Component plugged into the
   `cpu::minimizer::Minimize` preset.
3. **A new SCF inner-step helper** — a non-component class owned by
   `cpu::scf::SCFSolver` (inner solver, not a driver).

If you keep the asymmetries between these three in mind, every other
extension you ever write will fit somewhere on the same map.

## Case 1: Adding an MD Integrator

**Goal.** Implement a new symplectic integrator — say, the
Beeman-corrector form of velocity-Verlet — and let users pick it
inside `cpu::md::MD` exactly the way they pick the existing
`VelocityVerlet`.

**Where it lives.** A header at
`include/atomiverse/cpu/md/<name>.h` and an implementation at
`src/cpu/md/<name>.cpp`. The MD subdirectory is the right home because
the component will only ever participate in MD (its phase enum is
`md::Phase`). Putting it under `cpu/md/` also keeps it out of the
minimizer/SCF dependency footprint.

**The skeleton.** Inherit the matching kind (`Integrator`, `Potential`, …),
declare default bindings if the phase is fixed, and override `launch`:

```cpp
#pragma once

#include <string>
#include <vector>

#include "atomiverse/core/binding.h"
#include "atomiverse/cpu/component/integrator.h"
#include "atomiverse/facade/stage.h"

namespace ATV { namespace cpu { namespace md {

class Beeman : public cpu::Integrator {
public:
    explicit Beeman(double dt);

    std::string name() const override { return "beeman"; }
    // role() defaults to 0; override only for Deferred / FrameImmutable
    std::vector<DefaultBinding> default_bindings() const override {
        return {
            ATV::make_binding(ATV::md::Phase::initial_integrate, 0),
            ATV::make_binding(ATV::md::Phase::final_integrate, 0)
        };
    }
    void build (const cpu::BuildContext&)  override;  // optional, for prev-force buffer
    void launch(const cpu::LaunchContext&) override;

private:
    double dt_;
    std::vector<double> fx_prev_, fy_prev_, fz_prev_;  // Beeman needs f_{n-1}
};

}}}
```

Two architectural choices to make consciously:

1. **Default bindings or explicit?** If your integrator has an
   unambiguous home in the LAMMPS phase order (`initial_integrate` and
   `final_integrate` for any predictor-corrector form), declare
   `default_bindings()`. The user then registers it with
   `sim.add(Beeman{dt})` and never has to think about phases. If your
   "integrator" is really a partial step that needs to sit in a phase
   the user must choose (rare, but possible for, e.g., a Suzuki-Yoshida
   chain), return `{}` from `default_bindings()` and let the user
   write `sim.on(...).add(...)` explicitly.
2. **What state lives on the component vs. the frame?** Beeman needs
   $f_{n-1}$ — the previous step's forces. That is a *derived* quantity
   (it can always be reconstructed by reading current forces and
   keeping a copy), so it lives on the component (`fx_prev_` etc.),
   not on the frame. The general rule from [CPU Internals](cpu-internals.md):
   only fundamental physical quantities go on `cpu::Frame`.

**Allocation discipline.** If you store per-step buffers, allocate them
in `build()` (sized to `frame.capacity()`), not in `launch()`. The
zero-allocation invariant in `launch()` is verified by
`tests/cpu/test_md_no_alloc.cpp`, and breaking it will fail review.

**Validation comes for free.** `cpu::md::MD::validate_components`
walks the registered set and requires at least one `Integrator` (and an
energy/force kind). Inheriting `cpu::Integrator` makes Beeman a valid
replacement for `VelocityVerlet`. No driver-side change is needed.

**Test recipe.** Add `tests/cpu/md/test_beeman.cpp` that builds a small
NVE system (typically the FCC Argon fixture from `test_md.cpp`),
substitutes Beeman for VelocityVerlet, runs a few thousand steps, and
asserts:

- Total energy stays bounded (the canonical NVE drift test).
- Position trajectories agree with VelocityVerlet to within
  integrator-difference tolerance over a short interval.
- Time reversibility on a single round trip — propagate $N$ steps,
  flip velocities, propagate $N$ more, compare positions.

Hot-swap testing is the right way to validate "this integrator
implements the MD contract correctly". If your integrator passes
the same fixtures `VelocityVerlet` already passes, the wiring is
correct.

## Case 2: Adding a Minimizer

**Goal.** Add a FIRE (Fast Inertial Relaxation Engine; Bitzek et al.,
PRL 97, 170201, 2006) optimizer alongside the existing
`cpu::minimizer::LBFGS` and `cpu::minimizer::CG`.

**Where it lives.** `include/atomiverse/cpu/minimizer/fire.h` and
`src/cpu/minimizer/fire.cpp`. Same reasoning as above: minimizer
components live under `cpu/minimizer/`.

**The skeleton.** FIRE is structurally similar to LBFGS — it computes a
search direction *and* publishes the numbers by which the loop will be
judged. It does **not** judge them itself. Mirror the existing pattern:

```cpp
#pragma once

#include <string>
#include <vector>

#include "atomiverse/cpu/component/optimizer.h"
#include "atomiverse/cpu/minimizer/lbfgs.h"   // for OptRecord
#include "atomiverse/facade/stage.h"

namespace ATV { namespace cpu { namespace minimizer {

class FIRE : public cpu::Optimizer {
public:
    FIRE(double dt0,           // initial pseudo-timestep (fs)
         double dt_max,        // step ceiling
         int    n_min,         // accelerate after this many P>0 hits
         double f_inc,         // dt growth factor
         double f_dec,         // dt shrink factor on P<0
         double alpha0,        // initial mixing coefficient
         double f_alpha);      // alpha decay factor

    std::string   name() const override { return "fire"; }
    // Optimizer kind; role defaults to 0
    std::vector<DefaultBinding> default_bindings() const override {
        return {
            ATV::make_binding(ATV::minimizer::Phase::search_direction, 0),
            ATV::make_binding(ATV::minimizer::Phase::convergence_check, 0)
        };
    }
    void build (const cpu::BuildContext&)  override;
    void launch(const cpu::LaunchContext&) override;

    const std::vector<OptRecord>& trace() const { return trace_; }

private:
    // FIRE state
    std::vector<double> v_;             // pseudo-velocity, length 3N
    double dt_, dt_max_, alpha_, alpha0_, f_alpha_, f_inc_, f_dec_;
    int    n_min_, n_pos_steps_ = 0;
    // Measurement state (to difference against the previous iteration)
    double energy_prev_ = 0;
    int    step_ = 0;
    std::vector<double> x_prev_;
    std::vector<OptRecord> trace_;
};

}}}
```

Notice what is *not* there: no `ftol`, no `rms_dtol`, no `etol`, and no
`converged()` accessor. **An algorithm never judges itself.** At its
`convergence_check` binding, FIRE *measures* — max force, RMS force,
max and RMS displacement since the previous iteration, |ΔE| — and
publishes them into `ctx.loop.metrics` (`ATV::ConvergenceMetrics`):

```cpp
if (ATV::ConvergenceMetrics* m = ctx.runtime.loop.metrics) {
    m->energy            = e;
    m->max_force         = fmax;
    m->rms_force         = frms;
    m->max_displacement  = dmax;
    m->rms_displacement  = drms;
    m->energy_change     = std::abs(e - energy_prev_);
    m->has_displacement  = (step_ > 0);   // no previous geometry on the first step
}
```

Whether those numbers mean "converged" is decided by a `Criterion`.
`cpu::minimizer::Minimize` attaches `cpu::ForceConverged` by default,
which is the five-criteria test (max force, RMS force, max displacement,
RMS displacement, ΔE — **all five** must hold), with the tolerances now
living on the criterion where the policy belongs. A user who wants
different tolerances writes `opt.add_criterion(cpu::ForceConverged{0.01})`
and never touches the optimizer; a user who wants a genuinely different
criterion writes their own `Criterion`. `LBFGS::converged()` and
`CG::converged()` used to exist and are deleted — ask the loop:
`opt.converged()`, `opt.stop_reason()`.

This is why FIRE is easier to add than it used to be: the exotic
convergence definition that once demanded "its own convergence
component" is now just a criterion, and it is about ten lines.

**Lifecycle.** `build()` allocates `v_`, `x_prev_`, and `trace_` to
sizes proportional to `frame.capacity()`. `launch()` reads forces
from the frame, applies the FIRE update rules to `v_`, computes the new
positions, writes them back, and publishes the metrics above.

**Validation.** `cpu::minimizer::Minimize::validate_components` requires
at least one `Optimizer` and at least one energy/force kind
(`Potential`, `HF`, `Bias`, …). Inheriting `Optimizer` makes FIRE a
valid drop-in for `LBFGS` / `CG`. The driver does not have to change.

**Test recipe.** Extend `test_minimize_driver_swap_optimizer.cpp` to
run the same aspirin / RHF problem with all three optimizers and
assert each converges. Add a stand-alone test for FIRE-specific
behavior (the freeze-and-restart on $P < 0$, the $\alpha$ decay, the
$\Delta t$ ramp). Use a closed-form quadratic fixture if you want a
clean numeric oracle without touching SCF.

A subtlety worth flagging: FIRE has its own **internal** velocity
state $v$ that is *not* the same as the `kAtom_vx/vy/vz` columns (which would
be physical MD velocities). Keeping FIRE's $v$ on the component, not
on the frame, follows the data-residency rule — a different optimizer
might use different state, and the frame stays algorithm-agnostic.

## Case 3: Adding an SCF Inner-Step Helper

**Goal.** Add an alternative DIIS variant — say, ADIIS (Hu & Yang, JCP
132, 054109, 2010) — that the SCF cycle can use in place of the
standard Pulay DIIS for poorly-conditioned starts.

**Where it lives.** This is *not* a Component. It does not inherit
from `cpu::Component`, it does not have `default_bindings()`, and
it never gets registered into a Plan. It is an inner-step helper, and
inner-step helpers live in
[`include/atomiverse/cpu/scf/internal/`](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/cpu/scf/internal/)
in the namespace `cpu::scf::internal`.

The reason for this asymmetry — why optimizers are Components but
DIIS variants are not — is in [CPU Internals](cpu-internals.md): the
SCF cycle does not present a user-pluggable phase order. The user
chooses an SCF method through `SCFConfig` (HF vs KS, restricted vs
unrestricted, MINAO vs Hcore initial guess), not by inserting a
component into a Plan. The five inner steps are an implementation, not
an extension surface.

**The skeleton.** Mirror the shape of `internal::DIISMixer`:

```cpp
#pragma once

#include <deque>
#include <utility>

#include <xtensor/containers/xtensor.hpp>

#include "atomiverse/cpu/scf/dense_matrix.h"

namespace ATV { namespace cpu { namespace scf { namespace internal {

class ADIISMixer {
public:
    explicit ADIISMixer(int max_vectors = 6);

    /// @brief Spin-resolved entry point. Same signature as DIISMixer.
    std::pair<RealMatrix, RealMatrix> step(
        const RealMatrix& F_alpha, const RealMatrix& e_alpha,
        const RealMatrix& F_beta,  const RealMatrix& e_beta);

    /// @brief Single-channel convenience for restricted SCF.
    RealMatrix step(const RealMatrix& F, const RealMatrix& e);

    int  size() const;
    void reset();

private:
    int max_vectors_;
    std::deque<RealMatrix> Fa_hist_, Fb_hist_;
    std::deque<RealMatrix> ea_hist_, eb_hist_;
    // ADIIS-specific quadratic-program state…
};

}}}}
```

The contract is "given a history of Fock matrices and error vectors,
return an extrapolated Fock matrix". Match the existing `DIISMixer`
signature exactly — that minimizes the change inside `SCFSolver`.

**Wiring.** There are two ways to expose ADIIS to users.

1. **As a configuration switch.** Add a `DIISVariant` enum to
   `SCFConfig`:

   ```cpp
   enum class DIISVariant { Pulay, ADIIS };
   struct SCFConfig {
       // ...
       DIISVariant diis_variant = DIISVariant::Pulay;
   };
   ```

   Then `SCFSolver` holds both helpers and dispatches in its inner
   loop:

   ```cpp
   class SCFSolver {
       // ...
       internal::DIISMixer  pulay_;
       internal::ADIISMixer adiis_;
       // pick one in run() based on config_.diis_variant
   };
   ```

   This is the right pattern when the variants are semantically
   interchangeable and the user is choosing between them on a
   problem-by-problem basis.

2. **As an inner replacement (no public switch).** If ADIIS is
   strictly better and you want to retire the Pulay form entirely,
   replace the `internal::DIISMixer` member with `internal::ADIISMixer`
   and make sure the existing tests (`tests/cpu/scf/*`) still pass.
   This is the right pattern when the new helper is a strict
   improvement and the old one no longer pulls its weight.

The first option is more common in early development; the second is
the correct end state once the new helper has demonstrated dominance
across the test corpus.

**Test recipe.** Add `tests/cpu/scf/test_adiis_mixer.cpp` that
exercises the helper directly — feed it a synthetic sequence of Fock
matrices with a known answer (say, a Pulay-difficult harmonic case the
literature has documented), check that the extrapolated Fock matches
the reference. Then add a higher-level test that runs `SCFSolver` on a
small molecule (H2O, STO-3G) once with each variant, and checks that
both reach the same converged total energy to $10^{-9}$ Hartree.

Because the helper is in the `internal::` namespace, *production*
facade-public headers (`cpu/driver/evaluate.h`, `cpu/driver/single_point.h`,
`cpu/md/md.h`, `cpu/minimizer/lbfgs.h`) must not include from
`cpu/scf/internal/`. Tests can include these headers directly. The
boundary keeps `internal::` types from leaking into user-visible APIs.

## Choosing Where Your Extension Belongs

Three questions, in order:

1. **Does the extension live in a phase the user can already see?**
   - If yes (force, integrate, search_direction, end_of_step, …): it
     is a Component. Inherit the matching kind (`Potential`,
     `Integrator`, `Optimizer`, `HF`, …), declare default bindings,
     plug into the existing driver.
   - If no: it is an inner-solver helper. Pick the right inner solver
     class (today: `cpu::scf::SCFSolver`), put the helper in that
     solver's `internal::` directory, give it a `prepare()` /
     `step()` / `reset()` lifecycle, and own it as a member.
2. **Does the extension change a *termination* rule, or just an
   *algorithm*?**
   - Algorithm change: Component, plug in. (FIRE replacing L-BFGS;
     ADIIS replacing Pulay; CSVR replacing Nose-Hoover.)
   - Termination change: a `cpu::Criterion`, attached with
     `add_criterion(...)`. Not a driver class. ("Stop above 2000 K",
     "give up after an hour", "stop when every replica is converged".)
   - Accept/reject change: a `cpu::OutcomePolicy`, attached with
     `set_outcome(...)`. (A GCMC insertion rule, a REMD exchange.) Only
     a genuinely new *stage sequence* — a new phase enum — justifies a
     new driver class, and that is worth a spec before code.
3. **Does the extension cross the host/device boundary?**
   - If yes: read [Architecture](architecture.md) and consider whether
     the right home is the CUDA backend, the CPU backend, or a small
     helper on both with an `AsyncBridge` between them.
   - If no: stay on the CPU side. The CPU subsystem boundaries (md /
     minimizer / scf) under `include/atomiverse/cpu/` are the right
     organizing axis.

If after applying this rubric you still are not sure, search
`.claude/specs/` for a related spec. Several of the historical
extensions in this codebase have a written design rationale, and
matching one of them is usually faster than improvising.

## Documentation Touchpoints

When you ship a new component, update at least three places:

1. The Doxygen `@brief` and `@param` lines on every public method.
   Per CLAUDE.md, every public symbol needs `@brief`, `@param` with
   units, and `@return`.
2. The CLAUDE.md component table in
   [`/CLAUDE.md`](https://github.com/MolCrafts/Atomiverse/blob/dev/CLAUDE.md) — under "Implemented components" —
   so that downstream readers see the new option without searching the
   tree.
3. A reference in the relevant tutorial. If you add FIRE, the
   geometry-optimization tutorial
   ([Chapter 12](../tutorial/ch12-geometry-optimization.md)) gains a
   sentence or two about it. If you add ADIIS, the SCF tutorial chain
   ([ch02-rhf.md](../tutorial/ch02-rhf.md) onward) gets a footnote on the
   alternate accelerator.

The tutorial reference is what users see first; the CLAUDE.md table is
what AI-driven workflows see; the Doxygen blocks are what API readers
see. Skipping any of the three leaves the new component invisible to
one population of consumers.
