/// @file cpu/record/thermo.cpp
/// @brief CPU thermodynamic output component.

#include "atomiverse/cpu/record/thermo.h"
#include "atomiverse/core/log.h"

#include <utility>

namespace ATV {
namespace cpu {

Thermo::Thermo(int freq, std::vector<std::string> keywords)
    : freq_(freq),
      style_(keywords.empty() ? ThermoStyle{} : ThermoStyle{keywords}) {}

// LAMMPS src/output.cpp Output::setup / Output::write, src/verlet.cpp
// Verlet::run and src/min.cpp Min::run, e8c4c278a8. Copyright (2003) Sandia
// Corporation. GPL-2.0-only. `update->ntimestep` is incremented at the top
// of every step, so the end-of-step line at timestep N reports the state
// after N steps. Setup prints the header and the current timestep at the
// start of every run (`thermo->header(); thermo->compute(0)`); the loop
// prints when the timestep is a multiple of `thermo_every` and on the run's
// last step (`next_thermo = MIN(next_thermo, update->laststep)`; with
// `thermo_every == 0` only the last step); a minimization stopped early
// prints its final step; `last_thermo` keeps a step from printing twice
// within a run.

void Thermo::setup_output(const cpu::LaunchContext& ctx) {
    print_(ctx, ctx.step, /*header=*/true);
}

void Thermo::launch(const cpu::LaunchContext& ctx) {
    // A single pass (Evaluate, LAMMPS `run 0`) is its own setup output.
    const bool single = single_pass_output(ctx);
    const BigInt step = loop_output_step(ctx);
    const bool on_cadence = freq_ > 0 && (step % freq_) == 0;
    if (!single && !on_cadence && !ctx.runtime.md.is_last_step)
        return;
    if (step == last_step_)
        return;
    print_(ctx, step, /*header=*/!header_printed_);
}

void Thermo::final_output(const cpu::LaunchContext& ctx) {
    if (ctx.step == last_step_)
        return;
    print_(ctx, ctx.step, /*header=*/!header_printed_);
}

void Thermo::print_(const cpu::LaunchContext& ctx, BigInt step, bool header) {
    last_step_ = step; // every rank: the never-twice rule stays rank-identical
    if (ctx.runtime.mpi_rank != 0)
        return;
    const auto& f = ctx.frame;
    ThermoSample sample;
    sample.step = step;
    sample.pe = f.meta.get(kEnergy);
    sample.ke = f.meta.get(kKinetic);
    // The integrator's conserved quantity (extended Hamiltonian for
    // Nosé-Hoover NVT/NPT); 0 when no integrator populates it.
    sample.conserved = f.meta.get(kConserved);
    sample.volume = f.box.volume();
    const auto virial = f.meta.get(kStress);
    const auto kinetic = f.meta.get(kStressKinetic);
    for (int a = 0; a < 6; ++a) {
        sample.virial[a] = virial[a];
        sample.kinetic[a] = kinetic[a];
    }
    // Ndof has ONE owner (Driver::publish_dof_): 3N - 3 - n_constraints.
    sample.ndof = ctx.runtime.md.n_dof_global;
    if (header) {
        ATV_INFO("{}", style_.header());
        header_printed_ = true;
    }
    ATV_INFO("{}", style_.line(sample));
}

} // namespace cpu
} // namespace ATV
