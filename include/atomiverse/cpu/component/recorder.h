/// @file cpu/component/recorder.h
/// @brief cpu::Recorder — external I/O sinks.
#pragma once

#include "atomiverse/cpu/component/component.h"

namespace ATV {
namespace cpu {

/// @brief True when `ctx` belongs to a single-pass plan (`cpu::Evaluate`,
///        LAMMPS `run 0`): the pass itself is the setup output, so its
///        records are labelled `ctx.step` and every cadence admits them.
bool single_pass_output(const LaunchContext& ctx);

/// @brief Step label of an in-loop recorder write [completed iterations]
///        (LAMMPS `update->ntimestep`, incremented at the top of every
///        `Verlet::run` step / `Min::iterate` iteration): `ctx.step + 1` at
///        the end of iteration `ctx.step` of an iterating loop (MD,
///        minimization, …), `ctx.step` in a single pass.
inline BigInt loop_output_step(const LaunchContext& ctx) {
    return single_pass_output(ctx) ? ctx.step : ctx.step + 1;
}

/// @brief Writes outside the process (files, logs, …).
class Recorder : public Component {
  public:
    Role role() const override {
        return RoleFlag::FrameImmutable | RoleFlag::Deferred
               | RoleFlag::ExternalIO;
    }
};

} // namespace cpu
} // namespace ATV
