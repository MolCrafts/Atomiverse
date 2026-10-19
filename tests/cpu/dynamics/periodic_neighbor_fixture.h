#pragma once

#include <span>

#include "atomiverse/core/ghost_contract.h"
#include "atomiverse/cpu/communication/comm.h"
#include "atomiverse/cpu/driver/runtime.h"
#include "atomiverse/cpu/frame/pbc.h"

namespace ATV::test {

// Direct-kernel fixture: use the real communication owner to prepare periodic
// image rows, without running a trajectory or copying the halo algorithm.
//
// `ghosts` is the kernel under test's own `declare_ghosts()` table (e.g. the
// charge a Coulomb style reads on a neighbour); the fixture folds it onto the
// engine base set the way the Driver does, so the border pass carries it.
struct PeriodicNeighborFixture {
    cpu::Runtime runtime;
    cpu::Context& context;
    GhostContract contract;

    explicit PeriodicNeighborFixture(cpu::Frame& frame, double reach,
                                     std::span<const GhostField> ghosts = {})
        : context(runtime.context()) {
        runtime.init();
        runtime.set_frame(frame);
        contract.declare(base_fields::kEngine);
        contract.declare(ghosts);
        context.md.ghost = &contract;
        cpu::md::wrap_owned(frame, frame.nlocal());
        context.md.comm->set_ghost_cutoff(reach);
        const cpu::LaunchContext launch{context, frame, 0, 0};
        context.md.comm->setup(launch);
        context.md.comm->exchange(launch);
        context.md.comm->borders(launch);
    }
};

} // namespace ATV::test
