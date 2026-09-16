/// @file src/cpu/driver/dof.cpp
/// @brief publish_global_dof — the single owner of the DOF count.

#include "atomiverse/cpu/driver/dof.h"

#include <stdexcept>
#include <string>

#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/frame/frame.h"

#ifdef ATV_HAS_MPI
#include "atomiverse/cpu/communication/domain.h"
#endif

#ifdef ATV_HAS_MPI
#include <mpi.h>
#endif

namespace ATV {
namespace cpu {

void publish_global_dof(Context& ctx, const Frame& frame, int n_constraints) {
    long long n_global = frame.nlocal();

#ifdef ATV_HAS_MPI
    // Multi-rank: reduce the OWNER count, taken from the decomposition — NOT
    // frame.nlocal().
    //
    // `partition_atoms` compacts this rank's owned atoms to the front and records
    // the count in `dd->n_local`, but it deliberately leaves Frame metadata
    // at the full allocation (the tail is slack for ghosts). So on every rank
    // `frame.nlocal()` is still the GLOBAL atom count, and reducing it would
    // multiply the system size by the rank count. That is not a hypothetical: doing
    // it gave Ndof 8x too large on 8 ranks, which made the MTK barostat's target
    // kinetic energy 8x too large, and the NPT test equilibrated to 2404 K instead
    // of 300 K — exactly 8x the setpoint.
    //
    // This is the same source the thermostats each used before Ndof had an owner.
    if (ctx.mpi_size > 1 && ctx.md.dd != nullptr
        && ctx.md.dd->cart_comm != MPI_COMM_NULL) {
        long long n_local = ctx.md.dd->n_local;
        MPI_Allreduce(&n_local, &n_global, 1, MPI_LONG_LONG, MPI_SUM,
                      ctx.md.dd->cart_comm);
    }
#endif

    // 3 for centre-of-mass removal; the rest to the constraints.
    const long long ndof = 3 * n_global - 3 - n_constraints;

    // Over-constrained is a real error and throws. A DOF-less system with NO
    // constraints is not: a one-atom frame has 3N-3 = 0 by construction, and such
    // frames are legitimate (synthetic single-component tests, a lone atom in an
    // Evaluate). Those consumers already guard on `ndof > 0`, and always did — so
    // publish the honest count and let them, rather than refusing to run a driver
    // that has nothing to do with constraints.
    if (n_constraints > 0 && ndof <= 0)
        throw std::runtime_error(
            "publish_global_dof: the constraints leave no degrees of freedom ("
            + std::to_string(ndof) + ") — " + std::to_string(n_constraints)
            + " constraints on " + std::to_string(n_global) + " atoms");

    ctx.md.n_dof_global = static_cast<int>(ndof > 0 ? ndof : 0);
}

} // namespace cpu
} // namespace ATV
