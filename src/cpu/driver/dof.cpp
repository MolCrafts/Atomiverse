/// @file src/cpu/driver/dof.cpp
/// @brief publish_global_dof — the single owner of the DOF count.

#include "atomiverse/cpu/driver/dof.h"
#include "atomiverse/core/dynamics_math.h"

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
    // Frame records the owned rows on this rank; reduce owners only.
    if (ctx.mpi_size > 1 && ctx.md.dd != nullptr
        && ctx.md.dd->cart_comm != MPI_COMM_NULL) {
        long long n_local = frame.nlocal();
        MPI_Allreduce(&n_local, &n_global, 1, MPI_LONG_LONG, MPI_SUM,
                      ctx.md.dd->cart_comm);
    }
#endif

    // 3 for centre-of-mass removal; the rest to the constraints.
    const long long ndof = ATV::md::degrees_of_freedom(n_global, n_constraints);

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

void publish_constraints_by_type(Context& ctx, std::vector<BigInt>& tallies,
                                 bool known) {
#ifdef ATV_HAS_MPI
    if (ctx.mpi_size > 1 && ctx.md.dd != nullptr
        && ctx.md.dd->cart_comm != MPI_COMM_NULL) {
        int known_local = known ? 1 : 0;
        int known_all = 0;
        MPI_Allreduce(&known_local, &known_all, 1, MPI_INT, MPI_MIN,
                      ctx.md.dd->cart_comm);
        known = known_all != 0;
        long long size_local = static_cast<long long>(tallies.size());
        long long size_all = 0;
        MPI_Allreduce(&size_local, &size_all, 1, MPI_LONG_LONG, MPI_MAX,
                      ctx.md.dd->cart_comm);
        tallies.resize(static_cast<std::size_t>(size_all), 0);
        std::vector<long long> send(tallies.begin(), tallies.end());
        std::vector<long long> recv(send.size(), 0);
        if (!send.empty())
            MPI_Allreduce(send.data(), recv.data(),
                          static_cast<int>(send.size()), MPI_LONG_LONG, MPI_SUM,
                          ctx.md.dd->cart_comm);
        tallies.assign(recv.begin(), recv.end());
    }
#endif
    ctx.md.n_constraints_by_type = tallies.empty() ? nullptr : tallies.data();
    ctx.md.n_constraint_types = static_cast<int>(tallies.size());
    ctx.md.constraints_by_type_known = known;
}

} // namespace cpu
} // namespace ATV
