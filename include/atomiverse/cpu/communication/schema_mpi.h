/// @file cpu/communication/schema_mpi.h
/// @brief Collective validation for native-dtype Frame wire plans.
#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

#include "atomiverse/cpu/communication/atom_pack.h"

#ifdef ATV_HAS_MPI
#include <mpi.h>

namespace ATV {
namespace cpu {
namespace comm {

/// Every rank completes both reductions before any rank can throw. A true
/// schema mismatch therefore fails collectively before point-to-point or
/// bulk-payload communication begins.
inline void require_collective_fingerprint(std::uint64_t fingerprint,
                                           MPI_Comm communicator,
                                           std::string_view operation) {
    std::uint64_t minimum = 0;
    std::uint64_t maximum = 0;
    MPI_Allreduce(&fingerprint, &minimum, 1, MPI_UINT64_T, MPI_MIN,
                  communicator);
    MPI_Allreduce(&fingerprint, &maximum, 1, MPI_UINT64_T, MPI_MAX,
                  communicator);
    if (minimum == maximum)
        return;
    throw std::runtime_error(std::string(operation)
                             + ": collective Frame schema mismatch "
                               "(fingerprint range "
                             + std::to_string(minimum) + " .. "
                             + std::to_string(maximum) + ")");
}

inline void require_collective_schema(const PackPlan& plan,
                                      MPI_Comm communicator,
                                      std::string_view operation) {
    require_collective_fingerprint(plan.fingerprint, communicator, operation);
}

/// @brief Collective agreement on a wire plan's schema, with a
///        discriminator so two uses of the same plan shape stay distinct.
/// @param plan Compiled wire plan.
/// @param communicator Communicator every participant calls this on.
/// @param operation Name used in the failure message.
/// @param discriminator Extra value folded into the digest.
inline void require_collective_plan(const PackPlan& plan, MPI_Comm communicator,
                                    std::string_view operation,
                                    std::uint64_t discriminator = 0) {
    require_collective_fingerprint(plan.wire_fingerprint(discriminator),
                                   communicator, operation);
}

inline int checked_mpi_byte_count(std::size_t bytes,
                                  std::string_view operation) {
    if (bytes > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::length_error(std::string(operation)
                                + ": byte count exceeds INT_MAX");
    return static_cast<int>(bytes);
}

} // namespace comm
} // namespace cpu
} // namespace ATV

#endif // ATV_HAS_MPI
