/// @file cpu/communication/gather_for_io.cpp
/// @brief Shared owner-gather preamble for root-writing recorders.

#include "atomiverse/cpu/communication/gather_for_io.h"

#include "atomiverse/cpu/communication/gather.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/frame/frame.h"

namespace ATV {
namespace cpu {
namespace md {
namespace mpi {

const cpu::Frame* gather_for_root_io(const cpu::Frame& frame,
                                     const cpu::Context& runtime,
                                     cpu::Frame& gathered) {
    const auto* dd = runtime.md.dd;
    const bool multi = dd && runtime.mpi_size > 1;
    if (!multi)
        return &frame;

    // Collective: every rank participates before the rank gate.
    const auto g = gather_to_root(frame, *dd);
    if (runtime.mpi_rank != 0)
        return nullptr;
    gathered = to_frame(g);
    gathered.meta.copy_serializable_from(frame.meta);
    return &gathered;
}

} // namespace mpi
} // namespace md
} // namespace cpu
} // namespace ATV
